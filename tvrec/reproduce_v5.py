"""Fixed V5 replay in a NEW working directory; no search or Kaggle access.

This entry point is supplied for future reproduction, not run during finalization.
Input ingestion must first be run with `tvrec.pipeline prepare` in that directory.
"""
import gc
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .advanced import connect, metadata, metrics, save_predictions
from .core import write_submission
from .focused import extra_dataset
from .group_rank import fit_ranker, labels_for_pool, prediction_lists, validate_fold_groups
from .labels import cached_sessions, event_frame, training_features
from .ltr_experiment import matrix
from .models import ContentProfiles, ImplicitALS, matrix_from_features, popularity_scores
from .pipeline import LOW, SPLIT, TEST, END, save
from .rerank import PLAN, truth_for
from .rerank_data import EARLIER, available_items, cached_feature_dataset, feature_vocabulary


def base_scores(c, users, name, cutoff, upper):
    """Rebuild only the frozen four generators at this task's own cutoff."""
    training = event_frame(c, LOW, cutoff)
    items = available_items(c, cutoff, upper, name)
    f = training_features(training, minimum=.5, strength='frequency', cutoff=cutoff)
    model_users = sorted(set(training.user_id) | set(users))
    model_items = sorted(set(int(i) for i in training.item) | set(items))
    ui = {u: n for n, u in enumerate(model_users)}
    ii = {i: n for n, i in enumerate(model_items)}
    als = ImplicitALS(factors=32, alpha=10, regularization=1, iterations=12, seed=42)
    als.fit(matrix_from_features(f, model_users, model_items))
    mt, ma = metadata(c, cutoff, upper, model_items)
    cbf = ContentProfiles(rich=True, max_features=40000).fit_items(mt, ma)
    with threadpool_limits(limits=1):
        cf = np.maximum((als.user_factors[[ui[u] for u in users]] @
                         als.item_factors[[ii[i] for i in items]].T).astype(np.float32), 0)
        content, _ = cbf.score(matrix_from_features(f, users, model_items),
                               [ii[i] for i in items])
    scores = {'repeat': matrix_from_features(f, users, items).toarray().astype(np.float32),
              'popularity': popularity_scores(f, users, items), 'cf': cf, 'cbf': content}
    if name == 'final':
        folder = Path('cache/final/hybrid_v1')
        folder.mkdir(parents=True, exist_ok=True)
        save(folder/'mappings.json', {'template_users': users, 'test_candidates': items})
        np.savez_compressed(folder/'component_scores.npz', **scores)
    elif name in ('main', 'earlier'):
        folder = Path('cache/improvements')
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder/(name+'_v1_scores.npz'), **scores)
    else:
        folder = Path('cache/rerank')
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder/(name+'_base_scores.npz'), cf=cf, cbf=content)
    del training, f, als, cbf, scores
    gc.collect()


def fit(c, folds):
    xs, ys, groups = [], [], []
    for fold in folds:
        X, meta, z, _ = matrix(fold['name'])
        truth, _ = truth_for(c, fold['name'], fold['cutoff'], fold['target_end'], meta['users'])
        y = labels_for_pool(meta, z, truth)
        validate_fold_groups(z['user_index'], z['item_index'], np.array(meta['pool_lengths']), y)
        xs.append(X)
        ys.append(y)
        groups.extend(meta['pool_lengths'])
    model = fit_ranker(np.concatenate(xs), np.concatenate(ys), groups,
                       parameters={'lambdarank_truncation_level': 5})
    return model


def main():
    out = Path('artifacts/reproduction')
    # A second replay must use another fresh directory; archived CSVs are immutable.
    if out.exists() or Path('artifacts/final/submission_hybrid_v5.csv').exists():
        raise RuntimeError('Use a fresh replay directory, never the release or research workspace')
    out.mkdir(parents=True)
    c = connect(Path('cache'))
    cached_sessions(c, Path('cache'), out)
    users = [u for u, in c.execute('SELECT user_id FROM base.template ORDER BY source_row').fetchall()]
    folds = PLAN['training_folds'] + [
        {'name': 'earlier', 'cutoff': EARLIER, 'target_end': SPLIT},
        {'name': 'main', 'cutoff': SPLIT, 'target_end': TEST}]
    final = {'name': 'final', 'cutoff': TEST, 'target_end': END}
    vocab = feature_vocabulary(c)
    for fold in folds + [final]:
        base_scores(c, users, fold['name'], fold['cutoff'], fold['target_end'])
        cached_feature_dataset(c, Path('cache'), users, fold['name'],
                               fold['cutoff'], fold['target_end'], vocab)
        extra_dataset(c, fold['name'], out, blocks=['loyalty'])
    early = fit(c, folds[:2])
    results = []
    for fold in folds[2:]:
        X, meta, z, _ = matrix(fold['name'])
        pred = prediction_lists(meta, z, early.predict(X))
        truth, _ = truth_for(c, fold['name'], fold['cutoff'], fold['target_end'], users)
        save_predictions(out/(fold['name']+'_v5.csv'), pred, truth, users,
                         {u for n, u in enumerate(users) if not z['cold'][n]})
        results.append({'window': fold['name'], **metrics(truth, pred, users,
                        {u for n, u in enumerate(users) if not z['cold'][n]})})
    del early, X
    gc.collect()
    model = fit(c, folds)
    X, meta, z, names = matrix('final')
    pred = prediction_lists(meta, z, model.predict(X))
    write_submission(out/'submission_hybrid_v5.csv', users, pred, set(meta['items']))
    model.save_model(str(out/'lambda_top5_final.txt'))
    save(out/'replay.json', {'fixed_configuration_only': True, 'feature_names': names,
                            'holdout_results': results, 'training_folds': folds,
                            'no_kaggle_access': True,
                            'note': 'Compare output SHA-256 with release; cross-platform identity not promised.'})
    c.close()
    print(json.dumps({'output': str(out), 'holdout_results': results}, indent=2))


if __name__ == '__main__':
    main()
