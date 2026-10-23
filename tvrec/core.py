"""Small, independently testable metric, interval and output primitives."""
import csv
import heapq
import os
from pathlib import Path

import numpy as np


def ap_at_5(truth, predictions):
    """Binary relevance, min(5, |unique truth|) denominator; empty truth -> 0.

    Duplicate predictions consume a position but never earn a second hit.
    """
    truth = set(truth)
    if not truth:
        return 0.0
    hits = 0
    total = 0.0
    seen = set()
    for rank, item in enumerate(predictions[:5], 1):
        if item in truth and item not in seen:
            hits += 1
            total += hits / rank
        seen.add(item)
    return total / min(5, len(truth))


def map_at_5(truth, predictions, users):
    users = list(users)
    return (sum(ap_at_5(truth.get(u, set()), predictions[u]) for u in users)
            / len(users)) if users else None


def overlap_seconds(view_start, view_stop, airing_start, airing_end):
    return max(0, min(view_stop, airing_end) - max(view_start, airing_start))


def disjoint_airings(rows):
    """Resolve schedule conflicts without multiplying seconds.

    rows: (start, end, source_row_id). Latest start wins, then lowest source id.
    All rows participate, including unlabelled ones. Half-open intervals.
    Returns (segment_start, segment_end, source_row_id).
    """
    events = {}
    for start, end, sid in rows:
        if end > start:
            events.setdefault(start, []).append((end, sid))
            events.setdefault(end, [])
    heap = []
    result = []
    last = None
    for timestamp in sorted(events):
        while heap and heap[0][2] <= timestamp:
            heapq.heappop(heap)
        for end, sid in events[timestamp]:
            heapq.heappush(heap, (-timestamp, sid, end))
        if last is not None and last < timestamp:
            # The previous iteration's winner owns [last, timestamp).
            if winner is not None:
                if result and result[-1][1] == last and result[-1][2] == winner:
                    result[-1] = (result[-1][0], timestamp, winner)
                else:
                    result.append((last, timestamp, winner))
        winner = heap[0][1] if heap else None
        last = timestamp
    return result


def window_overlap(start, stop, lower, upper):
    """Clip first, before deriving interactions or features (prevents leakage)."""
    return max(start, lower), min(stop, upper)


def match_disjoint(vs, ve, starts, ends, lower, upper):
    """Vectorized binary-search join; return view index, segment index, overlap bounds.

    Schedule arrays must be sorted and non-overlapping, as produced by the sweep.
    Complexity O(V log S + actual overlaps), with no Cartesian product.
    """
    vs = np.maximum(np.asarray(vs), lower)
    ve = np.minimum(np.asarray(ve), upper)
    starts, ends = np.asarray(starts), np.asarray(ends)
    left = np.searchsorted(ends, vs, side='right')
    right = np.searchsorted(starts, ve, side='left')
    n = np.where(ve > vs, np.maximum(0, right-left), 0)
    vi = np.repeat(np.arange(len(n)), n)
    offsets = np.repeat(np.cumsum(n)-n, n)
    si = np.repeat(left, n)+np.arange(len(vi))-offsets
    first = np.maximum(vs[vi], starts[si])
    last = np.minimum(ve[vi], ends[si])
    return vi, si, first, last


def validate_submission(template, rows, candidates):
    if len(rows) != len(template):
        raise ValueError('wrong number of rows')
    for expected, (user, value) in zip(template, rows):
        if not isinstance(user, str) or user != expected:
            raise ValueError('user ID or row order changed')
        tokens = value.split(' ')
        if len(tokens) != 5 or len(set(tokens)) != 5:
            raise ValueError('need exactly five distinct tokens')
        if any(not t.isascii() or not t.isdecimal() or int(t) <= 0
               or int(t) not in candidates for t in tokens):
            raise ValueError('invalid or unavailable programme')


def write_submission(path, template, predictions, candidates):
    rows = [(u, ' '.join(str(int(i)) for i in predictions[u])) for u in template]
    validate_submission(template, rows, candidates)
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', 'tv_show_id'])
        writer.writerows(rows)
    with temporary.open(newline='') as f:
        reader = csv.DictReader(f)
        reread = [(r['user_id'], r['tv_show_id']) for r in reader]
    validate_submission(template, reread, candidates)
    os.replace(temporary, path)
