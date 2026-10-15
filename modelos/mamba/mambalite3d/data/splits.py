"""Subset selection and train/validation/test splits.

Two modes (config `data.split`):
- "stratified": every center is divided train/val/test with the same fractions (in-distribution test).
- "center": the cases of `test_centers` form the test set (held-out institution, measures
  generalisation to an unseen scanner/protocol); the other centers are divided train/val so that the
  train:val ratio matches the requested fractions.
- "loco": leave-one-center-out cross-validation, fold f tests on the f-th center (sorted names).

Cross-validation (`n_folds` + `fold`): in "stratified" mode each center is cut into n_folds parts and
fold f uses part f as test (every case is tested exactly once over the folds); the rest is divided
train/val with the train:val ratio of `fractions`. In "loco" mode n_folds is the number of centers.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import List, Optional, Sequence, Tuple

from .io import CaseFiles


def select_subset(
    cases: Sequence[CaseFiles], per_center: Optional[int], centers: Optional[Sequence[str]], seed: int
) -> List[CaseFiles]:
    by_center = defaultdict(list)
    for c in cases:
        if c.label is None:
            continue
        if centers and c.center not in centers:
            continue
        by_center[c.center].append(c)
    rng = random.Random(seed)
    chosen = []
    for center in sorted(by_center):
        items = sorted(by_center[center], key=lambda c: c.case_id)
        rng.shuffle(items)
        chosen.extend(items if per_center is None else items[:per_center])
    return sorted(chosen, key=lambda c: c.case_id)


def _split_center(items, fracs, rng, min_one):
    """Splits one center's cases into len(fracs) parts; parts after the first get >= 1 case if min_one."""
    items = sorted(items, key=lambda c: c.case_id)
    rng.shuffle(items)
    n = len(items)
    counts = [int(round(n * f)) for f in fracs[1:]]
    if min_one and n >= len(fracs):
        counts = [max(k, 1) for k in counts]
    while sum(counts) > n - 1 and any(counts):  # keep at least one training case
        counts[counts.index(max(counts))] -= 1
    parts, start = [], n - sum(counts)
    parts.append(items[:start])
    for k in counts:
        parts.append(items[start : start + k])
        start += k
    return parts


def split_cases(cases: Sequence[CaseFiles], split_cfg: dict, seed: int, fold: Optional[int] = None):
    """Returns (train, val, test) lists of CaseFiles, deterministic for a given seed (and fold)."""
    mode = split_cfg.get("mode", "stratified")
    f_tr, f_va, f_te = (float(x) for x in split_cfg.get("fractions", [0.7, 0.2, 0.1]))
    n_folds = split_cfg.get("n_folds")
    fold = split_cfg.get("fold") if fold is None else fold
    by_center = defaultdict(list)
    for c in cases:
        by_center[c.center].append(c)
    rng = random.Random(seed + 1)
    train, val, test = [], [], []
    key = lambda c: c.case_id  # noqa: E731
    f_va_rest = f_va / (f_tr + f_va)
    if n_folds and mode in ("stratified", "loco"):
        centers = sorted(by_center)
        if mode == "loco":
            n_folds = len(centers)
        fold = 0 if fold is None else int(fold)
        if not 0 <= fold < n_folds:
            raise ValueError(f"fold {fold} out of range for {n_folds} folds")
        for i, center in enumerate(centers):
            items = sorted(by_center[center], key=key)
            random.Random(f"{seed}-{center}").shuffle(items)  # same order for every fold
            if mode == "loco":
                if i == fold:
                    test += items
                    continue
                rest = items
            else:
                test += [c for j, c in enumerate(items) if j % n_folds == fold]
                rest = [c for j, c in enumerate(items) if j % n_folds != fold]
            if rest:
                tr, va = _split_center(rest, [1 - f_va_rest, f_va_rest], random.Random(f"{seed}-{center}-{fold}"), min_one=True)
                train += tr
                val += va
        return sorted(train, key=key), sorted(val, key=key), sorted(test, key=key)
    if mode == "loco":
        raise ValueError("mode 'loco' needs n_folds (any value; it is set to the number of centers)")
    if mode == "stratified":
        for center in sorted(by_center):
            tr, va, te = _split_center(by_center[center], [f_tr, f_va, f_te], rng, min_one=f_te > 0)
            train += tr
            val += va
            test += te
    elif mode == "center":
        test_centers = set(split_cfg.get("test_centers") or [])
        unknown = test_centers - set(by_center)
        if not test_centers or unknown:
            raise ValueError(f"split.test_centers must name existing centers {sorted(by_center)}; got {sorted(test_centers)}")
        for center in sorted(by_center):
            if center in test_centers:
                test += by_center[center]
                continue
            tr, va = _split_center(by_center[center], [1 - f_va_rest, f_va_rest], rng, min_one=True)
            train += tr
            val += va
    else:
        raise ValueError(f"unknown split mode {mode!r} (use 'stratified', 'center' or 'loco')")
    return sorted(train, key=key), sorted(val, key=key), sorted(test, key=key)


def pick_stratified(cases: Sequence[CaseFiles], k: Optional[int], seed: int) -> List[CaseFiles]:
    """Up to k cases taken round-robin across centers (for the quick validation during training)."""
    if not k or k >= len(cases):
        return list(cases)
    by_center = defaultdict(list)
    for c in cases:
        by_center[c.center].append(c)
    rng = random.Random(seed + 2)
    queues = []
    for center in sorted(by_center):
        items = sorted(by_center[center], key=lambda c: c.case_id)
        rng.shuffle(items)
        queues.append(items)
    chosen = []
    while len(chosen) < k:
        for q in queues:
            if q and len(chosen) < k:
                chosen.append(q.pop())
    return sorted(chosen, key=lambda c: c.case_id)
