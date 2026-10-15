from pathlib import Path

import pytest

from mambalite3d.data.io import CaseFiles
from mambalite3d.data.splits import pick_stratified, split_cases

HECKTOR_2025 = {"CHUM": 56, "CHUP": 72, "CHUS": 72, "HGJ": 55, "HMR": 18, "MDA": 396, "USZ": 11}


def _cases(counts=HECKTOR_2025):
    out = []
    for center, n in counts.items():
        for i in range(n):
            cid = f"{center}-{i:03d}"
            out.append(CaseFiles(cid, center, Path(cid + "__CT"), Path(cid + "__PT"), Path(cid)))
    return out


def _ids(xs):
    return {c.case_id for c in xs}


def test_stratified_70_20_10():
    cases = _cases()
    tr, va, te = split_cases(cases, {"mode": "stratified", "fractions": [0.7, 0.2, 0.1]}, seed=1)
    assert len(tr) + len(va) + len(te) == len(cases)
    assert not (_ids(tr) & _ids(va)) and not (_ids(tr) & _ids(te)) and not (_ids(va) & _ids(te))
    assert abs(len(tr) / len(cases) - 0.7) < 0.02 and abs(len(va) / len(cases) - 0.2) < 0.02 and abs(len(te) / len(cases) - 0.1) < 0.02
    for center in HECKTOR_2025:  # every center appears in every split
        assert all(any(c.center == center for c in part) for part in (tr, va, te))
    # deterministic
    tr2, va2, te2 = split_cases(cases, {"mode": "stratified", "fractions": [0.7, 0.2, 0.1]}, seed=1)
    assert _ids(tr) == _ids(tr2) and _ids(te) == _ids(te2)


def test_center_holdout():
    cases = _cases()
    tr, va, te = split_cases(cases, {"mode": "center", "fractions": [0.7, 0.2, 0.1], "test_centers": ["CHUP"]}, seed=1)
    assert {c.center for c in te} == {"CHUP"} and len(te) == 72
    assert "CHUP" not in {c.center for c in tr} | {c.center for c in va}
    assert len(tr) + len(va) + len(te) == len(cases)
    assert abs(len(va) / (len(tr) + len(va)) - 0.2 / 0.9) < 0.02  # train:val = 70:20


def test_center_holdout_rejects_unknown_center():
    with pytest.raises(ValueError):
        split_cases(_cases(), {"mode": "center", "test_centers": ["XYZ"]}, seed=0)


def test_small_subset_keeps_one_case_per_split():
    tr, va, te = split_cases(_cases({"A": 3, "B": 5}), {"mode": "stratified", "fractions": [0.7, 0.2, 0.1]}, seed=0)
    assert [len([c for c in p if c.center == "A"]) for p in (tr, va, te)] == [1, 1, 1]
    assert [len([c for c in p if c.center == "B"]) for p in (tr, va, te)] == [3, 1, 1]


def test_pick_stratified_round_robin():
    cases = _cases({"A": 10, "B": 10, "C": 1})
    picked = pick_stratified(cases, 7, seed=0)
    centers = [c.center for c in picked]
    assert len(picked) == 7 and centers.count("C") == 1 and centers.count("A") == 3 and centers.count("B") == 3


@pytest.mark.parametrize("mode", ["stratified", "loco"])
def test_cross_validation_folds_partition_the_data(mode):
    cases = _cases()
    tested = []
    for k in range(7):
        tr, va, te = split_cases(cases, {"mode": mode, "fractions": [0.7, 0.2, 0.1], "n_folds": 7}, seed=3, fold=k)
        a, b, c = _ids(tr), _ids(va), _ids(te)
        assert not (a & b or a & c or b & c) and len(a | b | c) == len(cases)
        tested += sorted(c)
        if mode == "loco":
            assert len({x.center for x in te}) == 1
        else:
            assert {x.center for x in te} == set(HECKTOR_2025)
            assert abs(len(va) / (len(tr) + len(va)) - 0.2 / 0.9) < 0.03
    assert sorted(tested) == sorted(_ids(cases))  # every case is tested exactly once


def test_fold_out_of_range():
    with pytest.raises(ValueError):
        split_cases(_cases(), {"mode": "stratified", "n_folds": 7}, seed=0, fold=7)
