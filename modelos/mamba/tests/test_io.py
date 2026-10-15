import numpy as np
import SimpleITK as sitk

from mambalite3d.config import DEFAULTS
from mambalite3d.data.io import compute_roi, find_cases, label_coverage, load_case, read_lps, restore_to_original
from mambalite3d.data.sampler import CaseStore, PatchSampler


def _write_case(root, cid, spacing=(1.0, 1.0, 3.0), size=(160, 160, 120), origin=(-80.0, -80.0, -300.0), body_r=60.0, table=False):
    """Synthetic head-and-neck case: body cylinder, hot brain at the top, a tumour and a node.

    With table=True a thin wide slab (the scanner table) is added behind the patient.
    """
    d = root / "Task" / cid
    d.mkdir(parents=True)
    z, y, x = np.meshgrid(
        origin[2] + np.arange(size[2]) * spacing[2],
        origin[1] + np.arange(size[1]) * spacing[1],
        origin[0] + np.arange(size[0]) * spacing[0],
        indexing="ij",
    )
    body = x**2 + y**2 <= body_r**2
    brain = (x**2 + y**2 + ((z - 20) / 1.2) ** 2) <= 50**2
    tumour = (x - 5) ** 2 + (y + 10) ** 2 + (z + 120) ** 2 <= 12**2
    node = (x - 35) ** 2 + (y - 5) ** 2 + (z + 140) ** 2 <= 6**2
    ct = np.where(body, 40, -1000).astype(np.int16)
    if table:
        ct[(y >= body_r + 2) & (y <= body_r + 10) & (np.abs(x) <= 78)] = 200
    pet = np.where(body, 1.0, 0.0) + np.where(brain & body, 7.0, 0.0) + np.where(tumour, 12.0, 0.0) + np.where(node, 6.0, 0.0)
    lab = np.zeros(ct.shape, np.uint8)
    lab[tumour] = 1
    lab[node] = 2

    def img(a):
        im = sitk.GetImageFromArray(a)
        im.SetSpacing(spacing)
        im.SetOrigin(origin)
        return im

    sitk.WriteImage(img(ct), str(d / f"{cid}__CT.nii.gz"))
    sitk.WriteImage(img(pet.astype(np.float32)), str(d / f"{cid}__PT.nii.gz"))
    sitk.WriteImage(img(lab), str(d / f"{cid}.nii.gz"))
    return root / "Task"


def test_load_case_roi_and_restore(tmp_path):
    task = _write_case(tmp_path, "CHUM-001")
    cases = find_cases(task)
    assert [c.case_id for c in cases] == ["CHUM-001"] and cases[0].center == "CHUM"
    data_cfg = DEFAULTS["data"]
    case = load_case(cases[0], data_cfg)
    assert case.image.shape[0] == 2 and case.image.dtype == np.float16
    assert case.label.shape == case.image.shape[1:]
    assert set(np.unique(case.label)) == {0, 1, 2}
    assert 1 in case.fg_coords and 2 in case.fg_coords
    assert label_coverage(case) == 1.0
    ct = case.image[0].astype(np.float32)
    assert ct.min() >= 0.0 and ct.max() <= 1.0
    pred_o, gt_o, vox = restore_to_original(case, case.label)
    assert vox == 3.0
    assert np.array_equal(pred_o, gt_o)


def test_store_and_sampler(tmp_path):
    task = _write_case(tmp_path, "CHUS-002")
    store = CaseStore(find_cases(task), DEFAULTS["data"], threads=1, log=lambda *_: None)
    store.preload()
    sampler = PatchSampler(store, (10, 40, 40), fg_fraction=0.5, seed=0)
    imgs, labs = sampler.sample(4)
    assert imgs.shape == (4, 2, 10, 40, 40) and labs.shape == (4, 1, 10, 40, 40)
    assert labs[2:].max() > 0  # forced foreground samples contain a lesion
    store.close()


def test_roi_ignores_scanner_table(tmp_path):
    task = _write_case(tmp_path, "CHUP-003", body_r=50.0, table=True)
    files = find_cases(task)[0]
    roi_cfg = DEFAULTS["data"]["roi"]
    spacing = DEFAULTS["data"]["target_spacing_xyz"]
    ct, pet = read_lps(files.ct), read_lps(files.pet)
    lo, hi = compute_roi(ct, pet, roi_cfg, spacing)
    # left-right extent follows the body (radius 50) + margin + half coarse voxel + grid rounding,
    # not the 156 mm wide table
    assert hi[0] < 50 + 8 + 2 + 4 + 1 and lo[0] > -(50 + 8 + 2 + 4 + 1)
    lo2, hi2 = compute_roi(ct, pet, dict(roi_cfg, main_body=False), spacing)
    assert hi2[0] > 75 and lo2[0] < -75  # without the component filter the table widens the ROI
    case = load_case(files, DEFAULTS["data"])
    assert label_coverage(case) == 1.0


def test_iter_cases_streams_in_order(tmp_path):
    from mambalite3d.data.sampler import iter_cases

    for cid in ("CHUM-010", "CHUM-011", "CHUM-012"):
        _write_case(tmp_path, cid)
    files = find_cases(tmp_path / "Task")
    ids = [c.case_id for c in iter_cases(files, DEFAULTS["data"], threads=2)]
    assert ids == ["CHUM-010", "CHUM-011", "CHUM-012"]


def test_rolling_cache_swaps_cases(tmp_path):
    import time

    for cid in ("CHUV-001", "CHUV-002", "CHUV-003", "CHUV-004"):
        _write_case(tmp_path, cid, size=(80, 80, 60), origin=(-40.0, -40.0, -150.0), body_r=35.0)
    files = find_cases(tmp_path / "Task")
    store = CaseStore(files, DEFAULTS["data"], max_cases=2, threads=2, seed=0, log=lambda *_: None)
    store.preload()
    first = set(c.case_id for c in store.cases())
    assert len(first) == 2
    assert store.refresh(2) == 0  # schedules 2 background loads
    deadline = time.time() + 60
    while not all(f.done() for f in store.pending.values()) and time.time() < deadline:
        time.sleep(0.1)
    assert store.refresh(2) == 2
    now = set(c.case_id for c in store.cases())
    assert len(now) == 2 and not (now & first)  # both old cases were replaced by the new ones
    store.close()
