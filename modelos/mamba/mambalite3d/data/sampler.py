"""Cases kept in RAM (no disk cache) and random patch sampling with foreground oversampling."""
from __future__ import annotations

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence

import numpy as np

from .io import CaseData, CaseFiles, load_case


class CaseStore:
    """Holds preprocessed cases in RAM.

    With `max_cases` smaller than the number of cases it behaves like a rolling cache: `refresh()`
    loads new cases in background threads and swaps them in, so every case is seen over time while
    RAM stays bounded and nothing is written to disk.
    """

    def __init__(
        self,
        cases: Sequence[CaseFiles],
        data_cfg: dict,
        max_cases: Optional[int] = None,
        threads: int = 2,
        seed: int = 0,
        log=print,
    ):
        self.all_cases = list(cases)
        self.data_cfg = data_cfg
        self.max_cases = len(self.all_cases) if not max_cases else min(max_cases, len(self.all_cases))
        self.pool = ThreadPoolExecutor(max_workers=max(1, threads))
        self.rng = random.Random(seed)
        self.loaded: Dict[str, CaseData] = {}
        self.pending = {}
        self.lock = threading.Lock()
        self.log = log
        self.load_seconds: List[float] = []

    def _load(self, files: CaseFiles) -> CaseData:
        t0 = time.time()
        case = load_case(files, self.data_cfg, with_label=True, seed=self.rng.randint(0, 2**31 - 1))
        self.load_seconds.append(time.time() - t0)
        return case

    def preload(self) -> None:
        order = self.all_cases[:]
        self.rng.shuffle(order)
        first = order[: self.max_cases]
        futures = [self.pool.submit(self._load, f) for f in first]
        for i, fut in enumerate(futures, 1):
            case = fut.result()
            self.loaded[case.case_id] = case
            if i % 5 == 0 or i == len(futures):
                self.log(f"  cargados {i}/{len(futures)} casos")

    def refresh(self, n_replace: int) -> int:
        """Collects finished background loads and schedules `n_replace` new ones (rolling cache).

        Returns the number of new cases swapped into the cache by this call.
        """
        if self.max_cases >= len(self.all_cases):
            return 0
        with self.lock:
            done = [cid for cid, fut in self.pending.items() if fut.done()]
            old_ids = list(self.loaded)
            self.rng.shuffle(old_ids)  # evict random cases that were already in the cache
            for cid in done:
                case = self.pending.pop(cid).result()
                if len(self.loaded) >= self.max_cases and old_ids:
                    del self.loaded[old_ids.pop()]
                self.loaded[case.case_id] = case
            candidates = [c for c in self.all_cases if c.case_id not in self.loaded and c.case_id not in self.pending]
            self.rng.shuffle(candidates)
            for files in candidates[: max(0, n_replace - len(self.pending))]:
                self.pending[files.case_id] = self.pool.submit(self._load, files)
        return len(done)

    def cases(self) -> List[CaseData]:
        return list(self.loaded.values())

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


def iter_cases(files: Sequence[CaseFiles], data_cfg: dict, threads: int = 2, with_label: bool = True):
    """Yields cases one at a time while the next ones load in background threads.

    RAM holds at most `threads` + 1 cases, so a whole validation/test set can be evaluated
    without keeping it in memory (and without writing anything to disk).
    """
    files = list(files)
    threads = max(1, threads)
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(load_case, f, data_cfg, with_label) for f in files[:threads]]
        for i in range(len(files)):
            case = futures[i].result()
            futures[i] = None
            if len(futures) < len(files):
                futures.append(pool.submit(load_case, files[len(futures)], data_cfg, with_label))
            yield case


def _extract(arr: np.ndarray, center: Sequence[int], size: Sequence[int], pad_value) -> np.ndarray:
    """Crops `size` voxels around `center` from the last three axes, padding outside the volume."""
    spatial = arr.shape[-3:]
    lo = [int(center[i]) - size[i] // 2 for i in range(3)]
    hi = [lo[i] + size[i] for i in range(3)]
    src_lo = [max(lo[i], 0) for i in range(3)]
    src_hi = [min(hi[i], spatial[i]) for i in range(3)]
    dst_lo = [src_lo[i] - lo[i] for i in range(3)]
    lead = arr.shape[:-3]
    out = np.empty(lead + tuple(size), dtype=arr.dtype)
    if isinstance(pad_value, (list, tuple)):
        for c, v in enumerate(pad_value):
            out[c] = v
    else:
        out[...] = pad_value
    if all(src_hi[i] > src_lo[i] for i in range(3)):
        out[
            ...,
            dst_lo[0] : dst_lo[0] + src_hi[0] - src_lo[0],
            dst_lo[1] : dst_lo[1] + src_hi[1] - src_lo[1],
            dst_lo[2] : dst_lo[2] + src_hi[2] - src_lo[2],
        ] = arr[..., src_lo[0] : src_hi[0], src_lo[1] : src_hi[1], src_lo[2] : src_hi[2]]
    return out


class PatchSampler:
    """nnU-Net-like sampling: the last round(B * fg_fraction) samples of a batch are centred on a
    random foreground voxel (random class among those present); the others anywhere in the ROI."""

    def __init__(self, store: CaseStore, ext_zyx: Sequence[int], fg_fraction: float = 0.33, seed: int = 0):
        self.store = store
        self.ext = [int(s) for s in ext_zyx]
        self.fg_fraction = fg_fraction
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_size: int):
        cases = self.store.cases()
        n_fg = int(round(batch_size * self.fg_fraction)) if self.fg_fraction > 0 else 0
        imgs = np.empty((batch_size, 2, *self.ext), dtype=np.float16)
        labs = np.zeros((batch_size, 1, *self.ext), dtype=np.uint8)
        for b in range(batch_size):
            case = cases[self.rng.integers(len(cases))]
            shape = case.image.shape[1:]
            force_fg = b >= batch_size - n_fg
            classes = [c for c in (1, 2) if c in case.fg_coords]
            if force_fg and classes:
                coords = case.fg_coords[classes[self.rng.integers(len(classes))]]
                center = coords[self.rng.integers(len(coords))]
            else:
                center = [self.rng.integers(0, s) for s in shape]
            imgs[b] = _extract(case.image, center, self.ext, case.pad_values)
            if case.label is not None:
                labs[b, 0] = _extract(case.label, center, self.ext, 0)
        return imgs, labs
