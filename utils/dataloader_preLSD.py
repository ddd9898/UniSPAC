"""
3D EM dataloaders that read pre-computed LSD caches instead of computing LSD on-the-fly.

Workflow:
    1. Run ``process_lsd.py`` once to precompute and store full-volume LSD .npy files, or
       convert them to chunked Zarr with ``tran_lsd_to_zarr.py`` for better I/O.
    2. Use the Dataset_3D_* classes in this module (drop-in replacements for those in
       ``dataloader.py``) to train with the cached LSD (``lsd_storage=\"zarr\"`` or ``\"npy\"``).

Cache format: for each (dataset_key, source_name, orientation) the cache is a numpy array
of shape ``(10, A, B, Z_full)`` where A/B are the two spatial "crop" axes and Z_full is the
full "slice" axis length.  The last spatial axis always corresponds to the first axis of the
stored ``images[ii]`` slice-stack, so slicing into it at ``z_offset + z0`` gives the patch.

Orientations:
    xy  →  full_ZHW processed as (H, W, Z)  →  cache (10, H, W, Z)
    xz  →  full_ZHW.T(1,0,2) → (H,Z,W), then T(1,2,0) → (Z, W, H)  →  cache (10, Z, W, H)
    yz  →  full_ZHW.T(1,2,0) → (H,W,Z), then T(1,2,0) → (W, Z, H)  →  cache (10, W, Z, H)

In all cases the last axis of the cache is the "slicing" dimension and z_offset + z0 gives
the correct position in that axis.
"""
from __future__ import annotations

import logging
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import tifffile
import zarr
from PIL import Image
from skimage.measure import label as connected_components

from lsd.train import local_shape_descriptor

from utils.dataloader import (
    AC3_DANIEL_SEG_BASENAME,
    AC3_DEFAULT_DATA_DIR,
    AC3_NUM_SLICES,
    AC4_DANIEL_SEG_BASENAME,
    AC4_DEFAULT_DATA_DIR,
    AC4_NUM_SLICES,
    AXONEM_H_DEFAULT_DATA_DIR,
    AXONEM_M_DEFAULT_DATA_DIR,
    CREMI_HDF_FILES,
    FIB25_ZARR_FILES,
    HEMI_DEFAULT_DATA_DIR,
    HEMI_ZARR_FILES,
    ISBI2012_DEFAULT_DATA_DIR,
    ISBI2012_TIFF_PAIRS,
    MICRONS_BASIL_ZARR_DIR,
    MICRONS_MINNIE_ZARR_DIR,
    MICRONS_PINKY_ZARR_DIR,
    ZEBRA_PATCH_FILES,
    ZEBRAFINCH_DEFAULT_DATA_DIR,
    VNC_DEFAULT_DATA_DIR,
    EM3DDataset,
    _build_3d_patch_index,
    _membrane_png_for_index,
    _vnc_raw_index_paths,
    _axonem_h5_pick_main_3d_dataset,
    affinity_3d,
    axonem_h5_paired_volume_keys,
    collate_fn_3D_hemi_Train,
    erode_instance_labels,
    gaussian_point_map_3d_try,
    load_ac3_volume,
    load_ac4_volume,
    load_axonem_h5_volume,
    load_isbi2012_volume_pair,
    load_microns_neuron_zarr_volume,
    load_vnc_stack1_volume,
    mask_and_points_3d,
    microns_neuron_zarr_stems,
    normalize_minmax,
    padding_for_crop,
    pinky_zarr_volume_keys,
)

_LOG = logging.getLogger(__name__)

# Full LSD float32 volumes keyed by cache path (train/val/test datasets share one copy).
_LSD_RAM_GLOBAL: Dict[str, Optional[np.ndarray]] = {}

# ---------------------------------------------------------------------------
# Optional __getitem__ profiling (see set_getitem_profile)
# ---------------------------------------------------------------------------

_PROFILE_ENV = "UNISPACE_PROFILE_DATA_N"
_item_prof_done = 0
_item_prof_sums: defaultdict = defaultdict(float)


def set_getitem_profile(num_samples: int) -> None:
    """Time the first ``num_samples`` ``__getitem__`` calls per DataLoader worker.

    Writes human-readable lines to stderr (visible even when root logging is WARNING).
    Sets env UNISPACE_PROFILE_DATA_N so fork/spawn DataLoader workers inherit the limit.
    """
    global _item_prof_done, _item_prof_sums
    n = max(0, int(num_samples))
    if n > 0:
        os.environ[_PROFILE_ENV] = str(n)
        _item_prof_done = 0
        _item_prof_sums = defaultdict(float)
    else:
        os.environ.pop(_PROFILE_ENV, None)


def _profile_getitem_limit() -> int:
    try:
        return max(0, int(os.environ.get(_PROFILE_ENV, "0")))
    except ValueError:
        return 0


def _profile_getitem_finish(stages: dict[str, float]) -> None:
    global _item_prof_done, _item_prof_sums
    lim = _profile_getitem_limit()
    if lim <= 0 or _item_prof_done >= lim:
        return
    _item_prof_done += 1
    for k, v in stages.items():
        _item_prof_sums[k] += v
    pid = os.getpid()
    if _item_prof_done <= min(5, lim):
        detail = " ".join(
            "%s=%.1fms" % (k, stages[k] * 1000) for k in sorted(stages)
        )
        print(
            "[data_profile pid=%d] __getitem__ %d/%d %s"
            % (pid, _item_prof_done, lim, detail),
            file=sys.stderr,
            flush=True,
        )
    if _item_prof_done == lim:
        av = {k: _item_prof_sums[k] / lim for k in _item_prof_sums}
        detail = " ".join(
            "%s=%.1fms" % (k, av[k] * 1000) for k in sorted(av)
        )
        print(
            "[data_profile pid=%d] __getitem__ AVG over %d: %s"
            % (pid, lim, detail),
            file=sys.stderr,
            flush=True,
        )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LSD_CACHE_DIR = "/data/UniSPAC/LSD_cache"

ORIENTATION_AXIS_PERMUTATIONS = {
    "xy": (0, 1, 2),
    "xz": (2, 1, 0),
    "yz": (1, 2, 0),
}


def _descriptor_channel_permutation(axis_permutation: Tuple[int, int, int]) -> Tuple[int, ...]:
    means = list(axis_permutation)
    variances = [3 + axis for axis in axis_permutation]
    pair_to_channel = {
        (0, 1): 6,
        (0, 2): 7,
        (1, 2): 8,
    }
    pearsons = [
        pair_to_channel[tuple(sorted((axis_permutation[0], axis_permutation[1])))],
        pair_to_channel[tuple(sorted((axis_permutation[0], axis_permutation[2])))],
        pair_to_channel[tuple(sorted((axis_permutation[1], axis_permutation[2])))],
    ]
    return tuple(means + variances + pearsons + [9])


ORIENTATION_CHANNEL_PERMUTATIONS = {
    orientation: _descriptor_channel_permutation(axis_permutation)
    for orientation, axis_permutation in ORIENTATION_AXIS_PERMUTATIONS.items()
}

# ---------------------------------------------------------------------------
# Cache path utilities
# ---------------------------------------------------------------------------


def make_lsd_volume_spec(
    *,
    dataset_key: str,
    source_name: str,
    split: str,
    n_val: int,
    orientation: str,
    volume_shape_zyx: tuple,
) -> dict:
    """Return a spec dict identifying a single LSD cache entry."""
    return {
        "dataset_key": dataset_key,
        "source_name": source_name,
        "split": split,
        "n_val": n_val,
        "orientation": orientation,
        "volume_shape_zyx": volume_shape_zyx,
    }


def _lsd_cache_basename(spec: dict) -> str:
    src = spec["source_name"]
    for ext in (".zarr", ".hdf", ".h5", ".tif", ".tiff"):
        if src.lower().endswith(ext):
            src = src[: -len(ext)]
    src = src.replace("/", "_").replace("\\", "_")
    return "{}__{}__{}".format(spec["dataset_key"], src, spec["orientation"])


def lsd_cache_path(cache_dir: str, spec: dict) -> str:
    """Stable filesystem path for a cached LSD numpy array (.npy)."""
    fname = "{}.npy".format(_lsd_cache_basename(spec))
    return os.path.join(cache_dir, fname)


def lsd_cache_zarr_path(cache_dir: str, spec: dict) -> str:
    """Stable filesystem path for a chunked LSD zarr store (.zarr directory)."""
    fname = "{}.zarr".format(_lsd_cache_basename(spec))
    return os.path.join(cache_dir, fname)


def lsd_cache_storage_path(cache_dir: str, spec: dict, *, storage: str = "zarr") -> str:
    """Path for precomputed LSD: ``storage`` is ``\"zarr\"`` or ``\"npy\"``."""
    if storage == "zarr":
        return lsd_cache_zarr_path(cache_dir, spec)
    if storage == "npy":
        return lsd_cache_path(cache_dir, spec)
    raise ValueError("storage must be 'zarr' or 'npy', got {!r}".format(storage))


# ---------------------------------------------------------------------------
# Dataset-source enumeration and raw/label loading
# (used by process_lsd.py to know which files to precompute)
# ---------------------------------------------------------------------------


def list_dataset_source_names(
    dataset_key: str,
    data_dir: str,
    *,
    zebra_data_idxs=None,
) -> List[str]:
    """Return source names for a given dataset key (one per physical volume file/cube)."""
    if dataset_key == "hemi":
        return list(HEMI_ZARR_FILES)
    if dataset_key == "fib25":
        return list(FIB25_ZARR_FILES)
    if dataset_key == "cremi":
        return list(CREMI_HDF_FILES)
    if dataset_key == "vnc":
        return ["stack1"]
    if dataset_key == "isbi2012":
        return [pair[0] for pair in ISBI2012_TIFF_PAIRS]
    if dataset_key == "ac3":
        return ["ac3"]
    if dataset_key == "ac4":
        return ["ac4"]
    if dataset_key == "basil":
        return list(microns_neuron_zarr_stems(data_dir))
    if dataset_key == "minnie":
        return list(microns_neuron_zarr_stems(data_dir))
    if dataset_key == "pinky":
        return list(pinky_zarr_volume_keys(data_dir))
    if dataset_key == "axonem_h":
        return list(axonem_h5_paired_volume_keys(data_dir))
    if dataset_key == "axonem_m":
        return list(axonem_h5_paired_volume_keys(data_dir))
    if dataset_key == "zebrafinch":
        idxs = (
            zebra_data_idxs
            if zebra_data_idxs is not None
            else range(len(ZEBRA_PATCH_FILES))
        )
        return [ZEBRA_PATCH_FILES[i] for i in idxs]
    raise ValueError("Unknown dataset_key: {}".format(dataset_key))


def load_dataset_source_volume(
    dataset_key: str,
    data_dir: str,
    source_name: str,
    load_raw: bool = True,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Load raw+label for one source as (Z, H, W) numpy arrays.

    Applies the same pre-processing (bbox crop, deletion of bad slices, etc.) that the
    training dataloaders in ``dataloader.py`` apply, so that the LSD cache is computed in
    exactly the same coordinate space as the training patches.
    """
    if dataset_key == "hemi":
        root = zarr.open(os.path.join(data_dir, source_name), mode="r")
        lab = root["volumes"]["labels"]["neuron_ids"]
        # Same 128-voxel border crop applied in Dataset_3D_hemi_Train
        raw = None
        if load_raw:
            raw = np.asarray(
                root["volumes"]["raw"][
                    128 : int(root["volumes"]["raw"].shape[0] - 128),
                    128 : int(root["volumes"]["raw"].shape[1] - 128),
                    128 : int(root["volumes"]["raw"].shape[2] - 128),
                ]
            )
        lab = np.asarray(
            lab[
                128 : int(lab.shape[0] - 128),
                128 : int(lab.shape[1] - 128),
                128 : int(lab.shape[2] - 128),
            ]
        )
        lab = connected_components(lab).astype(np.uint16)
        return raw, lab

    if dataset_key == "fib25":
        root = zarr.open(os.path.join(data_dir, source_name), mode="r")
        raw = np.asarray(root["volumes"]["raw"]) if load_raw else None
        lab = connected_components(
            np.asarray(root["volumes"]["labels"]["neuron_ids"])
        ).astype(np.uint16)
        return raw, lab

    if dataset_key == "cremi":
        path = os.path.join(data_dir, source_name)
        with h5py.File(path, "r") as f:
            raw = np.asarray(f["volumes"]["raw"]) if load_raw else None
            lab = np.asarray(f["volumes"]["labels"]["neuron_ids"])
        if source_name == "sample_C_20160501.hdf":
            if raw is not None:
                raw = np.delete(raw, [14, 74], 0)
            lab = np.delete(lab, [14, 74], 0)
        lab = connected_components(lab).astype(np.uint16)
        return raw, lab

    if dataset_key == "vnc":
        if load_raw:
            return load_vnc_stack1_volume(data_dir)
        idx_map = _vnc_raw_index_paths(data_dir)
        indices = sorted(idx_map.keys())
        if not indices:
            raise FileNotFoundError("No .tif slices under %s/raw" % data_dir)
        labs = []
        for i in indices:
            mem = np.array(Image.open(_membrane_png_for_index(data_dir, i)))
            if mem.ndim == 3:
                mem = mem[..., 0]
            labs.append((mem > 0).astype(np.uint16))
        return None, np.stack(labs, axis=0)

    if dataset_key == "isbi2012":
        for raw_name, lab_name in ISBI2012_TIFF_PAIRS:
            if raw_name == source_name:
                raw = None
                if load_raw:
                    raw = np.asarray(tifffile.imread(os.path.join(data_dir, raw_name)), dtype=np.float32)
                lab_tif = np.asarray(tifffile.imread(os.path.join(data_dir, lab_name)))
                fg = (lab_tif != 0).astype(np.uint8)
                lab = connected_components(fg).astype(np.uint16)
                if raw is not None and raw.shape != lab_tif.shape:
                    raise ValueError(
                        "ISBI-2012 raw %r shape %s != labels %r shape %s"
                        % (raw_name, raw.shape, lab_name, lab_tif.shape)
                    )
                return raw, lab
        raise ValueError("Unknown ISBI2012 source: {}".format(source_name))

    if dataset_key == "ac3":
        if load_raw:
            return load_ac3_volume(data_dir)
        seg_dir = os.path.join(data_dir, "ac3_dbseg_images")
        if not os.path.isdir(seg_dir):
            raise FileNotFoundError("AC3 segmentation dir not found: %s" % seg_dir)
        labs: List[np.ndarray] = []
        color_to_id: dict = {}
        next_id = 1
        for z in range(AC3_NUM_SLICES):
            s_num = AC3_NUM_SLICES - z
            seg_path = os.path.join(seg_dir, AC3_DANIEL_SEG_BASENAME % s_num)
            if not os.path.isfile(seg_path):
                raise FileNotFoundError("Missing AC3 Daniel seg slice: %s" % seg_path)
            rgb = np.asarray(Image.open(seg_path))
            if rgb.ndim == 2:
                rgb = np.stack([rgb, rgb, rgb], axis=-1)
            elif rgb.shape[-1] >= 4:
                rgb = rgb[..., :3]
            r = rgb[..., 0].astype(np.uint32, copy=False)
            g = rgb[..., 1].astype(np.uint32, copy=False)
            b = rgb[..., 2].astype(np.uint32, copy=False)
            packed = (r << 16) | (g << 8) | b
            flat = packed.ravel()
            uniq, inv = np.unique(flat, return_inverse=True)
            gids = np.zeros(len(uniq), dtype=np.uint16)
            for j, u in enumerate(uniq):
                if u == 0:
                    continue
                if u not in color_to_id:
                    if next_id > 65535:
                        raise ValueError(
                            "AC3: more than 65535 distinct RGB labels; cannot store as uint16"
                        )
                    color_to_id[u] = next_id
                    next_id += 1
                gids[j] = color_to_id[u]
            labs.append(gids[inv].reshape(packed.shape))
        return None, np.stack(labs, axis=0)

    if dataset_key == "ac4":
        if load_raw:
            return load_ac4_volume(data_dir)
        seg_dir = os.path.join(data_dir, "ac4_seg_daniel")
        if not os.path.isdir(seg_dir):
            raise FileNotFoundError("AC4 Daniel seg dir not found: %s" % seg_dir)
        labs: List[np.ndarray] = []
        color_to_id: dict = {}
        next_id = 1
        for z in range(AC4_NUM_SLICES):
            s_num = AC4_NUM_SLICES - z
            seg_path = os.path.join(seg_dir, AC4_DANIEL_SEG_BASENAME % s_num)
            if not os.path.isfile(seg_path):
                raise FileNotFoundError("Missing AC4 Daniel seg slice: %s" % seg_path)
            rgb = np.asarray(Image.open(seg_path))
            if rgb.ndim == 2:
                rgb = np.stack([rgb, rgb, rgb], axis=-1)
            elif rgb.shape[-1] >= 4:
                rgb = rgb[..., :3]
            r = rgb[..., 0].astype(np.uint32, copy=False)
            g = rgb[..., 1].astype(np.uint32, copy=False)
            b = rgb[..., 2].astype(np.uint32, copy=False)
            packed = (r << 16) | (g << 8) | b
            flat = packed.ravel()
            uniq, inv = np.unique(flat, return_inverse=True)
            gids = np.zeros(len(uniq), dtype=np.uint16)
            for j, u in enumerate(uniq):
                if u == 0:
                    continue
                if u not in color_to_id:
                    if next_id > 65535:
                        raise ValueError(
                            "AC4: more than 65535 distinct RGB labels; cannot store as uint16"
                        )
                    color_to_id[u] = next_id
                    next_id += 1
                gids[j] = color_to_id[u]
            labs.append(gids[inv].reshape(packed.shape))
        return None, np.stack(labs, axis=0)

    if dataset_key in ("basil", "minnie", "pinky"):
        if load_raw:
            return load_microns_neuron_zarr_volume(data_dir, source_name)
        path = os.path.join(os.path.abspath(data_dir), source_name + ".zarr")
        if not os.path.isdir(path):
            raise FileNotFoundError("MICrONS neuron zarr not found: %s" % path)
        root = zarr.open(path, mode="r")
        lab = np.asarray(root["volumes"]["labels"])
        mx = int(lab.max())
        if mx >= 65536:
            raise ValueError("MICrONS zarr %s: label max %d exceeds uint16" % (source_name, mx))
        return None, lab.astype(np.uint16)

    if dataset_key in ("axonem_h", "axonem_m"):
        if load_raw:
            return load_axonem_h5_volume(data_dir, source_name)
        d = os.path.abspath(data_dir)
        seg_path = os.path.join(d, "seg_%s_pad.h5" % source_name)
        if not os.path.isfile(seg_path):
            raise FileNotFoundError("AxonEM seg not found: %s" % seg_path)
        seg_ds = _axonem_h5_pick_main_3d_dataset(seg_path)
        with h5py.File(seg_path, "r") as f:
            lab = np.asarray(f[seg_ds])
        mx = int(lab.max())
        if mx >= 65536:
            raise ValueError("AxonEM %s: label max %d exceeds uint16" % (source_name, mx))
        if np.any(lab > 0):
            zz, yy, xx = np.where(lab > 0)
            z0, z1 = int(zz.min()), int(zz.max()) + 1
            y0, y1 = int(yy.min()), int(yy.max()) + 1
            x0, x1 = int(xx.min()), int(xx.max()) + 1
            Z, Y, X = lab.shape
            margin = 8
            z0 = max(0, z0 - margin)
            y0 = max(0, y0 - margin)
            x0 = max(0, x0 - margin)
            z1 = min(Z, z1 + margin)
            y1 = min(Y, y1 + margin)
            x1 = min(X, x1 + margin)
            lab = lab[z0:z1, y0:y1, x0:x1]
        return None, lab.astype(np.uint16)

    if dataset_key == "zebrafinch":
        root = zarr.open(os.path.join(data_dir, source_name), mode="r")
        lab = connected_components(
            np.asarray(root["volumes"]["labels"]["neuron_ids"])
        ).astype(np.uint16)
        raw = None
        if load_raw:
            raw = np.asarray(
                root["volumes"]["raw"][
                    100 : 100 + lab.shape[0],
                    200 : 200 + lab.shape[1],
                    200 : 200 + lab.shape[2],
                ],
                dtype=np.float32,
            )
        return raw, lab

    raise ValueError("Unknown dataset_key: {}".format(dataset_key))


# ---------------------------------------------------------------------------
# Augmentation helpers that keep raw / lab / lsd in sync
# ---------------------------------------------------------------------------


def prepare_3d_pair_with_lsd(
    raw: np.ndarray,
    lab: np.ndarray,
    lsd: Optional[np.ndarray],
    crop_size: int,
    pad_total: int,
    *,
    augment: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Crop and (optionally) augment raw, lab, and LSD patch simultaneously.

    Args:
        raw:  (H, W, Z) float32 – normalised EM volume patch
        lab:  (H, W, Z) uint16  – instance labels (not yet eroded)
        lsd:  (10, H_full, W_full, Z) float32 or None – LSD sub-volume extracted
              from the full-volume cache at the correct Z slice range.  May be
              wider/taller than the target crop_size (we crop it here).
        crop_size: target spatial size (square HW crop)
        pad_total:  padding target (same convention as ``prepare_2d_pair``)
        augment:    if True apply random crop + geometric+brightness augmentations;
                    if False use centre crop only.

    Returns:
        (raw_out, lab_out, lsd_out) all with spatial dims (crop_size, crop_size, Z).
        lsd_out is None if lsd was None on input.
    """
    H, W = raw.shape[:2]

    # Pad raw/lab (and LSD) so they are at least crop_size in HW
    pad_h = max(0, crop_size - H)
    pad_w = max(0, crop_size - W)
    if pad_h > 0 or pad_w > 0:
        raw = np.pad(raw, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
        lab = np.pad(lab, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
        if lsd is not None:
            lsd = np.pad(
                lsd, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode="constant"
            )
        H = H + pad_h
        W = W + pad_w

    # Spatial crop
    if augment:
        y0 = random.randint(0, H - crop_size)
        x0 = random.randint(0, W - crop_size)
    else:
        y0 = (H - crop_size) // 2
        x0 = (W - crop_size) // 2

    raw = raw[y0 : y0 + crop_size, x0 : x0 + crop_size]
    lab = lab[y0 : y0 + crop_size, x0 : x0 + crop_size]
    if lsd is not None:
        lsd = np.array(
            lsd[:, y0 : y0 + crop_size, x0 : x0 + crop_size], dtype=np.float32
        )

    if not augment:
        return raw, lab, lsd

    # --- Geometric augmentations (applied identically to raw, lab, lsd) ---

    # HorizontalFlip: flip H axis (axis 0)
    if random.random() < 0.3:
        raw = raw[::-1].copy()
        lab = lab[::-1].copy()
        if lsd is not None:
            lsd = lsd[:, ::-1].copy()

    # VerticalFlip: flip W axis (axis 1)
    if random.random() < 0.3:
        raw = raw[:, ::-1].copy()
        lab = lab[:, ::-1].copy()
        if lsd is not None:
            lsd = lsd[:, :, ::-1].copy()

    # RandomRotate90: rotate in HW plane
    if random.random() < 0.3:
        k = random.randint(1, 3)
        raw = np.rot90(raw, k, axes=(0, 1)).copy()
        lab = np.rot90(lab, k, axes=(0, 1)).copy()
        if lsd is not None:
            # LSD spatial dims are axes 1 and 2
            lsd = np.rot90(lsd, k, axes=(1, 2)).copy()

    # Transpose: swap H and W
    if random.random() < 0.3:
        raw = np.transpose(raw, (1, 0, 2)).copy()
        lab = np.transpose(lab, (1, 0, 2)).copy()
        if lsd is not None:
            lsd = np.transpose(lsd, (0, 2, 1, 3)).copy()

    # RandomBrightnessContrast: raw only, same range as albumentations default
    if random.random() < 0.3:
        alpha = random.uniform(0.8, 1.2)
        beta = random.uniform(-0.2, 0.2)
        raw = np.clip(raw.astype(np.float32) * alpha + beta, 0.0, 1.0).astype(
            np.float32
        )

    return raw, lab, lsd


def _sample_3d_pair_plan(
    raw_shape: Tuple[int, int, int],
    crop_size: int,
    *,
    augment: bool = True,
) -> dict:
    """Sample crop/augmentation parameters once so raw/lab/LSD stay aligned."""
    H, W = raw_shape[:2]
    pad_h = max(0, crop_size - H)
    pad_w = max(0, crop_size - W)
    H_pad = H + pad_h
    W_pad = W + pad_w

    if augment:
        y0 = random.randint(0, H_pad - crop_size)
        x0 = random.randint(0, W_pad - crop_size)
        flip_h = random.random() < 0.3
        flip_w = random.random() < 0.3
        rot_k = random.randint(1, 3) if random.random() < 0.3 else 0
        do_transpose = random.random() < 0.3
        if random.random() < 0.3:
            brightness = (
                random.uniform(0.8, 1.2),
                random.uniform(-0.2, 0.2),
            )
        else:
            brightness = None
    else:
        y0 = (H_pad - crop_size) // 2
        x0 = (W_pad - crop_size) // 2
        flip_h = False
        flip_w = False
        rot_k = 0
        do_transpose = False
        brightness = None

    return {
        "crop_size": crop_size,
        "pad_h": pad_h,
        "pad_w": pad_w,
        "y0": y0,
        "x0": x0,
        "flip_h": flip_h,
        "flip_w": flip_w,
        "rot_k": rot_k,
        "transpose_hw": do_transpose,
        "brightness": brightness,
    }


def _apply_3d_pair_plan(
    raw: np.ndarray,
    lab: np.ndarray,
    lsd: Optional[np.ndarray],
    plan: dict,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Apply a pre-sampled crop/augmentation plan to raw/lab/LSD."""
    crop_size = int(plan["crop_size"])
    pad_h = int(plan["pad_h"])
    pad_w = int(plan["pad_w"])
    y0 = int(plan["y0"])
    x0 = int(plan["x0"])

    if pad_h > 0 or pad_w > 0:
        raw = np.pad(raw, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
        lab = np.pad(lab, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
        if lsd is not None and (lsd.shape[1] != crop_size or lsd.shape[2] != crop_size):
            lsd = np.pad(
                lsd,
                ((0, 0), (0, crop_size - lsd.shape[1]), (0, crop_size - lsd.shape[2]), (0, 0)),
                mode="constant",
            )

    raw = raw[y0 : y0 + crop_size, x0 : x0 + crop_size]
    lab = lab[y0 : y0 + crop_size, x0 : x0 + crop_size]
    if lsd is not None and (lsd.shape[1] != crop_size or lsd.shape[2] != crop_size):
        lsd = np.array(lsd[:, y0 : y0 + crop_size, x0 : x0 + crop_size], dtype=np.float32)

    if plan["flip_h"]:
        raw = raw[::-1].copy()
        lab = lab[::-1].copy()
        if lsd is not None:
            lsd = lsd[:, ::-1].copy()

    if plan["flip_w"]:
        raw = raw[:, ::-1].copy()
        lab = lab[:, ::-1].copy()
        if lsd is not None:
            lsd = lsd[:, :, ::-1].copy()

    if plan["rot_k"]:
        raw = np.rot90(raw, plan["rot_k"], axes=(0, 1)).copy()
        lab = np.rot90(lab, plan["rot_k"], axes=(0, 1)).copy()
        if lsd is not None:
            lsd = np.rot90(lsd, plan["rot_k"], axes=(1, 2)).copy()

    if plan["transpose_hw"]:
        raw = np.transpose(raw, (1, 0, 2)).copy()
        lab = np.transpose(lab, (1, 0, 2)).copy()
        if lsd is not None:
            lsd = np.transpose(lsd, (0, 2, 1, 3)).copy()

    if plan["brightness"] is not None:
        alpha, beta = plan["brightness"]
        raw = np.clip(raw.astype(np.float32) * alpha + beta, 0.0, 1.0).astype(np.float32)

    return raw, lab, lsd


def _read_lsd_crop(
    lsd_mmap,
    *,
    orientation: str,
    full_z0: int,
    full_z1: int,
    y0: int,
    x0: int,
    crop_size: int,
    source_h: int,
    source_w: int,
) -> np.ndarray:
    """Read only the spatial crop needed for one training sample."""
    y1 = min(y0 + crop_size, source_h)
    x1 = min(x0 + crop_size, source_w)

    if orientation == "xy":
        lsd_patch = np.array(
            lsd_mmap[:, y0:y1, x0:x1, full_z0:full_z1], dtype=np.float32
        )
    elif orientation == "xz":
        xy_patch = np.array(
            lsd_mmap[:, full_z0:full_z1, x0:x1, y0:y1], dtype=np.float32
        )
        lsd_patch = np.take(
            xy_patch, ORIENTATION_CHANNEL_PERMUTATIONS[orientation], axis=0
        )
        lsd_patch = np.transpose(
            lsd_patch,
            axes=(0,) + tuple(axis + 1 for axis in ORIENTATION_AXIS_PERMUTATIONS[orientation]),
        )
    else:  # yz
        xy_patch = np.array(
            lsd_mmap[:, full_z0:full_z1, y0:y1, x0:x1], dtype=np.float32
        )
        lsd_patch = np.take(
            xy_patch, ORIENTATION_CHANNEL_PERMUTATIONS[orientation], axis=0
        )
        lsd_patch = np.transpose(
            lsd_patch,
            axes=(0,) + tuple(axis + 1 for axis in ORIENTATION_AXIS_PERMUTATIONS[orientation]),
        )

    pad_h = crop_size - lsd_patch.shape[1]
    pad_w = crop_size - lsd_patch.shape[2]
    if pad_h > 0 or pad_w > 0:
        lsd_patch = np.pad(
            lsd_patch, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode="constant"
        )
    return lsd_patch


# ---------------------------------------------------------------------------
# Extended EM3D dataset that reads LSD from cache
# ---------------------------------------------------------------------------


class EM3DDatasetPreLSD(EM3DDataset):
    """EM3DDataset variant that reads pre-computed LSD from .npy files.

    ``lsd_paths`` is a list parallel to ``images``/``masks``, where each entry
    is either ``(npy_file_path, z_offset)`` or ``(xy_npy_file_path, z_offset,
    orientation)``. ``z_offset`` is the offset into the oriented cache's last
    axis that maps index 0 of ``images[ii]`` to the corresponding position in
    the cache array. When ``orientation`` is ``xz`` or ``yz``, the cache file
    still points to the canonical ``xy`` LSD volume and is re-oriented on load.

    With ``preload_lsd_to_ram=True``, each unique cache file is loaded once into
    ``_LSD_RAM_GLOBAL`` (shared across train/val/test dataset instances).
    """

    def __init__(
        self,
        images: List[np.ndarray],
        masks: List[np.ndarray],
        idxs: List[Tuple[int, int]],
        split: str,
        crop_size: int,
        padding_size: int,
        num_slices: int,
        require_lsd: bool,
        augment: Optional[bool],
        lsd_paths: List[Tuple],
        lsd_cache_dir: str = DEFAULT_LSD_CACHE_DIR,
        preload_lsd_to_ram: bool = False,
        need_point_map: bool = True,
    ):
        super().__init__(
            images,
            masks,
            idxs,
            split,
            crop_size,
            padding_size,
            num_slices,
            require_lsd,
            augment=augment,
        )
        self.lsd_paths = lsd_paths
        self.lsd_cache_dir = lsd_cache_dir
        self.need_point_map = need_point_map
        self.minimal_output = False
        # Per-instance lazy open (zarr / memmap) keyed by path or ("__no_lsd_path__", ii).
        self._lsd_mmaps: dict = {}
        # Dedupe LSD read warnings per DataLoader worker process.
        self._lsd_warned: set = set()
        if preload_lsd_to_ram and require_lsd:
            self._preload_lsd_volumes_to_ram()

    def _lsd_warn_once(self, key: Tuple, fmt: str, *args, exc_info: bool = False) -> None:
        if key in self._lsd_warned:
            return
        self._lsd_warned.add(key)
        _LOG.warning(fmt, *args, exc_info=exc_info)

    def _lsd_entry(self, ii: int) -> Tuple[str, int, str]:
        entry = self.lsd_paths[ii]
        if len(entry) == 2:
            path, z_offset = entry
            orientation = "xy"
        else:
            path, z_offset, orientation = entry
        return path, int(z_offset), str(orientation)

    def _lsd_mmap_cache_key(self, ii: int) -> object:
        path, _, _ = self._lsd_entry(ii)
        if path:
            return path
        return ("__no_lsd_path__", ii)

    def _preload_lsd_volumes_to_ram(self) -> None:
        """Load each unique LSD file into `_LSD_RAM_GLOBAL` (deduped across datasets)."""
        global _LSD_RAM_GLOBAL
        seen: set = set()
        paths: List[str] = []
        for ii in range(len(self.lsd_paths)):
            path, _, _ = self._lsd_entry(ii)
            if path and path not in seen:
                seen.add(path)
                paths.append(path)
        if not paths:
            return
        _LOG.info(
            "Preloading %d unique LSD cache file(s) into RAM (covers %d volume views)...",
            len(paths),
            len(self.lsd_paths),
        )
        t0 = time.time()
        total_bytes = 0
        loaded_here = 0
        for path in paths:
            if path in _LSD_RAM_GLOBAL:
                continue
            try:
                if path.endswith(".zarr"):
                    if not os.path.isdir(path):
                        self._lsd_warn_once(
                            ("lsd", "zarr_not_dir", path),
                            "LSD zarr path is not a directory: %s",
                            path,
                        )
                        _LSD_RAM_GLOBAL[path] = None
                        continue
                    arr = np.asarray(zarr.open(path, mode="r")[:], dtype=np.float32)
                elif os.path.isfile(path):
                    arr = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
                else:
                    self._lsd_warn_once(
                        ("lsd", "file_missing", path),
                        "LSD cache file not found (expected .npy or .zarr): %s",
                        path,
                    )
                    _LSD_RAM_GLOBAL[path] = None
                    continue
                _LSD_RAM_GLOBAL[path] = arr
                total_bytes += int(arr.nbytes)
                loaded_here += 1
            except Exception:
                _LSD_RAM_GLOBAL[path] = None
                _LOG.warning("LSD preload failed for %s", path, exc_info=True)
        _LOG.info(
            "LSD RAM: +%d file(s) loaded in this dataset init, +%.2f GiB; "
            "global cache entries: %d (non-null arrays: %d)",
            loaded_here,
            total_bytes / (1024.0 ** 3),
            len(_LSD_RAM_GLOBAL),
            sum(1 for v in _LSD_RAM_GLOBAL.values() if v is not None),
        )
        _LOG.info("LSD preload phase wall time: %.1fs", time.time() - t0)

    def _get_lsd_mmap(self, ii: int):
        """Return ndarray (RAM), zarr array, or memmap for volume ``ii``."""
        path, _, _ = self._lsd_entry(ii)
        if path and path in _LSD_RAM_GLOBAL:
            return _LSD_RAM_GLOBAL[path]

        key = self._lsd_mmap_cache_key(ii)
        if key not in self._lsd_mmaps:
            self._lsd_mmaps[key] = None
            if not path:
                self._lsd_warn_once(
                    ("lsd", "empty_path", ii),
                    "LSD cache path is empty for volume index %s (sample idx maps to this volume).",
                    ii,
                )
                return None
            try:
                if path.endswith(".zarr"):
                    if os.path.isdir(path):
                        self._lsd_mmaps[key] = zarr.open(path, mode="r")
                    else:
                        self._lsd_warn_once(
                            ("lsd", "zarr_not_dir", path),
                            "LSD zarr path is not a directory: %s",
                            path,
                        )
                elif os.path.isfile(path):
                    self._lsd_mmaps[key] = np.load(path, mmap_mode="r")
                else:
                    self._lsd_warn_once(
                        ("lsd", "file_missing", path),
                        "LSD cache file not found (expected .npy or existing .zarr dir): %s",
                        path,
                    )
            except Exception:
                self._lsd_mmaps[key] = None
                self._lsd_warn_once(
                    ("lsd", "open_failed", path),
                    "Failed to open LSD cache at %s",
                    path,
                    exc_info=True,
                )
        return self._lsd_mmaps[key]

    def __getitem__(self, idx: int):
        if not self.require_lsd:
            # No LSD needed – use base class path unchanged.
            return super().__getitem__(idx)

        lim = _profile_getitem_limit()
        profiling = lim > 0 and _item_prof_done < lim
        if profiling:
            t_start = time.perf_counter()
            t_mark = t_start
            stages: dict[str, float] = {}

        ii, z0 = self.idxs[idx]
        raw = self.images[ii]
        lab = self.masks[ii]

        raw = raw[z0 : z0 + self.num_slices]
        lab = lab[z0 : z0 + self.num_slices]
        raw = raw.transpose(1, 2, 0)   # → (H, W, num_slices)
        lab = lab.transpose(1, 2, 0)
        raw = normalize_minmax(raw)
        plan = _sample_3d_pair_plan(raw.shape, self.crop_size, augment=self.augment)

        if profiling:
            t_now = time.perf_counter()
            stages["slice_norm"] = t_now - t_mark
            t_mark = t_now

        # --- Attempt to load LSD sub-patch from cache ---
        lsd_patch: Optional[np.ndarray] = None
        lsd_mmap = self._get_lsd_mmap(ii)
        if profiling:
            t_now = time.perf_counter()
            stages["lsd_mmap"] = t_now - t_mark
            t_mark = t_now

        if lsd_mmap is not None:
            _, z_offset, orientation = self._lsd_entry(ii)
            full_z0 = z_offset + z0
            full_z1 = full_z0 + self.num_slices
            source_axis_limit = lsd_mmap.shape[3] if orientation == "xy" else lsd_mmap.shape[1]
            if full_z1 <= source_axis_limit:
                try:
                    lsd_patch = _read_lsd_crop(
                        lsd_mmap,
                        orientation=orientation,
                        full_z0=full_z0,
                        full_z1=full_z1,
                        y0=plan["y0"],
                        x0=plan["x0"],
                        crop_size=self.crop_size,
                        source_h=raw.shape[0],
                        source_w=raw.shape[1],
                    )
                except Exception:
                    lsd_patch = None
                    self._lsd_warn_once(
                        ("lsd", "read_slice", ii, orientation),
                        "Failed reading LSD slice from cache (volume=%s orient=%s z=[%s,%s) shape=%s): see traceback.",
                        ii,
                        orientation,
                        full_z0,
                        full_z1,
                        getattr(lsd_mmap, "shape", None),
                        exc_info=True,
                    )
            else:
                self._lsd_warn_once(
                    ("lsd", "z_oob", ii),
                    "LSD cache Z-range out of bounds for volume %s: need z in [0,%s), have patch [%s,%s) (z_offset=%s num_slices=%s orient=%s path=%s).",
                    ii,
                    source_axis_limit,
                    full_z0,
                    full_z1,
                    z_offset,
                    self.num_slices,
                    orientation,
                    self._lsd_entry(ii)[0],
                )

        if profiling:
            t_now = time.perf_counter()
            stages["lsd_read"] = t_now - t_mark
            t_mark = t_now

        # --- Crop + augment (lsd_patch may be None → falls back later) ---
        raw, lab, lsd_out = _apply_3d_pair_plan(raw, lab, lsd_patch, plan)

        if profiling:
            t_now = time.perf_counter()
            stages["prepare_aug"] = t_now - t_mark
            t_mark = t_now

        raw = np.expand_dims(raw, axis=0)
        lab = erode_instance_labels(lab, iterations=1, border_value=1)

        if profiling:
            t_now = time.perf_counter()
            stages["erode"] = t_now - t_mark
            t_mark = t_now

        affinity = affinity_3d(lab)

        if profiling:
            t_now = time.perf_counter()
            stages["affinity"] = t_now - t_mark
            t_mark = t_now

        if lsd_out is None:
            # Cache miss: fall back to on-the-fly computation.
            self._lsd_warn_once(
                ("lsd", "on_the_fly", ii),
                "LSD cache unavailable for volume %s; computing LSD on-the-fly for this sample (path=%s).",
                ii,
                self._lsd_entry(ii)[0] or "(empty)",
            )
            lsd_out = local_shape_descriptor.get_local_shape_descriptors(
                segmentation=lab, sigma=(5,) * 3, voxel_size=(1,) * 3
            ).astype(np.float32)

        if profiling:
            t_now = time.perf_counter()
            stages["lsd_onthefly"] = t_now - t_mark
            t_mark = t_now

        if self.minimal_output:
            if profiling:
                stages["prompt_pointmap"] = 0.0
                stages["total"] = time.perf_counter() - t_start
                _profile_getitem_finish(stages)
            return raw, affinity, lsd_out

        if self.need_point_map:
            mask_3d, pp, pl = mask_and_points_3d(lab)
            point_map = gaussian_point_map_3d_try(
                pp, pl, self.crop_size, self.crop_size, theta=30
            )
        else:
            # Collate still stacks mask_3d / point_map; ACRLSD neo preLSD training ignores them.
            mask_3d = np.zeros(lab.shape, dtype=np.uint8)
            point_map = np.zeros(
                (self.crop_size, self.crop_size), dtype=np.float32
            )

        if profiling:
            t_now = time.perf_counter()
            stages["prompt_pointmap"] = t_now - t_mark
            stages["total"] = t_now - t_start
            _profile_getitem_finish(stages)

        return raw, lab, mask_3d, affinity, point_map, lsd_out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _lsd_path(
    cache_dir: str,
    dataset_key: str,
    source_name: str,
    orientation: str,
    *,
    lsd_storage: str = "zarr",
) -> str:
    spec = make_lsd_volume_spec(
        dataset_key=dataset_key,
        source_name=source_name,
        split="full",
        n_val=0,
        orientation=orientation,
        volume_shape_zyx=(),
    )
    return lsd_cache_storage_path(cache_dir, spec, storage=lsd_storage)


def _append_volume(
    images: list,
    masks: list,
    lsd_paths: list,
    raw: np.ndarray,
    lab: np.ndarray,
    split: str,
    n_val: int,
    cache_path: str,
    z_offset: int,
    orientation: str = "xy",
) -> None:
    """Add one oriented volume to images/masks/lsd_paths with correct split slice."""
    if split == "train":
        images.append(raw[n_val:])
        masks.append(lab[n_val:])
    else:
        images.append(raw[:n_val])
        masks.append(lab[:n_val])
    lsd_paths.append((cache_path, z_offset, orientation))


def _append_volume_with_orientations(
    images: list,
    masks: list,
    lsd_paths: list,
    raw: np.ndarray,
    lab: np.ndarray,
    split: str,
    n_val: int,
    cache_dir: str,
    dataset_key: str,
    source_name: str,
    require_xz_yz: bool,
    *,
    lsd_storage: str = "zarr",
) -> None:
    """Append xy (always) and optionally xz + yz oriented volumes."""
    z_offset = n_val if split == "train" else 0

    # xy view (default, no transpose)
    _append_volume(
        images, masks, lsd_paths,
        raw, lab, split, n_val,
        _lsd_path(cache_dir, dataset_key, source_name, "xy", lsd_storage=lsd_storage),
        z_offset,
        "xy",
    )

    if require_xz_yz:
        # xz view: transpose(1, 0, 2) → first axis is H, slicing along H
        _append_volume(
            images, masks, lsd_paths,
            raw.transpose(1, 0, 2), lab.transpose(1, 0, 2),
            split, n_val,
            _lsd_path(cache_dir, dataset_key, source_name, "xy", lsd_storage=lsd_storage),
            z_offset,
            "xz",
        )
        # yz view: transpose(1, 2, 0) → first axis is H, slicing along H
        _append_volume(
            images, masks, lsd_paths,
            raw.transpose(1, 2, 0), lab.transpose(1, 2, 0),
            split, n_val,
            _lsd_path(cache_dir, dataset_key, source_name, "xy", lsd_storage=lsd_storage),
            z_offset,
            "yz",
        )


# ---------------------------------------------------------------------------
# Dataset_3D_*_Train  (mirror of dataloader.py but using EM3DDatasetPreLSD)
# ---------------------------------------------------------------------------


class Dataset_3D_hemi_Train(EM3DDatasetPreLSD):
    def __init__(
        self,
        data_dir=HEMI_DEFAULT_DATA_DIR,
        split="train",
        crop_size=None,
        num_slices=8,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
        augment: Optional[bool] = None,
        lsd_cache_dir: str = DEFAULT_LSD_CACHE_DIR,
        lsd_storage: str = "zarr",
        preload_lsd_to_ram: bool = False,
        need_point_map: bool = True,
    ):
        images, masks, lsd_paths = [], [], []
        for name in HEMI_ZARR_FILES:
            raw, lab = load_dataset_source_volume("hemi", data_dir, name)
            print("data {}: raw shape={}, label shape={}".format(name, raw.shape, lab.shape))
            _append_volume_with_orientations(
                images, masks, lsd_paths, raw, lab, split, n_val,
                lsd_cache_dir, "hemi", name, require_xz_yz,
                lsd_storage=lsd_storage,
            )
        idxs = _build_3d_patch_index(images, masks, num_slices)
        super().__init__(
            images, masks, idxs, split, crop_size, padding_size, num_slices,
            require_lsd, augment, lsd_paths, lsd_cache_dir,
            preload_lsd_to_ram,
            need_point_map,
        )


class Dataset_3D_fib25_Train(EM3DDatasetPreLSD):
    def __init__(
        self,
        data_dir=None,
        split="train",
        crop_size=None,
        num_slices=8,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
        augment: Optional[bool] = None,
        lsd_cache_dir: str = DEFAULT_LSD_CACHE_DIR,
        lsd_storage: str = "zarr",
        preload_lsd_to_ram: bool = False,
        need_point_map: bool = True,
    ):
        from utils.dataloader import FIB25_DEFAULT_DATA_DIR
        if data_dir is None:
            data_dir = FIB25_DEFAULT_DATA_DIR
        images, masks, lsd_paths = [], [], []
        for name in FIB25_ZARR_FILES:
            raw, lab = load_dataset_source_volume("fib25", data_dir, name)
            print("data {}: raw shape={}, label shape={}".format(name, raw.shape, lab.shape))
            _append_volume_with_orientations(
                images, masks, lsd_paths, raw, lab, split, n_val,
                lsd_cache_dir, "fib25", name, require_xz_yz,
                lsd_storage=lsd_storage,
            )
        idxs = _build_3d_patch_index(images, masks, num_slices)
        super().__init__(
            images, masks, idxs, split, crop_size, padding_size, num_slices,
            require_lsd, augment, lsd_paths, lsd_cache_dir,
            preload_lsd_to_ram,
            need_point_map,
        )


class Dataset_3D_cremi_Train(EM3DDatasetPreLSD):
    def __init__(
        self,
        data_dir=None,
        split="train",
        crop_size=None,
        num_slices=8,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
        augment: Optional[bool] = None,
        lsd_cache_dir: str = DEFAULT_LSD_CACHE_DIR,
        lsd_storage: str = "zarr",
        preload_lsd_to_ram: bool = False,
        need_point_map: bool = True,
    ):
        from utils.dataloader import CREMI_DEFAULT_DATA_DIR
        if data_dir is None:
            data_dir = CREMI_DEFAULT_DATA_DIR
        images, masks, lsd_paths = [], [], []
        for name in CREMI_HDF_FILES:
            raw, lab = load_dataset_source_volume("cremi", data_dir, name)
            print("data {}: raw shape={}, label shape={}".format(name, raw.shape, lab.shape))
            _append_volume_with_orientations(
                images, masks, lsd_paths, raw, lab, split, n_val,
                lsd_cache_dir, "cremi", name, require_xz_yz,
                lsd_storage=lsd_storage,
            )
        idxs = _build_3d_patch_index(images, masks, num_slices)
        super().__init__(
            images, masks, idxs, split, crop_size, padding_size, num_slices,
            require_lsd, augment, lsd_paths, lsd_cache_dir,
            preload_lsd_to_ram,
            need_point_map,
        )


class Dataset_3D_VNC_Train(EM3DDatasetPreLSD):
    def __init__(
        self,
        data_dir=VNC_DEFAULT_DATA_DIR,
        split="train",
        crop_size=None,
        num_slices=8,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: Optional[int] = None,
        augment: Optional[bool] = None,
        lsd_cache_dir: str = DEFAULT_LSD_CACHE_DIR,
        lsd_storage: str = "zarr",
        preload_lsd_to_ram: bool = False,
        need_point_map: bool = True,
    ):
        raw, lab = load_dataset_source_volume("vnc", data_dir, "stack1")
        z = raw.shape[0]
        if n_val is None:
            n_val = min(8, max(1, z // 5))
        print("data VNC stack1: raw shape={}, label shape={}".format(raw.shape, lab.shape))
        images, masks, lsd_paths = [], [], []
        _append_volume_with_orientations(
            images, masks, lsd_paths, raw, lab, split, n_val,
            lsd_cache_dir, "vnc", "stack1", require_xz_yz,
            lsd_storage=lsd_storage,
        )
        idxs = _build_3d_patch_index(images, masks, num_slices)
        super().__init__(
            images, masks, idxs, split, crop_size, padding_size, num_slices,
            require_lsd, augment, lsd_paths, lsd_cache_dir,
            preload_lsd_to_ram,
            need_point_map,
        )


class Dataset_3D_isbi2012_Train(EM3DDatasetPreLSD):
    def __init__(
        self,
        data_dir=ISBI2012_DEFAULT_DATA_DIR,
        split="train",
        crop_size=None,
        num_slices=8,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
        augment: Optional[bool] = None,
        lsd_cache_dir: str = DEFAULT_LSD_CACHE_DIR,
        lsd_storage: str = "zarr",
        preload_lsd_to_ram: bool = False,
        need_point_map: bool = True,
    ):
        images, masks, lsd_paths = [], [], []
        for raw_name, _ in ISBI2012_TIFF_PAIRS:
            raw, lab = load_dataset_source_volume("isbi2012", data_dir, raw_name)
            print("data ISBI-2012 {}: raw shape={}, label shape={}".format(raw_name, raw.shape, lab.shape))
            _append_volume_with_orientations(
                images, masks, lsd_paths, raw, lab, split, n_val,
                lsd_cache_dir, "isbi2012", raw_name, require_xz_yz,
                lsd_storage=lsd_storage,
            )
        idxs = _build_3d_patch_index(images, masks, num_slices)
        super().__init__(
            images, masks, idxs, split, crop_size, padding_size, num_slices,
            require_lsd, augment, lsd_paths, lsd_cache_dir,
            preload_lsd_to_ram,
            need_point_map,
        )


class Dataset_3D_ac3_Train(EM3DDatasetPreLSD):
    def __init__(
        self,
        data_dir=AC3_DEFAULT_DATA_DIR,
        split="train",
        crop_size=None,
        num_slices=8,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
        stack_slices: int = AC3_NUM_SLICES,
        augment: Optional[bool] = None,
        lsd_cache_dir: str = DEFAULT_LSD_CACHE_DIR,
        lsd_storage: str = "zarr",
        preload_lsd_to_ram: bool = False,
        need_point_map: bool = True,
    ):
        raw, lab = load_ac3_volume(data_dir, num_slices=stack_slices)
        print("data AC3: raw shape={}, label shape={}".format(raw.shape, lab.shape))
        images, masks, lsd_paths = [], [], []
        _append_volume_with_orientations(
            images, masks, lsd_paths, raw, lab, split, n_val,
            lsd_cache_dir, "ac3", "ac3", require_xz_yz,
            lsd_storage=lsd_storage,
        )
        idxs = _build_3d_patch_index(images, masks, num_slices)
        super().__init__(
            images, masks, idxs, split, crop_size, padding_size, num_slices,
            require_lsd, augment, lsd_paths, lsd_cache_dir,
            preload_lsd_to_ram,
            need_point_map,
        )


class Dataset_3D_ac4_Train(EM3DDatasetPreLSD):
    def __init__(
        self,
        data_dir=AC4_DEFAULT_DATA_DIR,
        split="train",
        crop_size=None,
        num_slices=8,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
        stack_slices: int = AC4_NUM_SLICES,
        augment: Optional[bool] = None,
        lsd_cache_dir: str = DEFAULT_LSD_CACHE_DIR,
        lsd_storage: str = "zarr",
        preload_lsd_to_ram: bool = False,
        need_point_map: bool = True,
    ):
        raw, lab = load_ac4_volume(data_dir, num_slices=stack_slices)
        print("data AC4: raw shape={}, label shape={}".format(raw.shape, lab.shape))
        images, masks, lsd_paths = [], [], []
        _append_volume_with_orientations(
            images, masks, lsd_paths, raw, lab, split, n_val,
            lsd_cache_dir, "ac4", "ac4", require_xz_yz,
            lsd_storage=lsd_storage,
        )
        idxs = _build_3d_patch_index(images, masks, num_slices)
        super().__init__(
            images, masks, idxs, split, crop_size, padding_size, num_slices,
            require_lsd, augment, lsd_paths, lsd_cache_dir,
            preload_lsd_to_ram,
            need_point_map,
        )


def _microns_neuron_zarr_preLSD(
    data_dir: str,
    dataset_key: str,
    volume_keys: Sequence[str],
    split: str,
    n_val: int,
    require_xz_yz: bool,
    lsd_cache_dir: str,
    subset_label: str,
    *,
    lsd_storage: str = "zarr",
) -> Tuple[list, list, list]:
    images, masks, lsd_paths = [], [], []
    for name in volume_keys:
        raw, lab = load_microns_neuron_zarr_volume(data_dir, name)
        print(
            "data MICrONS {} {}: raw shape={}, label shape={}".format(
                subset_label, name, raw.shape, lab.shape
            )
        )
        _append_volume_with_orientations(
            images, masks, lsd_paths, raw, lab, split, n_val,
            lsd_cache_dir, dataset_key, name, require_xz_yz,
            lsd_storage=lsd_storage,
        )
    return images, masks, lsd_paths


class Dataset_3D_basil_Train(EM3DDatasetPreLSD):
    def __init__(
        self,
        data_dir=MICRONS_BASIL_ZARR_DIR,
        split="train",
        crop_size=None,
        num_slices=8,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
        volume_keys: Optional[Sequence[str]] = None,
        augment: Optional[bool] = None,
        lsd_cache_dir: str = DEFAULT_LSD_CACHE_DIR,
        lsd_storage: str = "zarr",
        preload_lsd_to_ram: bool = False,
        need_point_map: bool = True,
    ):
        keys = tuple(volume_keys) if volume_keys is not None else microns_neuron_zarr_stems(data_dir)
        images, masks, lsd_paths = _microns_neuron_zarr_preLSD(
            data_dir, "basil", keys, split, n_val, require_xz_yz, lsd_cache_dir, "basil",
            lsd_storage=lsd_storage,
        )
        idxs = _build_3d_patch_index(images, masks, num_slices)
        super().__init__(
            images, masks, idxs, split, crop_size, padding_size, num_slices,
            require_lsd, augment, lsd_paths, lsd_cache_dir,
            preload_lsd_to_ram,
            need_point_map,
        )


class Dataset_3D_minnie_Train(EM3DDatasetPreLSD):
    def __init__(
        self,
        data_dir=MICRONS_MINNIE_ZARR_DIR,
        split="train",
        crop_size=None,
        num_slices=8,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
        volume_keys: Optional[Sequence[str]] = None,
        augment: Optional[bool] = None,
        lsd_cache_dir: str = DEFAULT_LSD_CACHE_DIR,
        lsd_storage: str = "zarr",
        preload_lsd_to_ram: bool = False,
        need_point_map: bool = True,
    ):
        keys = tuple(volume_keys) if volume_keys is not None else microns_neuron_zarr_stems(data_dir)
        images, masks, lsd_paths = _microns_neuron_zarr_preLSD(
            data_dir, "minnie", keys, split, n_val, require_xz_yz, lsd_cache_dir, "minnie",
            lsd_storage=lsd_storage,
        )
        idxs = _build_3d_patch_index(images, masks, num_slices)
        super().__init__(
            images, masks, idxs, split, crop_size, padding_size, num_slices,
            require_lsd, augment, lsd_paths, lsd_cache_dir,
            preload_lsd_to_ram,
            need_point_map,
        )


class Dataset_3D_pinky_Train(EM3DDatasetPreLSD):
    def __init__(
        self,
        data_dir=MICRONS_PINKY_ZARR_DIR,
        split="train",
        crop_size=None,
        num_slices=8,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
        volume_keys: Optional[Sequence[str]] = None,
        augment: Optional[bool] = None,
        lsd_cache_dir: str = DEFAULT_LSD_CACHE_DIR,
        lsd_storage: str = "zarr",
        preload_lsd_to_ram: bool = False,
        need_point_map: bool = True,
    ):
        keys = tuple(volume_keys) if volume_keys is not None else pinky_zarr_volume_keys(data_dir)
        images, masks, lsd_paths = _microns_neuron_zarr_preLSD(
            data_dir, "pinky", keys, split, n_val, require_xz_yz, lsd_cache_dir, "pinky",
            lsd_storage=lsd_storage,
        )
        idxs = _build_3d_patch_index(images, masks, num_slices)
        super().__init__(
            images, masks, idxs, split, crop_size, padding_size, num_slices,
            require_lsd, augment, lsd_paths, lsd_cache_dir,
            preload_lsd_to_ram,
            need_point_map,
        )


def _axonem_h5_preLSD(
    data_dir: str,
    dataset_key: str,
    volume_keys: Sequence[str],
    split: str,
    n_val: int,
    require_xz_yz: bool,
    lsd_cache_dir: str,
    subset_label: str,
    *,
    lsd_storage: str = "zarr",
) -> Tuple[list, list, list]:
    images, masks, lsd_paths = [], [], []
    for name in volume_keys:
        raw, lab = load_axonem_h5_volume(data_dir, name)
        print(
            "data AxonEM %s %s: raw shape=%s, label shape=%s"
            % (subset_label, name, raw.shape, lab.shape)
        )
        # AxonEM copies: to avoid memory sharing issues across workers
        raw_f = np.array(raw, dtype=np.float32, copy=True)
        lab_u = np.array(lab, copy=True)
        del raw, lab
        _append_volume_with_orientations(
            images, masks, lsd_paths, raw_f, lab_u, split, n_val,
            lsd_cache_dir, dataset_key, name, require_xz_yz,
            lsd_storage=lsd_storage,
        )
    return images, masks, lsd_paths


class Dataset_3D_axonem_h_Train(EM3DDatasetPreLSD):
    def __init__(
        self,
        data_dir=AXONEM_H_DEFAULT_DATA_DIR,
        split="train",
        crop_size=None,
        num_slices=8,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
        volume_keys: Optional[Sequence[str]] = None,
        augment: Optional[bool] = None,
        lsd_cache_dir: str = DEFAULT_LSD_CACHE_DIR,
        lsd_storage: str = "zarr",
        preload_lsd_to_ram: bool = False,
        need_point_map: bool = True,
    ):
        keys = tuple(volume_keys) if volume_keys is not None else axonem_h5_paired_volume_keys(data_dir)
        images, masks, lsd_paths = _axonem_h5_preLSD(
            data_dir, "axonem_h", keys, split, n_val, require_xz_yz, lsd_cache_dir, "H",
            lsd_storage=lsd_storage,
        )
        idxs = _build_3d_patch_index(images, masks, num_slices)
        super().__init__(
            images, masks, idxs, split, crop_size, padding_size, num_slices,
            require_lsd, augment, lsd_paths, lsd_cache_dir,
            preload_lsd_to_ram,
            need_point_map,
        )


class Dataset_3D_axonem_m_Train(EM3DDatasetPreLSD):
    def __init__(
        self,
        data_dir=AXONEM_M_DEFAULT_DATA_DIR,
        split="train",
        crop_size=None,
        num_slices=8,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
        volume_keys: Optional[Sequence[str]] = None,
        augment: Optional[bool] = None,
        lsd_cache_dir: str = DEFAULT_LSD_CACHE_DIR,
        lsd_storage: str = "zarr",
        preload_lsd_to_ram: bool = False,
        need_point_map: bool = True,
    ):
        keys = tuple(volume_keys) if volume_keys is not None else axonem_h5_paired_volume_keys(data_dir)
        images, masks, lsd_paths = _axonem_h5_preLSD(
            data_dir, "axonem_m", keys, split, n_val, require_xz_yz, lsd_cache_dir, "M",
            lsd_storage=lsd_storage,
        )
        idxs = _build_3d_patch_index(images, masks, num_slices)
        super().__init__(
            images, masks, idxs, split, crop_size, padding_size, num_slices,
            require_lsd, augment, lsd_paths, lsd_cache_dir,
            preload_lsd_to_ram,
            need_point_map,
        )


class Dataset_3D_zebrafinch_Train_CL(EM3DDatasetPreLSD):
    def __init__(
        self,
        data_dir=ZEBRAFINCH_DEFAULT_DATA_DIR,
        split="train",
        crop_size=None,
        num_slices=8,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        data_idxs=tuple(range(len(ZEBRA_PATCH_FILES))),
        n_val: int = 8,
        augment: Optional[bool] = None,
        lsd_cache_dir: str = DEFAULT_LSD_CACHE_DIR,
        lsd_storage: str = "zarr",
        preload_lsd_to_ram: bool = False,
        need_point_map: bool = True,
    ):
        images, masks, lsd_paths = [], [], []
        for patch_idx in data_idxs:
            name = ZEBRA_PATCH_FILES[patch_idx]
            raw, lab = load_dataset_source_volume("zebrafinch", data_dir, name)
            print("data {}: raw shape={}, label shape={}".format(name, raw.shape, lab.shape))
            _append_volume_with_orientations(
                images, masks, lsd_paths, raw, lab, split, n_val,
                lsd_cache_dir, "zebrafinch", name, require_xz_yz,
                lsd_storage=lsd_storage,
            )
        idxs = _build_3d_patch_index(images, masks, num_slices)
        super().__init__(
            images, masks, idxs, split, crop_size, padding_size, num_slices,
            require_lsd, augment, lsd_paths, lsd_cache_dir,
            preload_lsd_to_ram,
            need_point_map,
        )
