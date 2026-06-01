"""
Unified EM segmentation dataloaders (hemi / FIB-25 / CREMI / VNC / ISBI-2012 / AC3 / AC4 / zebrafinch / MICrONS Neuron Zarr / AxonEM H5).

Shared: erosion, padding, albumentations 2D/3D crop path, affinity, prompts, LSD, collate.
Per-dataset code is limited to volume loading and slice indexing rules.

Smoke tests: from repo root, with conda env ``UniSPAC`` active (or ``conda run -n UniSPAC python ...``), run
``python utils/dataloader.py vnc_2d`` / ``basil_2d`` / ``minnie_2d`` / ``pinky_2d`` / ``axonem_h_2d`` / ``axonem_m_2d`` or ``all``
(writes ``*_smoke.png`` next to this file; missing data prints FAILED and continues when using ``all``).
"""
from __future__ import annotations

import glob
import os
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import albumentations as A
import h5py
import numpy as np
import tifffile
import zarr
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt, find_objects
from scipy.stats import multivariate_normal
from skimage.measure import label as connected_components
from torch.utils.data import Dataset

from lsd.train import local_shape_descriptor

from utils.segem2d_interactive_sampling import prepare_2d_instance_centric_pair

# ---------------------------------------------------------------------------
# Constants (file lists must match legacy dataloader_*.py)
# ---------------------------------------------------------------------------

CREMI_HDF_FILES = (
    "sample_A_20160501.hdf",
    "sample_B_20160501.hdf",
    "sample_C_20160501.hdf",
)

FIB25_ZARR_FILES = (
    "trvol-250-1.zarr",
    "trvol-250-2.zarr",
    "tstvol-520-1.zarr",
    "tstvol-520-2.zarr",
)

HEMI_ZARR_FILES = (
    "eb-inner-groundtruth-with-context-x20172-y2322-z14332.zarr",
    "eb-outer-groundtruth-with-context-x20532-y3512-z14332.zarr",
    "fb-inner-groundtruth-with-context-x17342-y4052-z14332.zarr",
    "fb-outer-groundtruth-with-context-x13542-y2462-z14332.zarr",
    "lh-groundtruth-with-context-x7737-y20781-z12444.zarr",
    "lobula-groundtruth-with-context-x3648-y12800-z29056.zarr",
    "pb-groundtruth-with-context-x8472-y2372-z9372.zarr",
    "pb-groundtruth-with-context-x8472-y2892-z9372.zarr",
)

ZEBRA_PATCH_FILES = (
    "gt_z2834-2984_y5311-5461_x5077-5227.zarr",
    "gt_z2868-3018_y5744-5894_x5157-5307.zarr",
    "gt_z2874-3024_y5707-5857_x5304-5454.zarr",
    "gt_z2934-3084_y5115-5265_x5140-5290.zarr",
    "gt_z3096-3246_y5954-6104_x5813-5963.zarr",
    "gt_z3118-3268_y6538-6688_x6100-6250.zarr",
    "gt_z3126-3276_y6857-7007_x5694-5844.zarr",
    "gt_z3436-3586_y599-749_x2779-2929.zarr",
    "gt_z3438-3588_y2775-2925_x3476-3626.zarr",
    "gt_z3456-3606_y3188-3338_x4043-4193.zarr",
    "gt_z3492-3642_y7888-8038_x8374-8524.zarr",
    "gt_z3492-3642_y841-991_x381-531.zarr",
    "gt_z3596-3746_y3888-4038_x3661-3811.zarr",
    "gt_z3604-3754_y4101-4251_x3493-3643.zarr",
    "gt_z3608-3758_y3829-3979_x3423-3573.zarr",
    "gt_z3702-3852_y9605-9755_x2244-2394.zarr",
    "gt_z3710-3860_y8691-8841_x2889-3039.zarr",
    "gt_z3722-3872_y4548-4698_x2879-3029.zarr",
    "gt_z3734-3884_y4315-4465_x2209-2359.zarr",
    "gt_z3914-4064_y9035-9185_x2573-2723.zarr",
    "gt_z4102-4252_y6330-6480_x1899-2049.zarr",
    "gt_z4312-4462_y9341-9491_x2419-2569.zarr",
    "gt_z4440-4590_y7294-7444_x2350-2500.zarr",
    "gt_z4801-4951_y10154-10304_x1972-2122.zarr",
    "gt_z4905-5055_y928-1078_x1729-1879.zarr",
    "gt_z4951-5101_y9415-9565_x2272-2422.zarr",
    "gt_z5001-5151_y9426-9576_x2197-2347.zarr",
    "gt_z5119-5247_y1023-1279_x1663-1919.zarr",
    "gt_z5405-5555_y10490-10640_x3406-3556.zarr",
    "gt_z734-884_y9561-9711_x563-713.zarr",
    "gt_z255-383_y1407-1663_x1535-1791.zarr",
    "gt_z2559-2687_y4991-5247_x4863-5119.zarr",
    "gt_z2815-2943_y5631-5887_x4607-4863.zarr",
)

# Default dataset roots (override via Dataset(..., data_dir=...) or symlink ./data).
HEMI_DEFAULT_DATA_DIR = "./data/funke/hemi/training/"
FIB25_DEFAULT_DATA_DIR = "./data/funke/fib25/training/"
CREMI_DEFAULT_DATA_DIR = "./data/CREMI/"
VNC_DEFAULT_DATA_DIR = "./data/groundtruth-drosophila-vnc-master/stack1/"
ZEBRAFINCH_DEFAULT_DATA_DIR = "./data/funke/zebrafinch/training/"

ISBI2012_DEFAULT_DATA_DIR = "./data/ISBI-2012/"
ISBI2012_TIFF_PAIRS = (
    ("train-volume.tif", "train-labels.tif"),
    ("test-volume.tif", "test-labels.tif"),
)

AC3_DEFAULT_DATA_DIR = "./data/AC3/"
AC3_NUM_SLICES = 256
AC3_EM_SLICE_BASENAME = "Thousand_highmag_256slices_2kcenter_1k_inv_%04d.png"
AC3_DANIEL_SEG_BASENAME = "ac3_daniel_s%04d.png"

AC4_DEFAULT_DATA_DIR = "./data/AC4/"
AC4_NUM_SLICES = 100
AC4_EM_SLICE_BASENAME = "affinecropped4_inv_%04d.png"
AC4_DANIEL_SEG_BASENAME = "ac4_daniel_s%04d.png"

# MICrONS Neuron: bbox-cropped stacks from ``data/MICrONS/transfer_zarr.py`` → ``Neuron_zarr/{basil,minnie,pinky}/``.
MICRONS_NEURON_ZARR_ROOT = "./data/MICrONS/Neuron_zarr/"
MICRONS_BASIL_ZARR_DIR = "./data/MICrONS/Neuron_zarr/basil/"
MICRONS_MINNIE_ZARR_DIR = "./data/MICrONS/Neuron_zarr/minnie/"
MICRONS_PINKY_ZARR_DIR = "./data/MICrONS/Neuron_zarr/pinky/"
MICRONS_VOXEL_SIZE_NM_ZYX = (40.0, 4.0, 4.0)
# Legacy alias (pinky now reads zarr, not TIFF under Neuron/pinky/).
PINKY_DEFAULT_DATA_DIR = MICRONS_PINKY_ZARR_DIR
# Nine cuboid stacks for pinky CL indexing (cf. ``ZEBRA_PATCH_FILES``).
PINKY_PATCH_FILES = (
    "pinky_vol101",
    "pinky_vol102",
    "pinky_vol103",
    "pinky_vol104",
    "pinky_vol201",
    "pinky_vol401",
    "pinky_vol501",
    "pinky_vol502",
    "pinky_vol503",
)
# Preferred pinky zarr order when all exist (two stitched + nine cuboids).
PINKY_ALL_VOLUME_KEYS = (
    "pinky_stitched_vol19-vol34",
    "pinky_stitched_vol40-vol41",
) + PINKY_PATCH_FILES

# AxonEM EM30: ``im_*_pad.h5`` + ``seg_*_pad.h5`` per cuboid (see ``data/AxonEM/.../plot.ipynb``).
# EM30-H: nine paired stacks (extra unpaired EM-only cuboids are ignored). EM30-M: nine with labels;
# ``im_*`` without ``seg_*`` is skipped. Dense labels cover only a central bbox inside the padded volume;
# we crop to the foreground bounding box before building slices / 3D patches.
AXONEM_H_DEFAULT_DATA_DIR = "./data/AxonEM/EM30-H-axon-train-9vol/"
AXONEM_M_DEFAULT_DATA_DIR = "./data/AxonEM/EM30-M-axon-train-9vol/"


# ---------------------------------------------------------------------------
# Geometry / labels (shared)
# ---------------------------------------------------------------------------


def erode_instance_labels(labels: np.ndarray, iterations: int = 1, border_value: int = 1) -> np.ndarray:
    """Erode each instance (non-zero label) by face connectivity; same semantics as scipy per-ID path.

    For ``iterations == 1`` and ``border_value in (0, 1)``, uses neighbor-equality (O(volume))
    instead of one ``binary_erosion`` per instance ID — same approach as ``process_lsd.py``.
    """
    labels = np.asarray(labels)
    if iterations == 1 and border_value in (0, 1):
        keep = labels != 0
        for axis in range(labels.ndim):
            lhs = [slice(None)] * labels.ndim
            rhs = [slice(None)] * labels.ndim
            lhs[axis] = slice(1, None)
            rhs[axis] = slice(None, -1)
            lhs_t = tuple(lhs)
            rhs_t = tuple(rhs)
            same = labels[lhs_t] == labels[rhs_t]
            keep[lhs_t] &= same
            keep[rhs_t] &= same

        if border_value == 0:
            for axis in range(labels.ndim):
                lower = [slice(None)] * labels.ndim
                upper = [slice(None)] * labels.ndim
                lower[axis] = 0
                upper[axis] = -1
                keep[tuple(lower)] = False
                keep[tuple(upper)] = False

        eroded = labels.copy()
        eroded[np.logical_not(keep)] = 0
        return eroded

    foreground = np.zeros_like(labels, dtype=bool)
    for lid in np.unique(labels):
        if lid == 0:
            continue
        lm = labels == lid
        eroded = binary_erosion(lm, iterations=iterations, border_value=border_value)
        foreground = np.logical_or(eroded, foreground)
    out = labels.copy()
    out[np.logical_not(foreground)] = 0
    return out


def padding_for_crop(crop_size: Optional[int], padding_unit: int) -> int:
    if crop_size is None:
        return 0
    q = int(crop_size / padding_unit)
    if crop_size % padding_unit != 0:
        return padding_unit * (q + 1)
    return crop_size


def normalize_minmax(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    mn = float(np.min(data))
    mx = float(np.max(data))
    if mx <= mn:
        return np.zeros_like(data, dtype=np.float32)
    return (data - mn) / (mx - mn)


def _random_xy_from_mask(mask: np.ndarray) -> Optional[List[int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    idx = random.randrange(len(xs))
    return [int(xs[idx]), int(ys[idx])]


def _mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(np.min(ys)), int(np.min(xs)), int(np.max(ys)) + 1, int(np.max(xs)) + 1


def _select_component_center_xy(mask: np.ndarray, dist_transform: np.ndarray) -> Optional[List[int]]:
    if not np.any(mask):
        return None
    max_dist = float(np.max(dist_transform))
    if max_dist > 0:
        coords_yx = np.column_stack(np.where(dist_transform == max_dist))
    else:
        coords_yx = np.column_stack(np.where(mask))
    if len(coords_yx) == 0:
        return None

    centroid = np.mean(coords_yx.astype(np.float32), axis=0)
    nearest_idx = int(np.argmin(np.sum((coords_yx - centroid[None, :]) ** 2, axis=1)))
    y, x = coords_yx[nearest_idx]
    return [int(x), int(y)]


def _sample_grid_points_xy_from_mask(
    candidate_mask: np.ndarray,
    bbox_yx: Tuple[int, int, int, int],
    stride: int,
) -> List[List[int]]:
    if stride < 1:
        raise ValueError("stride must be >= 1, got %r" % (stride,))
    y0, x0, y1, x1 = bbox_yx
    ys = np.arange(y0, y1, stride, dtype=np.int32)
    xs = np.arange(x0, x1, stride, dtype=np.int32)
    out = []
    for y in ys:
        for x in xs:
            if candidate_mask[y, x]:
                out.append([int(x), int(y)])
    return out


def sample_positive_target_points_2d(
    labels: np.ndarray,
    *,
    point_thre: float = 0.2,
    point_grid_stride: int = 16,
    max_points: int = 8,
    single_point_prob: float = 0.8,
    area_ref: float = 256.0,
    max_candidate_pool: int = 2048,
    rng=None,
):
    """Pick random foreground instance, then random interior prompt pixels (away from boundary).

    ``point_thre`` is a ratio of the max EDT depth: keep pixels with dist >= max(max*thre, 1).
    ``point_grid_stride`` is ignored (kept for call compatibility).

    ~``single_point_prob`` of the time use one prompt; otherwise sample 2..n_cap points without
    replacement, where n_cap scales with sqrt(instance area) and is capped by ``max_points``.
    EDT and candidate indexing run on a tight bbox crop for speed.
    """
    del point_grid_stride  # legacy API; interior sampling does not use a grid
    labels = np.asarray(labels)
    fg_flat = labels.ravel()
    fg_flat = fg_flat[fg_flat != 0]
    if fg_flat.size == 0:
        return None, None
    fg_ids = np.unique(fg_flat)
    if rng is None:
        target_id = int(np.random.choice(fg_ids))
    else:
        target_id = int(rng.choice(fg_ids))

    mask = labels == target_id
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None, None

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = np.asarray(mask[y0:y1, x0:x1], dtype=np.uint8)
    h_c, w_c = crop.shape
    padded = np.pad(crop, 1, mode="constant", constant_values=0)
    dist = distance_transform_edt(padded)[1:-1, 1:-1]
    md = float(dist.max())
    ratio = float(point_thre)
    if md > 0:
        d_min = max(1.0, md * ratio)
        cand = (dist >= d_min) & (crop.astype(bool))
        if not np.any(cand):
            d_min = 1.0
            cand = (dist >= d_min) & (crop.astype(bool))
        if not np.any(cand):
            cand = crop.astype(bool)
    else:
        cand = crop.astype(bool)

    flat = np.flatnonzero(cand.ravel())
    if flat.size == 0:
        return None, None

    n_pix = int(crop.sum())
    n_cap = min(int(max_points), max(1, 1 + int(np.sqrt(float(n_pix) / float(area_ref)))))
    if rng is None:
        u = float(np.random.random())
    else:
        u = float(rng.random())
    if u < float(single_point_prob):
        n_take = 1
    else:
        if rng is None:
            n_take = int(np.random.randint(2, n_cap + 1)) if n_cap >= 2 else 1
        else:
            n_take = int(rng.integers(2, n_cap + 1)) if n_cap >= 2 else 1

    n_take = min(n_take, flat.size)
    if flat.size > int(max_candidate_pool):
        if rng is None:
            keep = np.random.choice(flat.size, size=int(max_candidate_pool), replace=False)
        else:
            keep = rng.choice(flat.size, size=int(max_candidate_pool), replace=False)
        flat = flat[keep]

    if rng is None:
        pick = np.random.choice(flat.size, size=n_take, replace=False)
    else:
        pick = rng.choice(flat.size, size=n_take, replace=False)
    sel = flat[pick]
    yy_loc = (sel // w_c).astype(np.int64)
    xx_loc = (sel % w_c).astype(np.int64)
    points_xy = [[int(x0 + int(xx_loc[i])), int(y0 + int(yy_loc[i]))] for i in range(n_take)]

    return points_xy, mask


def sample_no_prompt_all_2d(labels: np.ndarray):
    labels = np.asarray(labels)
    h, w = labels.shape
    return np.ones((h, w), dtype=np.float32), (labels != 0).astype(np.uint8)


def sample_positive_target_prompt_2d(
    labels: np.ndarray,
    *,
    point_thre: float = 0.2,
    point_grid_stride: int = 16,
    max_points: int = 8,
    single_point_prob: float = 0.8,
    area_ref: float = 256.0,
    max_candidate_pool: int = 2048,
    theta: float = 30.0,
    rng=None,
):
    labels = np.asarray(labels)
    h, w = labels.shape
    points_xy, mask = sample_positive_target_points_2d(
        labels,
        point_thre=point_thre,
        point_grid_stride=point_grid_stride,
        max_points=max_points,
        single_point_prob=single_point_prob,
        area_ref=area_ref,
        max_candidate_pool=max_candidate_pool,
        rng=rng,
    )
    if points_xy is None or mask is None:
        return np.ones((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.uint8), False, []

    point_map = gaussian_point_map(points_xy, [1] * len(points_xy), h, w, theta=theta).astype(np.float32)
    return point_map, mask.astype(np.uint8), True, points_xy


def affinity_2d(labels: np.ndarray) -> np.ndarray:
    shifted = np.pad(labels, ((0, 1), (0, 1)), mode="edge")
    ax = np.expand_dims(((labels - shifted[1:, :-1]) != 0) + 0, axis=0)
    ay = np.expand_dims(((labels - shifted[:-1, 1:]) != 0) + 0, axis=0)
    return np.concatenate([ax, ay], axis=0).astype(np.float32)


def affinity_3d(labels: np.ndarray) -> np.ndarray:
    shifted = np.pad(labels, ((0, 1), (0, 1), (0, 1)), mode="edge")
    ax = np.expand_dims(((labels - shifted[1:, :-1, :-1]) != 0) + 0, axis=0)
    ay = np.expand_dims(((labels - shifted[:-1, 1:, :-1]) != 0) + 0, axis=0)
    az = np.expand_dims(((labels - shifted[:-1, :-1, 1:]) != 0) + 0, axis=0)
    bg = labels == 0
    ax[0][bg] = 1
    ay[0][bg] = 1
    az[0][bg] = 1
    return np.concatenate([ax, ay, az], axis=0).astype(np.float32)


def _expand_bbox_2d(shape: Tuple[int, int], bbox: Tuple[slice, slice], halo: Tuple[int, int]) -> Tuple[slice, slice]:
    out = []
    for axis, (slc, pad) in enumerate(zip(bbox, halo)):
        start = max(0, int(slc.start) - int(pad))
        stop = min(int(shape[axis]), int(slc.stop) + int(pad))
        out.append(slice(start, stop))
    return tuple(out)


def compute_2d_lsds_bbox_local(
    segmentation: np.ndarray,
    *,
    sigma: Tuple[float, float] = (5.0, 5.0),
) -> np.ndarray:
    """Compute 2D LSDs per-instance on expanded local boxes instead of full-frame.

    This matches the spirit of ``process_lsd.py``: most EM slices contain sparse instances,
    so computing descriptors on each label's halo-expanded ROI is much cheaper than running
    LSD on the entire crop every time.
    """
    segmentation = np.asarray(segmentation)
    descriptors = np.zeros((6,) + segmentation.shape, dtype=np.float32)
    if segmentation.size == 0:
        return descriptors

    unique_ids, inverse = np.unique(segmentation, return_inverse=True)
    if unique_ids.size == 0 or (unique_ids.size == 1 and unique_ids[0] == 0):
        return descriptors

    # Compress sparse instance ids to a dense label space so find_objects scales
    # with objects-in-crop instead of max(label_id).
    dense_seg = inverse.reshape(segmentation.shape)
    if unique_ids[0] != 0:
        dense_seg = dense_seg + 1
    dense_seg = np.asarray(dense_seg, dtype=np.int32)

    halo = tuple(int(np.ceil(3.0 * float(s))) for s in sigma)
    object_slices = find_objects(dense_seg)
    for label_id, bbox in enumerate(object_slices, start=1):
        if bbox is None:
            continue
        expanded = _expand_bbox_2d(dense_seg.shape, bbox, halo)
        local_seg = dense_seg[expanded]
        local_mask = local_seg == label_id
        if not np.any(local_mask):
            continue
        local_desc = local_shape_descriptor.get_local_shape_descriptors(
            segmentation=local_mask.astype(np.uint8, copy=False),
            sigma=sigma,
            voxel_size=(1.0, 1.0),
        ).astype(np.float32, copy=False)
        target = descriptors[(slice(None),) + expanded]
        target[:, local_mask] = local_desc[:, local_mask]
    return descriptors


DEFAULT_APPROX_2D_LSD_CACHE_DIR = os.path.abspath("./LSD_cache")
APPROX_2D_LSD_CHANNELS_BY_ORIENTATION = {
    # Canonical 3D cache axes are (Y, X, Z); current 2D xz/yz slices are (Z, X) / (Z, Y).
    "xy": (0, 1, 3, 4, 6, 9),
    "xz": (2, 1, 5, 4, 8, 9),
    "yz": (2, 0, 5, 3, 7, 9),
}


def _approx_2d_lsd_cache_xy_path(cache_dir: str, dataset_key: str, source_name: str) -> str:
    src = os.path.splitext(str(source_name))[0].replace(os.sep, "__").replace("/", "__")
    return os.path.join(cache_dir, "{}__{}__xy.zarr".format(dataset_key, src))


def _read_approx_2d_lsd_from_3d_cache(lsd_arr, orientation: str, slice_idx: int) -> np.ndarray:
    if orientation == "xy":
        lsd_10 = np.array(lsd_arr[:, :, :, slice_idx], dtype=np.float32)
    elif orientation == "xz":
        lsd_10 = np.transpose(np.array(lsd_arr[:, slice_idx, :, :], dtype=np.float32), (0, 2, 1))
    elif orientation == "yz":
        lsd_10 = np.transpose(np.array(lsd_arr[:, :, slice_idx, :], dtype=np.float32), (0, 2, 1))
    else:
        raise ValueError("Unsupported approximate 2D LSD orientation: {!r}".format(orientation))
    return np.take(lsd_10, APPROX_2D_LSD_CHANNELS_BY_ORIENTATION[orientation], axis=0)


def get_prompt_2d(labels: np.ndarray):
    """Standard random prompt (ACRLSD / segEM2d)."""
    points_pos, points_lab, boxes = [], [], []
    mask = np.zeros_like(labels, dtype=bool)
    fg_ids = [int(lid) for lid in np.unique(labels) if lid != 0]
    if not fg_ids:
        return None, None, None, mask
    if np.random.rand() < 0.5:
        mask = labels != 0
        return None, None, None, mask

    p_contain = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    contain = []
    for _ in range(10):
        contain = [lid for lid in fg_ids if np.random.rand() < p_contain]
        if contain:
            break
    if not contain:
        contain = [random.choice(fg_ids)]
    exclude = [lid for lid in fg_ids if lid not in contain]

    point_style = random.choice(["+", "-", "+-"])
    for lid in contain:
        m = labels == lid
        mask = np.logical_or(mask, m)
        pt = _random_xy_from_mask(m)
        if pt is not None and ("+" in point_style or len(contain) == 1):
            points_pos.append(pt)
            points_lab.append(1)
        ys, xs = np.where(m)
        if len(xs):
            boxes.append([int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys))])

    for lid in exclude:
        if "-" not in point_style:
            continue
        m = labels == lid
        pt = _random_xy_from_mask(m)
        if pt is not None:
            points_pos.append(pt)
            points_lab.append(0)

    return points_pos, points_lab, boxes, mask


def get_prompt_2d_sam(labels: np.ndarray):
    """SAM path: p_default fixed to 1 (legacy)."""
    points_pos, points_lab, boxes = [], [], []
    mask = np.zeros_like(labels, dtype=bool)
    p_default = 1
    fg_ids = [int(lid) for lid in np.unique(labels) if lid != 0]
    point_style = random.choice(["+", "-", "+-"])
    if not fg_ids:
        return [[0, 0]], [0], [[0, 0, 0, 0]], mask
    if p_default < 0.5:
        mask = labels != 0
        return [[0, 0]], [0], [[0, 0, 0, 0]], mask

    p_contain = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    contain = []
    for _ in range(10):
        contain = [lid for lid in fg_ids if np.random.rand() < p_contain]
        if contain:
            break
    if not contain:
        contain = [random.choice(fg_ids)]
    exclude = [lid for lid in fg_ids if lid not in contain]

    for lid in contain:
        m = labels == lid
        mask = np.logical_or(mask, m)
        pt = _random_xy_from_mask(m)
        if pt is not None and ("+" in point_style or len(contain) == 1):
            points_pos.append(pt)
            points_lab.append(1)
        ys, xs = np.where(m)
        if len(xs):
            boxes.append([int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys))])

    for lid in exclude:
        if "-" not in point_style:
            continue
        m = labels == lid
        pt = _random_xy_from_mask(m)
        if pt is not None:
            points_pos.append(pt)
            points_lab.append(0)

    return points_pos, points_lab, boxes, mask


def gaussian_point_map(points_pos, points_lab, h: int, w: int, theta: float = 10.0) -> np.ndarray:
    if points_pos is None:
        return np.ones((h, w), dtype=np.float32)
    total = np.zeros((h, w), dtype=np.float32)
    seen = set()
    xg, yg = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    inv_two_theta = 0.5 / float(theta)
    for n, (X, Y) in enumerate(points_pos):
        if (X, Y) in seen:
            continue
        seen.add((X, Y))
        g = np.exp(-(((xg - float(X)) ** 2 + (yg - float(Y)) ** 2) * inv_two_theta))
        peak = float(np.max(g))
        if peak > 0.0:
            g *= 1.0 / peak
        total = total + g * (points_lab[n] * 2 - 1)
    if points_lab is not None and len(points_lab) and np.max(points_lab) == 0:
        total = total * 2 + 1
    return total


def gaussian_point_map_3d_try(points_pos, points_lab, h: int, w: int, theta: float = 10.0) -> np.ndarray:
    """3D dataloader legacy: try/except around empty Points_lab max."""
    if points_pos is None:
        return np.ones((h, w), dtype=np.float32)
    total = np.zeros((h, w), dtype=np.float32)
    seen = set()
    xg, yg = np.meshgrid(np.arange(w), np.arange(h))
    pos = np.dstack((xg, yg))
    for n, (X, Y) in enumerate(points_pos):
        if (X, Y) in seen:
            continue
        seen.add((X, Y))
        rv = multivariate_normal(mean=[X, Y], cov=[[theta, 0], [0, theta]])
        m = rv.pdf(pos)
        m = m * (1.0 / np.max(m))
        total = total + m * (points_lab[n] * 2 - 1)
    try:
        if np.max(points_lab) == 0:
            total = total * 2 + 1
    except Exception:
        pass
    return total


def mask_and_points_3d(labels: np.ndarray):
    """Prompt / mask over 3D block; points from first z-slice (legacy)."""
    points_pos, points_lab = [], []
    mask_3d = np.zeros_like(labels, dtype=bool)
    p_default = np.random.rand()
    fg_ids = [int(lid) for lid in np.unique(labels[:, :, 0]) if lid != 0]
    if not fg_ids:
        return mask_3d, None, None

    if p_default < 0.5:
        p_contain = 1.0
    else:
        p_contain = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    contain = []
    for _ in range(10):
        contain = [lid for lid in fg_ids if np.random.rand() < p_contain]
        if contain:
            break
    if not contain:
        contain = [random.choice(fg_ids)]
    exclude = [lid for lid in fg_ids if lid not in contain]

    for lid in contain:
        mask_3d = np.logical_or(mask_3d, labels == lid)

    if p_default < 0.5:
        return mask_3d, None, None

    point_style = random.choice(["+", "-", "+-"])
    for lid in contain:
        m = labels[:, :, 0] == lid
        pt = _random_xy_from_mask(m)
        if pt is not None and "+" in point_style:
            points_pos.append(pt)
            points_lab.append(1)

    for lid in exclude:
        if "-" not in point_style:
            continue
        m = labels[:, :, 0] == lid
        pt = _random_xy_from_mask(m)
        if pt is not None:
            points_pos.append(pt)
            points_lab.append(0)

    return mask_3d, points_pos, points_lab


def augment_2d_pair(
    raw: np.ndarray,
    mask: np.ndarray,
    crop_size: int,
    pad_total: int,
    check_shapes: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    kw = {} if check_shapes else {"is_check_shapes": False}
    transform = A.Compose(
        [
            A.PadIfNeeded(min_height=crop_size, min_width=crop_size, p=1, border_mode=0),
            A.RandomCrop(width=crop_size, height=crop_size),
            A.PadIfNeeded(min_height=pad_total, min_width=pad_total, p=1, border_mode=0),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.Transpose(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
        ],
        **kw,
    )
    out = transform(image=raw, mask=mask)
    return out["image"], out["mask"]


def prepare_2d_pair(
    raw: np.ndarray,
    mask: np.ndarray,
    crop_size: int,
    pad_total: int,
    check_shapes: bool = True,
    *,
    augment: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    if augment:
        return augment_2d_pair(raw, mask, crop_size, pad_total, check_shapes)
    kw = {} if check_shapes else {"is_check_shapes": False}
    transform = A.Compose(
        [
            A.PadIfNeeded(min_height=crop_size, min_width=crop_size, p=1, border_mode=0),
            A.CenterCrop(width=crop_size, height=crop_size),
        ],
        **kw,
    )
    out = transform(image=raw, mask=mask)
    return out["image"], out["mask"]


def _prepare_2d_with_lsd(
    raw: np.ndarray,
    mask: np.ndarray,
    lsd: Optional[np.ndarray],
    crop_size: int,
    *,
    augment: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Crop and augment raw, mask, and precomputed 2D LSD simultaneously.

    Mirrors the 3D preLSD augmentation strategy: the same spatial transforms
    (random crop, flips, rot90, transpose, brightness) are applied identically
    to raw, mask, and the cached LSD map so they stay aligned.

    Applying spatial transforms to the LSD array without adjusting directional
    channels is an approximation (same convention used by the 3D preLSD path).

    Args:
        raw:       (H, W) float32 – normalised EM slice.
        mask:      (H, W) uint16  – instance labels (not yet eroded).
        lsd:       (6, H, W) float32 or None – precomputed full-slice LSD.
        crop_size: target spatial size (square HW output).
        augment:   if True apply random crop + geometric+brightness augmentation;
                   if False use centre crop only.

    Returns:
        (raw, mask, lsd) with spatial dims (crop_size, crop_size).
        lsd is None when input lsd is None.
    """
    H, W = raw.shape[:2]

    pad_h = max(0, crop_size - H)
    pad_w = max(0, crop_size - W)
    if pad_h > 0 or pad_w > 0:
        raw = np.pad(raw, ((0, pad_h), (0, pad_w)), mode="constant")
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")
        if lsd is not None:
            lsd = np.pad(lsd, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
        H += pad_h
        W += pad_w

    if augment:
        y0 = random.randint(0, H - crop_size)
        x0 = random.randint(0, W - crop_size)
    else:
        y0 = (H - crop_size) // 2
        x0 = (W - crop_size) // 2

    raw = raw[y0 : y0 + crop_size, x0 : x0 + crop_size]
    mask = mask[y0 : y0 + crop_size, x0 : x0 + crop_size]
    if lsd is not None:
        lsd = np.array(lsd[:, y0 : y0 + crop_size, x0 : x0 + crop_size], dtype=np.float32)

    if not augment:
        return raw, mask, lsd

    if random.random() < 0.3:  # HorizontalFlip
        raw = raw[::-1].copy()
        mask = mask[::-1].copy()
        if lsd is not None:
            lsd = lsd[:, ::-1].copy()

    if random.random() < 0.3:  # VerticalFlip
        raw = raw[:, ::-1].copy()
        mask = mask[:, ::-1].copy()
        if lsd is not None:
            lsd = lsd[:, :, ::-1].copy()

    if random.random() < 0.3:  # RandomRotate90
        k = random.randint(1, 3)
        raw = np.rot90(raw, k).copy()
        mask = np.rot90(mask, k).copy()
        if lsd is not None:
            lsd = np.rot90(lsd, k, axes=(1, 2)).copy()

    if random.random() < 0.3:  # Transpose (swap H and W)
        raw = np.transpose(raw).copy()
        mask = np.transpose(mask).copy()
        if lsd is not None:
            lsd = np.transpose(lsd, (0, 2, 1)).copy()

    if random.random() < 0.3:  # RandomBrightnessContrast (raw only)
        alpha = random.uniform(0.8, 1.2)
        beta = random.uniform(-0.2, 0.2)
        raw = np.clip(raw * alpha + beta, 0.0, 1.0).astype(np.float32)

    return raw, mask, lsd


# ---------------------------------------------------------------------------
# VNC stack1 I/O
# ---------------------------------------------------------------------------


def _membrane_png_for_index(stack1_dir: str, i: int) -> str:
    mem = os.path.join(stack1_dir, "membranes")
    if not os.path.isdir(mem):
        raise FileNotFoundError("VNC stack1 membranes dir not found: %s" % mem)
    for name in ("%02d.png" % i, "%d.png" % i, "%08d.png" % i):
        p = os.path.join(mem, name)
        if os.path.isfile(p):
            return p
    for p in glob.glob(os.path.join(mem, "*.png")) + glob.glob(os.path.join(mem, "*.PNG")):
        stem, _ = os.path.splitext(os.path.basename(p))
        if stem.isdigit() and int(stem) == i:
            return p
    raise FileNotFoundError("Missing membrane mask for slice index %d under %s" % (i, mem))


def _vnc_raw_index_paths(stack1_dir: str) -> dict:
    raw_dir = os.path.join(stack1_dir, "raw")
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError("VNC stack1 raw dir not found: %s" % raw_dir)
    out = {}
    for p in glob.glob(os.path.join(raw_dir, "*.tif")) + glob.glob(os.path.join(raw_dir, "*.TIF")):
        stem, _ = os.path.splitext(os.path.basename(p))
        if stem.isdigit():
            out[int(stem)] = p
    return out


def load_vnc_stack1_volume(stack1_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    idx_map = _vnc_raw_index_paths(stack1_dir)
    indices = sorted(idx_map.keys())
    if not indices:
        raise FileNotFoundError("No .tif slices under %s/raw" % stack1_dir)
    raws, labs = [], []
    for i in indices:
        r = tifffile.imread(idx_map[i])
        if r.ndim > 2:
            r = np.squeeze(r)
        mem = np.array(Image.open(_membrane_png_for_index(stack1_dir, i)))
        if mem.ndim == 3:
            mem = mem[..., 0]
        labs.append((mem > 0).astype(np.uint16))
        raws.append(r)
    shapes = {x.shape for x in raws}
    if len(shapes) != 1:
        raise ValueError("VNC stack1 raw slices have inconsistent shapes: %s" % shapes)
    return np.stack(raws, axis=0), np.stack(labs, axis=0)


def load_isbi2012_volume_pair(data_dir: str, raw_name: str, lab_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load one ISBI-2012 stack as (Z, H, W): float32 raw + uint16 instance labels.

    Labels in the TIFF are binary foreground; instances are 3D connected components (skimage ``label``).
    """
    raw = np.asarray(tifffile.imread(os.path.join(data_dir, raw_name)), dtype=np.float32)
    lab_tif = np.asarray(tifffile.imread(os.path.join(data_dir, lab_name)))
    if raw.shape != lab_tif.shape:
        raise ValueError(
            "ISBI-2012 raw %r shape %s != labels %r shape %s"
            % (raw_name, raw.shape, lab_name, lab_tif.shape)
        )
    fg = (lab_tif != 0).astype(np.uint8)
    lab = connected_components(fg).astype(np.uint16)
    return raw, lab


def microns_neuron_zarr_stems(zarr_subset_dir: str) -> Tuple[str, ...]:
    """Sorted volume keys (stems of ``*.zarr`` directories) under ``.../Neuron_zarr/{subset}/``."""
    d = os.path.abspath(zarr_subset_dir)
    if not os.path.isdir(d):
        return ()
    out: List[str] = []
    for p in sorted(glob.glob(os.path.join(d, "*.zarr"))):
        if os.path.isdir(p):
            out.append(os.path.basename(p)[:-5])
    return tuple(out)


def pinky_zarr_volume_keys(zarr_subset_dir: str) -> Tuple[str, ...]:
    """Pinky keys in ``PINKY_ALL_VOLUME_KEYS`` order, skipping missing stores."""
    have = frozenset(microns_neuron_zarr_stems(zarr_subset_dir))
    return tuple(k for k in PINKY_ALL_VOLUME_KEYS if k in have)


def load_microns_neuron_zarr_volume(zarr_subset_dir: str, volume_key: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load one MICrONS Neuron zarr as (Z, H, W): float32 EM + uint16 instance labels.

    Expects ``data/MICrONS/transfer_zarr.py`` layout: ``{volume_key}.zarr/volumes/raw`` and
    ``.../labels`` (mask already applied; bbox crop already applied).
    """
    path = os.path.join(os.path.abspath(zarr_subset_dir), volume_key + ".zarr")
    if not os.path.isdir(path):
        raise FileNotFoundError("MICrONS neuron zarr not found: %s" % path)
    root = zarr.open(path, mode="r")
    raw = np.asarray(root["volumes"]["raw"], dtype=np.float32)
    lab = np.asarray(root["volumes"]["labels"])
    if raw.shape != lab.shape:
        raise ValueError("MICrONS zarr %s: raw %s != labels %s" % (volume_key, raw.shape, lab.shape))
    mx = int(lab.max())
    if mx >= 65536:
        raise ValueError("MICrONS zarr %s: label max %d exceeds uint16" % (volume_key, mx))
    return raw, lab.astype(np.uint16)


def _axonem_h5_list_3d_datasets(h5_path: str) -> List[Tuple[str, Tuple[int, ...], str]]:
    out: List[Tuple[str, Tuple[int, ...], str]] = []
    with h5py.File(h5_path, "r") as f:

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                out.append((name, tuple(obj.shape), str(obj.dtype)))

        f.visititems(visitor)
    return out


def _axonem_h5_pick_main_3d_dataset(h5_path: str) -> str:
    dss = _axonem_h5_list_3d_datasets(h5_path)
    cands = [(n, sh, dt) for (n, sh, dt) in dss if len(sh) == 3 and all(s > 0 for s in sh)]
    if not cands:
        raise RuntimeError("AxonEM %s: no 3D dataset. Found: %s" % (h5_path, dss))
    cands.sort(key=lambda x: np.prod(x[1]), reverse=True)
    return cands[0][0]


def _axonem_volume_key_from_stem(stem: str, prefix: str) -> str:
    """stem like ``im_0-0-0_pad`` or ``seg_0-0-0_pad`` → key ``0-0-0``."""
    p = prefix + "_"
    if not stem.startswith(p):
        raise ValueError("AxonEM: expected stem %r to start with %r" % (stem, p))
    rest = stem[len(p) :]
    if not rest.endswith("_pad"):
        raise ValueError("AxonEM: expected stem %r to end with '_pad'" % stem)
    return rest[: -len("_pad")]


def axonem_h5_paired_volume_keys(data_dir: str) -> Tuple[str, ...]:
    """Sorted cuboid keys that have both ``im_*_pad.h5`` and ``seg_*_pad.h5``."""
    d = os.path.abspath(data_dir)
    im_map: dict = {}
    for p in glob.glob(os.path.join(d, "im_*_pad.h5")):
        stem, _ = os.path.splitext(os.path.basename(p))
        im_map[_axonem_volume_key_from_stem(stem, "im")] = p
    seg_map: dict = {}
    for p in glob.glob(os.path.join(d, "seg_*_pad.h5")):
        stem, _ = os.path.splitext(os.path.basename(p))
        seg_map[_axonem_volume_key_from_stem(stem, "seg")] = p
    common = sorted(set(im_map.keys()) & set(seg_map.keys()))
    return tuple(common)


def _crop_zyx_to_label_bbox(
    raw: np.ndarray, lab: np.ndarray, margin: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """Shrink padded AxonEM stacks to the bounding box of ``lab > 0`` (annotated core)."""
    fg = lab > 0
    if not np.any(fg):
        return raw, lab
    zz, yy, xx = np.where(fg)
    z0, z1 = int(zz.min()), int(zz.max()) + 1
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    Z, Y, X = raw.shape
    z0 = max(0, z0 - margin)
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    z1 = min(Z, z1 + margin)
    y1 = min(Y, y1 + margin)
    x1 = min(X, x1 + margin)
    return raw[z0:z1, y0:y1, x0:x1], lab[z0:z1, y0:y1, x0:x1]


def load_axonem_h5_volume(data_dir: str, volume_key: str) -> Tuple[np.ndarray, np.ndarray]:
    """One AxonEM cuboid as (Z, Y, X): float32 EM + uint16 labels (same H5 layout as ``plot.ipynb``: ``main`` volume).

    EM30-H carries multi-instance ids; EM30-M H5s use binary ``{0,1}`` foreground masks (still valid training labels).
    """
    d = os.path.abspath(data_dir)
    im_path = os.path.join(d, "im_%s_pad.h5" % volume_key)
    seg_path = os.path.join(d, "seg_%s_pad.h5" % volume_key)
    if not os.path.isfile(im_path):
        raise FileNotFoundError("AxonEM EM not found: %s" % im_path)
    if not os.path.isfile(seg_path):
        raise FileNotFoundError("AxonEM seg not found: %s" % seg_path)
    im_ds = _axonem_h5_pick_main_3d_dataset(im_path)
    seg_ds = _axonem_h5_pick_main_3d_dataset(seg_path)
    with h5py.File(im_path, "r") as f:
        raw = np.asarray(f[im_ds], dtype=np.float32)
    with h5py.File(seg_path, "r") as f:
        lab = np.asarray(f[seg_ds])
    if raw.shape != lab.shape:
        z = min(raw.shape[0], lab.shape[0])
        y = min(raw.shape[1], lab.shape[1])
        x = min(raw.shape[2], lab.shape[2])
        raw = raw[:z, :y, :x]
        lab = lab[:z, :y, :x]
    raw, lab = _crop_zyx_to_label_bbox(raw, lab, margin=8)
    mx = int(lab.max())
    if mx >= 65536:
        raise ValueError("AxonEM %s: label max %d exceeds uint16" % (volume_key, mx))
    return raw, lab.astype(np.uint16)


def load_ac3_volume(ac3_dir: str, num_slices: int = AC3_NUM_SLICES) -> Tuple[np.ndarray, np.ndarray]:
    """Load AC3 as (Z, H, W): float32 EM + uint16 instance labels.

    Raw slices live under ``ac3_dir/ac3_EM/`` as
    ``Thousand_highmag_256slices_2kcenter_1k_inv_{z:04d}.png`` for z = 0 .. num_slices-1.
    Daniel RGB instance labels under ``ac3_dir/ac3_dbseg_images/`` use inverted Z:
    EM index z pairs with ``ac3_daniel_s{num_slices - z:04d}.png`` (see ``data/AC3/plot.ipynb``).

    Each distinct non-black RGB color is assigned a stable instance id across the whole stack
    (0 = background / unlabeled). Labels are sparse per slice.
    """
    em_dir = os.path.join(ac3_dir, "ac3_EM")
    seg_dir = os.path.join(ac3_dir, "ac3_dbseg_images")
    if not os.path.isdir(em_dir):
        raise FileNotFoundError("AC3 EM dir not found: %s" % em_dir)
    if not os.path.isdir(seg_dir):
        raise FileNotFoundError("AC3 segmentation dir not found: %s" % seg_dir)

    raws: List[np.ndarray] = []
    labs: List[np.ndarray] = []
    color_to_id: dict = {}
    next_id = 1
    for z in range(num_slices):
        em_path = os.path.join(em_dir, AC3_EM_SLICE_BASENAME % z)
        s_num = num_slices - z
        seg_path = os.path.join(seg_dir, AC3_DANIEL_SEG_BASENAME % s_num)
        if not os.path.isfile(em_path):
            raise FileNotFoundError("Missing AC3 EM slice: %s" % em_path)
        if not os.path.isfile(seg_path):
            raise FileNotFoundError("Missing AC3 Daniel seg slice: %s" % seg_path)
        raw = np.asarray(Image.open(em_path), dtype=np.float32)
        if raw.ndim != 2:
            raw = np.squeeze(raw)
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
        lab = gids[inv].reshape(packed.shape)
        raws.append(raw)
        labs.append(lab)
    raw_vol = np.stack(raws, axis=0)
    lab_vol = np.stack(labs, axis=0)
    shapes = {x.shape for x in raws}
    if len(shapes) != 1:
        raise ValueError("AC3 EM slices have inconsistent shapes: %s" % shapes)
    if raw_vol.shape != lab_vol.shape:
        raise ValueError("AC3 raw shape %s != label shape %s" % (raw_vol.shape, lab_vol.shape))
    return raw_vol, lab_vol


def load_ac4_volume(ac4_dir: str, num_slices: int = AC4_NUM_SLICES) -> Tuple[np.ndarray, np.ndarray]:
    """Load AC4 as (Z, H, W): float32 EM + uint16 instance labels.

    Raw slices live under ``ac4_dir/ac4_EM/`` as ``affinecropped4_inv_{k:04d}.png`` with **1-based**
    ``k = 1 .. num_slices`` (stack index z = 0 .. num_slices-1 uses ``k = z + 1``).

    Daniel RGB labels under ``ac4_dir/ac4_seg_daniel/`` use inverted Z vs those EM names:
    slice z pairs with ``ac4_daniel_s{num_slices - z:04d}.png`` (e.g. ``inv_0001`` ↔ ``s0100`` when
    num_slices is 100; see ``data/AC4/plot.ipynb``).

    Each distinct non-black RGB color is assigned a stable instance id across the whole stack
    (0 = background / unlabeled).
    """
    em_dir = os.path.join(ac4_dir, "ac4_EM")
    seg_dir = os.path.join(ac4_dir, "ac4_seg_daniel")
    if not os.path.isdir(em_dir):
        raise FileNotFoundError("AC4 EM dir not found: %s" % em_dir)
    if not os.path.isdir(seg_dir):
        raise FileNotFoundError("AC4 Daniel seg dir not found: %s" % seg_dir)

    raws: List[np.ndarray] = []
    labs: List[np.ndarray] = []
    color_to_id: dict = {}
    next_id = 1
    for z in range(num_slices):
        k = z + 1
        em_path = os.path.join(em_dir, AC4_EM_SLICE_BASENAME % k)
        s_num = num_slices - z
        seg_path = os.path.join(seg_dir, AC4_DANIEL_SEG_BASENAME % s_num)
        if not os.path.isfile(em_path):
            raise FileNotFoundError("Missing AC4 EM slice: %s" % em_path)
        if not os.path.isfile(seg_path):
            raise FileNotFoundError("Missing AC4 Daniel seg slice: %s" % seg_path)
        raw = np.asarray(Image.open(em_path), dtype=np.float32)
        if raw.ndim != 2:
            raw = np.squeeze(raw)
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
        lab = gids[inv].reshape(packed.shape)
        raws.append(raw)
        labs.append(lab)
    raw_vol = np.stack(raws, axis=0)
    lab_vol = np.stack(labs, axis=0)
    shapes = {x.shape for x in raws}
    if len(shapes) != 1:
        raise ValueError("AC4 EM slices have inconsistent shapes: %s" % shapes)
    if raw_vol.shape != lab_vol.shape:
        raise ValueError("AC4 raw shape %s != label shape %s" % (raw_vol.shape, lab_vol.shape))
    return raw_vol, lab_vol


# ---------------------------------------------------------------------------
# Slice list builders
# ---------------------------------------------------------------------------


def _append_xy_slices(
    images: list,
    masks: list,
    raw: np.ndarray,
    lab: np.ndarray,
    split: str,
    n_val: int,
    *,
    lsd_entries: Optional[list] = None,
    z_raw_fn: Optional[Callable[[int, np.ndarray, np.ndarray], np.ndarray]] = None,
    skip_empty_z: bool = False,
    lsd_cache_path: Optional[str] = None,
) -> None:
    z = lab.shape[0]
    if split == "train":
        z_lo, z_hi = n_val, z
    else:
        z_lo, z_hi = 0, n_val
    _lazy = _is_lazy_source(raw)
    for n in range(z_lo, z_hi):
        sl = lab[n]
        if skip_empty_z and sl.max() == 0:
            continue
        if _lazy:
            images.append(_LazySlice(_mk_loader_xy(raw, n, z_raw_fn, lab)))
        elif z_raw_fn is None:
            images.append(raw[n])
        else:
            images.append(z_raw_fn(n, raw, lab))
        masks.append(sl)
        if lsd_entries is not None and lsd_cache_path is not None:
            lsd_entries.append((lsd_cache_path, "xy", n))


def _append_xz_yz(
    images: list,
    masks: list,
    raw: np.ndarray,
    lab: np.ndarray,
    split: str,
    n_val: int,
    *,
    lsd_entries: Optional[list] = None,
    y_raw_fn=None,
    x_raw_fn=None,
    skip_empty_y: bool = False,
    skip_empty_x: bool = False,
    lsd_cache_path: Optional[str] = None,
) -> None:
    if split == "train":
        y_lo, y_hi = n_val, lab.shape[1]
        x_lo, x_hi = n_val, lab.shape[2]
    else:
        y_lo, y_hi = 0, n_val
        x_lo, x_hi = 0, n_val

    _lazy = _is_lazy_source(raw)

    for n in range(y_lo, y_hi):
        sl = lab[:, n, :]
        if skip_empty_y and sl.max() == 0:
            continue
        if _lazy:
            images.append(_LazySlice(_mk_loader_xz(raw, n, y_raw_fn, lab)))
        elif y_raw_fn is None:
            images.append(raw[:, n, :])
        else:
            images.append(y_raw_fn(n, raw, lab))
        masks.append(sl)
        if lsd_entries is not None and lsd_cache_path is not None:
            lsd_entries.append((lsd_cache_path, "xz", n))

    for n in range(x_lo, x_hi):
        sl = lab[:, :, n]
        if skip_empty_x and sl.max() == 0:
            continue
        if _lazy:
            images.append(_LazySlice(_mk_loader_yz(raw, n, x_raw_fn, lab)))
        elif x_raw_fn is None:
            images.append(raw[:, :, n])
        else:
            images.append(x_raw_fn(n, raw, lab))
        masks.append(sl)
        if lsd_entries is not None and lsd_cache_path is not None:
            lsd_entries.append((lsd_cache_path, "yz", n))


# ---------------------------------------------------------------------------
# Lazy slice loading — avoids keeping large 3D raw arrays in RAM.
# ---------------------------------------------------------------------------


class _LazySlice:
    """Deferred 2D raw slice.  Stores a zero-arg callable that returns float32 ndarray.

    Implements ``__array__`` so ``np.asarray(lazy_slice)`` transparently triggers the
    actual read.  This keeps zarr / h5py data off the heap until ``__getitem__`` runs.
    """

    __slots__ = ("_loader",)

    def __init__(self, loader):
        self._loader = loader

    def __array__(self, dtype=None):
        arr = self._loader()
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if dtype is not None and np.dtype(dtype) != np.float32:
            arr = arr.astype(dtype)
        return arr

    def __repr__(self):
        return "_LazySlice(<deferred>)"


class _ZarrCropView:
    """Lazy read-only crop of a zarr.Array — no data read until indexing.

    Supports the three 2D-slice access patterns used by ``_append_xy_slices`` /
    ``_append_xz_yz``:
        view[n]          → XY slice  (axis-0 integer)
        view[:, n, :]    → XZ slice  (axis-1 integer)
        view[:, :, n]    → YZ slice  (axis-2 integer)
    """

    __slots__ = ("_z", "_z0", "_z1", "_y0", "_y1", "_x0", "_x1")

    def __init__(self, zarr_arr, z0: int, z1: int, y0: int, y1: int, x0: int, x1: int):
        self._z = zarr_arr
        self._z0, self._z1 = int(z0), int(z1)
        self._y0, self._y1 = int(y0), int(y1)
        self._x0, self._x1 = int(x0), int(x1)

    @property
    def shape(self):
        return (self._z1 - self._z0, self._y1 - self._y0, self._x1 - self._x0)

    @property
    def dtype(self):
        return self._z.dtype

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self._z[self._z0 + int(key), self._y0 : self._y1, self._x0 : self._x1]
        if isinstance(key, tuple) and len(key) == 3:
            k0, k1, k2 = key
            if k0 == slice(None) and isinstance(k1, (int, np.integer)) and k2 == slice(None):
                return self._z[self._z0 : self._z1, self._y0 + int(k1), self._x0 : self._x1]
            if k0 == slice(None) and k1 == slice(None) and isinstance(k2, (int, np.integer)):
                return self._z[self._z0 : self._z1, self._y0 : self._y1, self._x0 + int(k2)]
        raise IndexError(f"_ZarrCropView: unsupported index {key!r}")


def _is_lazy_source(arr) -> bool:
    """True when indexing ``arr`` triggers I/O rather than returning a numpy view."""
    if isinstance(arr, _ZarrCropView):
        return True
    try:
        import zarr as _zarr_mod

        return isinstance(arr, _zarr_mod.Array)
    except ImportError:
        return False


def _mk_loader_xy(raw, n: int, fn=None, lab=None):
    """Factory: returns a zero-arg callable that loads one XY raw slice."""
    if fn is not None:
        def _l():
            return np.asarray(fn(n, raw, lab), dtype=np.float32)
    else:
        def _l():
            return np.asarray(raw[n], dtype=np.float32)
    return _l


def _mk_loader_xz(raw, n: int, fn=None, lab=None):
    """Factory: XZ slice (axis-1)."""
    if fn is not None:
        def _l():
            return np.asarray(fn(n, raw, lab), dtype=np.float32)
    else:
        def _l():
            return np.asarray(raw[:, n, :], dtype=np.float32)
    return _l


def _mk_loader_yz(raw, n: int, fn=None, lab=None):
    """Factory: YZ slice (axis-2)."""
    if fn is not None:
        def _l():
            return np.asarray(fn(n, raw, lab), dtype=np.float32)
    else:
        def _l():
            return np.asarray(raw[:, :, n], dtype=np.float32)
    return _l


def _zebra_z_raw(n: int, raw: np.ndarray, lab: np.ndarray) -> np.ndarray:
    return raw[100 + n, 200 : (200 + lab.shape[1]), 200 : (200 + lab.shape[2])]


def _zebra_y_raw(n: int, raw: np.ndarray, lab: np.ndarray) -> np.ndarray:
    return raw[100 : (100 + lab.shape[0]), 200 + n, 200 : (200 + lab.shape[2])]


def _zebra_x_raw(n: int, raw: np.ndarray, lab: np.ndarray) -> np.ndarray:
    return raw[100 : (100 + lab.shape[0]), 200 : (200 + lab.shape[1]), 200 + n]


# ---------------------------------------------------------------------------
# 2D Dataset (standard + SAM + zebra avalanche)
# ---------------------------------------------------------------------------


class EM2DDataset(Dataset):
    """One sample = one 2D slice (or xz/yz)."""

    def __init__(
        self,
        images: Sequence[np.ndarray],
        masks: Sequence[np.ndarray],
        split: str,
        crop_size: Optional[int],
        padding_size: int,
        require_lsd: bool,
        *,
        lsd_entries: Optional[Sequence[Tuple[str, str, int]]] = None,
        aug_check_shapes: bool = True,
        sam: bool = False,
        avalanche: bool = False,
        augment: Optional[bool] = None,
        crop_mode: str = "random",
        extra_item_meta: bool = False,
        segem_crop_kwargs: Optional[Dict[str, object]] = None,
    ):
        self.images = list(images)
        self.masks = list(masks)
        self.lsd_entries = list(lsd_entries) if lsd_entries is not None else None
        self.split = split
        self.crop_size = crop_size
        self.padding_size = padding_size
        self.require_lsd = require_lsd
        self._aug_check = aug_check_shapes
        self.sam = sam
        self.avalanche = avalanche
        self.augment = (split == "train") if augment is None else augment
        self.crop_mode = str(crop_mode)
        self.extra_item_meta = bool(extra_item_meta)
        self.segem_crop_kwargs = dict(segem_crop_kwargs) if segem_crop_kwargs is not None else {}
        self._pad_total = padding_for_crop(self.crop_size, self.padding_size)
        # Populated by preload_lsd_cache(); maps cache_path -> opened zarr array.
        self._lsd_cache: Optional[Dict[str, object]] = None
        # Set to True externally (e.g. set_concat_dataset_attr) to activate the
        # minimal (raw, affinity, lsd) output used by train_ACRLSD_2d_neo.py.
        self.minimal_output: bool = False
        self._minimal_live_cache: Optional[Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = (
            {} if not self.augment else None
        )

    def preload_lsd_cache(self) -> None:
        """Open approximate 2D LSD cache handles once per dataset."""
        if not self.require_lsd:
            return
        if self._lsd_cache is not None:
            return
        if not self.lsd_entries:
            raise RuntimeError("require_lsd=True but no lsd_entries metadata is available")
        unique_paths = sorted({entry[0] for entry in self.lsd_entries})
        print(
            "[EM2DDataset] opening approx 2D LSD cache files for {} slices ...".format(len(self.masks)),
            flush=True,
        )
        self._lsd_cache = {}
        for path in unique_paths:
            if not os.path.exists(path):
                raise FileNotFoundError("Approximate 2D LSD cache not found: {}".format(path))
            self._lsd_cache[path] = zarr.open(path, mode="r")

    def __len__(self):
        return len(self.images)

    def _require_crop_size(self) -> int:
        if self.crop_size is None:
            raise ValueError(
                "crop_size=None is only supported for direct full-slice access via dataset.images/masks; "
                "set an integer crop_size before using __getitem__."
            )
        return self.crop_size

    def _getitem_minimal_prelsd(self, idx: int):
        """Fast path: reads approximate 2D LSD from existing 3D cache, returns (raw, affinity, lsd) only."""
        raw = np.asarray(self.images[idx], dtype=np.float32)
        labels = np.asarray(self.masks[idx], dtype=np.uint16).copy()
        raw = normalize_minmax(raw)
        cache_path, orientation, slice_idx = self.lsd_entries[idx]
        lsd = _read_approx_2d_lsd_from_3d_cache(self._lsd_cache[cache_path], orientation, int(slice_idx))
        crop_size = self._require_crop_size()
        raw, labels, lsd = _prepare_2d_with_lsd(
            raw, labels, lsd, crop_size, augment=self.augment
        )
        raw = np.expand_dims(raw, axis=0)
        labels = erode_instance_labels(labels, iterations=1, border_value=1)
        affinity = affinity_2d(labels)
        return raw, affinity, lsd

    def _getitem_minimal_live_lsd(self, idx: int):
        """Fast path: compute only the targets needed by ACRLSD training."""
        if self._minimal_live_cache is not None and idx in self._minimal_live_cache:
            return self._minimal_live_cache[idx]

        raw = np.asarray(self.images[idx], dtype=np.float32)
        labels = np.asarray(self.masks[idx], dtype=np.uint16).copy()
        raw = normalize_minmax(raw)
        crop_size = self._require_crop_size()
        raw, labels = prepare_2d_pair(
            raw, labels, crop_size, self._pad_total, self._aug_check, augment=self.augment
        )
        raw = np.expand_dims(raw, axis=0)
        labels = erode_instance_labels(labels, iterations=1, border_value=1)
        affinity = affinity_2d(labels).astype(np.float32, copy=False)
        lsds = compute_2d_lsds_bbox_local(labels, sigma=(5.0, 5.0)).astype(np.float32, copy=False)
        sample = (raw.astype(np.float32, copy=False), affinity, lsds)
        if self._minimal_live_cache is not None:
            self._minimal_live_cache[idx] = sample
        return sample

    def _getitem_standard(self, idx: int):
        raw = np.asarray(self.images[idx], dtype=np.float32)
        labels = np.asarray(self.masks[idx], dtype=np.uint16).copy()
        raw = normalize_minmax(raw)
        centric_tid = None
        crop_size = self._require_crop_size()
        if self.crop_mode == "instance_centric":
            raw, labels, centric_tid = prepare_2d_instance_centric_pair(
                raw,
                labels,
                crop_size,
                self._pad_total,
                self._aug_check,
                augment=self.augment,
                **self.segem_crop_kwargs,
            )
        else:
            raw, labels = prepare_2d_pair(
                raw, labels, crop_size, self._pad_total, self._aug_check, augment=self.augment
            )
        raw = np.expand_dims(raw, axis=0)
        labels = erode_instance_labels(labels, iterations=1, border_value=1)
        if centric_tid is not None and not np.any(labels == centric_tid):
            centric_tid = None
        affinity = affinity_2d(labels)
        pp, pl, _boxes, pmask = get_prompt_2d(labels)
        point_map = gaussian_point_map(pp, pl, crop_size, crop_size, theta=30)
        crop_meta = {"centric_target_id": centric_tid} if centric_tid is not None else {}
        if self.require_lsd:
            lsds = compute_2d_lsds_bbox_local(labels, sigma=(5.0, 5.0)).astype(np.float32)
            if self.extra_item_meta:
                return raw, labels, point_map, pmask, affinity, lsds, crop_meta
            return raw, labels, point_map, pmask, affinity, lsds
        if self.extra_item_meta:
            return raw, labels, point_map, pmask, affinity, crop_meta
        return raw, labels, point_map, pmask, affinity

    def _getitem_sam(self, idx: int):
        raw = np.asarray(self.images[idx], dtype=np.float32)
        labels = np.asarray(self.masks[idx], dtype=np.uint16).copy()
        crop_size = self._require_crop_size()
        raw, labels = prepare_2d_pair(
            raw, labels, crop_size, self._pad_total, self._aug_check, augment=self.augment
        )
        raw = np.expand_dims(raw, axis=0)
        labels = erode_instance_labels(labels, iterations=1, border_value=1)
        affinity = affinity_2d(labels)
        pp, pl, boxes, pmask = get_prompt_2d_sam(labels)
        point_map = gaussian_point_map(pp, pl, crop_size, crop_size, theta=30)
        if self.require_lsd:
            lsds = compute_2d_lsds_bbox_local(labels, sigma=(5.0, 5.0)).astype(np.float32)
            return raw, labels, pp, pl, boxes, point_map, pmask, affinity, lsds
        return raw, labels, pp, pl, boxes, point_map, pmask, affinity

    def __getitem__(self, idx: int):
        if self.minimal_output:
            if self._lsd_cache is not None and self.lsd_entries is not None:
                return self._getitem_minimal_prelsd(idx)
            return self._getitem_minimal_live_lsd(idx)
        if self.avalanche:
            return self._getitem_avalanche(idx)
        if self.sam:
            return self._getitem_sam(idx)
        return self._getitem_standard(idx)

    def _getitem_avalanche(self, idx: int):
        """Exact tuple layout expected by train_segEM2d_CL_avalanche validation loop."""
        raw = np.asarray(self.images[idx], dtype=np.float32)
        labels = np.asarray(self.masks[idx], dtype=np.uint16).copy()
        raw = normalize_minmax(raw)
        crop_size = self._require_crop_size()
        raw, labels = prepare_2d_pair(
            raw, labels, crop_size, self._pad_total, self._aug_check, augment=self.augment
        )
        raw = np.expand_dims(raw, axis=0)
        labels = erode_instance_labels(labels, iterations=1, border_value=1)
        affinity = affinity_2d(labels)
        pp, pl, _b, pmask = get_prompt_2d(labels)
        point_map = gaussian_point_map(pp, pl, crop_size, crop_size, theta=30)
        if self.require_lsd:
            lsds = compute_2d_lsds_bbox_local(labels, sigma=(5.0, 5.0)).astype(np.float32)
            return (
                raw.astype(np.float32),
                1,
                labels.astype(np.int32),
                point_map.astype(np.float32),
                pmask.astype(np.float32),
                affinity.astype(np.float32),
                lsds.astype(np.float32),
            )
        return (
            raw.astype(np.float32),
            1,
            labels.astype(np.int32),
            point_map.astype(np.float32),
            pmask.astype(np.float32),
            affinity.astype(np.float32),
        )


class Dataset_2D_hemi_Train(EM2DDataset):
    def __init__(
        self,
        data_dir=HEMI_DEFAULT_DATA_DIR,
        split="train",
        crop_size=None,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
        augment: Optional[bool] = None,
        lsd_cache_dir: str = DEFAULT_APPROX_2D_LSD_CACHE_DIR,
    ):
        images, masks, lsd_entries = [], [], []
        for name in HEMI_ZARR_FILES:
            root = zarr.open(os.path.join(data_dir, name), mode="r")
            raw_zarr = root["volumes"]["raw"]
            lab_zarr = root["volumes"]["labels"]["neuron_ids"]
            # Compute crop bounds without reading any raw data
            z0, z1 = 128, int(raw_zarr.shape[0]) - 128
            y0, y1 = 128, int(raw_zarr.shape[1]) - 128
            x0, x1 = 128, int(raw_zarr.shape[2]) - 128
            # Labels must be materialised for 3-D connected-components; raw stays lazy
            lab = connected_components(
                np.asarray(lab_zarr[z0:z1, y0:y1, x0:x1])
            ).astype(np.uint16)
            raw = _ZarrCropView(raw_zarr, z0, z1, y0, y1, x0, x1)
            print("data {}: raw shape={}, label shape = {}".format(name, raw.shape, lab.shape))
            lsd_cache_path = _approx_2d_lsd_cache_xy_path(lsd_cache_dir, "hemi", name)
            _append_xy_slices(
                images, masks, raw, lab, split, n_val, lsd_entries=lsd_entries, lsd_cache_path=lsd_cache_path
            )
            if require_xz_yz:
                _append_xz_yz(
                    images, masks, raw, lab, split, n_val, lsd_entries=lsd_entries, lsd_cache_path=lsd_cache_path
                )
        super().__init__(images, masks, split, crop_size, padding_size, require_lsd, lsd_entries=lsd_entries, augment=augment)


class Dataset_2D_fib25_Train(EM2DDataset):
    def __init__(
        self,
        data_dir=FIB25_DEFAULT_DATA_DIR,
        split="train",
        crop_size=None,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
        augment: Optional[bool] = None,
        lsd_cache_dir: str = DEFAULT_APPROX_2D_LSD_CACHE_DIR,
    ):
        images, masks, lsd_entries = [], [], []
        for name in FIB25_ZARR_FILES:
            root = zarr.open(os.path.join(data_dir, name), mode="r")
            raw = root["volumes"]["raw"]
            lab = connected_components(np.asarray(root["volumes"]["labels"]["neuron_ids"])).astype(np.uint16)
            print("data {}: raw shape={}, label shape = {}".format(name, raw.shape, lab.shape))
            lsd_cache_path = _approx_2d_lsd_cache_xy_path(lsd_cache_dir, "fib25", name)
            _append_xy_slices(
                images, masks, raw, lab, split, n_val, lsd_entries=lsd_entries, lsd_cache_path=lsd_cache_path
            )
            if require_xz_yz:
                _append_xz_yz(
                    images, masks, raw, lab, split, n_val, lsd_entries=lsd_entries, lsd_cache_path=lsd_cache_path
                )
        super().__init__(images, masks, split, crop_size, padding_size, require_lsd, lsd_entries=lsd_entries, augment=augment)


class Dataset_2D_cremi_Train(EM2DDataset):
    def __init__(
        self,
        data_dir=CREMI_DEFAULT_DATA_DIR,
        split="train",
        crop_size=None,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
        augment: Optional[bool] = None,
        lsd_cache_dir: str = DEFAULT_APPROX_2D_LSD_CACHE_DIR,
    ):
        images, masks, lsd_entries = [], [], []
        for name in CREMI_HDF_FILES:
            path = os.path.join(data_dir, name)
            with h5py.File(path, "r") as f:
                vol = f["volumes"]
                raw = np.asarray(vol["raw"])
                lab = np.asarray(vol["labels"]["neuron_ids"])
            if name == "sample_C_20160501.hdf":
                raw = np.delete(raw, [14, 74], 0)
                lab = np.delete(lab, [14, 74], 0)
            lab = connected_components(lab).astype(np.uint16)
            print("data {}: raw shape={}, label shape = {}".format(name, raw.shape, lab.shape))
            lsd_cache_path = _approx_2d_lsd_cache_xy_path(lsd_cache_dir, "cremi", name)
            _append_xy_slices(
                images, masks, raw, lab, split, n_val, lsd_entries=lsd_entries, lsd_cache_path=lsd_cache_path
            )
            if require_xz_yz:
                _append_xz_yz(
                    images, masks, raw, lab, split, n_val, lsd_entries=lsd_entries, lsd_cache_path=lsd_cache_path
                )
        super().__init__(images, masks, split, crop_size, padding_size, require_lsd, lsd_entries=lsd_entries, augment=augment)


class Dataset_2D_VNC_Train(EM2DDataset):
    def __init__(
        self,
        data_dir=VNC_DEFAULT_DATA_DIR,
        split="train",
        crop_size=None,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: Optional[int] = None,
        augment: Optional[bool] = None,
        lsd_cache_dir: str = DEFAULT_APPROX_2D_LSD_CACHE_DIR,
    ):
        stack1 = os.path.abspath(data_dir)
        raw, lab = load_vnc_stack1_volume(stack1)
        z = raw.shape[0]
        if n_val is None:
            n_val = min(8, max(1, z // 5))
        print("data VNC stack1: raw shape={}, label shape = {}".format(raw.shape, lab.shape))
        images, masks, lsd_entries = [], [], []
        lsd_cache_path = _approx_2d_lsd_cache_xy_path(lsd_cache_dir, "vnc", "stack1")
        _append_xy_slices(
            images, masks, raw, lab, split, n_val, lsd_entries=lsd_entries, lsd_cache_path=lsd_cache_path
        )
        if require_xz_yz:
            _append_xz_yz(
                images, masks, raw, lab, split, n_val, lsd_entries=lsd_entries, lsd_cache_path=lsd_cache_path
            )
        super().__init__(images, masks, split, crop_size, padding_size, require_lsd, lsd_entries=lsd_entries, augment=augment)


class Dataset_2D_isbi2012_Train(EM2DDataset):
    """ISBI 2012 SNEMI3D: train + test volumes, dense binary labels → 3D CC instance ids."""

    def __init__(
        self,
        data_dir=ISBI2012_DEFAULT_DATA_DIR,
        split="train",
        crop_size=None,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
        augment: Optional[bool] = None,
        lsd_cache_dir: str = DEFAULT_APPROX_2D_LSD_CACHE_DIR,
    ):
        images, masks, lsd_entries = [], [], []
        for raw_name, lab_name in ISBI2012_TIFF_PAIRS:
            raw, lab = load_isbi2012_volume_pair(data_dir, raw_name, lab_name)
            print("data ISBI-2012 {}: raw shape={}, label shape = {}".format(raw_name, raw.shape, lab.shape))
            lsd_cache_path = _approx_2d_lsd_cache_xy_path(lsd_cache_dir, "isbi2012", raw_name)
            _append_xy_slices(
                images, masks, raw, lab, split, n_val, lsd_entries=lsd_entries, lsd_cache_path=lsd_cache_path
            )
            if require_xz_yz:
                _append_xz_yz(
                    images, masks, raw, lab, split, n_val, lsd_entries=lsd_entries, lsd_cache_path=lsd_cache_path
                )
        super().__init__(images, masks, split, crop_size, padding_size, require_lsd, lsd_entries=lsd_entries, augment=augment)


class Dataset_2D_ac3_Train(EM2DDataset):
    """AC3: 256 PNG EM slices + Daniel RGB neuron instances (Z order inverted vs EM filenames).

    Skips xy (and xz/yz) slices with no foreground so prompt sampling never deadlocks on empty labels.
    """

    def __init__(
        self,
        data_dir=AC3_DEFAULT_DATA_DIR,
        split="train",
        crop_size=None,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
        num_slices: int = AC3_NUM_SLICES,
        augment: Optional[bool] = None,
        lsd_cache_dir: str = DEFAULT_APPROX_2D_LSD_CACHE_DIR,
    ):
        images, masks, lsd_entries = [], [], []
        raw, lab = load_ac3_volume(data_dir, num_slices=num_slices)
        print("data AC3: raw shape={}, label shape = {}".format(raw.shape, lab.shape))
        lsd_cache_path = _approx_2d_lsd_cache_xy_path(lsd_cache_dir, "ac3", "ac3")
        _append_xy_slices(
            images,
            masks,
            raw,
            lab,
            split,
            n_val,
            lsd_entries=lsd_entries,
            skip_empty_z=True,
            lsd_cache_path=lsd_cache_path,
        )
        if require_xz_yz:
            _append_xz_yz(
                images,
                masks,
                raw,
                lab,
                split,
                n_val,
                lsd_entries=lsd_entries,
                skip_empty_y=True,
                skip_empty_x=True,
                lsd_cache_path=lsd_cache_path,
            )
        super().__init__(images, masks, split, crop_size, padding_size, require_lsd, lsd_entries=lsd_entries, augment=augment)


class Dataset_2D_ac4_Train(EM2DDataset):
    """AC4: 100 PNG EM slices (1-based filenames) + Daniel RGB instances (Z inverted vs EM).

    Skips empty slices like AC3 so prompt sampling stays well-defined.
    """

    def __init__(
        self,
        data_dir=AC4_DEFAULT_DATA_DIR,
        split="train",
        crop_size=None,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
        num_slices: int = AC4_NUM_SLICES,
        augment: Optional[bool] = None,
        lsd_cache_dir: str = DEFAULT_APPROX_2D_LSD_CACHE_DIR,
    ):
        images, masks, lsd_entries = [], [], []
        raw, lab = load_ac4_volume(data_dir, num_slices=num_slices)
        print("data AC4: raw shape={}, label shape = {}".format(raw.shape, lab.shape))
        lsd_cache_path = _approx_2d_lsd_cache_xy_path(lsd_cache_dir, "ac4", "ac4")
        _append_xy_slices(
            images,
            masks,
            raw,
            lab,
            split,
            n_val,
            lsd_entries=lsd_entries,
            skip_empty_z=True,
            lsd_cache_path=lsd_cache_path,
        )
        if require_xz_yz:
            _append_xz_yz(
                images,
                masks,
                lsd_entries,
                raw,
                lab,
                split,
                n_val,
                skip_empty_y=True,
                skip_empty_x=True,
                lsd_cache_path=lsd_cache_path,
            )
        super().__init__(images, masks, split, crop_size, padding_size, require_lsd, lsd_entries=lsd_entries, augment=augment)


def _append_microns_neuron_zarr_xy_slices(
    images: list,
    masks: list,
    lsd_entries: Optional[list],
    zarr_subset_dir: str,
    volume_keys: Sequence[str],
    split: str,
    n_val: int,
    require_xz_yz: bool,
    subset_label: str,
    cache_dir: str,
) -> None:
    for name in volume_keys:
        raw, lab = load_microns_neuron_zarr_volume(zarr_subset_dir, name)
        lsd_cache_path = _approx_2d_lsd_cache_xy_path(cache_dir, subset_label, name)
        print(
            "data MICrONS {} {}: raw shape={}, label shape = {}".format(
                subset_label, name, raw.shape, lab.shape
            )
        )
        _append_xy_slices(
            images,
            masks,
            raw,
            lab,
            split,
            n_val,
            lsd_entries=lsd_entries,
            skip_empty_z=True,
            lsd_cache_path=lsd_cache_path,
        )
        if require_xz_yz:
            _append_xz_yz(
                images,
                masks,
                raw,
                lab,
                split,
                n_val,
                lsd_entries=lsd_entries,
                skip_empty_y=True,
                skip_empty_x=True,
                lsd_cache_path=lsd_cache_path,
            )


def _append_axonem_h5_xy_slices(
    images: list,
    masks: list,
    lsd_entries: Optional[list],
    data_dir: str,
    volume_keys: Sequence[str],
    split: str,
    n_val: int,
    require_xz_yz: bool,
    subset_label: str,
    cache_dir: str,
) -> None:
    # Copy each 2D plane so the full (Z,Y,X) block can be freed; views would keep every volume in RAM.
    for name in volume_keys:
        raw, lab = load_axonem_h5_volume(data_dir, name)
        dataset_key = "axonem_h" if subset_label == "H" else "axonem_m"
        lsd_cache_path = _approx_2d_lsd_cache_xy_path(cache_dir, dataset_key, name)
        print(
            "data AxonEM %s %s: raw shape=%s, label shape = %s"
            % (subset_label, name, raw.shape, lab.shape)
        )
        z = lab.shape[0]
        if split == "train":
            z_lo, z_hi = n_val, z
        else:
            z_lo, z_hi = 0, n_val
        for n in range(z_lo, z_hi):
            sl = lab[n]
            if sl.max() == 0:
                continue
            images.append(np.array(raw[n], dtype=np.float32, copy=True))
            masks.append(np.array(sl, copy=True))
            if lsd_entries is not None:
                lsd_entries.append((lsd_cache_path, "xy", n))
        if require_xz_yz:
            if split == "train":
                y_lo, y_hi = n_val, lab.shape[1]
                x_lo, x_hi = n_val, lab.shape[2]
            else:
                y_lo, y_hi = 0, n_val
                x_lo, x_hi = 0, n_val
            for n in range(y_lo, y_hi):
                sl = lab[:, n, :]
                if sl.max() == 0:
                    continue
                images.append(np.array(raw[:, n, :], dtype=np.float32, copy=True))
                masks.append(np.array(sl, copy=True))
                if lsd_entries is not None:
                    lsd_entries.append((lsd_cache_path, "xz", n))
            for n in range(x_lo, x_hi):
                sl = lab[:, :, n]
                if sl.max() == 0:
                    continue
                images.append(np.array(raw[:, :, n], dtype=np.float32, copy=True))
                masks.append(np.array(sl, copy=True))
                if lsd_entries is not None:
                    lsd_entries.append((lsd_cache_path, "yz", n))
        del raw, lab


class Dataset_2D_basil_Train(EM2DDataset):
    """MICrONS basil: bbox-cropped stacks under ``Neuron_zarr/basil/*.zarr``."""

    def __init__(
        self,
        data_dir=MICRONS_BASIL_ZARR_DIR,
        split="train",
        crop_size=None,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
        volume_keys: Optional[Sequence[str]] = None,
        augment: Optional[bool] = None,
        lsd_cache_dir: str = DEFAULT_APPROX_2D_LSD_CACHE_DIR,
    ):
        images, masks, lsd_entries = [], [], []
        keys = tuple(volume_keys) if volume_keys is not None else microns_neuron_zarr_stems(data_dir)
        _append_microns_neuron_zarr_xy_slices(
            images, masks, lsd_entries, data_dir, keys, split, n_val, require_xz_yz, "basil", lsd_cache_dir
        )
        super().__init__(images, masks, split, crop_size, padding_size, require_lsd, lsd_entries=lsd_entries, augment=augment)


class Dataset_2D_minnie_Train(EM2DDataset):
    """MICrONS minnie: bbox-cropped stacks under ``Neuron_zarr/minnie/*.zarr``."""

    def __init__(
        self,
        data_dir=MICRONS_MINNIE_ZARR_DIR,
        split="train",
        crop_size=None,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
        volume_keys: Optional[Sequence[str]] = None,
        augment: Optional[bool] = None,
        lsd_cache_dir: str = DEFAULT_APPROX_2D_LSD_CACHE_DIR,
    ):
        images, masks, lsd_entries = [], [], []
        keys = tuple(volume_keys) if volume_keys is not None else microns_neuron_zarr_stems(data_dir)
        _append_microns_neuron_zarr_xy_slices(
            images, masks, lsd_entries, data_dir, keys, split, n_val, require_xz_yz, "minnie", lsd_cache_dir
        )
        super().__init__(images, masks, split, crop_size, padding_size, require_lsd, lsd_entries=lsd_entries, augment=augment)


class Dataset_2D_pinky_Train(EM2DDataset):
    """MICrONS pinky: bbox-cropped stacks under ``Neuron_zarr/pinky/*.zarr`` (see ``transfer_zarr.py``)."""

    def __init__(
        self,
        data_dir=MICRONS_PINKY_ZARR_DIR,
        split="train",
        crop_size=None,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
        volume_keys: Optional[Sequence[str]] = None,
        augment: Optional[bool] = None,
        lsd_cache_dir: str = DEFAULT_APPROX_2D_LSD_CACHE_DIR,
    ):
        images, masks, lsd_entries = [], [], []
        keys = tuple(volume_keys) if volume_keys is not None else pinky_zarr_volume_keys(data_dir)
        _append_microns_neuron_zarr_xy_slices(
            images, masks, lsd_entries, data_dir, keys, split, n_val, require_xz_yz, "pinky", lsd_cache_dir
        )
        super().__init__(images, masks, split, crop_size, padding_size, require_lsd, lsd_entries=lsd_entries, augment=augment)


class Dataset_2D_axonem_h_Train(EM2DDataset):
    """AxonEM EM30-H: paired ``im_*_pad.h5`` / ``seg_*_pad.h5`` under ``EM30-H-axon-train-9vol/``."""

    def __init__(
        self,
        data_dir=AXONEM_H_DEFAULT_DATA_DIR,
        split="train",
        crop_size=None,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
        volume_keys: Optional[Sequence[str]] = None,
        augment: Optional[bool] = None,
        lsd_cache_dir: str = DEFAULT_APPROX_2D_LSD_CACHE_DIR,
    ):
        images, masks, lsd_entries = [], [], []
        keys = tuple(volume_keys) if volume_keys is not None else axonem_h5_paired_volume_keys(data_dir)
        _append_axonem_h5_xy_slices(
            images, masks, lsd_entries, data_dir, keys, split, n_val, require_xz_yz, "H", lsd_cache_dir
        )
        super().__init__(images, masks, split, crop_size, padding_size, require_lsd, lsd_entries=lsd_entries, augment=augment)


class Dataset_2D_axonem_m_Train(EM2DDataset):
    """AxonEM EM30-M: paired H5 stacks under ``EM30-M-axon-train-9vol/`` (cuboids without ``seg_*`` omitted).

    Same I/O as ``plot.ipynb`` (largest 3D ``main`` volume, ``im_*``/``seg_*`` pairing). Released M labels are
    binary ``{0,1}`` in the H5, unlike EM30-H instance ids — smoke PNGs may look uniform on the label panel if
    the random crop is entirely foreground (see ``_plot_2d_batch``).
    """

    def __init__(
        self,
        data_dir=AXONEM_M_DEFAULT_DATA_DIR,
        split="train",
        crop_size=None,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
        volume_keys: Optional[Sequence[str]] = None,
        augment: Optional[bool] = None,
        lsd_cache_dir: str = DEFAULT_APPROX_2D_LSD_CACHE_DIR,
    ):
        images, masks, lsd_entries = [], [], []
        keys = tuple(volume_keys) if volume_keys is not None else axonem_h5_paired_volume_keys(data_dir)
        _append_axonem_h5_xy_slices(
            images, masks, lsd_entries, data_dir, keys, split, n_val, require_xz_yz, "M", lsd_cache_dir
        )
        super().__init__(images, masks, split, crop_size, padding_size, require_lsd, lsd_entries=lsd_entries, augment=augment)


def _microns_neuron_zarr_subset(
    names: Sequence[str],
    zarr_subset_dir: str,
    split: str,
    require_xz_yz: bool,
    avalanche: bool,
    subset_label: str,
    lsd_entries: Optional[list] = None,
    cache_dir: str = DEFAULT_APPROX_2D_LSD_CACHE_DIR,
):
    images, masks, targets = [], [], []
    for patch_idx, name in enumerate(names):
        n_before = len(images)
        raw, lab = load_microns_neuron_zarr_volume(zarr_subset_dir, name)
        lsd_cache_path = _approx_2d_lsd_cache_xy_path(cache_dir, subset_label, name)
        print(
            "data MICrONS {} {}: raw shape={}, label shape = {}".format(
                subset_label, name, raw.shape, lab.shape
            )
        )
        _append_xy_slices(
            images,
            masks,
            raw,
            lab,
            split,
            8,
            lsd_entries=lsd_entries,
            skip_empty_z=True,
            lsd_cache_path=lsd_cache_path,
        )
        if require_xz_yz:
            _append_xz_yz(
                images,
                masks,
                raw,
                lab,
                split,
                8,
                lsd_entries=lsd_entries,
                skip_empty_y=True,
                skip_empty_x=True,
                lsd_cache_path=lsd_cache_path,
            )
        if avalanche:
            n_added = len(images) - n_before
            targets.extend([patch_idx] * n_added)
    return images, masks, (targets if avalanche else None)


class Dataset_2D_pinky_Train_CL(EM2DDataset):
    """Pinky CL: subset of cuboid stacks indexed via ``PINKY_PATCH_FILES`` (cf. zebrafinch CL)."""

    def __init__(
        self,
        data_dir=MICRONS_PINKY_ZARR_DIR,
        split="train",
        crop_size=None,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        data_idxs=(0, 1, 2),
        augment: Optional[bool] = None,
        lsd_cache_dir: str = DEFAULT_APPROX_2D_LSD_CACHE_DIR,
    ):
        names = [PINKY_PATCH_FILES[i] for i in data_idxs]
        lsd_entries = []
        images, masks, _ = _microns_neuron_zarr_subset(
            names, data_dir, split, require_xz_yz, avalanche=False, subset_label="pinky", lsd_entries=lsd_entries, cache_dir=lsd_cache_dir
        )
        super().__init__(
            images, masks, split, crop_size, padding_size, require_lsd, lsd_entries=lsd_entries, aug_check_shapes=False, augment=augment
        )


class Dataset_2D_pinky_Train_CL_avalanche(EM2DDataset):
    def __init__(
        self,
        data_dir=MICRONS_PINKY_ZARR_DIR,
        split="train",
        crop_size=None,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        data_idxs=(0, 1, 2),
    ):
        names = [PINKY_PATCH_FILES[i] for i in data_idxs]
        images, masks, targets = _microns_neuron_zarr_subset(
            names, data_dir, split, require_xz_yz, avalanche=True, subset_label="pinky"
        )
        super().__init__(
            images,
            masks,
            split,
            crop_size,
            padding_size,
            require_lsd,
            aug_check_shapes=False,
            avalanche=True,
        )
        self.targets = targets


def _append_zebra_xz_yz(
    images: list,
    masks: list,
    raw: np.ndarray,
    lab: np.ndarray,
    split: str,
    n_val: int,
    *,
    lsd_entries: Optional[list] = None,
    lsd_cache_path: Optional[str] = None,
) -> None:
    """Match legacy zebrafinch ordering (val interleaves y then x per index n)."""
    _lazy = _is_lazy_source(raw)
    if split == "train":
        for n in range(n_val, lab.shape[1]):
            if lab[:, n, :].max() == 0:
                continue
            if _lazy:
                images.append(_LazySlice(_mk_loader_xz(raw, n, _zebra_y_raw, lab)))
            else:
                images.append(_zebra_y_raw(n, raw, lab))
            masks.append(lab[:, n, :])
            if lsd_entries is not None and lsd_cache_path is not None:
                lsd_entries.append((lsd_cache_path, "xz", n))
        for n in range(n_val, lab.shape[2]):
            if lab[:, :, n].max() == 0:
                continue
            if _lazy:
                images.append(_LazySlice(_mk_loader_yz(raw, n, _zebra_x_raw, lab)))
            else:
                images.append(_zebra_x_raw(n, raw, lab))
            masks.append(lab[:, :, n])
            if lsd_entries is not None and lsd_cache_path is not None:
                lsd_entries.append((lsd_cache_path, "yz", n))
    else:
        for n in range(n_val):
            if lab[:, n, :].max() != 0:
                if _lazy:
                    images.append(_LazySlice(_mk_loader_xz(raw, n, _zebra_y_raw, lab)))
                else:
                    images.append(_zebra_y_raw(n, raw, lab))
                masks.append(lab[:, n, :])
                if lsd_entries is not None and lsd_cache_path is not None:
                    lsd_entries.append((lsd_cache_path, "xz", n))
            if lab[:, :, n].max() != 0:
                if _lazy:
                    images.append(_LazySlice(_mk_loader_yz(raw, n, _zebra_x_raw, lab)))
                else:
                    images.append(_zebra_x_raw(n, raw, lab))
                masks.append(lab[:, :, n])
                if lsd_entries is not None and lsd_cache_path is not None:
                    lsd_entries.append((lsd_cache_path, "yz", n))


def _zebra_subset(
    names: Sequence[str],
    data_dir: str,
    split: str,
    require_xz_yz: bool,
    avalanche: bool,
    n_val: int = 8,
    lsd_entries: Optional[list] = None,
    cache_dir: str = DEFAULT_APPROX_2D_LSD_CACHE_DIR,
):
    images, masks, targets = [], [], []
    for patch_idx, name in enumerate(names):
        n_before = len(images)
        root = zarr.open(os.path.join(data_dir, name), mode="r")
        raw = root["volumes"]["raw"]
        lab = connected_components(np.asarray(root["volumes"]["labels"]["neuron_ids"])).astype(np.uint16)
        lsd_cache_path = _approx_2d_lsd_cache_xy_path(cache_dir, "zebrafinch", name)
        print("data {}: raw shape={}, label shape = {}".format(name, raw.shape, lab.shape))
        _append_xy_slices(
            images,
            masks,
            raw,
            lab,
            split,
            n_val,
            lsd_entries=lsd_entries,
            z_raw_fn=_zebra_z_raw,
            skip_empty_z=True,
            lsd_cache_path=lsd_cache_path,
        )
        if require_xz_yz:
            _append_zebra_xz_yz(
                images, masks, raw, lab, split, n_val, lsd_entries=lsd_entries, lsd_cache_path=lsd_cache_path
            )
        if avalanche:
            n_added = len(images) - n_before
            targets.extend([patch_idx] * n_added)
    return images, masks, (targets if avalanche else None)


class Dataset_2D_zebrafinch_Train_CL(EM2DDataset):
    def __init__(
        self,
        data_dir=ZEBRAFINCH_DEFAULT_DATA_DIR,
        split="train",
        crop_size=None,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        data_idxs=(0, 1, 2),
        n_val: int = 8,
        augment: Optional[bool] = None,
        lsd_cache_dir: str = DEFAULT_APPROX_2D_LSD_CACHE_DIR,
    ):
        names = [ZEBRA_PATCH_FILES[i] for i in data_idxs]
        lsd_entries = []
        images, masks, _ = _zebra_subset(
            names,
            data_dir,
            split,
            require_xz_yz,
            avalanche=False,
            n_val=n_val,
            lsd_entries=lsd_entries,
            cache_dir=lsd_cache_dir,
        )
        super().__init__(
            images, masks, split, crop_size, padding_size, require_lsd, lsd_entries=lsd_entries, aug_check_shapes=False, augment=augment
        )


class Dataset_2D_zebrafinch_Train_CL_avalanche(EM2DDataset):
    def __init__(
        self,
        data_dir=ZEBRAFINCH_DEFAULT_DATA_DIR,
        split="train",
        crop_size=None,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        data_idxs=(0, 1, 2),
        n_val: int = 8,
    ):
        names = [ZEBRA_PATCH_FILES[i] for i in data_idxs]
        images, masks, targets = _zebra_subset(names, data_dir, split, require_xz_yz, avalanche=True, n_val=n_val)
        super().__init__(
            images, masks, split, crop_size, padding_size, require_lsd, aug_check_shapes=False, avalanche=True
        )
        self.targets = targets


def _sam_init_from_lists(sam_cls, images, masks, **kwargs):
    obj = sam_cls.__new__(sam_cls)
    EM2DDataset.__init__(obj, images, masks, sam=True, **kwargs)
    return obj


class SAM_Dataset_2D_hemi_Train(EM2DDataset):
    def __init__(
        self,
        data_dir=HEMI_DEFAULT_DATA_DIR,
        split="train",
        crop_size=None,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
    ):
        images, masks = [], []
        for name in HEMI_ZARR_FILES:
            root = zarr.open(os.path.join(data_dir, name), mode="r")
            raw = root["volumes"]["raw"]
            lab = root["volumes"]["labels"]["neuron_ids"]
            raw = raw[128 : int(raw.shape[0] - 128), 128 : int(raw.shape[1] - 128), 128 : int(raw.shape[2] - 128)]
            lab = lab[128 : int(lab.shape[0] - 128), 128 : int(lab.shape[1] - 128), 128 : int(lab.shape[2] - 128)]
            lab = connected_components(np.asarray(lab)).astype(np.uint16)
            print("data {}: raw shape={}, label shape = {}".format(name, raw.shape, lab.shape))
            _append_xy_slices(images, masks, raw, lab, split, n_val)
            if require_xz_yz:
                _append_xz_yz(images, masks, raw, lab, split, n_val)
        super().__init__(images, masks, split, crop_size, padding_size, require_lsd, sam=True)


class SAM_Dataset_2D_fib25_Train(EM2DDataset):
    def __init__(
        self,
        data_dir=FIB25_DEFAULT_DATA_DIR,
        split="train",
        crop_size=None,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
    ):
        images, masks = [], []
        for name in FIB25_ZARR_FILES:
            root = zarr.open(os.path.join(data_dir, name), mode="r")
            raw = root["volumes"]["raw"]
            lab = connected_components(np.asarray(root["volumes"]["labels"]["neuron_ids"])).astype(np.uint16)
            print("data {}: raw shape={}, label shape = {}".format(name, raw.shape, lab.shape))
            _append_xy_slices(images, masks, raw, lab, split, n_val)
            if require_xz_yz:
                _append_xz_yz(images, masks, raw, lab, split, n_val)
        super().__init__(images, masks, split, crop_size, padding_size, require_lsd, sam=True)


class SAM_Dataset_2D_cremi_Train(EM2DDataset):
    def __init__(
        self,
        data_dir=CREMI_DEFAULT_DATA_DIR,
        split="train",
        crop_size=None,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
    ):
        images, masks = [], []
        for name in CREMI_HDF_FILES:
            path = os.path.join(data_dir, name)
            with h5py.File(path, "r") as f:
                vol = f["volumes"]
                raw = np.asarray(vol["raw"])
                lab = np.asarray(vol["labels"]["neuron_ids"])
            if name == "sample_C_20160501.hdf":
                raw = np.delete(raw, [14, 74], 0)
                lab = np.delete(lab, [14, 74], 0)
            lab = connected_components(lab).astype(np.uint16)
            print("data {}: raw shape={}, label shape = {}".format(name, raw.shape, lab.shape))
            _append_xy_slices(images, masks, raw, lab, split, n_val)
            if require_xz_yz:
                _append_xz_yz(images, masks, raw, lab, split, n_val)
        super().__init__(images, masks, split, crop_size, padding_size, require_lsd, sam=True)


class SAM_Dataset_2D_VNC_Train(EM2DDataset):
    def __init__(
        self,
        data_dir=VNC_DEFAULT_DATA_DIR,
        split="train",
        crop_size=None,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: Optional[int] = None,
    ):
        stack1 = os.path.abspath(data_dir)
        raw, lab = load_vnc_stack1_volume(stack1)
        z = raw.shape[0]
        if n_val is None:
            n_val = min(8, max(1, z // 5))
        print("data VNC stack1: raw shape={}, label shape = {}".format(raw.shape, lab.shape))
        images, masks = [], []
        _append_xy_slices(images, masks, raw, lab, split, n_val)
        if require_xz_yz:
            _append_xz_yz(images, masks, raw, lab, split, n_val)
        super().__init__(images, masks, split, crop_size, padding_size, require_lsd, sam=True)


class SAM_Dataset_2D_isbi2012_Train(EM2DDataset):
    def __init__(
        self,
        data_dir=ISBI2012_DEFAULT_DATA_DIR,
        split="train",
        crop_size=None,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
    ):
        images, masks = [], []
        for raw_name, lab_name in ISBI2012_TIFF_PAIRS:
            raw, lab = load_isbi2012_volume_pair(data_dir, raw_name, lab_name)
            print("data ISBI-2012 {}: raw shape={}, label shape = {}".format(raw_name, raw.shape, lab.shape))
            _append_xy_slices(images, masks, raw, lab, split, n_val)
            if require_xz_yz:
                _append_xz_yz(images, masks, raw, lab, split, n_val)
        super().__init__(images, masks, split, crop_size, padding_size, require_lsd, sam=True)


class SAM_Dataset_2D_ac3_Train(EM2DDataset):
    def __init__(
        self,
        data_dir=AC3_DEFAULT_DATA_DIR,
        split="train",
        crop_size=None,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
        num_slices: int = AC3_NUM_SLICES,
    ):
        images, masks = [], []
        raw, lab = load_ac3_volume(data_dir, num_slices=num_slices)
        print("data AC3: raw shape={}, label shape = {}".format(raw.shape, lab.shape))
        _append_xy_slices(images, masks, raw, lab, split, n_val, skip_empty_z=True)
        if require_xz_yz:
            _append_xz_yz(
                images,
                masks,
                raw,
                lab,
                split,
                n_val,
                skip_empty_y=True,
                skip_empty_x=True,
            )
        super().__init__(images, masks, split, crop_size, padding_size, require_lsd, sam=True)


class SAM_Dataset_2D_ac4_Train(EM2DDataset):
    def __init__(
        self,
        data_dir=AC4_DEFAULT_DATA_DIR,
        split="train",
        crop_size=None,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
        num_slices: int = AC4_NUM_SLICES,
    ):
        images, masks = [], []
        raw, lab = load_ac4_volume(data_dir, num_slices=num_slices)
        print("data AC4: raw shape={}, label shape = {}".format(raw.shape, lab.shape))
        _append_xy_slices(images, masks, raw, lab, split, n_val, skip_empty_z=True)
        if require_xz_yz:
            _append_xz_yz(
                images,
                masks,
                raw,
                lab,
                split,
                n_val,
                skip_empty_y=True,
                skip_empty_x=True,
            )
        super().__init__(images, masks, split, crop_size, padding_size, require_lsd, sam=True)


class SAM_Dataset_2D_basil_Train(EM2DDataset):
    def __init__(
        self,
        data_dir=MICRONS_BASIL_ZARR_DIR,
        split="train",
        crop_size=None,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
        volume_keys: Optional[Sequence[str]] = None,
    ):
        images, masks = [], []
        keys = tuple(volume_keys) if volume_keys is not None else microns_neuron_zarr_stems(data_dir)
        _append_microns_neuron_zarr_xy_slices(
            images, masks, data_dir, keys, split, n_val, require_xz_yz, "basil"
        )
        super().__init__(images, masks, split, crop_size, padding_size, require_lsd, sam=True)


class SAM_Dataset_2D_minnie_Train(EM2DDataset):
    def __init__(
        self,
        data_dir=MICRONS_MINNIE_ZARR_DIR,
        split="train",
        crop_size=None,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
        volume_keys: Optional[Sequence[str]] = None,
    ):
        images, masks = [], []
        keys = tuple(volume_keys) if volume_keys is not None else microns_neuron_zarr_stems(data_dir)
        _append_microns_neuron_zarr_xy_slices(
            images, masks, data_dir, keys, split, n_val, require_xz_yz, "minnie"
        )
        super().__init__(images, masks, split, crop_size, padding_size, require_lsd, sam=True)


class SAM_Dataset_2D_pinky_Train(EM2DDataset):
    def __init__(
        self,
        data_dir=MICRONS_PINKY_ZARR_DIR,
        split="train",
        crop_size=None,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
        volume_keys: Optional[Sequence[str]] = None,
    ):
        images, masks = [], []
        keys = tuple(volume_keys) if volume_keys is not None else pinky_zarr_volume_keys(data_dir)
        _append_microns_neuron_zarr_xy_slices(
            images, masks, data_dir, keys, split, n_val, require_xz_yz, "pinky"
        )
        super().__init__(images, masks, split, crop_size, padding_size, require_lsd, sam=True)


class SAM_Dataset_2D_axonem_h_Train(EM2DDataset):
    def __init__(
        self,
        data_dir=AXONEM_H_DEFAULT_DATA_DIR,
        split="train",
        crop_size=None,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
        volume_keys: Optional[Sequence[str]] = None,
    ):
        images, masks = [], []
        keys = tuple(volume_keys) if volume_keys is not None else axonem_h5_paired_volume_keys(data_dir)
        _append_axonem_h5_xy_slices(
            images, masks, data_dir, keys, split, n_val, require_xz_yz, "H"
        )
        super().__init__(images, masks, split, crop_size, padding_size, require_lsd, sam=True)


class SAM_Dataset_2D_axonem_m_Train(EM2DDataset):
    def __init__(
        self,
        data_dir=AXONEM_M_DEFAULT_DATA_DIR,
        split="train",
        crop_size=None,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
        volume_keys: Optional[Sequence[str]] = None,
    ):
        images, masks = [], []
        keys = tuple(volume_keys) if volume_keys is not None else axonem_h5_paired_volume_keys(data_dir)
        _append_axonem_h5_xy_slices(
            images, masks, data_dir, keys, split, n_val, require_xz_yz, "M"
        )
        super().__init__(images, masks, split, crop_size, padding_size, require_lsd, sam=True)


# ---------------------------------------------------------------------------
# 3D Dataset
# ---------------------------------------------------------------------------


class EM3DDataset(Dataset):
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
        augment: Optional[bool] = None,
    ):
        self.images = images
        self.masks = masks
        self.idxs = idxs
        self.split = split
        self.crop_size = crop_size
        self.padding_size = padding_size
        self.num_slices = num_slices
        self.require_lsd = require_lsd
        self.augment = (split == "train") if augment is None else augment

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx: int):
        ii, z0 = self.idxs[idx]
        raw = self.images[ii]
        lab = self.masks[ii]
        raw = raw[z0 : z0 + self.num_slices]
        lab = lab[z0 : z0 + self.num_slices]
        raw = raw.transpose(1, 2, 0)
        lab = lab.transpose(1, 2, 0)
        raw = normalize_minmax(raw)
        pad = padding_for_crop(self.crop_size, self.padding_size)
        raw, lab = prepare_2d_pair(raw, lab, self.crop_size, pad, True, augment=self.augment)
        raw = np.expand_dims(raw, axis=0)
        lab = erode_instance_labels(lab, iterations=1, border_value=1)
        affinity = affinity_3d(lab)
        if self.require_lsd:
            lsds = local_shape_descriptor.get_local_shape_descriptors(
                segmentation=lab, sigma=(5,) * 3, voxel_size=(1,) * 3
            ).astype(np.float32)
        mask_3d, pp, pl = mask_and_points_3d(lab)
        point_map = gaussian_point_map_3d_try(pp, pl, self.crop_size, self.crop_size, theta=30)
        if self.require_lsd:
            return raw, lab, mask_3d, affinity, point_map, lsds
        return raw, lab, mask_3d, affinity, point_map


def _build_3d_patch_index(images, masks, num_slices: int) -> List[Tuple[int, int]]:
    idxs = []
    for ii in range(len(images)):
        vol = images[ii]
        msk = masks[ii]
        for z0 in range(vol.shape[0] - num_slices + 1):
            if not np.any(msk[z0]):
                continue
            idxs.append((ii, z0))
    return idxs


class Dataset_3D_hemi_Train(EM3DDataset):
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
    ):
        images, masks = [], []
        for name in HEMI_ZARR_FILES:
            root = zarr.open(os.path.join(data_dir, name), mode="r")
            raw = root["volumes"]["raw"]
            lab = root["volumes"]["labels"]["neuron_ids"]
            raw = raw[128 : int(raw.shape[0] - 128), 128 : int(raw.shape[1] - 128), 128 : int(raw.shape[2] - 128)]
            lab = lab[128 : int(lab.shape[0] - 128), 128 : int(lab.shape[1] - 128), 128 : int(lab.shape[2] - 128)]
            raw = np.asarray(raw)
            lab = connected_components(np.asarray(lab)).astype(np.uint16)
            print("data {}: raw shape={}, label shape = {}".format(name, raw.shape, lab.shape))
            if split == "train":
                images.append(raw[n_val:])
                masks.append(lab[n_val:])
            else:
                images.append(raw[:n_val])
                masks.append(lab[:n_val])
            if require_xz_yz:
                if split == "train":
                    images.append(raw.transpose(1, 0, 2)[n_val:])
                    masks.append(lab.transpose(1, 0, 2)[n_val:])
                    images.append(raw.transpose(1, 2, 0)[n_val:])
                    masks.append(lab.transpose(1, 2, 0)[n_val:])
                else:
                    images.append(raw.transpose(1, 0, 2)[:n_val])
                    masks.append(lab.transpose(1, 0, 2)[:n_val])
                    images.append(raw.transpose(1, 2, 0)[:n_val])
                    masks.append(lab.transpose(1, 2, 0)[:n_val])
        idxs = _build_3d_patch_index(images, masks, num_slices)
        super().__init__(images, masks, idxs, split, crop_size, padding_size, num_slices, require_lsd, augment=augment)


class Dataset_3D_fib25_Train(EM3DDataset):
    def __init__(
        self,
        data_dir=FIB25_DEFAULT_DATA_DIR,
        split="train",
        crop_size=None,
        num_slices=8,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
        augment: Optional[bool] = None,
    ):
        images, masks = [], []
        for name in FIB25_ZARR_FILES:
            root = zarr.open(os.path.join(data_dir, name), mode="r")
            raw = np.asarray(root["volumes"]["raw"])
            lab = connected_components(np.asarray(root["volumes"]["labels"]["neuron_ids"])).astype(np.uint16)
            print("data {}: raw shape={}, label shape = {}".format(name, raw.shape, lab.shape))
            if split == "train":
                images.append(raw[n_val:])
                masks.append(lab[n_val:])
            else:
                images.append(raw[:n_val])
                masks.append(lab[:n_val])
            if require_xz_yz:
                if split == "train":
                    images.append(raw.transpose(1, 0, 2)[n_val:])
                    masks.append(lab.transpose(1, 0, 2)[n_val:])
                    images.append(raw.transpose(1, 2, 0)[n_val:])
                    masks.append(lab.transpose(1, 2, 0)[n_val:])
                else:
                    images.append(raw.transpose(1, 0, 2)[:n_val])
                    masks.append(lab.transpose(1, 0, 2)[:n_val])
                    images.append(raw.transpose(1, 2, 0)[:n_val])
                    masks.append(lab.transpose(1, 2, 0)[:n_val])
        idxs = _build_3d_patch_index(images, masks, num_slices)
        super().__init__(images, masks, idxs, split, crop_size, padding_size, num_slices, require_lsd, augment=augment)


class Dataset_3D_cremi_Train(EM3DDataset):
    def __init__(
        self,
        data_dir=CREMI_DEFAULT_DATA_DIR,
        split="train",
        crop_size=None,
        num_slices=8,
        padding_size=8,
        require_lsd=False,
        require_xz_yz=False,
        n_val: int = 8,
        augment: Optional[bool] = None,
    ):
        images, masks = [], []
        for name in CREMI_HDF_FILES:
            path = os.path.join(data_dir, name)
            with h5py.File(path, "r") as f:
                vol = f["volumes"]
                raw = np.asarray(vol["raw"])
                lab = np.asarray(vol["labels"]["neuron_ids"])
            if name == "sample_C_20160501.hdf":
                raw = np.delete(raw, [14, 74], 0)
                lab = np.delete(lab, [14, 74], 0)
            lab = connected_components(lab).astype(np.uint16)
            print("data {}: raw shape={}, label shape = {}".format(name, raw.shape, lab.shape))
            if split == "train":
                images.append(raw[n_val:])
                masks.append(lab[n_val:])
            else:
                images.append(raw[:n_val])
                masks.append(lab[:n_val])
            if require_xz_yz:
                if split == "train":
                    images.append(raw.transpose(1, 0, 2)[n_val:])
                    masks.append(lab.transpose(1, 0, 2)[n_val:])
                    images.append(raw.transpose(1, 2, 0)[n_val:])
                    masks.append(lab.transpose(1, 2, 0)[n_val:])
                else:
                    images.append(raw.transpose(1, 0, 2)[:n_val])
                    masks.append(lab.transpose(1, 0, 2)[:n_val])
                    images.append(raw.transpose(1, 2, 0)[:n_val])
                    masks.append(lab.transpose(1, 2, 0)[:n_val])
        idxs = _build_3d_patch_index(images, masks, num_slices)
        super().__init__(images, masks, idxs, split, crop_size, padding_size, num_slices, require_lsd, augment=augment)


class Dataset_3D_VNC_Train(EM3DDataset):
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
    ):
        stack1 = os.path.abspath(data_dir)
        raw, lab = load_vnc_stack1_volume(stack1)
        z = raw.shape[0]
        if n_val is None:
            n_val = min(8, max(1, z // 5))
        print("data VNC stack1: raw shape={}, label shape = {}".format(raw.shape, lab.shape))
        images, masks = [], []
        if split == "train":
            images.append(raw[n_val:])
            masks.append(lab[n_val:])
        else:
            images.append(raw[:n_val])
            masks.append(lab[:n_val])
        if require_xz_yz:
            if split == "train":
                images.append(raw.transpose(1, 0, 2)[n_val:])
                masks.append(lab.transpose(1, 0, 2)[n_val:])
                images.append(raw.transpose(1, 2, 0)[n_val:])
                masks.append(lab.transpose(1, 2, 0)[n_val:])
            else:
                images.append(raw.transpose(1, 0, 2)[:n_val])
                masks.append(lab.transpose(1, 0, 2)[:n_val])
                images.append(raw.transpose(1, 2, 0)[:n_val])
                masks.append(lab.transpose(1, 2, 0)[:n_val])
        idxs = _build_3d_patch_index(images, masks, num_slices)
        super().__init__(images, masks, idxs, split, crop_size, padding_size, num_slices, require_lsd, augment=augment)


class Dataset_3D_isbi2012_Train(EM3DDataset):
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
    ):
        images, masks = [], []
        for raw_name, lab_name in ISBI2012_TIFF_PAIRS:
            raw, lab = load_isbi2012_volume_pair(data_dir, raw_name, lab_name)
            print("data ISBI-2012 {}: raw shape={}, label shape = {}".format(raw_name, raw.shape, lab.shape))
            if split == "train":
                images.append(raw[n_val:])
                masks.append(lab[n_val:])
            else:
                images.append(raw[:n_val])
                masks.append(lab[:n_val])
            if require_xz_yz:
                if split == "train":
                    images.append(raw.transpose(1, 0, 2)[n_val:])
                    masks.append(lab.transpose(1, 0, 2)[n_val:])
                    images.append(raw.transpose(1, 2, 0)[n_val:])
                    masks.append(lab.transpose(1, 2, 0)[n_val:])
                else:
                    images.append(raw.transpose(1, 0, 2)[:n_val])
                    masks.append(lab.transpose(1, 0, 2)[:n_val])
                    images.append(raw.transpose(1, 2, 0)[:n_val])
                    masks.append(lab.transpose(1, 2, 0)[:n_val])
        idxs = _build_3d_patch_index(images, masks, num_slices)
        super().__init__(images, masks, idxs, split, crop_size, padding_size, num_slices, require_lsd, augment=augment)


class Dataset_3D_ac3_Train(EM3DDataset):
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
    ):
        raw, lab = load_ac3_volume(data_dir, num_slices=stack_slices)
        print("data AC3: raw shape={}, label shape = {}".format(raw.shape, lab.shape))
        images, masks = [], []
        if split == "train":
            images.append(raw[n_val:])
            masks.append(lab[n_val:])
        else:
            images.append(raw[:n_val])
            masks.append(lab[:n_val])
        if require_xz_yz:
            if split == "train":
                images.append(raw.transpose(1, 0, 2)[n_val:])
                masks.append(lab.transpose(1, 0, 2)[n_val:])
                images.append(raw.transpose(1, 2, 0)[n_val:])
                masks.append(lab.transpose(1, 2, 0)[n_val:])
            else:
                images.append(raw.transpose(1, 0, 2)[:n_val])
                masks.append(lab.transpose(1, 0, 2)[:n_val])
                images.append(raw.transpose(1, 2, 0)[:n_val])
                masks.append(lab.transpose(1, 2, 0)[:n_val])
        idxs = _build_3d_patch_index(images, masks, num_slices)
        super().__init__(images, masks, idxs, split, crop_size, padding_size, num_slices, require_lsd, augment=augment)


class Dataset_3D_ac4_Train(EM3DDataset):
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
    ):
        raw, lab = load_ac4_volume(data_dir, num_slices=stack_slices)
        print("data AC4: raw shape={}, label shape = {}".format(raw.shape, lab.shape))
        images, masks = [], []
        if split == "train":
            images.append(raw[n_val:])
            masks.append(lab[n_val:])
        else:
            images.append(raw[:n_val])
            masks.append(lab[:n_val])
        if require_xz_yz:
            if split == "train":
                images.append(raw.transpose(1, 0, 2)[n_val:])
                masks.append(lab.transpose(1, 0, 2)[n_val:])
                images.append(raw.transpose(1, 2, 0)[n_val:])
                masks.append(lab.transpose(1, 2, 0)[n_val:])
            else:
                images.append(raw.transpose(1, 0, 2)[:n_val])
                masks.append(lab.transpose(1, 0, 2)[:n_val])
                images.append(raw.transpose(1, 2, 0)[:n_val])
                masks.append(lab.transpose(1, 2, 0)[:n_val])
        idxs = _build_3d_patch_index(images, masks, num_slices)
        super().__init__(images, masks, idxs, split, crop_size, padding_size, num_slices, require_lsd, augment=augment)


def _microns_neuron_zarr_3d_volumes(
    data_dir: str,
    volume_keys: Sequence[str],
    split: str,
    n_val: int,
    require_xz_yz: bool,
    subset_label: str,
) -> Tuple[list, list]:
    images, masks = [], []
    for name in volume_keys:
        raw, lab = load_microns_neuron_zarr_volume(data_dir, name)
        print(
            "data MICrONS {} {}: raw shape={}, label shape = {}".format(
                subset_label, name, raw.shape, lab.shape
            )
        )
        if split == "train":
            images.append(raw[n_val:])
            masks.append(lab[n_val:])
        else:
            images.append(raw[:n_val])
            masks.append(lab[:n_val])
        if require_xz_yz:
            if split == "train":
                images.append(raw.transpose(1, 0, 2)[n_val:])
                masks.append(lab.transpose(1, 0, 2)[n_val:])
                images.append(raw.transpose(1, 2, 0)[n_val:])
                masks.append(lab.transpose(1, 2, 0)[n_val:])
            else:
                images.append(raw.transpose(1, 0, 2)[:n_val])
                masks.append(lab.transpose(1, 0, 2)[:n_val])
                images.append(raw.transpose(1, 2, 0)[:n_val])
                masks.append(lab.transpose(1, 2, 0)[:n_val])
    return images, masks


class Dataset_3D_basil_Train(EM3DDataset):
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
    ):
        keys = tuple(volume_keys) if volume_keys is not None else microns_neuron_zarr_stems(data_dir)
        images, masks = _microns_neuron_zarr_3d_volumes(
            data_dir, keys, split, n_val, require_xz_yz, "basil"
        )
        idxs = _build_3d_patch_index(images, masks, num_slices)
        super().__init__(images, masks, idxs, split, crop_size, padding_size, num_slices, require_lsd, augment=augment)


class Dataset_3D_minnie_Train(EM3DDataset):
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
    ):
        keys = tuple(volume_keys) if volume_keys is not None else microns_neuron_zarr_stems(data_dir)
        images, masks = _microns_neuron_zarr_3d_volumes(
            data_dir, keys, split, n_val, require_xz_yz, "minnie"
        )
        idxs = _build_3d_patch_index(images, masks, num_slices)
        super().__init__(images, masks, idxs, split, crop_size, padding_size, num_slices, require_lsd, augment=augment)


class Dataset_3D_pinky_Train(EM3DDataset):
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
    ):
        keys = tuple(volume_keys) if volume_keys is not None else pinky_zarr_volume_keys(data_dir)
        images, masks = _microns_neuron_zarr_3d_volumes(
            data_dir, keys, split, n_val, require_xz_yz, "pinky"
        )
        idxs = _build_3d_patch_index(images, masks, num_slices)
        super().__init__(images, masks, idxs, split, crop_size, padding_size, num_slices, require_lsd, augment=augment)


def _axonem_h5_3d_volumes(
    data_dir: str,
    volume_keys: Sequence[str],
    split: str,
    n_val: int,
    require_xz_yz: bool,
    subset_label: str,
) -> Tuple[list, list]:
    images, masks = [], []
    for name in volume_keys:
        raw, lab = load_axonem_h5_volume(data_dir, name)
        print(
            "data AxonEM %s %s: raw shape=%s, label shape = %s"
            % (subset_label, name, raw.shape, lab.shape)
        )
        if split == "train":
            images.append(np.array(raw[n_val:], dtype=np.float32, copy=True))
            masks.append(np.array(lab[n_val:], copy=True))
        else:
            images.append(np.array(raw[:n_val], dtype=np.float32, copy=True))
            masks.append(np.array(lab[:n_val], copy=True))
        if require_xz_yz:
            if split == "train":
                images.append(np.array(raw.transpose(1, 0, 2)[n_val:], dtype=np.float32, copy=True))
                masks.append(np.array(lab.transpose(1, 0, 2)[n_val:], copy=True))
                images.append(np.array(raw.transpose(1, 2, 0)[n_val:], dtype=np.float32, copy=True))
                masks.append(np.array(lab.transpose(1, 2, 0)[n_val:], copy=True))
            else:
                images.append(np.array(raw.transpose(1, 0, 2)[:n_val], dtype=np.float32, copy=True))
                masks.append(np.array(lab.transpose(1, 0, 2)[:n_val], copy=True))
                images.append(np.array(raw.transpose(1, 2, 0)[:n_val], dtype=np.float32, copy=True))
                masks.append(np.array(lab.transpose(1, 2, 0)[:n_val], copy=True))
        del raw, lab
    return images, masks


class Dataset_3D_axonem_h_Train(EM3DDataset):
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
    ):
        keys = tuple(volume_keys) if volume_keys is not None else axonem_h5_paired_volume_keys(data_dir)
        images, masks = _axonem_h5_3d_volumes(
            data_dir, keys, split, n_val, require_xz_yz, "H"
        )
        idxs = _build_3d_patch_index(images, masks, num_slices)
        super().__init__(images, masks, idxs, split, crop_size, padding_size, num_slices, require_lsd, augment=augment)


class Dataset_3D_axonem_m_Train(EM3DDataset):
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
    ):
        keys = tuple(volume_keys) if volume_keys is not None else axonem_h5_paired_volume_keys(data_dir)
        images, masks = _axonem_h5_3d_volumes(
            data_dir, keys, split, n_val, require_xz_yz, "M"
        )
        idxs = _build_3d_patch_index(images, masks, num_slices)
        super().__init__(images, masks, idxs, split, crop_size, padding_size, num_slices, require_lsd, augment=augment)


class Dataset_3D_zebrafinch_Train_CL(EM3DDataset):
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
    ):
        images, masks = [], []
        for patch_idx in data_idxs:
            name = ZEBRA_PATCH_FILES[patch_idx]
            root = zarr.open(os.path.join(data_dir, name), mode="r")
            lab = connected_components(np.asarray(root["volumes"]["labels"]["neuron_ids"])).astype(np.uint16)
            raw = np.asarray(
                root["volumes"]["raw"][
                    100 : 100 + lab.shape[0],
                    200 : 200 + lab.shape[1],
                    200 : 200 + lab.shape[2],
                ],
                dtype=np.float32,
            )
            print("data {}: raw shape={}, label shape = {}".format(name, raw.shape, lab.shape))
            if split == "train":
                images.append(raw[n_val:])
                masks.append(lab[n_val:])
            else:
                images.append(raw[:n_val])
                masks.append(lab[:n_val])
            if require_xz_yz:
                if split == "train":
                    images.append(raw.transpose(1, 0, 2)[n_val:])
                    masks.append(lab.transpose(1, 0, 2)[n_val:])
                    images.append(raw.transpose(1, 2, 0)[n_val:])
                    masks.append(lab.transpose(1, 2, 0)[n_val:])
                else:
                    images.append(raw.transpose(1, 0, 2)[:n_val])
                    masks.append(lab.transpose(1, 0, 2)[:n_val])
                    images.append(raw.transpose(1, 2, 0)[:n_val])
                    masks.append(lab.transpose(1, 2, 0)[:n_val])
        idxs = _build_3d_patch_index(images, masks, num_slices)
        super().__init__(images, masks, idxs, split, crop_size, padding_size, num_slices, require_lsd, augment=augment)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def collate_fn_2D_hemi_Train(batch):
    raw = np.array([item[0] for item in batch]).astype(np.float32)
    labels = np.array([item[1] for item in batch]).astype(np.int32)
    point_map = np.array([item[2] for item in batch]).astype(np.float32)
    mask = np.array([item[3] for item in batch]).astype(np.uint8)
    affinity = np.array([item[4] for item in batch]).astype(np.float32)
    if len(batch[0]) == 6:
        lsds = np.array([item[5] for item in batch]).astype(np.float32)
        return raw, labels, point_map, mask, affinity, lsds
    return raw, labels, point_map, mask, affinity


collate_fn_2D_fib25_Train = collate_fn_2D_hemi_Train
collate_fn_2D_cremi_Train = collate_fn_2D_hemi_Train
collate_fn_2D_VNC_Train = collate_fn_2D_hemi_Train
collate_fn_2D_isbi2012_Train = collate_fn_2D_hemi_Train
collate_fn_2D_ac3_Train = collate_fn_2D_hemi_Train
collate_fn_2D_ac4_Train = collate_fn_2D_hemi_Train
collate_fn_2D_basil_Train = collate_fn_2D_hemi_Train
collate_fn_2D_minnie_Train = collate_fn_2D_hemi_Train
collate_fn_2D_pinky_Train = collate_fn_2D_hemi_Train
collate_fn_2D_pinky_Train_CL = collate_fn_2D_hemi_Train
collate_fn_2D_zebrafinch_Train_CL = collate_fn_2D_hemi_Train
collate_fn_2D_axonem_h_Train = collate_fn_2D_hemi_Train
collate_fn_2D_axonem_m_Train = collate_fn_2D_hemi_Train


def collate_fn_3D_hemi_Train(batch):
    raw = np.array([item[0] for item in batch]).astype(np.float32)
    labels = np.array([item[1] for item in batch]).astype(np.int32)
    mask_3d = np.array([item[2] for item in batch]).astype(np.uint8)
    affinity = np.array([item[3] for item in batch]).astype(np.float32)
    point_map = np.array([item[4] for item in batch]).astype(np.float32)
    if len(batch[0]) == 6:
        lsds = np.array([item[5] for item in batch]).astype(np.float32)
        return raw, labels, mask_3d, affinity, point_map, lsds
    return raw, labels, mask_3d, affinity, point_map


collate_fn_3D_fib25_Train = collate_fn_3D_hemi_Train
collate_fn_3D_cremi_Train = collate_fn_3D_hemi_Train
collate_fn_3D_VNC_Train = collate_fn_3D_hemi_Train
collate_fn_3D_isbi2012_Train = collate_fn_3D_hemi_Train
collate_fn_3D_ac3_Train = collate_fn_3D_hemi_Train
collate_fn_3D_ac4_Train = collate_fn_3D_hemi_Train
collate_fn_3D_basil_Train = collate_fn_3D_hemi_Train
collate_fn_3D_minnie_Train = collate_fn_3D_hemi_Train
collate_fn_3D_pinky_Train = collate_fn_3D_hemi_Train
collate_fn_3D_zebrafinch_Train_CL = collate_fn_3D_hemi_Train
collate_fn_3D_axonem_h_Train = collate_fn_3D_hemi_Train
collate_fn_3D_axonem_m_Train = collate_fn_3D_hemi_Train


def collate_fn_2D_SAM_train(batch):
    raw = np.array([item[0] for item in batch]).astype(np.uint8)
    labels = np.array([item[1] for item in batch]).astype(np.int32)
    max_p = max(len(item[2]) for item in batch)
    max_b = max(len(item[4]) for item in batch)
    points_pos = []
    points_lab = []
    boxes = []
    for item in batch:
        points_pos.append(np.pad(item[2], ((0, max_p - len(item[2])), (0, 0)), mode="edge"))
        points_lab.append(np.pad(item[3], (0, max_p - len(item[3])), mode="edge"))
        boxes.append(np.pad(item[4], ((0, max_b - len(item[4])), (0, 0)), mode="edge"))
    points_pos = np.array(points_pos)
    points_lab = np.array(points_lab)
    boxes = np.array(boxes)
    point_map = np.array([item[5] for item in batch]).astype(np.float32)
    mask = np.array([item[6] for item in batch]).astype(np.uint8)
    affinity = np.array([item[7] for item in batch]).astype(np.float32)
    if len(batch[0]) == 9:
        lsds = np.array([item[8] for item in batch]).astype(np.float32)
        return raw, labels, points_pos, points_lab, boxes, point_map, mask, affinity, lsds
    return raw, labels, points_pos, points_lab, boxes, point_map, mask, affinity


collate_fn_2D_hemi_Train_SAM = collate_fn_2D_SAM_train
collate_fn_2D_fib25_Train_SAM = collate_fn_2D_SAM_train
collate_fn_2D_cremi_Train_SAM = collate_fn_2D_SAM_train
collate_fn_2D_VNC_Train_SAM = collate_fn_2D_SAM_train
collate_fn_2D_isbi2012_Train_SAM = collate_fn_2D_SAM_train
collate_fn_2D_ac3_Train_SAM = collate_fn_2D_SAM_train
collate_fn_2D_ac4_Train_SAM = collate_fn_2D_SAM_train
collate_fn_2D_basil_Train_SAM = collate_fn_2D_SAM_train
collate_fn_2D_minnie_Train_SAM = collate_fn_2D_SAM_train
collate_fn_2D_pinky_Train_SAM = collate_fn_2D_SAM_train
collate_fn_2D_axonem_h_Train_SAM = collate_fn_2D_SAM_train
collate_fn_2D_axonem_m_Train_SAM = collate_fn_2D_SAM_train


# ---------------------------------------------------------------------------
# Smoke tests (run from project root: python utils/dataloader.py [DATASET])
# ---------------------------------------------------------------------------

def _smoke_out_path(filename: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


def _plot_2d_batch(
    raw_b, labels_b, pmask_b, aff_b, out_path: str, suptitle: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = raw_b.shape[0]
    _, axs = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axs = axs.reshape(1, -1)
    z0 = np.zeros_like(aff_b[0, 0])
    for i in range(n):
        axs[i, 0].imshow(raw_b[i, 0], cmap="gray", vmin=0, vmax=1)
        pm = np.asarray(pmask_b[i]).astype(np.float32)
        axs[i, 1].imshow(pm, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        li = np.asarray(labels_b[i])
        lmax = max(float(np.max(li)), 1.0) if li.size else 1.0
        axs[i, 2].imshow(li, cmap="turbo", vmin=0, vmax=lmax, interpolation="nearest")
        axs[i, 3].imshow(
            np.dstack([aff_b[i, 0], aff_b[i, 1], z0]),
            vmin=0,
            vmax=1,
        )
        for j in range(4):
            axs[i, j].axis("off")
    axs[0, 0].set_title("raw")
    axs[0, 1].set_title("prompt mask")
    axs[0, 2].set_title("instance labels")
    axs[0, 3].set_title("affinity (R=x, G=y)")
    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Wrote", out_path)


def _plot_3d_batch_midz(
    raw_b, labels_b, pmask_b, aff_b, out_path: str, suptitle: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = raw_b.shape[0]
    zmid = raw_b.shape[-1] // 2
    _, axs = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axs = axs.reshape(1, -1)
    for i in range(n):
        axs[i, 0].imshow(raw_b[i, 0, :, :, zmid], cmap="gray", vmin=0, vmax=1)
        pm = np.asarray(pmask_b[i, :, :, zmid]).astype(np.float32)
        axs[i, 1].imshow(pm, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        axs[i, 2].imshow(labels_b[i, :, :, zmid], cmap="gray", vmin=0, vmax=1)
        axs[i, 3].imshow(
            np.dstack(
                [
                    aff_b[i, 0, :, :, zmid],
                    aff_b[i, 1, :, :, zmid],
                    aff_b[i, 2, :, :, zmid],
                ]
            ),
            vmin=0,
            vmax=1,
        )
        for j in range(4):
            axs[i, j].axis("off")
    axs[0, 0].set_title("raw (z mid)")
    axs[0, 1].set_title("prompt mask (z mid)")
    axs[0, 2].set_title("labels (z mid)")
    axs[0, 3].set_title("affinity xyz as RGB (z mid)")
    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Wrote", out_path)


def _run_smoke_2d_standard(name: str, ds: Dataset, collate_fn, png: str) -> None:
    from torch.utils.data import DataLoader

    if len(ds) == 0:
        print("[%s] skip: empty dataset" % name)
        return
    bs = min(3, len(ds))
    loader = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate_fn, num_workers=0)
    batch = next(iter(loader))
    raw_b, labels_b, _, pmask_b, aff_b = batch[:5]
    _plot_2d_batch(raw_b, labels_b, pmask_b, aff_b, _smoke_out_path(png), name)


def _run_smoke_3d(name: str, ds: Dataset, collate_fn, png: str) -> None:
    from torch.utils.data import DataLoader

    if len(ds) == 0:
        print("[%s] skip: empty dataset" % name)
        return
    bs = min(3, len(ds))
    loader = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate_fn, num_workers=0)
    batch = next(iter(loader))
    raw_b, labels_b, pmask_b, aff_b = batch[0], batch[1], batch[2], batch[3]
    _plot_3d_batch_midz(raw_b, labels_b, pmask_b, aff_b, _smoke_out_path(png), name)


def _run_smoke_sam(name: str, ds: Dataset, png: str) -> None:
    from torch.utils.data import DataLoader

    if len(ds) == 0:
        print("[%s] skip: empty dataset" % name)
        return
    bs = min(3, len(ds))
    loader = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate_fn_2D_SAM_train, num_workers=0)
    raw_b, labels_b, _, _, _, _, pmask_b, aff_b = next(iter(loader))
    _plot_2d_batch(
        raw_b.astype(np.float32) / 255.0, labels_b, pmask_b, aff_b, _smoke_out_path(png), name
    )


def _run_smoke_zebra_avalanche(name: str, ds: Dataset, png: str) -> None:
    if len(ds) == 0:
        print("[%s] skip: empty dataset" % name)
        return
    k = min(3, len(ds))
    raws, labs, pmasks, affs = [], [], [], []
    for i in range(k):
        raw, _one, lab, _pm, mk, aff = ds[i][:6]
        raws.append(raw)
        labs.append(lab)
        pmasks.append(np.asarray(mk))
        affs.append(aff)
    raw_b = np.stack(raws)
    labels_b = np.stack(labs)
    pmask_b = np.stack(pmasks)
    aff_b = np.stack(affs)
    _plot_2d_batch(raw_b, labels_b, pmask_b, aff_b, _smoke_out_path(png), name)


def _smoke_dispatch() -> dict:
    """Lazy registry: value is zero-arg callable running one test."""
    return {
        "hemi_2d": lambda: _run_smoke_2d_standard(
            "hemi 2D",
            Dataset_2D_hemi_Train(split="train", crop_size=128),
            collate_fn_2D_hemi_Train,
            "hemi_dataloader_2d_smoke.png",
        ),
        "fib25_2d": lambda: _run_smoke_2d_standard(
            "fib25 2D",
            Dataset_2D_fib25_Train(split="train", crop_size=128),
            collate_fn_2D_fib25_Train,
            "fib25_dataloader_2d_smoke.png",
        ),
        "cremi_2d": lambda: _run_smoke_2d_standard(
            "cremi 2D",
            Dataset_2D_cremi_Train(split="train", crop_size=128),
            collate_fn_2D_cremi_Train,
            "cremi_dataloader_2d_smoke.png",
        ),
        "vnc_2d": lambda: _run_smoke_2d_standard(
            "vnc 2D",
            Dataset_2D_VNC_Train(split="train", crop_size=128),
            collate_fn_2D_VNC_Train,
            "vnc_dataloader_2d_smoke.png",
        ),
        "isbi2012_2d": lambda: _run_smoke_2d_standard(
            "ISBI-2012 2D",
            Dataset_2D_isbi2012_Train(split="train", crop_size=128),
            collate_fn_2D_isbi2012_Train,
            "isbi2012_dataloader_2d_smoke.png",
        ),
        "ac3_2d": lambda: _run_smoke_2d_standard(
            "AC3 2D",
            Dataset_2D_ac3_Train(split="train", crop_size=128),
            collate_fn_2D_ac3_Train,
            "ac3_dataloader_2d_smoke.png",
        ),
        "ac4_2d": lambda: _run_smoke_2d_standard(
            "AC4 2D",
            Dataset_2D_ac4_Train(split="train", crop_size=128),
            collate_fn_2D_ac4_Train,
            "ac4_dataloader_2d_smoke.png",
        ),
        "basil_2d": lambda: _run_smoke_2d_standard(
            "MICrONS basil 2D",
            Dataset_2D_basil_Train(split="train", crop_size=128),
            collate_fn_2D_basil_Train,
            "basil_dataloader_2d_smoke.png",
        ),
        "minnie_2d": lambda: _run_smoke_2d_standard(
            "MICrONS minnie 2D",
            Dataset_2D_minnie_Train(split="train", crop_size=128),
            collate_fn_2D_minnie_Train,
            "minnie_dataloader_2d_smoke.png",
        ),
        "pinky_2d": lambda: _run_smoke_2d_standard(
            "MICrONS pinky 2D",
            Dataset_2D_pinky_Train(split="train", crop_size=128),
            collate_fn_2D_pinky_Train,
            "pinky_dataloader_2d_smoke.png",
        ),
        "axonem_h_2d": lambda: _run_smoke_2d_standard(
            "AxonEM EM30-H 2D",
            Dataset_2D_axonem_h_Train(split="train", crop_size=128),
            collate_fn_2D_axonem_h_Train,
            "axonem_h_dataloader_2d_smoke.png",
        ),
        "axonem_m_2d": lambda: _run_smoke_2d_standard(
            "AxonEM EM30-M 2D",
            Dataset_2D_axonem_m_Train(split="train", crop_size=128),
            collate_fn_2D_axonem_m_Train,
            "axonem_m_dataloader_2d_smoke.png",
        ),
        "pinky_2d_cl": lambda: _run_smoke_2d_standard(
            "MICrONS pinky 2D CL",
            Dataset_2D_pinky_Train_CL(split="train", crop_size=128, data_idxs=(0, 1, 2)),
            collate_fn_2D_pinky_Train_CL,
            "pinky_dataloader_2d_cl_smoke.png",
        ),
        "pinky_avalanche_2d": lambda: _run_smoke_zebra_avalanche(
            "MICrONS pinky avalanche 2D",
            Dataset_2D_pinky_Train_CL_avalanche(
                split="train", crop_size=128, require_lsd=False, data_idxs=(0, 1, 2)
            ),
            "pinky_avalanche_dataloader_2d_smoke.png",
        ),
        "zebra_2d": lambda: _run_smoke_2d_standard(
            "zebrafinch 2D",
            Dataset_2D_zebrafinch_Train_CL(split="train", crop_size=128, data_idxs=(0, 1, 2)),
            collate_fn_2D_zebrafinch_Train_CL,
            "zebrafinch_dataloader_2d_smoke.png",
        ),
        "zebra_avalanche_2d": lambda: _run_smoke_zebra_avalanche(
            "zebrafinch avalanche 2D",
            Dataset_2D_zebrafinch_Train_CL_avalanche(
                split="train", crop_size=128, require_lsd=False, data_idxs=(0, 1, 2)
            ),
            "zebrafinch_avalanche_dataloader_2d_smoke.png",
        ),
        "hemi_3d": lambda: _run_smoke_3d(
            "hemi 3D",
            Dataset_3D_hemi_Train(split="train", crop_size=128, num_slices=8),
            collate_fn_3D_hemi_Train,
            "hemi_dataloader_3d_smoke.png",
        ),
        "fib25_3d": lambda: _run_smoke_3d(
            "fib25 3D",
            Dataset_3D_fib25_Train(split="train", crop_size=128, num_slices=8),
            collate_fn_3D_fib25_Train,
            "fib25_dataloader_3d_smoke.png",
        ),
        "cremi_3d": lambda: _run_smoke_3d(
            "cremi 3D",
            Dataset_3D_cremi_Train(split="train", crop_size=128, num_slices=8),
            collate_fn_3D_cremi_Train,
            "cremi_dataloader_3d_smoke.png",
        ),
        "vnc_3d": lambda: _run_smoke_3d(
            "vnc 3D",
            Dataset_3D_VNC_Train(split="train", crop_size=128, num_slices=8),
            collate_fn_3D_VNC_Train,
            "vnc_dataloader_3d_smoke.png",
        ),
        "isbi2012_3d": lambda: _run_smoke_3d(
            "ISBI-2012 3D",
            Dataset_3D_isbi2012_Train(split="train", crop_size=128, num_slices=8),
            collate_fn_3D_isbi2012_Train,
            "isbi2012_dataloader_3d_smoke.png",
        ),
        "ac3_3d": lambda: _run_smoke_3d(
            "AC3 3D",
            Dataset_3D_ac3_Train(split="train", crop_size=128, num_slices=8),
            collate_fn_3D_ac3_Train,
            "ac3_dataloader_3d_smoke.png",
        ),
        "ac4_3d": lambda: _run_smoke_3d(
            "AC4 3D",
            Dataset_3D_ac4_Train(split="train", crop_size=128, num_slices=8),
            collate_fn_3D_ac4_Train,
            "ac4_dataloader_3d_smoke.png",
        ),
        "basil_3d": lambda: _run_smoke_3d(
            "MICrONS basil 3D",
            Dataset_3D_basil_Train(split="train", crop_size=128, num_slices=8),
            collate_fn_3D_basil_Train,
            "basil_dataloader_3d_smoke.png",
        ),
        "minnie_3d": lambda: _run_smoke_3d(
            "MICrONS minnie 3D",
            Dataset_3D_minnie_Train(split="train", crop_size=128, num_slices=8),
            collate_fn_3D_minnie_Train,
            "minnie_dataloader_3d_smoke.png",
        ),
        "pinky_3d": lambda: _run_smoke_3d(
            "MICrONS pinky 3D",
            Dataset_3D_pinky_Train(split="train", crop_size=128, num_slices=8),
            collate_fn_3D_pinky_Train,
            "pinky_dataloader_3d_smoke.png",
        ),
        "axonem_h_3d": lambda: _run_smoke_3d(
            "AxonEM EM30-H 3D",
            Dataset_3D_axonem_h_Train(split="train", crop_size=128, num_slices=8),
            collate_fn_3D_axonem_h_Train,
            "axonem_h_dataloader_3d_smoke.png",
        ),
        "axonem_m_3d": lambda: _run_smoke_3d(
            "AxonEM EM30-M 3D",
            Dataset_3D_axonem_m_Train(split="train", crop_size=128, num_slices=8),
            collate_fn_3D_axonem_m_Train,
            "axonem_m_dataloader_3d_smoke.png",
        ),
        "hemi_sam": lambda: _run_smoke_sam(
            "hemi SAM 2D",
            SAM_Dataset_2D_hemi_Train(split="train", crop_size=128),
            "hemi_dataloader_2d_sam_smoke.png",
        ),
        "fib25_sam": lambda: _run_smoke_sam(
            "fib25 SAM 2D",
            SAM_Dataset_2D_fib25_Train(split="train", crop_size=128),
            "fib25_dataloader_2d_sam_smoke.png",
        ),
        "cremi_sam": lambda: _run_smoke_sam(
            "cremi SAM 2D",
            SAM_Dataset_2D_cremi_Train(split="train", crop_size=128),
            "cremi_dataloader_2d_sam_smoke.png",
        ),
        "vnc_sam": lambda: _run_smoke_sam(
            "vnc SAM 2D",
            SAM_Dataset_2D_VNC_Train(split="train", crop_size=128),
            "vnc_dataloader_2d_sam_smoke.png",
        ),
        "isbi2012_sam": lambda: _run_smoke_sam(
            "ISBI-2012 SAM 2D",
            SAM_Dataset_2D_isbi2012_Train(split="train", crop_size=128),
            "isbi2012_dataloader_2d_sam_smoke.png",
        ),
        "ac3_sam": lambda: _run_smoke_sam(
            "AC3 SAM 2D",
            SAM_Dataset_2D_ac3_Train(split="train", crop_size=128),
            "ac3_dataloader_2d_sam_smoke.png",
        ),
        "ac4_sam": lambda: _run_smoke_sam(
            "AC4 SAM 2D",
            SAM_Dataset_2D_ac4_Train(split="train", crop_size=128),
            "ac4_dataloader_2d_sam_smoke.png",
        ),
        "basil_sam": lambda: _run_smoke_sam(
            "MICrONS basil SAM 2D",
            SAM_Dataset_2D_basil_Train(split="train", crop_size=128),
            "basil_dataloader_2d_sam_smoke.png",
        ),
        "minnie_sam": lambda: _run_smoke_sam(
            "MICrONS minnie SAM 2D",
            SAM_Dataset_2D_minnie_Train(split="train", crop_size=128),
            "minnie_dataloader_2d_sam_smoke.png",
        ),
        "pinky_sam": lambda: _run_smoke_sam(
            "MICrONS pinky SAM 2D",
            SAM_Dataset_2D_pinky_Train(split="train", crop_size=128),
            "pinky_dataloader_2d_sam_smoke.png",
        ),
        "axonem_h_sam": lambda: _run_smoke_sam(
            "AxonEM EM30-H SAM 2D",
            SAM_Dataset_2D_axonem_h_Train(split="train", crop_size=128),
            "axonem_h_dataloader_2d_sam_smoke.png",
        ),
        "axonem_m_sam": lambda: _run_smoke_sam(
            "AxonEM EM30-M SAM 2D",
            SAM_Dataset_2D_axonem_m_Train(split="train", crop_size=128),
            "axonem_m_dataloader_2d_sam_smoke.png",
        ),
    }


if __name__ == "__main__":
    import argparse

    dispatch = _smoke_dispatch()
    all_keys = sorted(dispatch.keys())
    parser = argparse.ArgumentParser(
        description="Smoke-test one or all dataloaders; run from project root with ./data/... present."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="all",
        help="Dataset key or 'all'. Choices: %s" % ", ".join(all_keys),
    )
    args = parser.parse_args()
    keys = all_keys if args.dataset == "all" else [args.dataset]
    for k in keys:
        if k not in dispatch:
            raise SystemExit("Unknown dataset %r. Use one of: all, %s" % (k, ", ".join(all_keys)))
        print("=== smoke: %s ===" % k)
        try:
            dispatch[k]()
        except Exception as e:
            print("[%s] FAILED: %s" % (k, e))
            if args.dataset != "all":
                raise
