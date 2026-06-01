import argparse
import glob
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import tifffile
import zarr
from lsd.train import local_shape_descriptor
from PIL import Image
from scipy.ndimage import find_objects, gaussian_filter
from skimage.measure import label as connected_components
from tqdm.auto import tqdm

from utils.dataloader import erode_instance_labels
from utils.dataloader_preLSD import (
    AC3_DANIEL_SEG_BASENAME,
    AC3_NUM_SLICES,
    AC4_DANIEL_SEG_BASENAME,
    AC4_NUM_SLICES,
    DEFAULT_LSD_CACHE_DIR,
    ZEBRA_PATCH_FILES,
    list_dataset_source_names,
    lsd_cache_path,
)


DATASET_CONFIGS = (
    {"dataset_key": "vnc", "data_dir": "./data/groundtruth-drosophila-vnc-master/stack1/"},
    {"dataset_key": "hemi", "data_dir": "./data/funke/hemi/training/"},
    {"dataset_key": "fib25", "data_dir": "./data/funke/fib25/training/"},
    {"dataset_key": "cremi", "data_dir": "./data/CREMI/"},
    {"dataset_key": "isbi2012", "data_dir": "./data/ISBI-2012/"},
    {"dataset_key": "ac3", "data_dir": "./data/AC3/"},
    {"dataset_key": "ac4", "data_dir": "./data/AC4/"},
    {"dataset_key": "basil", "data_dir": "./data/MICrONS/Neuron_zarr/basil/"},
    {"dataset_key": "minnie", "data_dir": "./data/MICrONS/Neuron_zarr/minnie/"},
    {"dataset_key": "pinky", "data_dir": "./data/MICrONS/Neuron_zarr/pinky/"},
    {"dataset_key": "axonem_h", "data_dir": "./data/AxonEM/EM30-H-axon-train-9vol/"},
    {"dataset_key": "axonem_m", "data_dir": "./data/AxonEM/EM30-M-axon-train-9vol/"},
    {
        "dataset_key": "zebrafinch",
        "data_dir": "./data/funke/zebrafinch/training/",
        "zebra_data_idxs": tuple(range(len(ZEBRA_PATCH_FILES))),
    },
)

LSD_SIGMA = (5.0, 5.0, 5.0)
LSD_DOWNSAMPLE = 2
LSD_CHANNELS = 10
LSD_HALO = tuple(int(np.ceil(3.0 * s)) for s in LSD_SIGMA)
_COORD_CACHE: Dict[Tuple[Tuple[int, int, int], int], np.ndarray] = {}


def _task_label(task: Dict[str, object]) -> str:
    return "{}/{}".format(task["dataset_key"], task["source_name"])


def _format_shapes(shapes: Dict[str, Tuple[int, ...]]) -> str:
    parts = []
    for orientation in sorted(shapes):
        parts.append("{}={}".format(orientation, shapes[orientation]))
    return ", ".join(parts)


def _configure_native_threads(num_threads: int) -> None:
    if num_threads < 1:
        raise ValueError("--native-threads must be >= 1")
    value = str(num_threads)
    os.environ["OMP_NUM_THREADS"] = value
    os.environ["OPENBLAS_NUM_THREADS"] = value
    os.environ["MKL_NUM_THREADS"] = value
    os.environ["NUMEXPR_NUM_THREADS"] = value


def _available_cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def _resolve_num_workers(requested_workers: int, num_tasks: int) -> int:
    if requested_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if num_tasks <= 0:
        return 0
    if requested_workers == 0:
        # Full-volume LSD generation is memory heavy, so keep auto parallelism conservative.
        return min(num_tasks, _available_cpu_count(), 2)
    return min(num_tasks, requested_workers)


def _init_worker(native_threads: int) -> None:
    _configure_native_threads(native_threads)


def _cache_path(cache_dir: str, dataset_key: str, source_name: str, orientation: str) -> str:
    return lsd_cache_path(
        cache_dir,
        {
            "dataset_key": dataset_key,
            "source_name": source_name,
            "split": "full",
            "n_val": 0,
            "orientation": orientation,
            "volume_shape_zyx": (),
        },
    )


def _build_tasks(cache_dir: str, datasets: Sequence[str]) -> List[Dict[str, object]]:
    tasks: List[Dict[str, object]] = []
    wanted = set(datasets)
    for cfg in DATASET_CONFIGS:
        dataset_key = str(cfg["dataset_key"])
        if wanted and dataset_key not in wanted:
            continue
        data_dir = str(cfg["data_dir"])
        zebra_data_idxs = cfg.get("zebra_data_idxs", tuple(range(len(ZEBRA_PATCH_FILES))))
        for source_name in list_dataset_source_names(
            dataset_key,
            data_dir,
            zebra_data_idxs=zebra_data_idxs,
        ):
            tasks.append(
                {
                    "dataset_key": dataset_key,
                    "data_dir": data_dir,
                    "source_name": source_name,
                    "cache_paths": {
                        "xy": _cache_path(cache_dir, dataset_key, source_name, "xy"),
                    },
                }
            )
    return tasks


def _connected_components_uint16(labels: np.ndarray) -> np.ndarray:
    cc = connected_components(np.asarray(labels))
    mx = int(cc.max())
    if mx >= 65536:
        raise ValueError("Connected components exceed uint16 range: {}".format(mx))
    return cc.astype(np.uint16)


def _vnc_raw_index_paths(stack1_dir: str) -> Dict[int, str]:
    raw_dir = os.path.join(stack1_dir, "raw")
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError("VNC stack1 raw dir not found: %s" % raw_dir)
    out: Dict[int, str] = {}
    for path in glob.glob(os.path.join(raw_dir, "*.tif")) + glob.glob(os.path.join(raw_dir, "*.TIF")):
        stem, _ = os.path.splitext(os.path.basename(path))
        if stem.isdigit():
            out[int(stem)] = path
    return out


def _membrane_png_for_index(stack1_dir: str, i: int) -> str:
    mem_dir = os.path.join(stack1_dir, "membranes")
    if not os.path.isdir(mem_dir):
        raise FileNotFoundError("VNC stack1 membranes dir not found: %s" % mem_dir)
    for name in ("%02d.png" % i, "%d.png" % i, "%08d.png" % i):
        path = os.path.join(mem_dir, name)
        if os.path.isfile(path):
            return path
    for path in glob.glob(os.path.join(mem_dir, "*.png")) + glob.glob(os.path.join(mem_dir, "*.PNG")):
        stem, _ = os.path.splitext(os.path.basename(path))
        if stem.isdigit() and int(stem) == i:
            return path
    raise FileNotFoundError("Missing membrane mask for slice index %d under %s" % (i, mem_dir))


def _pick_main_3d_h5_dataset(h5_path: str) -> str:
    datasets: List[Tuple[str, Tuple[int, ...]]] = []
    with h5py.File(h5_path, "r") as f:

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                shape = tuple(int(s) for s in obj.shape)
                if len(shape) == 3 and all(s > 0 for s in shape):
                    datasets.append((name, shape))

        f.visititems(visitor)
    if not datasets:
        raise RuntimeError("No 3D dataset found in {}".format(h5_path))
    datasets.sort(key=lambda item: np.prod(item[1]), reverse=True)
    return datasets[0][0]


def _crop_label_to_foreground_bbox(lab: np.ndarray, margin: int = 8) -> np.ndarray:
    fg = lab > 0
    if not np.any(fg):
        return lab
    zz, yy, xx = np.where(fg)
    z0, z1 = int(zz.min()), int(zz.max()) + 1
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    z0 = max(0, z0 - margin)
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    z1 = min(lab.shape[0], z1 + margin)
    y1 = min(lab.shape[1], y1 + margin)
    x1 = min(lab.shape[2], x1 + margin)
    return lab[z0:z1, y0:y1, x0:x1]


def _pack_rgb_labels_to_instances(
    rgb: np.ndarray, dataset_name: str, color_to_id: Dict[int, int], next_id: int
) -> Tuple[np.ndarray, int]:
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
        key = int(u)
        if key == 0:
            continue
        if key not in color_to_id:
            if next_id > 65535:
                raise ValueError(
                    "{}: more than 65535 distinct RGB labels; cannot store as uint16".format(
                        dataset_name
                    )
                )
            color_to_id[key] = next_id
            next_id += 1
        gids[j] = color_to_id[key]
    return gids[inv].reshape(packed.shape), next_id


def _load_label_volume(task: Dict[str, object]) -> np.ndarray:
    dataset_key = str(task["dataset_key"])
    data_dir = str(task["data_dir"])
    source_name = str(task["source_name"])

    if dataset_key == "hemi":
        root = zarr.open(os.path.join(data_dir, source_name), mode="r")
        lab = np.asarray(root["volumes"]["labels"]["neuron_ids"])
        lab = lab[128 : lab.shape[0] - 128, 128 : lab.shape[1] - 128, 128 : lab.shape[2] - 128]
        return _connected_components_uint16(lab)

    if dataset_key == "fib25":
        root = zarr.open(os.path.join(data_dir, source_name), mode="r")
        return _connected_components_uint16(np.asarray(root["volumes"]["labels"]["neuron_ids"]))

    if dataset_key == "cremi":
        with h5py.File(os.path.join(data_dir, source_name), "r") as f:
            lab = np.asarray(f["volumes"]["labels"]["neuron_ids"])
        if source_name == "sample_C_20160501.hdf":
            lab = np.delete(lab, [14, 74], axis=0)
        return _connected_components_uint16(lab)

    if dataset_key == "vnc":
        idx_map = _vnc_raw_index_paths(data_dir)
        indices = sorted(idx_map.keys())
        if not indices:
            raise FileNotFoundError("No .tif slices under %s/raw" % data_dir)
        labs = []
        for i in indices:
            mem = np.asarray(Image.open(_membrane_png_for_index(data_dir, i)))
            if mem.ndim == 3:
                mem = mem[..., 0]
            labs.append((mem > 0).astype(np.uint16))
        return np.stack(labs, axis=0)

    if dataset_key == "isbi2012":
        lab_name = "train-labels.tif" if source_name == "train-volume.tif" else "test-labels.tif"
        lab_tif = np.asarray(tifffile.imread(os.path.join(data_dir, lab_name)))
        return _connected_components_uint16((lab_tif != 0).astype(np.uint8))

    if dataset_key == "ac3":
        seg_dir = os.path.join(data_dir, "ac3_dbseg_images")
        labs = []
        color_to_id: Dict[int, int] = {}
        next_id = 1
        for z in range(AC3_NUM_SLICES):
            s_num = AC3_NUM_SLICES - z
            seg_path = os.path.join(seg_dir, AC3_DANIEL_SEG_BASENAME % s_num)
            if not os.path.isfile(seg_path):
                raise FileNotFoundError("Missing AC3 Daniel seg slice: %s" % seg_path)
            lab, next_id = _pack_rgb_labels_to_instances(
                np.asarray(Image.open(seg_path)), "AC3", color_to_id, next_id
            )
            labs.append(lab)
        return np.stack(labs, axis=0)

    if dataset_key == "ac4":
        seg_dir = os.path.join(data_dir, "ac4_seg_daniel")
        labs = []
        color_to_id: Dict[int, int] = {}
        next_id = 1
        for z in range(AC4_NUM_SLICES):
            s_num = AC4_NUM_SLICES - z
            seg_path = os.path.join(seg_dir, AC4_DANIEL_SEG_BASENAME % s_num)
            if not os.path.isfile(seg_path):
                raise FileNotFoundError("Missing AC4 Daniel seg slice: %s" % seg_path)
            lab, next_id = _pack_rgb_labels_to_instances(
                np.asarray(Image.open(seg_path)), "AC4", color_to_id, next_id
            )
            labs.append(lab)
        return np.stack(labs, axis=0)

    if dataset_key in ("basil", "minnie", "pinky"):
        path = os.path.join(os.path.abspath(data_dir), source_name + ".zarr")
        if not os.path.isdir(path):
            raise FileNotFoundError("MICrONS neuron zarr not found: %s" % path)
        root = zarr.open(path, mode="r")
        lab = np.asarray(root["volumes"]["labels"])
        mx = int(lab.max())
        if mx >= 65536:
            raise ValueError("MICrONS zarr %s: label max %d exceeds uint16" % (source_name, mx))
        return lab.astype(np.uint16)

    if dataset_key in ("axonem_h", "axonem_m"):
        seg_path = os.path.join(os.path.abspath(data_dir), "seg_%s_pad.h5" % source_name)
        if not os.path.isfile(seg_path):
            raise FileNotFoundError("AxonEM seg not found: %s" % seg_path)
        seg_ds = _pick_main_3d_h5_dataset(seg_path)
        with h5py.File(seg_path, "r") as f:
            lab = np.asarray(f[seg_ds])
        mx = int(lab.max())
        if mx >= 65536:
            raise ValueError("AxonEM %s: label max %d exceeds uint16" % (source_name, mx))
        return _crop_label_to_foreground_bbox(lab.astype(np.uint16), margin=8)

    if dataset_key == "zebrafinch":
        root = zarr.open(os.path.join(data_dir, source_name), mode="r")
        lab = np.asarray(root["volumes"]["labels"]["neuron_ids"])
        return _connected_components_uint16(lab)

    raise ValueError("Unknown dataset_key: {}".format(dataset_key))


def _crop_with_halo(
    shape: Tuple[int, int, int], bbox: Tuple[slice, slice, slice], halo: Tuple[int, int, int]
) -> Tuple[slice, slice, slice]:
    slices = []
    for axis, slc in enumerate(bbox):
        start = max(0, int(slc.start) - int(halo[axis]))
        stop = min(int(shape[axis]), int(slc.stop) + int(halo[axis]))
        slices.append(slice(start, stop))
    return tuple(slices)  # type: ignore[return-value]


def _pad_spatial_to_multiple(
    array: np.ndarray, factor: int
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    shape = tuple(int(s) for s in array.shape)
    if factor <= 1:
        return array, shape
    pad_width = []
    for size in shape:
        pad_after = (factor - (size % factor)) % factor
        pad_width.append((0, pad_after))
    if not any(pad_after for _, pad_after in pad_width):
        return array, shape
    return np.pad(array, pad_width, mode="constant"), shape


def _get_downsampled_coords(sub_shape: Tuple[int, int, int], step: int) -> np.ndarray:
    key = (sub_shape, step)
    if key not in _COORD_CACHE:
        grid = np.meshgrid(
            np.arange(0, sub_shape[0] * step, step, dtype=np.float32),
            np.arange(0, sub_shape[1] * step, step, dtype=np.float32),
            np.arange(0, sub_shape[2] * step, step, dtype=np.float32),
            indexing="ij",
        )
        _COORD_CACHE[key] = np.asarray(grid, dtype=np.float32)
    return _COORD_CACHE[key]


def _downsampled_local_descriptor(
    local_mask: np.ndarray, sigma: Tuple[float, float, float], downsample: int
) -> np.ndarray:
    padded_mask, original_shape = _pad_spatial_to_multiple(local_mask.astype(np.float32, copy=False), downsample)
    sub_mask = padded_mask[::downsample, ::downsample, ::downsample]
    sub_shape = tuple(int(s) for s in sub_mask.shape)
    sub_sigma = tuple(float(s) / float(downsample) for s in sigma)
    coords = _get_downsampled_coords(sub_shape, downsample)

    count = gaussian_filter(sub_mask, sigma=sub_sigma, mode="constant", cval=0.0, truncate=3.0)
    count_safe = count.copy()
    count_safe[count_safe == 0] = 1.0

    masked_coords = coords * sub_mask[None, ...]
    mean = np.stack(
        [
            gaussian_filter(masked_coords[d], sigma=sub_sigma, mode="constant", cval=0.0, truncate=3.0)
            for d in range(3)
        ],
        axis=0,
    )
    mean /= count_safe[None, ...]
    mean_offset = mean - coords

    cov_inputs = (
        masked_coords[0] * masked_coords[0],
        masked_coords[1] * masked_coords[1],
        masked_coords[2] * masked_coords[2],
        masked_coords[0] * masked_coords[1],
        masked_coords[0] * masked_coords[2],
        masked_coords[1] * masked_coords[2],
    )
    covariance = np.stack(
        [
            gaussian_filter(arr, sigma=sub_sigma, mode="constant", cval=0.0, truncate=3.0)
            for arr in cov_inputs
        ],
        axis=0,
    )
    covariance /= count_safe[None, ...]
    covariance[0] -= mean[0] * mean[0]
    covariance[1] -= mean[1] * mean[1]
    covariance[2] -= mean[2] * mean[2]
    covariance[3] -= mean[0] * mean[1]
    covariance[4] -= mean[0] * mean[2]
    covariance[5] -= mean[1] * mean[2]

    variance = covariance[:3]
    variance[variance < 1e-3] = 1e-3
    pearson = covariance[3:].copy()
    pearson[0] /= np.sqrt(variance[0] * variance[1])
    pearson[1] /= np.sqrt(variance[0] * variance[2])
    pearson[2] /= np.sqrt(variance[1] * variance[2])
    variance[0] /= sigma[0] ** 2
    variance[1] /= sigma[1] ** 2
    variance[2] /= sigma[2] ** 2

    sub_descriptor = np.concatenate([mean_offset, variance, pearson, count_safe[None, ...]], axis=0)
    sub_descriptor[[0, 1, 2]] = sub_descriptor[[0, 1, 2]] / np.asarray(sigma, dtype=np.float32)[:, None, None, None] * 0.5 + 0.5
    sub_descriptor[[6, 7, 8]] = sub_descriptor[[6, 7, 8]] * 0.5 + 0.5
    np.clip(sub_descriptor, 0.0, 1.0, out=sub_descriptor)

    if downsample > 1:
        sub_descriptor = np.repeat(sub_descriptor, downsample, axis=1)
        sub_descriptor = np.repeat(sub_descriptor, downsample, axis=2)
        sub_descriptor = np.repeat(sub_descriptor, downsample, axis=3)

    cropped = sub_descriptor[
        :,
        : original_shape[0],
        : original_shape[1],
        : original_shape[2],
    ]
    return cropped * local_mask[None, ...]


def _compute_xy_lsds_bbox_local(
    segmentation: np.ndarray, progress_desc: str | None = None, show_progress: bool = True
) -> np.ndarray:
    segmentation = np.asarray(segmentation)
    descriptors = np.zeros((LSD_CHANNELS,) + segmentation.shape, dtype=np.float32)
    object_slices = find_objects(segmentation)
    total_labels = sum(1 for bbox in object_slices if bbox is not None)

    progress_iter = tqdm(
        enumerate(object_slices, start=1),
        total=len(object_slices),
        desc=progress_desc or "Volume LSD",
        leave=False,
        disable=(not show_progress) or total_labels == 0,
        unit="label",
    )
    for label, bbox in progress_iter:
        if bbox is None:
            continue
        expanded = _crop_with_halo(segmentation.shape, bbox, LSD_HALO)
        local_seg = segmentation[expanded]
        local_mask = local_seg == label
        if not np.any(local_mask):
            continue
        descriptors[(slice(None),) + expanded] += _downsampled_local_descriptor(
            local_mask=local_mask,
            sigma=LSD_SIGMA,
            downsample=LSD_DOWNSAMPLE,
        )

    return descriptors


def _compute_xy_lsds(task: Dict[str, object], lab: np.ndarray | None = None) -> np.ndarray:
    if lab is None:
        lab = _load_label_volume(task)
    lab = np.asarray(lab)
    if lab.size == 0:
        raise ValueError("Empty label volume for task {}".format(task))

    # Canonical cache is stored in (H, W, Z) layout.
    lab_hwz = lab.transpose(1, 2, 0)
    lab_hwz = erode_instance_labels(lab_hwz, iterations=1, border_value=1)
    return _compute_xy_lsds_bbox_local(
        lab_hwz,
        progress_desc="LSD {}".format(_task_label(task)),
        show_progress=True,
    )


def _expected_lsd_shape_from_lab(lab: np.ndarray) -> Tuple[int, ...]:
    return (10,) + lab.transpose(1, 2, 0).shape


def _compute_and_store_one(task: Dict[str, object], overwrite: bool) -> Dict[str, object]:
    print("Processing {} ...".format(_task_label(task)), flush=True)
    cache_paths = dict(task["cache_paths"])
    xy_path = str(cache_paths["xy"])
    need_xy = overwrite or (not os.path.isfile(xy_path))
    skipped_files = 0 if need_xy else 1
    if not need_xy:
        print("Skipping {} (all cache files already exist)".format(_task_label(task)), flush=True)
        return {
            "status": "skipped",
            "written_files": 0,
            "skipped_files": skipped_files,
            "task": task,
        }

    if (not overwrite) and os.path.isfile(xy_path):
        lsds_xy = np.load(xy_path, mmap_mode="r")
        xy_source = "cache"
    else:
        lab = _load_label_volume(task)
        lab = np.asarray(lab)
        print(
            "Computing {}: label_shape={} -> xy={}".format(
                _task_label(task),
                tuple(int(v) for v in lab.shape),
                tuple(int(v) for v in _expected_lsd_shape_from_lab(lab)),
            ),
            flush=True,
        )
        lsds_xy = _compute_xy_lsds(task, lab=lab)
        xy_source = "computed"

    os.makedirs(os.path.dirname(xy_path), exist_ok=True)
    np.save(xy_path, lsds_xy)
    written_files = 1
    output_shapes = {"xy": tuple(int(v) for v in lsds_xy.shape)}

    print(
        "Finished {}: {} (xy_source={})".format(
            _task_label(task), _format_shapes(output_shapes), xy_source
        ),
        flush=True,
    )
    return {
        "status": "written" if written_files else "skipped",
        "written_files": written_files,
        "skipped_files": skipped_files,
        "xy_source": xy_source,
        "shapes": output_shapes,
        "task": task,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute full-volume 3D LSD caches used by ACRLSD 3D."
    )
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_LSD_CACHE_DIR)
    parser.add_argument(
        "--n-val-holdout",
        type=int,
        default=32,
        help="Deprecated compatibility flag. Full-volume cache generation ignores train/val splits.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of concurrent volume tasks. 0 picks a conservative process count "
        "based on visible CPUs and memory-heavy workload characteristics. "
        "Set 1 to force serial execution.",
    )
    parser.add_argument(
        "--native-threads",
        type=int,
        default=1,
        help="Threads used by native numerical kernels inside each task.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Comma separated dataset keys, e.g. hemi,fib25,cremi . Default: all",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _configure_native_threads(args.native_threads)
    dataset_keys = ()
    if args.datasets.strip().lower() != "all":
        dataset_keys = tuple(k.strip() for k in args.datasets.split(",") if k.strip())
    tasks = _build_tasks(
        cache_dir=args.cache_dir,
        datasets=dataset_keys,
    )
    os.makedirs(args.cache_dir, exist_ok=True)
    total_cache_files = len(tasks)
    effective_num_workers = _resolve_num_workers(args.num_workers, len(tasks))
    print(
        "Total source volumes: {}, total cache files: {}, requested_workers={}, "
        "effective_workers={}, native_threads={}".format(
            len(tasks),
            total_cache_files,
            args.num_workers,
            effective_num_workers,
            args.native_threads,
        )
    )
    failures = []
    written = 0
    skipped = 0
    if effective_num_workers <= 1:
        for task in tqdm(tasks, desc="Precomputing LSD volumes"):
            try:
                result = _compute_and_store_one(task, args.overwrite)
                written += int(result["written_files"])
                skipped += int(result["skipped_files"])
            except Exception as exc:
                failures.append((task, exc, traceback.format_exc()))
                print("FAILED: {}".format(task))
                print(exc)
    else:
        with ProcessPoolExecutor(
            max_workers=effective_num_workers,
            initializer=_init_worker,
            initargs=(args.native_threads,),
        ) as executor:
            futures = {executor.submit(_compute_and_store_one, task, args.overwrite): task for task in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Precomputing LSD volumes"):
                task = futures[future]
                try:
                    result = future.result()
                    written += int(result["written_files"])
                    skipped += int(result["skipped_files"])
                except BrokenProcessPool as exc:
                    failures.append((task, exc, traceback.format_exc()))
                    print("FAILED: {}".format(task))
                    print(exc)
                    print(
                        "Worker pool died abruptly. This usually means the OS killed a worker "
                        "because memory usage was too high. Retry with fewer --num-workers "
                        "(for example 1 or 2), or process fewer datasets at a time."
                    )
                    break
                except Exception as exc:
                    failures.append((task, exc, traceback.format_exc()))
                    print("FAILED: {}".format(task))
                    print(exc)
    print("Finished. written={}, skipped={}, failed={}".format(written, skipped, len(failures)))
    if failures:
        for task, exc, tb in failures[:10]:
            print("=" * 80)
            print("Task:", task)
            print("Error:", exc)
            print(tb)
        raise SystemExit(1)
