import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage import measure
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils.dataloader import (  # noqa: E402
    ZEBRA_PATCH_FILES,
    Dataset_2D_VNC_Train,
    Dataset_2D_ac3_Train,
    Dataset_2D_ac4_Train,
    Dataset_2D_axonem_h_Train,
    Dataset_2D_axonem_m_Train,
    Dataset_2D_basil_Train,
    Dataset_2D_cremi_Train,
    Dataset_2D_fib25_Train,
    Dataset_2D_hemi_Train,
    Dataset_2D_isbi2012_Train,
    Dataset_2D_minnie_Train,
    Dataset_2D_pinky_Train,
    Dataset_2D_zebrafinch_Train_CL,
)

SPECIES_TO_SOURCES = {
    "drosophila": ("hemi", "fib25", "cremi", "vnc", "isbi2012"),
    "mouse": ("ac3", "ac4", "basil", "minnie", "pinky", "axonem_m"),
    "human": ("axonem_h",),
    "zebrafinch": ("zebrafinch",),
}


def build_source_factories():
    common = dict(split="train", crop_size=None, require_lsd=False, require_xz_yz=False, n_val=0, augment=False)
    return {
        "hemi": lambda: Dataset_2D_hemi_Train(data_dir="./data/funke/hemi/training/", **common),
        "fib25": lambda: Dataset_2D_fib25_Train(data_dir="./data/funke/fib25/training/", **common),
        "cremi": lambda: Dataset_2D_cremi_Train(data_dir="./data/CREMI/", **common),
        "vnc": lambda: Dataset_2D_VNC_Train(data_dir="./data/groundtruth-drosophila-vnc-master/stack1/", **common),
        "isbi2012": lambda: Dataset_2D_isbi2012_Train(data_dir="./data/ISBI-2012/", **common),
        "ac3": lambda: Dataset_2D_ac3_Train(data_dir="./data/AC3/", **common),
        "ac4": lambda: Dataset_2D_ac4_Train(data_dir="./data/AC4/", **common),
        "basil": lambda: Dataset_2D_basil_Train(data_dir="./data/MICrONS/Neuron_zarr/basil/", **common),
        "minnie": lambda: Dataset_2D_minnie_Train(data_dir="./data/MICrONS/Neuron_zarr/minnie/", **common),
        "pinky": lambda: Dataset_2D_pinky_Train(data_dir="./data/MICrONS/Neuron_zarr/pinky/", **common),
        "axonem_m": lambda: Dataset_2D_axonem_m_Train(data_dir="./data/AxonEM/EM30-M-axon-train-9vol/", **common),
        "axonem_h": lambda: Dataset_2D_axonem_h_Train(data_dir="./data/AxonEM/EM30-H-axon-train-9vol/", **common),
        "zebrafinch": lambda: Dataset_2D_zebrafinch_Train_CL(
            data_dir="./data/funke/zebrafinch/training/",
            data_idxs=tuple(range(len(ZEBRA_PATCH_FILES))),
            split="train",
            crop_size=None,
            require_lsd=False,
            require_xz_yz=False,
            n_val=0,
            augment=False,
        ),
    }


def stable_int_seed(base_seed, *parts):
    payload = "::".join([str(base_seed), *map(str, parts)]).encode("utf-8")
    return int(hashlib.blake2b(payload, digest_size=8).hexdigest(), 16) % (2**32)


def normalize_2d(arr):
    arr = np.asarray(arr)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D slice, got shape {arr.shape}")
    return arr


def save_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def iter_with_progress(items, desc, total=None):
    if tqdm is not None:
        return tqdm(items, desc=desc, total=total, leave=False)
    return items


def sample_grid_points_from_mask(candidate_mask, bbox, stride):
    if stride < 1:
        raise ValueError(f"point_grid_stride must be >= 1, got {stride}")

    y0, x0, y1, x1 = map(int, bbox)
    ys = np.arange(y0, y1, stride, dtype=np.int32)
    xs = np.arange(x0, x1, stride, dtype=np.int32)
    grid_points = [(int(y), int(x)) for y in ys for x in xs if candidate_mask[y, x]]

    if grid_points:
        return np.asarray(grid_points, dtype=np.int32)

    candidate_coords = np.column_stack(np.where(candidate_mask))
    if len(candidate_coords) == 0:
        return np.empty((0, 2), dtype=np.int32)

    center = np.array([(y0 + y1 - 1) / 2.0, (x0 + x1 - 1) / 2.0], dtype=np.float32)
    nearest_idx = int(np.argmin(np.sum((candidate_coords - center) ** 2, axis=1)))
    return candidate_coords[nearest_idx : nearest_idx + 1].astype(np.int32, copy=False)


def select_component_center(binary_mask, dist_transform, region):
    max_dist = float(dist_transform.max())
    if max_dist > 0:
        candidate_coords = np.column_stack(np.where(dist_transform == max_dist))
    else:
        candidate_coords = np.column_stack(np.where(binary_mask))

    if len(candidate_coords) == 0:
        raise ValueError("Cannot select center from an empty connected component.")

    centroid = np.asarray(region.centroid, dtype=np.float32)
    nearest_idx = int(np.argmin(np.sum((candidate_coords - centroid) ** 2, axis=1)))
    return candidate_coords[nearest_idx].astype(np.int32, copy=False)


def sample_positive_points(binary_mask, candidate_mask, dist_transform, region, stride, rng, num_random_points=4):
    center_point = select_component_center(binary_mask, dist_transform, region)
    grid_points = sample_grid_points_from_mask(candidate_mask, region.bbox, stride)

    if len(grid_points) == 0:
        random_pool = center_point[None, :]
    else:
        not_center = np.any(grid_points != center_point[None, :], axis=1)
        random_pool = grid_points[not_center]
        if len(random_pool) == 0:
            random_pool = grid_points

    sample_indices = rng.choice(len(random_pool), size=num_random_points, replace=len(random_pool) < num_random_points)
    sampled_points = random_pool[sample_indices].astype(np.int32, copy=False)
    return np.vstack([center_point[None, :], sampled_points]).astype(np.int32, copy=False)


def raw_to_uint8(raw):
    raw = normalize_2d(np.asarray(raw))
    if raw.dtype == np.uint8:
        return raw

    raw = raw.astype(np.float32, copy=False)
    finite_mask = np.isfinite(raw)
    if not np.any(finite_mask):
        return np.zeros(raw.shape, dtype=np.uint8)

    valid = raw[finite_mask]
    min_value = float(valid.min())
    max_value = float(valid.max())
    if max_value <= min_value:
        return np.zeros(raw.shape, dtype=np.uint8)

    scaled = np.zeros(raw.shape, dtype=np.float32)
    scaled[finite_mask] = (raw[finite_mask] - min_value) / (max_value - min_value)
    return np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)


def extract_slice_prompts(label_slice, point_thre, point_grid_stride, seed, dataset_name, layer_name):
    label_slice = normalize_2d(label_slice).astype(np.uint16, copy=False)
    connected = measure.label(label_slice, connectivity=1)
    max_length = 0.95 * max(label_slice.shape)
    saved_labels = np.zeros_like(connected, dtype=np.uint16)
    point_rows = []
    box_rows = []
    instance_count = 0

    for region in measure.regionprops(connected):
        height = region.bbox[2] - region.bbox[0]
        width = region.bbox[3] - region.bbox[1]
        if (
            region.area < 10
            or height < 5
            or width < 5
            or height > max_length
            or width > max_length
        ):
            continue

        instance_count += 1
        binary_mask = connected == region.label
        saved_labels[binary_mask] = instance_count

        y0, x0, y1, x1 = map(int, region.bbox)
        box_rows.append([instance_count, x0, y0, x1 - 1, y1 - 1])

        padded_mask = np.pad(binary_mask.astype(np.uint8), pad_width=1, mode="constant", constant_values=0)
        dist_transform = distance_transform_edt(padded_mask)[1:-1, 1:-1]
        max_dist = float(dist_transform.max())
        if max_dist <= 0:
            candidate_mask = binary_mask
        else:
            threshold = max_dist * point_thre
            candidate_mask = dist_transform > threshold
            if not np.any(candidate_mask):
                candidate_mask = binary_mask

        rng = np.random.default_rng(stable_int_seed(seed, dataset_name, layer_name, instance_count))
        coords = sample_positive_points(
            binary_mask=binary_mask,
            candidate_mask=candidate_mask,
            dist_transform=dist_transform,
            region=region,
            stride=point_grid_stride,
            rng=rng,
        )
        for point_id, (y, x) in enumerate(coords):
            point_rows.append([instance_count, point_id, int(x), int(y)])

    return saved_labels, point_rows, box_rows, instance_count


def slice_output_paths(dataset_root, layer_name):
    return {
        "raw": dataset_root / "raw" / f"{layer_name}.jpg",
        "label": dataset_root / "seg_label" / f"{layer_name}.tiff",
        "point": dataset_root / "point_prompts" / f"{layer_name}.csv",
        "box": dataset_root / "box_prompts" / f"{layer_name}.csv",
    }


def slice_is_complete(dataset_root, layer_name):
    paths = slice_output_paths(dataset_root, layer_name)
    return all(path.is_file() for path in paths.values())


def existing_slice_metadata(dataset_root, layer_name):
    paths = slice_output_paths(dataset_root, layer_name)
    raw = np.asarray(Image.open(paths["raw"]))
    label = tifffile.imread(paths["label"])
    instance_count = int(np.max(label))
    return [layer_name, int(raw.shape[1]), int(raw.shape[0]), instance_count]


def export_source_dataset(dataset_name, dataset, output_root, point_thre, point_grid_stride, seed, overwrite=False):
    dataset_root = output_root / dataset_name
    raw_root = dataset_root / "raw"
    label_root = dataset_root / "seg_label"
    point_root = dataset_root / "point_prompts"
    box_root = dataset_root / "box_prompts"
    for p in (raw_root, label_root, point_root, box_root):
        p.mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    exported = 0
    skipped = 0
    total = len(dataset.images)
    iterator = enumerate(zip(dataset.images, dataset.masks))
    iterator = iter_with_progress(iterator, desc=f"{dataset_name} slices", total=total)
    for idx, (raw_slice, label_slice) in iterator:
        raw = normalize_2d(np.asarray(raw_slice))
        labels = normalize_2d(np.asarray(label_slice, dtype=np.uint16))
        layer_name = f"{idx:06d}"
        if not overwrite and slice_is_complete(dataset_root, layer_name):
            metadata_rows.append(existing_slice_metadata(dataset_root, layer_name))
            skipped += 1
            continue
        saved_labels, point_rows, box_rows, instance_count = extract_slice_prompts(
            labels,
            point_thre=point_thre,
            point_grid_stride=point_grid_stride,
            seed=seed,
            dataset_name=dataset_name,
            layer_name=layer_name,
        )
        if instance_count == 0:
            continue

        paths = slice_output_paths(dataset_root, layer_name)
        raw_uint8 = raw_to_uint8(raw)
        Image.fromarray(raw_uint8).save(paths["raw"], format="JPEG", quality=95)
        tifffile.imwrite(paths["label"], saved_labels)
        save_csv(paths["point"], ["label_id", "point_id", "x", "y"], point_rows)
        save_csv(paths["box"], ["label_id", "x0", "y0", "x1", "y1"], box_rows)
        metadata_rows.append([layer_name, raw_uint8.shape[1], raw_uint8.shape[0], instance_count])
        exported += 1

    save_csv(dataset_root / "slices.csv", ["layer_name", "width", "height", "instance_count"], metadata_rows)
    return {
        "dataset_name": dataset_name,
        "total_layers": total,
        "exported_layers": exported,
        "skipped_layers": skipped,
        "metadata_layers": len(metadata_rows),
    }


def export_species(species, output_root, point_thre, point_grid_stride, seed, sources=None, overwrite=False):
    if species not in SPECIES_TO_SOURCES:
        raise ValueError(f"species must be one of {sorted(SPECIES_TO_SOURCES)}, got {species!r}")

    target_sources = tuple(sources) if sources else SPECIES_TO_SOURCES[species]
    factories = build_source_factories()
    missing = [source for source in target_sources if source not in factories]
    if missing:
        raise ValueError(f"Unknown sources: {missing}")

    species_root = Path(output_root) / f"{species}_test"
    species_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "species": species,
        "seed": int(seed),
        "point_threshold": float(point_thre),
        "point_grid_stride": int(point_grid_stride),
        "datasets": [],
    }
    source_iter = iter_with_progress(target_sources, desc=f"{species} datasets", total=len(target_sources))
    for source in source_iter:
        dataset = factories[source]()
        summary = export_source_dataset(
            source,
            dataset,
            species_root,
            point_thre=point_thre,
            point_grid_stride=point_grid_stride,
            seed=seed,
            overwrite=overwrite,
        )
        manifest["datasets"].append(summary)
        print(
            f"dataset {source}: total={summary['total_layers']} "
            f"exported={summary['exported_layers']} skipped={summary['skipped_layers']}"
        )

    with open(species_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"saved processed data to: {species_root}")


def parse_args():
    parser = argparse.ArgumentParser(description="Export leave-species-out test slices to SAM-ready JPG/TIFF/CSV data.")
    parser.add_argument("--species", type=str, default="human", choices=sorted(SPECIES_TO_SOURCES))
    parser.add_argument("--output-root", type=str, default="./compare/processed")
    parser.add_argument("--point-thre", type=float, default=0.2)
    parser.add_argument(
        "--point-grid-stride",
        type=int,
        default=16,
        help="Grid stride for sampling point prompts inside each neuron instance.",
    )
    parser.add_argument("--seed", type=int, default=1998)
    parser.add_argument("--sources", nargs="*", default=None, help="Optional subset of source keys within the species.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite slices even if JPG/TIFF/CSV outputs already exist. Default behavior supports resume by skipping complete slices.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_species(
        species=args.species,
        output_root=args.output_root,
        point_thre=args.point_thre,
        point_grid_stride=args.point_grid_stride,
        seed=args.seed,
        sources=args.sources,
        overwrite=args.overwrite,
    )
