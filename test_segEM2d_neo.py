from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import tifffile
import torch
from scipy import ndimage
from tqdm.auto import tqdm

from train_segEM2d_neo import (
    LEAVE_SPECIES_CHOICES,
    SegEM2dNeo,
    _infer_prompt_guidance_prior,
    _infer_use_teacher_lsd,
    set_seed,
)
from utils.dataloader import (
    affinity_2d,
    compute_2d_lsds_bbox_local,
    erode_instance_labels,
    gaussian_point_map,
    normalize_minmax,
)

# Ablation upper bound: use at most this many point prompts per neuron (per label_id in CSV).
MAX_PROMPT_POINTS_PER_NEURON = 5


def natural_sort_key(text: str):
    return [int(chunk) if chunk.isdigit() else chunk for chunk in re.split(r"(\d+)", str(text))]


def resolve_processed_root(input_root: str, species: str) -> Path:
    input_path = Path(input_root).resolve()
    species_path = input_path / f"{species}_test"
    if (input_path / "manifest.json").is_file():
        return input_path
    if species_path.is_dir():
        return species_path
    return input_path


def discover_datasets(processed_root: Path) -> list[str]:
    manifest_path = processed_root / "manifest.json"
    if manifest_path.is_file():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        return [item["dataset_name"] for item in manifest.get("datasets", [])]

    datasets = []
    for child in sorted(processed_root.iterdir()):
        if child.is_dir() and (child / "raw").is_dir() and (child / "seg_label").is_dir():
            datasets.append(child.name)
    return datasets


def read_points_csv(csv_path: Path) -> dict[int, np.ndarray]:
    """Group rows by label_id. For each neuron, point order is CSV top-to-bottom among its rows only."""
    grouped: dict[int, list[list[int]]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_id = int(row["label_id"])
            grouped.setdefault(label_id, []).append([int(row["x"]), int(row["y"])])
    return {
        label_id: np.asarray(points, dtype=np.int32)
        for label_id, points in sorted(grouped.items(), key=lambda item: item[0])
    }


def read_label_slice(path: Path) -> np.ndarray:
    label = tifffile.imread(path)
    label = np.asarray(label)
    if label.ndim > 2:
        label = np.squeeze(label)
    if label.ndim != 2:
        raise ValueError(f"Expected 2D label slice, got {label.shape} from {path}")
    return label.astype(np.int32, copy=False)


def resolve_raw_slice_path(raw_root: Path, layer_name: str) -> Path:
    candidates = (
        raw_root / f"{layer_name}.jpg",
        raw_root / f"{layer_name}.jpeg",
        raw_root / f"{layer_name}.tiff",
        raw_root / f"{layer_name}.tif",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Cannot find raw slice for layer {layer_name} under {raw_root}")


def read_raw_slice(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        raw = np.asarray(Image.open(path))
    else:
        raw = tifffile.imread(path)
    raw = np.asarray(raw)
    if raw.ndim > 2:
        raw = np.squeeze(raw)
    if raw.ndim != 2:
        raise ValueError(f"Expected 2D raw slice, got {raw.shape} from {path}")
    return raw.astype(np.float32, copy=False)


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def find_default_checkpoint(checkpoint_dir: Path, species: str) -> Path:
    pattern = f"segEM2d_leaveout_{species}_*_Best_in_val.model"
    candidates = sorted(checkpoint_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint matched {pattern} under {checkpoint_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_model(args, device: torch.device):
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else find_default_checkpoint(Path(args.checkpoint_dir), args.species)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    backbone_config = checkpoint.get("backbone_config", {}) if isinstance(checkpoint, dict) else {}

    backbone_checkpoint = (
        args.backbone_checkpoint
        or checkpoint.get("backbone_checkpoint")
        or f"./output/checkpoints/ACRLSD_2D_leaveout_{args.species}_holdoutVal16_neo_Best_in_val.model"
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected segEM2d checkpoint to be a dict containing mask head weights.")

    mask_width = int(config.get("mask_width", 96))
    mask_head_state_dict = checkpoint.get("ema_state_dict") if not args.no_ema else None
    if mask_head_state_dict is None:
        mask_head_state_dict = checkpoint.get("mask_head_state_dict")
    if mask_head_state_dict is None:
        mask_head_state_dict = checkpoint.get("prompt_head_state_dict")
    if mask_head_state_dict is None:
        mask_head_state_dict = checkpoint.get("model_state_dict")
    if mask_head_state_dict is None:
        raise KeyError(f"No usable mask-head weights found in {checkpoint_path}")
    mask_head_state_dict = strip_module_prefix(mask_head_state_dict)
    use_prompt_guidance_prior = _infer_prompt_guidance_prior(backbone_config, mask_head_state_dict, config, default=False)
    use_teacher_lsd = _infer_use_teacher_lsd(
        backbone_config,
        mask_head_state_dict,
        config,
        use_prompt_guidance_prior=use_prompt_guidance_prior,
    )

    model = SegEM2dNeo(
        device=device,
        backbone_checkpoint=backbone_checkpoint,
        mask_width=mask_width,
        use_teacher_lsd=use_teacher_lsd,
        use_prompt_guidance_prior=use_prompt_guidance_prior,
        prompt_guidance_iters=int(config.get("prompt_guidance_iters", 64)),
    ).to(device)
    model.mask_head.load_state_dict(mask_head_state_dict, strict=True)
    model.eval()
    return model, checkpoint_path, checkpoint, config


def ordered_prompt_prefix(points_xy: np.ndarray, k: int) -> np.ndarray:
    """First k prompts for one neuron: ``points_xy`` is that label_id's list from the CSV (scan order).

    Not the first k rows of the whole file. At most ``MAX_PROMPT_POINTS_PER_NEURON`` and ``len(points_xy)``.
    """
    points_xy = np.asarray(points_xy, dtype=np.int32)
    if points_xy.size == 0 or k <= 0:
        return points_xy[:0]
    n = min(int(k), len(points_xy), MAX_PROMPT_POINTS_PER_NEURON)
    return points_xy[:n]


def normalize_to_uint8(array: np.ndarray, *, prefer_unit_range: bool = True) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    if array.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    lo = float(array.min())
    hi = float(array.max())

    if prefer_unit_range and lo >= 0.0 and hi <= 1.0 + 1e-6:
        scaled = np.clip(array, 0.0, 1.0) * 255.0
        return scaled.astype(np.uint8)

    if hi <= lo + 1e-8:
        return np.zeros_like(array, dtype=np.uint8)

    scaled = (array - lo) / (hi - lo)
    return np.clip(scaled * 255.0, 0.0, 255.0).astype(np.uint8)


def build_targets_from_labels(labels_hw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lab = erode_instance_labels(np.asarray(labels_hw, dtype=np.uint16), iterations=1, border_value=1)
    aff = affinity_2d(lab).astype(np.float32, copy=False)
    lsd = compute_2d_lsds_bbox_local(lab, sigma=(5.0, 5.0)).astype(np.float32, copy=False)
    return aff, lsd


def affinity_to_panel(affinity: np.ndarray) -> np.ndarray:
    affinity = np.asarray(affinity, dtype=np.float32)
    if affinity.ndim == 3 and affinity.shape[0] == 2:
        chan0, chan1 = affinity[0], affinity[1]
    elif affinity.ndim == 3 and affinity.shape[-1] == 2:
        chan0, chan1 = affinity[..., 0], affinity[..., 1]
    else:
        raise ValueError(f"Expected affinity with 2 channels, got shape {affinity.shape}")

    vis0 = normalize_to_uint8(chan0)
    vis1 = normalize_to_uint8(chan1)
    gap = np.full((vis0.shape[0], 4), 255, dtype=np.uint8)
    return np.concatenate([vis0, gap, vis1], axis=1)


def lsd_to_panel(lsd: np.ndarray) -> np.ndarray:
    lsd = np.asarray(lsd, dtype=np.float32)
    if lsd.ndim != 3 or lsd.shape[0] != 6:
        raise ValueError(f"Expected LSD with 6 channels, got shape {lsd.shape}")

    rows = []
    gap = np.full((lsd.shape[1], 4), 255, dtype=np.uint8)
    for row_idx in range(2):
        row_images = [normalize_to_uint8(lsd[row_idx * 3 + col_idx]) for col_idx in range(3)]
        rows.append(np.concatenate([row_images[0], gap, row_images[1], gap, row_images[2]], axis=1))

    row_gap = np.full((4, rows[0].shape[1]), 255, dtype=np.uint8)
    return np.concatenate([rows[0], row_gap, rows[1]], axis=0)


def compute_teacher_cache_segem2d(
    model: SegEM2dNeo,
    raw: np.ndarray,
    device: torch.device,
    *,
    amp_enabled: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One backbone forward per slice; reused for all neurons and all k (same as inside SegEM2dNeo.forward)."""
    raw_t = torch.from_numpy(raw).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
        teacher_1 = model._teacher_forward(raw_t)
    return raw_t, teacher_1


def teacher_cache_to_pred_affinity_lsd(teacher_1: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    aff = teacher_1["affinity_prob"].float().cpu().numpy()[0]
    lsd = teacher_1["lsd_prob"].float().cpu().numpy()[0]
    return aff.astype(np.float32, copy=False), lsd.astype(np.float32, copy=False)


def _mask_input_from_teacher_batch(
    model: SegEM2dNeo,
    raw_b: torch.Tensor,
    x_prompt_b: torch.Tensor,
    teacher_b: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Match SegEM2dNeo.forward mask stack for batch size B."""
    return model._build_mask_input(raw_b, x_prompt_b, teacher_b)


def fuse_layer_predictions_batched(
    model: SegEM2dNeo,
    raw: np.ndarray,
    raw_t_11: torch.Tensor,
    teacher_1: dict[str, torch.Tensor],
    ordered_items: list[tuple[int, np.ndarray]],
    device: torch.device,
    gt_label: np.ndarray,
    *,
    point_theta: float,
    threshold: float,
    amp_enabled: bool,
    instance_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge instance masks in dict iteration order; batch only the mask_head forward."""
    pred_label = np.zeros_like(gt_label, dtype=np.uint16)
    pred_score = np.zeros_like(raw, dtype=np.float32)
    if not ordered_items:
        return pred_label, pred_score

    h, w = raw.shape
    bs = len(ordered_items) if instance_batch_size <= 0 else instance_batch_size
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    for start in range(0, len(ordered_items), bs):
        chunk = ordered_items[start : start + bs]
        b = len(chunk)
        maps = np.empty((b, h, w), dtype=np.float32)
        for i, (_, sp) in enumerate(chunk):
            maps[i] = gaussian_point_map(sp.tolist(), [1] * len(sp), h, w, theta=point_theta).astype(np.float32)

        x_prompt_b = torch.from_numpy(maps).to(device=device, dtype=torch.float32).unsqueeze(1)
        raw_b = raw_t_11.expand(b, -1, -1, -1)
        teacher_b = {
            k: v.expand(b, *v.shape[1:]) if isinstance(v, torch.Tensor) else v for k, v in teacher_1.items()
        }
        mask_in = _mask_input_from_teacher_batch(model, raw_b, x_prompt_b, teacher_b)

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
            mask_logits = model.mask_head(mask_in)
        mask_prob = torch.sigmoid(mask_logits).float().cpu().numpy()[:, 0]

        for i, (label_id, selected_points) in enumerate(chunk):
            pred_mask = mask_prob[i] >= threshold
            pred_mask = keep_prompt_connected_component(pred_mask, selected_points)
            update_mask = pred_mask & (pred_label == 0)
            pred_label[update_mask] = label_id
            pred_score[update_mask] = mask_prob[i][update_mask]

    return pred_label, pred_score


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred_mask = np.asarray(pred_mask, dtype=bool)
    gt_mask = np.asarray(gt_mask, dtype=bool)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return float(intersection / union) if union > 0 else 1.0


def keep_prompt_connected_component(
    pred_mask: np.ndarray,
    points_xy: np.ndarray,
    *,
    neighborhood_radius: int = 3,
) -> np.ndarray:
    pred_mask = np.asarray(pred_mask, dtype=bool)
    if pred_mask.sum() == 0 or len(points_xy) == 0:
        return pred_mask

    labeled, num_components = ndimage.label(pred_mask)
    if num_components <= 1:
        return pred_mask

    h, w = pred_mask.shape
    keep_ids = set()

    for x, y in points_xy:
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        x0 = max(0, x - neighborhood_radius)
        x1 = min(w, x + neighborhood_radius + 1)
        y0 = max(0, y - neighborhood_radius)
        y1 = min(h, y + neighborhood_radius + 1)
        local_ids = np.unique(labeled[y0:y1, x0:x1])
        keep_ids.update(int(v) for v in local_ids if int(v) > 0)

    if keep_ids:
        return np.isin(labeled, list(keep_ids))

    fg_yx = np.column_stack(np.where(pred_mask))
    prompt_yx = points_xy[:, [1, 0]]
    sq_dists = ((fg_yx[:, None, :] - prompt_yx[None, :, :]) ** 2).sum(axis=2)
    nearest_fg_idx = int(np.argmin(sq_dists.min(axis=1)))
    nearest_y, nearest_x = fg_yx[nearest_fg_idx]
    nearest_component = int(labeled[nearest_y, nearest_x])
    return labeled == nearest_component


def save_json(path: Path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def layer_k_outputs_complete(
    output_dataset_root: Path,
    layer_name: str,
    k: int,
    *,
    save_probability: bool,
) -> bool:
    pred_path = output_dataset_root / f"pred_labels_n{k}" / f"{layer_name}.tiff"
    overview_path = output_dataset_root / f"overviews_n{k}" / f"{layer_name}.png"
    if not pred_path.is_file() or pred_path.stat().st_size <= 0:
        return False
    if not overview_path.is_file() or overview_path.stat().st_size <= 0:
        return False
    if save_probability:
        prob_path = output_dataset_root / f"pred_scores_n{k}" / f"{layer_name}.tiff"
        if not prob_path.is_file() or prob_path.stat().st_size <= 0:
            return False
    return True


def append_layer_k_eval_metrics(
    *,
    pred_label: np.ndarray,
    gt_label: np.ndarray,
    layer_name: str,
    k: int,
    per_k_layer_rows: dict[int, list],
    per_k_slice_ious: dict[int, list],
    per_k_instance_ious: dict[int, list],
    per_k_fg_ious: dict[int, list],
    per_k_dataset_detail: dict[int, dict],
    all_slice_ious_by_k: dict[int, list],
    all_instance_ious_by_k: dict[int, list],
    all_fg_ious_by_k: dict[int, list],
) -> None:
    label_ids = sorted(int(v) for v in np.unique(gt_label) if int(v) != 0)
    instance_metrics = []
    for label_id in label_ids:
        iou = compute_iou(pred_label == label_id, gt_label == label_id)
        instance_metrics.append({"label_id": label_id, "iou": iou})
        per_k_instance_ious[k].append(iou)
        all_instance_ious_by_k[k].append(iou)

    fg_iou = compute_iou(pred_label > 0, gt_label > 0)
    per_k_fg_ious[k].append(fg_iou)
    all_fg_ious_by_k[k].append(fg_iou)

    mean_instance_iou = (
        float(np.mean([item["iou"] for item in instance_metrics])) if instance_metrics else float("nan")
    )
    per_k_slice_ious[k].append(mean_instance_iou)
    all_slice_ious_by_k[k].append(mean_instance_iou)
    per_k_layer_rows[k].append(
        {
            "layer_name": layer_name,
            "num_instances": len(instance_metrics),
            "mean_instance_iou": mean_instance_iou,
            "fg_iou": fg_iou,
        }
    )
    per_k_dataset_detail[k]["layers"][layer_name] = {
        "num_instances": len(instance_metrics),
        "mean_instance_iou": mean_instance_iou,
        "fg_iou": fg_iou,
        "instance_metrics": instance_metrics,
    }


def label_id_to_rgb_tuple(label_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(10007 + int(label_id))
    return tuple(int(x) for x in rng.integers(48, 256, size=3))


def instance_label_to_rgb(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    h, w = labels.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for lid in [int(v) for v in np.unique(labels) if int(v) != 0]:
        rgb[labels == lid] = label_id_to_rgb_tuple(lid)
    return rgb


def raw_with_green_prompt_triangles(
    raw: np.ndarray,
    label_to_selected: dict[int, np.ndarray],
    *,
    tri_half_w: int = 5,
    tri_h: int = 7,
    fill_color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw small upward green triangles at each prompt (apex above the point)."""
    g = normalize_to_uint8(raw.astype(np.float32))
    rgb = np.stack([g, g, g], axis=-1).copy()
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    for _lid, pts in sorted(label_to_selected.items()):
        for xy in pts:
            x, y = float(xy[0]), float(xy[1])
            apex = (int(round(x)), int(round(y - tri_h)))
            left = (int(round(x - tri_half_w)), int(round(y + tri_h // 2)))
            right = (int(round(x + tri_half_w)), int(round(y + tri_h // 2)))
            draw.polygon([apex, left, right], outline=fill_color, fill=fill_color)
    return np.asarray(img)


_NEAREST = getattr(getattr(Image, "Resampling", Image), "NEAREST")


def resize_for_overview(panel: np.ndarray, *, max_side: int, is_rgb: bool) -> np.ndarray:
    if max_side <= 0 or panel.size == 0:
        return panel
    h, w = panel.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return panel
    scale = max_side / float(m)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    mode = "RGB" if is_rgb else "L"
    img = Image.fromarray(panel, mode=mode)
    img = img.resize((nw, nh), resample=_NEAREST)
    return np.asarray(img)


def save_segem2d_overview_png(
    path: Path,
    *,
    raw_prompt_rgb: np.ndarray,
    gt_seg_rgb: np.ndarray,
    pred_seg_rgb: np.ndarray,
    aff_pred_panel: np.ndarray,
    aff_gt_panel: np.ndarray,
    lsd_pred_panel: np.ndarray,
    lsd_gt_panel: np.ndarray,
    overview_max_side: int,
    dpi: int,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    titles = (
        "raw + prompts",
        "seg label",
        "seg pred (prompts)",
        "affinity pred",
        "affinity GT",
        "LSD pred",
        "LSD GT",
    )
    rgb_flags = (True, True, True, False, False, False, False)
    panels_in = (
        raw_prompt_rgb,
        gt_seg_rgb,
        pred_seg_rgb,
        aff_pred_panel,
        aff_gt_panel,
        lsd_pred_panel,
        lsd_gt_panel,
    )
    panels = tuple(
        resize_for_overview(p, max_side=overview_max_side, is_rgb=rgb)
        for p, rgb in zip(panels_in, rgb_flags)
    )

    fig, axes = plt.subplots(3, 3, figsize=(16, 16))
    axes_flat = axes.ravel()
    for i in range(7):
        ax = axes_flat[i]
        p = panels[i]
        if rgb_flags[i]:
            ax.imshow(p)
        else:
            ax.imshow(p, cmap="gray", vmin=0, vmax=255)
        ax.set_title(titles[i], fontsize=10)
        ax.axis("off")
    axes_flat[7].axis("off")
    axes_flat[8].axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def build_default_log_path(species: str, output_dir: str) -> Path:
    output_name = Path(output_dir).name or "segEM2d_eval"
    return Path("./output/log") / f"log_test_segEM2d_neo_{species}_{output_name}.txt"


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("test_segEM2d_neo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Run segEM2d neo on full slices; ablate k prompts per neuron (default k=1..5). "
        "For each neuron, use the first k rows with that label_id in the CSV (file order for that id only). "
        "Each k writes pred_labels_n{k}/, overviews_n{k}/, metrics_n{k}/."
    )
    parser.add_argument("--input", type=str, default="./compare/processed")
    parser.add_argument("--output", type=str, default="./compare/output/segEM2d_eval")
    parser.add_argument("--species", type=str, default="human", choices=LEAVE_SPECIES_CHOICES)
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional subset of dataset directories to evaluate.")
    parser.add_argument("--checkpoint", type=str, default=None, help="segEM2d checkpoint path. Default: species Best_in_val.")
    parser.add_argument("--checkpoint-dir", type=str, default="./output/checkpoints")
    parser.add_argument("--backbone-checkpoint", type=str, default=None, help="Optional override for frozen ACRLSD backbone.")
    parser.add_argument(
        "--prompt-counts",
        type=int,
        nargs="*",
        default=None,
        help="Ablation: for each k, use the first k point prompts recorded for each neuron (same label_id rows, "
        f"CSV order within that neuron; max {MAX_PROMPT_POINTS_PER_NEURON}). Default: 1 2 3 4 5. "
        "Each k writes pred_labels_n{k}/, overviews_n{k}/, metrics_n{k}/.",
    )
    parser.add_argument("--point-theta", type=float, default=30.0, help="Gaussian theta for point prompt map generation.")
    parser.add_argument(
        "--instance-batch-size",
        type=int,
        default=32,
        help="Mask-head forward batch size (neurons per GPU step). 0 = all neurons in one batch. "
        "Lower if GPU OOM; higher may improve throughput.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary threshold for predicted instance masks.")
    parser.add_argument("--seed", type=int, default=1998)
    parser.add_argument("--device", type=str, default=None, help="cuda:0 / cpu. Default: cuda:0 if available else cpu.")
    parser.add_argument("--max-layers", type=int, default=0, help="Debug option: only evaluate the first N layers per dataset.")
    parser.add_argument("--no-ema", action="store_true", help="Load raw mask_head_state_dict instead of ema_state_dict.")
    parser.add_argument("--save-probability", action="store_true", help="Also save per-layer max probability maps as float32 TIFFs.")
    parser.add_argument(
        "--overview-dpi",
        type=int,
        default=120,
        help="PNG dpi for overviews/{layer}.png (3×3 layout, 7 panels + 2 empty cells).",
    )
    parser.add_argument(
        "--overview-max-side",
        type=int,
        default=2048,
        help="Max longer side when resizing panels inside the PNG (0 = no resize; full-res PNGs can be huge).",
    )
    parser.add_argument("--log-file", type=str, default=None, help="Optional log file path. Default: output/log/log_test_segEM2d_neo_*.txt")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip inference for a layer/k when pred TIFF, overview PNG (and pred_scores TIFF if --save-probability) "
        "already exist under --output; still aggregates metrics from disk. If every k is done for a layer, skips "
        "backbone/teacher for that layer entirely.",
    )
    return parser


def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    amp_enabled = device.type == "cuda"
    log_path = Path(args.log_file) if args.log_file else build_default_log_path(args.species, args.output)
    logger = setup_logger(log_path)

    raw_prompt_counts = [int(x) for x in (args.prompt_counts or [1, 2, 3, 4, 5])]
    overflow = sorted({k for k in raw_prompt_counts if k > MAX_PROMPT_POINTS_PER_NEURON})
    if overflow:
        logger.warning(
            "Clamping --prompt-counts %s to max %d prompts per neuron.",
            overflow,
            MAX_PROMPT_POINTS_PER_NEURON,
        )
    prompt_counts = sorted({min(k, MAX_PROMPT_POINTS_PER_NEURON) for k in raw_prompt_counts if k > 0})
    if not prompt_counts:
        raise SystemExit("No positive values in --prompt-counts (after clamping).")

    processed_root = resolve_processed_root(args.input, args.species)
    datasets = args.datasets if args.datasets else discover_datasets(processed_root)
    if not datasets:
        raise SystemExit(f"No processed datasets found under {processed_root}")

    model, checkpoint_path, checkpoint, config = load_model(args, device)
    logger.info("Processed root: %s", processed_root)
    logger.info("Checkpoint:     %s", checkpoint_path)
    logger.info("Backbone ckpt:  %s", checkpoint.get("backbone_checkpoint", args.backbone_checkpoint))
    logger.info("Device:         %s", device)
    logger.info("Datasets:       %s", datasets)
    logger.info("Log file:       %s", log_path.resolve())
    logger.info(
        "Prompt ablation k in %s (per neuron: first k rows with that label_id in CSV, max %d).",
        prompt_counts,
        MAX_PROMPT_POINTS_PER_NEURON,
    )
    logger.info(
        "Speed: one teacher forward per slice; batched mask_head (instance-batch-size=%d, 0=all).",
        int(args.instance_batch_size),
    )
    logger.info("Full slice eval (no crop); raw normalized with normalize_minmax.")
    if args.resume:
        logger.info(
            "Resume: skip inference when pred_labels_n{k}/<layer>.tiff, overviews_n{k}/<layer>.png "
            "(and pred_scores_n{k} if --save-probability) already exist; full layer skip skips backbone when all k done."
        )

    output_root = Path(args.output) / f"{args.species}_test"
    output_root.mkdir(parents=True, exist_ok=True)

    run_summary = {
        "species": args.species,
        "processed_root": str(processed_root),
        "checkpoint": str(checkpoint_path),
        "backbone_checkpoint": checkpoint.get("backbone_checkpoint", args.backbone_checkpoint),
        "prompt_counts": prompt_counts,
        "max_prompt_points_per_neuron": MAX_PROMPT_POINTS_PER_NEURON,
        "point_theta": float(args.point_theta),
        "instance_batch_size": int(args.instance_batch_size),
        "threshold": float(args.threshold),
        "full_slice": True,
        "overview_dpi": int(args.overview_dpi),
        "overview_max_side": int(args.overview_max_side),
        "outputs": (
            "Per k: pred_labels_n{k}/, overviews_n{k}/, metrics_n{k}/; "
            "per neuron, first k prompts among rows with that label_id (CSV order for that id), k<="
            f"{MAX_PROMPT_POINTS_PER_NEURON}."
        ),
        "overviews": (
            "per-dataset overviews_n{k}/{layer}.png — 3×3 (7 panels): raw+green prompt triangles, seg GT, seg pred (prompts), "
            "affinity pred/GT, LSD pred/GT (teacher backbone)"
        ),
        "datasets": {},
        "config": config,
    }

    all_slice_ious_by_k = {k: [] for k in prompt_counts}
    all_instance_ious_by_k = {k: [] for k in prompt_counts}
    all_fg_ious_by_k = {k: [] for k in prompt_counts}

    for dataset_name in datasets:
        dataset_root = processed_root / dataset_name
        raw_root = dataset_root / "raw"
        label_root = dataset_root / "seg_label"
        point_root = dataset_root / "point_prompts"

        output_dataset_root = output_root / dataset_name
        for k in prompt_counts:
            (output_dataset_root / f"pred_labels_n{k}").mkdir(parents=True, exist_ok=True)
            (output_dataset_root / f"overviews_n{k}").mkdir(parents=True, exist_ok=True)
            (output_dataset_root / f"metrics_n{k}").mkdir(parents=True, exist_ok=True)
            if args.save_probability:
                (output_dataset_root / f"pred_scores_n{k}").mkdir(parents=True, exist_ok=True)

        layer_paths = sorted(point_root.glob("*.csv"), key=lambda p: natural_sort_key(p.stem))
        if args.max_layers > 0:
            layer_paths = layer_paths[: args.max_layers]

        per_k_layer_rows = {k: [] for k in prompt_counts}
        per_k_slice_ious = {k: [] for k in prompt_counts}
        per_k_instance_ious = {k: [] for k in prompt_counts}
        per_k_fg_ious = {k: [] for k in prompt_counts}
        per_k_dataset_detail = {k: {"layers": {}} for k in prompt_counts}

        for point_csv_path in tqdm(layer_paths, desc=f"{dataset_name} layers"):
            layer_name = point_csv_path.stem

            if args.resume and all(
                layer_k_outputs_complete(
                    output_dataset_root, layer_name, k, save_probability=args.save_probability
                )
                for k in prompt_counts
            ):
                gt_label = read_label_slice(label_root / f"{layer_name}.tiff")
                shapes_ok = True
                pred_by_k: dict[int, np.ndarray] = {}
                for k in prompt_counts:
                    pred_path = output_dataset_root / f"pred_labels_n{k}" / f"{layer_name}.tiff"
                    pred_layer = read_label_slice(pred_path)
                    if pred_layer.shape != gt_label.shape:
                        shapes_ok = False
                        break
                    pred_by_k[k] = pred_layer
                if shapes_ok:
                    for k in prompt_counts:
                        append_layer_k_eval_metrics(
                            pred_label=pred_by_k[k],
                            gt_label=gt_label,
                            layer_name=layer_name,
                            k=k,
                            per_k_layer_rows=per_k_layer_rows,
                            per_k_slice_ious=per_k_slice_ious,
                            per_k_instance_ious=per_k_instance_ious,
                            per_k_fg_ious=per_k_fg_ious,
                            per_k_dataset_detail=per_k_dataset_detail,
                            all_slice_ious_by_k=all_slice_ious_by_k,
                            all_instance_ious_by_k=all_instance_ious_by_k,
                            all_fg_ious_by_k=all_fg_ious_by_k,
                        )
                    continue

            raw = read_raw_slice(resolve_raw_slice_path(raw_root, layer_name))
            gt_label = read_label_slice(label_root / f"{layer_name}.tiff")
            points_by_label = read_points_csv(point_csv_path)

            if raw.shape != gt_label.shape:
                raise ValueError(
                    "raw shape {} != label shape {} for layer {!r}".format(raw.shape, gt_label.shape, layer_name)
                )
            raw = normalize_minmax(raw).astype(np.float32, copy=False)

            aff_gt, lsd_gt = build_targets_from_labels(np.asarray(gt_label, dtype=np.uint16))
            raw_t_11, teacher_1 = compute_teacher_cache_segem2d(model, raw, device, amp_enabled=amp_enabled)
            aff_pred, lsd_pred = teacher_cache_to_pred_affinity_lsd(teacher_1)

            for k in prompt_counts:
                pred_root_k = output_dataset_root / f"pred_labels_n{k}"
                prob_root_k = output_dataset_root / f"pred_scores_n{k}"
                overviews_root_k = output_dataset_root / f"overviews_n{k}"

                if args.resume and layer_k_outputs_complete(
                    output_dataset_root, layer_name, k, save_probability=args.save_probability
                ):
                    pred_resume = read_label_slice(pred_root_k / f"{layer_name}.tiff")
                    if pred_resume.shape == gt_label.shape:
                        append_layer_k_eval_metrics(
                            pred_label=pred_resume,
                            gt_label=gt_label,
                            layer_name=layer_name,
                            k=k,
                            per_k_layer_rows=per_k_layer_rows,
                            per_k_slice_ious=per_k_slice_ious,
                            per_k_instance_ious=per_k_instance_ious,
                            per_k_fg_ious=per_k_fg_ious,
                            per_k_dataset_detail=per_k_dataset_detail,
                            all_slice_ious_by_k=all_slice_ious_by_k,
                            all_instance_ious_by_k=all_instance_ious_by_k,
                            all_fg_ious_by_k=all_fg_ious_by_k,
                        )
                        continue
                    logger.warning(
                        "Resume: pred shape %s != gt shape %s for layer %r k=%s; re-running inference.",
                        pred_resume.shape,
                        gt_label.shape,
                        layer_name,
                        k,
                    )

                label_to_selected: dict[int, np.ndarray] = {}
                for label_id, points_xy in points_by_label.items():
                    selected_points = ordered_prompt_prefix(points_xy, k)
                    if len(selected_points) == 0:
                        continue
                    label_to_selected[label_id] = selected_points

                ordered_items = list(label_to_selected.items())
                pred_label, pred_score = fuse_layer_predictions_batched(
                    model,
                    raw,
                    raw_t_11,
                    teacher_1,
                    ordered_items,
                    device,
                    gt_label,
                    point_theta=args.point_theta,
                    threshold=args.threshold,
                    amp_enabled=amp_enabled,
                    instance_batch_size=args.instance_batch_size,
                )

                tifffile.imwrite(pred_root_k / f"{layer_name}.tiff", pred_label.astype(np.uint16))
                if args.save_probability:
                    tifffile.imwrite(prob_root_k / f"{layer_name}.tiff", pred_score.astype(np.float32))

                save_segem2d_overview_png(
                    overviews_root_k / f"{layer_name}.png",
                    raw_prompt_rgb=raw_with_green_prompt_triangles(raw, label_to_selected),
                    gt_seg_rgb=instance_label_to_rgb(gt_label),
                    pred_seg_rgb=instance_label_to_rgb(pred_label),
                    aff_pred_panel=affinity_to_panel(aff_pred),
                    aff_gt_panel=affinity_to_panel(aff_gt),
                    lsd_pred_panel=lsd_to_panel(lsd_pred),
                    lsd_gt_panel=lsd_to_panel(lsd_gt),
                    overview_max_side=args.overview_max_side,
                    dpi=args.overview_dpi,
                )

                append_layer_k_eval_metrics(
                    pred_label=pred_label,
                    gt_label=gt_label,
                    layer_name=layer_name,
                    k=k,
                    per_k_layer_rows=per_k_layer_rows,
                    per_k_slice_ious=per_k_slice_ious,
                    per_k_instance_ious=per_k_instance_ious,
                    per_k_fg_ious=per_k_fg_ious,
                    per_k_dataset_detail=per_k_dataset_detail,
                    all_slice_ious_by_k=all_slice_ious_by_k,
                    all_instance_ious_by_k=all_instance_ious_by_k,
                    all_fg_ious_by_k=all_fg_ious_by_k,
                )

        by_prompt_count: dict[str, dict] = {}
        for k in prompt_counts:
            layer_rows = per_k_layer_rows[k]
            metrics_root_k = output_dataset_root / f"metrics_n{k}"
            csv_path = metrics_root_k / "layer_metrics.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "layer_name",
                        "num_instances",
                        "mean_instance_iou",
                        "fg_iou",
                    ],
                )
                writer.writeheader()
                writer.writerows(layer_rows)

            k_summary = {
                "prompt_count": k,
                "num_layers": len(layer_rows),
                "mean_slice_iou": float(np.mean(per_k_slice_ious[k])) if per_k_slice_ious[k] else float("nan"),
                "mean_layer_instance_iou": (
                    float(np.mean([row["mean_instance_iou"] for row in layer_rows])) if layer_rows else float("nan")
                ),
                "mean_instance_iou": (
                    float(np.mean(per_k_instance_ious[k])) if per_k_instance_ious[k] else float("nan")
                ),
                "mean_fg_iou": float(np.mean(per_k_fg_ious[k])) if per_k_fg_ious[k] else float("nan"),
                "layers": per_k_dataset_detail[k]["layers"],
            }
            save_json(metrics_root_k / "test_eval.json", k_summary)
            by_prompt_count[str(k)] = k_summary

        num_layers_ref = len(per_k_layer_rows[prompt_counts[0]]) if prompt_counts else 0
        dataset_summary = {
            "dataset_name": dataset_name,
            "prompt_counts": prompt_counts,
            "num_layers": num_layers_ref,
            "by_prompt_count": by_prompt_count,
        }
        run_summary["datasets"][dataset_name] = dataset_summary

        k_parts = " ".join(
            f"k={k}:slice={by_prompt_count[str(k)]['mean_slice_iou']:.4f} fg={by_prompt_count[str(k)]['mean_fg_iou']:.4f}"
            for k in prompt_counts
        )
        logger.info("%s: layers=%d | %s", dataset_name, num_layers_ref, k_parts)

    by_prompt_count_global: dict[str, dict[str, float]] = {}
    for k in prompt_counts:
        by_prompt_count_global[str(k)] = {
            "mean_slice_iou": (
                float(np.mean(all_slice_ious_by_k[k])) if all_slice_ious_by_k[k] else float("nan")
            ),
            "mean_instance_iou": (
                float(np.mean(all_instance_ious_by_k[k])) if all_instance_ious_by_k[k] else float("nan")
            ),
            "mean_fg_iou": float(np.mean(all_fg_ious_by_k[k])) if all_fg_ious_by_k[k] else float("nan"),
        }
    run_summary["by_prompt_count"] = by_prompt_count_global

    save_json(output_root / "summary.json", run_summary)
    logger.info("========================================")
    for k in prompt_counts:
        g = by_prompt_count_global[str(k)]
        logger.info(
            "Overall k=%d: mean slice IoU=%.4f mean inst IoU=%.4f mean fg IoU=%.4f",
            k,
            g["mean_slice_iou"],
            g["mean_instance_iou"],
            g["mean_fg_iou"],
        )
    logger.info(
        "Saved per-k outputs under: %s/<dataset>/{pred_labels_n{k},overviews_n{k},metrics_n{k}}/",
        output_root,
    )
    logger.info("Saved outputs to:          %s", output_root)


if __name__ == "__main__":
    main()
