"""
Interactive segEM2d training: instance-centric crops, pos/neg prompts, multi-round correction sampling.
This is the default data path for ``train_segEM2d_neo.py`` (no legacy random-crop / positive-only branch).

Designed to avoid importing ``utils.dataloader`` (``dataloader`` may import this module for cropping).

Conceptual training sample (training script returns a subset via the existing 7-tuple batch):

.. code-block:: text

    {
        "image": (1, H, W) float32,           # normalized EM patch
        "target_mask": (H, W) uint8 binary,   # single target neuron after erosion
        "all_instance_mask": labels tensor,   # same as batch ``labels`` (instance ids)
        "point_map": (H, W) float32,          # signed Gaussian map (+ pos, − neg)
        "positive_points" / "negative_points": # derivable from interaction_history
        "interaction_history": [...],         # in meta from build_interactive_prompt_episode
        "meta": { "target_instance_id", "replay_round", "difficulty", ... },
    }

Public entry points: ``InteractiveSegEMConfig``, ``prepare_2d_instance_centric_pair``,
``sample_initial_prompts``, ``sample_correction_prompts``, ``build_interactive_prompt_episode``,
``build_seg_interactive_sample``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import albumentations as A
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, label as cc_label

# ---------------------------------------------------------------------------
# Gaussian prompt raster (same semantics as dataloader.gaussian_point_map)
# ---------------------------------------------------------------------------


def gaussian_point_map(points_pos: Sequence[Sequence[int]], points_lab: Sequence[int], h: int, w: int, theta: float) -> np.ndarray:
    if points_pos is None or len(points_pos) == 0:
        return np.ones((h, w), dtype=np.float32)
    total = np.zeros((h, w), dtype=np.float32)
    seen = set()
    xg, yg = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    inv_two_theta = 0.5 / float(theta)
    for n, (X, Y) in enumerate(points_pos):
        key = (int(X), int(Y))
        if key in seen:
            continue
        seen.add(key)
        g = np.exp(-(((xg - float(X)) ** 2 + (yg - float(Y)) ** 2) * inv_two_theta))
        peak = float(np.max(g))
        if peak > 0.0:
            g *= 1.0 / peak
        total = total + g * (int(points_lab[n]) * 2 - 1)
    if points_lab is not None and len(points_lab) and int(np.max(points_lab)) == 0:
        total = total * 2 + 1
    return total.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Instance-centric crop (replaces RandomCrop while keeping post-crop augment)
# ---------------------------------------------------------------------------


def _pad_min_hw(raw: np.ndarray, mask: np.ndarray, min_h: int, min_w: int) -> Tuple[np.ndarray, np.ndarray]:
    H, W = raw.shape[:2]
    pad_h = max(0, min_h - H)
    pad_w = max(0, min_w - W)
    if pad_h == 0 and pad_w == 0:
        return raw, mask
    raw = np.pad(raw, ((0, pad_h), (0, pad_w)), mode="constant")
    mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")
    return raw, mask


def _pick_crop_top_left(
    H: int,
    W: int,
    S: int,
    ty0: int,
    ty1: int,
    tx0: int,
    tx1: int,
    union_y0: int,
    union_y1: int,
    union_x0: int,
    union_x1: int,
    rng: np.random.Generator,
    center_jitter_px: int,
) -> Tuple[int, int]:
    """Pick (y0, x0) for an SxS window covering the target bbox; bias context toward union bbox."""
    th, tw = ty1 - ty0, tx1 - tx0
    if th > S or tw > S:
        cy = (ty0 + ty1) // 2 + int(rng.integers(-center_jitter_px, center_jitter_px + 1))
        cx = (tx0 + tx1) // 2 + int(rng.integers(-center_jitter_px, center_jitter_px + 1))
        y0 = int(np.clip(cy - S // 2, 0, H - S))
        x0 = int(np.clip(cx - S // 2, 0, W - S))
        return y0, x0

    ymin = max(0, ty1 - S)
    ymax = min(ty0, H - S)
    xmin = max(0, tx1 - S)
    xmax = min(tx0, W - S)
    if ymin > ymax or xmin > xmax:
        y0 = int(np.clip(ty0 - (S - th) // 2, 0, H - S))
        x0 = int(np.clip(tx0 - (S - tw) // 2, 0, W - S))
        return y0, x0

    uy0 = max(0, min(union_y0, H - 1))
    uy1 = min(H, max(union_y1, 0))
    ux0 = max(0, min(union_x0, W - 1))
    ux1 = min(W, max(union_x1, 0))
    best_score = -1.0
    best: Optional[Tuple[int, int]] = None
    for _ in range(48):
        y0 = int(rng.integers(ymin, ymax + 1))
        x0 = int(rng.integers(xmin, xmax + 1))
        iy0, iy1 = y0, y0 + S
        ix0, ix1 = x0, x0 + S
        inter = max(0, min(iy1, uy1) - max(iy0, uy0)) * max(0, min(ix1, ux1) - max(ix0, ux0))
        score = float(inter) + 1e-3 * float(rng.random())
        if score > best_score:
            best_score = score
            best = (y0, x0)
    assert best is not None
    jy = int(rng.integers(-center_jitter_px, center_jitter_px + 1))
    jx = int(rng.integers(-center_jitter_px, center_jitter_px + 1))
    y0 = int(np.clip(best[0] + jy, ymin, ymax))
    x0 = int(np.clip(best[1] + jx, xmin, xmax))
    return y0, x0


def prepare_2d_instance_centric_pair(
    raw: np.ndarray,
    mask: np.ndarray,
    crop_size: int,
    pad_total: int,
    check_shapes: bool = True,
    *,
    augment: bool = True,
    rng: Optional[np.random.Generator] = None,
    neighbor_halo_px: int = 28,
    center_jitter_px: int = 10,
    touch_dilate: int = 2,
    max_touching_neighbors: int = 8,
) -> Tuple[np.ndarray, np.ndarray, Optional[int]]:
    """Pad like ``augment_2d_pair``, then crop around a random FG instance + touching neighbors.

    Returns ``(raw_crop, mask_crop, target_instance_id)``; the id is the crop-anchor object (for prompt alignment).
    """
    if rng is None:
        rng = np.random.default_rng()

    raw = np.asarray(raw, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.uint16)
    raw, mask = _pad_min_hw(raw, mask, crop_size, crop_size)
    H, W = raw.shape[:2]
    fg = mask[mask != 0]
    if fg.size == 0:
        r, m = _fallback_center_crop(raw, mask, crop_size, pad_total, check_shapes, augment=augment)
        return r, m, None

    fg_ids = np.unique(fg)
    target_id = int(rng.choice(fg_ids))
    tgt = mask == target_id
    ys, xs = np.where(tgt)
    ty0, ty1 = int(ys.min()), int(ys.max()) + 1
    tx0, tx1 = int(xs.min()), int(xs.max()) + 1

    touch = binary_dilation(tgt, iterations=int(max(1, touch_dilate)))
    neighbor_ids = [int(l) for l in np.unique(mask[touch]) if l != 0 and int(l) != target_id]
    rng.shuffle(neighbor_ids)
    neighbor_ids = neighbor_ids[:max_touching_neighbors]

    union = tgt.copy()
    for nid in neighbor_ids:
        union |= mask == nid
    uy, ux = np.where(union)
    union_y0, union_y1 = int(uy.min()), int(uy.max()) + 1
    union_x0, union_x1 = int(ux.min()), int(ux.max()) + 1
    union_y0 = max(0, union_y0 - neighbor_halo_px)
    union_y1 = min(H, union_y1 + neighbor_halo_px)
    union_x0 = max(0, union_x0 - neighbor_halo_px)
    union_x1 = min(W, union_x1 + neighbor_halo_px)

    S = crop_size
    y0, x0 = _pick_crop_top_left(
        H, W, S, ty0, ty1, tx0, tx1, union_y0, union_y1, union_x0, union_x1, rng, center_jitter_px
    )
    raw_c = raw[y0 : y0 + S, x0 : x0 + S].copy()
    mask_c = mask[y0 : y0 + S, x0 : x0 + S].copy()

    kw = {} if check_shapes else {"is_check_shapes": False}
    if augment:
        post = A.Compose(
            [
                A.PadIfNeeded(min_height=pad_total, min_width=pad_total, p=1, border_mode=0),
                A.HorizontalFlip(p=0.3),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.Transpose(p=0.3),
                A.RandomBrightnessContrast(p=0.3),
            ],
            **kw,
        )
    else:
        post = A.Compose(
            [
                A.PadIfNeeded(min_height=pad_total, min_width=pad_total, p=1, border_mode=0),
            ],
            **kw,
        )
    out = post(image=raw_c, mask=mask_c)
    return out["image"], out["mask"], target_id


def _fallback_center_crop(
    raw: np.ndarray,
    mask: np.ndarray,
    crop_size: int,
    pad_total: int,
    check_shapes: bool,
    *,
    augment: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    kw = {} if check_shapes else {"is_check_shapes": False}
    if augment:
        t = A.Compose(
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
    else:
        t = A.Compose(
            [
                A.PadIfNeeded(min_height=crop_size, min_width=crop_size, p=1, border_mode=0),
                A.CenterCrop(width=crop_size, height=crop_size),
                A.PadIfNeeded(min_height=pad_total, min_width=pad_total, p=1, border_mode=0),
            ],
            **kw,
        )
    o = t(image=raw, mask=mask)
    return o["image"], o["mask"]


# ---------------------------------------------------------------------------
# Prompt sampling
# ---------------------------------------------------------------------------


def _interior_candidates(target_bool: np.ndarray, point_thre: float, max_pool: int, rng: np.random.Generator) -> np.ndarray:
    ys, xs = np.where(target_bool)
    if ys.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = target_bool[y0:y1, x0:x1]
    h_c, w_c = crop.shape
    padded = np.pad(crop.astype(np.uint8), 1, mode="constant", constant_values=0)
    dist = distance_transform_edt(padded)[1:-1, 1:-1]
    md = float(dist.max())
    if md > 0:
        d_min = max(1.0, md * float(point_thre))
        cand = (dist >= d_min) & crop
        if not np.any(cand):
            d_min = 1.0
            cand = (dist >= d_min) & crop
        if not np.any(cand):
            cand = crop
    else:
        cand = crop
    yy, xx = np.where(cand)
    if yy.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    pts = np.stack([yy + y0, xx + x0], axis=1).astype(np.int64)
    if pts.shape[0] > max_pool:
        keep = rng.choice(pts.shape[0], size=max_pool, replace=False)
        pts = pts[keep]
    return pts


def _pick_diverse_points(
    cand_yx: np.ndarray,
    n_take: int,
    dist_map_full: np.ndarray,
    rng: np.random.Generator,
) -> List[List[int]]:
    """Greedy farthest-point sampling in (y,x) using distance transform as auxiliary spacing."""
    if cand_yx.shape[0] == 0 or n_take <= 0:
        return []
    first_i = int(rng.integers(0, cand_yx.shape[0]))
    chosen = [first_i]
    cy, cx = cand_yx[first_i]
    min_d2 = (cand_yx[:, 0] - cy) ** 2 + (cand_yx[:, 1] - cx) ** 2
    for _ in range(1, min(n_take, cand_yx.shape[0])):
        weighted = min_d2.astype(np.float64) / (1.0 + dist_map_full[cand_yx[:, 0], cand_yx[:, 1]].astype(np.float64) + 1e-3)
        nxt = int(np.argmax(weighted))
        chosen.append(nxt)
        cy, cx = cand_yx[nxt]
        min_d2 = np.minimum(min_d2, (cand_yx[:, 0] - cy) ** 2 + (cand_yx[:, 1] - cx) ** 2)
    out = []
    for i in chosen:
        y, x = int(cand_yx[i, 0]), int(cand_yx[i, 1])
        out.append([x, y])
    return out


def _neighbor_interior_points(
    labels: np.ndarray,
    neighbor_id: int,
    n: int,
    point_thre: float,
    max_pool: int,
    rng: np.random.Generator,
) -> List[List[int]]:
    m = labels == neighbor_id
    if not np.any(m):
        return []
    cand = _interior_candidates(m, point_thre, max_pool, rng)
    if cand.shape[0] == 0:
        return []
    padded = np.pad(m.astype(np.uint8), 1, mode="constant", constant_values=0)
    dist = distance_transform_edt(padded)[1:-1, 1:-1]
    n_take = min(n, cand.shape[0])
    pick = _pick_diverse_points(cand, n_take, dist, rng)
    return pick


def _largest_cc_seed(mask_bool: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, int]:
    if not np.any(mask_bool):
        return mask_bool, 0
    lab, n = cc_label(mask_bool.astype(np.uint8))
    if n <= 0:
        return mask_bool, 0
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    lid = int(np.argmax(counts))
    return lab == lid, int(counts[lid])


def _min_dist_to_points(y: int, x: int, pos_xy: Sequence[Sequence[int]], neg_xy: Sequence[Sequence[int]]) -> float:
    best = 1e18
    for px, py in list(pos_xy) + list(neg_xy):
        d = (px - x) ** 2 + (py - y) ** 2
        if d < best:
            best = d
    return float(np.sqrt(best + 1e-6))


@dataclass
class InteractiveSegEMConfig:
    # Positive count distribution (length 4: 1,2,3,4+)
    positive_count_probs: Tuple[float, float, float, float] = (0.5, 0.3, 0.15, 0.05)
    max_positive_points: int = 8
    point_thre: float = 0.2
    max_candidate_pool: int = 2048
    theta: float = 30.0
    # Initial negatives
    p_initial_negative: float = 0.35
    max_initial_negatives: int = 2
    # Multi-round
    max_interaction_rounds: int = 3
    max_correction_points_per_round: int = 2
    merge_fp_vs_fn_weight: float = 1.35
    # Simulated prediction
    sim_merge_pixel_prob: float = 0.45
    sim_fn_erosion_max: int = 4
    p_sim_mode_merge: float = 0.55
    # Dedup
    min_new_point_sep_px: float = 4.0
    # Instance crop (only used when crop_mode=instance_centric on dataset)
    neighbor_halo_px: int = 28
    center_jitter_px: int = 10
    touch_dilate: int = 2
    max_touching_neighbors: int = 8


def _sample_positive_count(cfg: InteractiveSegEMConfig, rng: np.random.Generator) -> int:
    p = np.asarray(cfg.positive_count_probs, dtype=np.float64)
    p = p / p.sum()
    u = float(rng.random())
    c = np.cumsum(p)
    k = int(np.searchsorted(c, u, side="right"))
    k = int(np.clip(k, 0, 3))
    if k < 3:
        return k + 1
    return int(rng.integers(4, cfg.max_positive_points + 1))


def sample_initial_prompts(
    labels: np.ndarray,
    target_id: int,
    cfg: InteractiveSegEMConfig,
    rng: np.random.Generator,
) -> Tuple[List[List[int]], List[int], List[List[int]], List[int]]:
    """Returns (pos_xy, pos_lab, neg_xy, neg_lab) with labs 1/0 for gaussian map."""
    h, w = labels.shape
    tgt = labels == target_id
    padded = np.pad(tgt.astype(np.uint8), 1, mode="constant", constant_values=0)
    dist_tgt = distance_transform_edt(padded)[1:-1, 1:-1]

    n_pos = _sample_positive_count(cfg, rng)
    cand = _interior_candidates(tgt, cfg.point_thre, cfg.max_candidate_pool, rng)
    pos_xy = _pick_diverse_points(cand, min(n_pos, cand.shape[0]), dist_tgt, rng)
    pos_lab = [1] * len(pos_xy)

    neg_xy: List[List[int]] = []
    neg_lab: List[int] = []
    if float(rng.random()) < cfg.p_initial_negative and cfg.max_initial_negatives > 0:
        touch = binary_dilation(tgt, iterations=max(1, cfg.touch_dilate))
        nids = [int(l) for l in np.unique(labels[touch]) if l != 0 and int(l) != target_id]
        rng.shuffle(nids)
        for nid in nids[: cfg.max_initial_negatives]:
            pts = _neighbor_interior_points(
                labels, nid, 1, cfg.point_thre, cfg.max_candidate_pool, rng
            )
            for p in pts:
                if _min_dist_to_points(p[1], p[0], pos_xy, neg_xy) >= cfg.min_new_point_sep_px:
                    neg_xy.append(p)
                    neg_lab.append(0)
                    break
    return pos_xy, pos_lab, neg_xy, neg_lab


def simulate_prediction_mask(
    target_mask: np.ndarray,
    labels: np.ndarray,
    target_id: int,
    cfg: InteractiveSegEMConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Binary mask approximating a plausible bad prediction (merge / miss)."""
    gt = target_mask.astype(bool)
    pred = gt.copy()
    touch = binary_dilation(gt, iterations=max(1, cfg.touch_dilate))
    neighbor_ids = [int(l) for l in np.unique(labels[touch]) if l != 0 and int(l) != target_id]

    if float(rng.random()) < cfg.p_sim_mode_merge and neighbor_ids:
        nid = int(rng.choice(neighbor_ids))
        nm = labels == nid
        if np.any(nm):
            sel = rng.random(nm.shape) < cfg.sim_merge_pixel_prob
            pred = np.logical_or(pred, np.logical_and(nm, sel))

    if float(rng.random()) < (1.0 - cfg.p_sim_mode_merge) or not neighbor_ids:
        k = int(rng.integers(1, cfg.sim_fn_erosion_max + 1))
        pred = binary_erosion(pred.astype(np.uint8), iterations=k).astype(bool)

    return pred.astype(np.uint8)


def sample_correction_prompts(
    pred_mask: np.ndarray,
    gt_target: np.ndarray,
    labels: np.ndarray,
    target_id: int,
    existing_pos: List[List[int]],
    existing_neg: List[List[int]],
    cfg: InteractiveSegEMConfig,
    rng: np.random.Generator,
) -> Tuple[List[List[int]], List[int]]:
    """New points (xy + lab) for one correction step; merge-aware negatives."""
    pred = pred_mask.astype(bool)
    gt = gt_target.astype(bool)
    fp = np.logical_and(pred, np.logical_not(gt))
    fn = np.logical_and(gt, np.logical_not(pred))
    fp_area = int(fp.sum())
    fn_area = int(fn.sum())
    budget = int(cfg.max_correction_points_per_round)
    new_xy: List[List[int]] = []
    new_lab: List[int] = []

    merge_bias = fp_area > cfg.merge_fp_vs_fn_weight * max(1, fn_area)
    if merge_bias and fp_area > 0:
        fp_cc, _ = _largest_cc_seed(fp, rng)
        best_nid = -1
        best_ov = 0
        for nid in np.unique(labels):
            if nid == 0 or int(nid) == target_id:
                continue
            nm = labels == int(nid)
            ov = int(np.logical_and(fp_cc, nm).sum())
            if ov > best_ov:
                best_ov = ov
                best_nid = int(nid)
        placed = 0
        if best_nid > 0 and best_ov > 0:
            neigh = labels == best_nid
            interior = np.logical_and(neigh, fp_cc)
            if not np.any(interior):
                interior = np.logical_and(neigh, fp)
            ys, xs = np.where(interior)
            if ys.size > 0:
                pick = int(rng.integers(0, ys.size))
                y, x = int(ys[pick]), int(xs[pick])
                if _min_dist_to_points(y, x, existing_pos + new_xy, existing_neg) >= cfg.min_new_point_sep_px:
                    new_xy.append([x, y])
                    new_lab.append(0)
                    placed += 1
        if placed == 0 and fp_area > 0:
            fp_cc2, _ = _largest_cc_seed(fp, rng)
            ys, xs = np.where(fp_cc2)
            if ys.size > 0:
                pick = int(rng.integers(0, ys.size))
                y, x = int(ys[pick]), int(xs[pick])
                if _min_dist_to_points(y, x, existing_pos + new_xy, existing_neg) >= cfg.min_new_point_sep_px:
                    new_xy.append([x, y])
                    new_lab.append(0)
                    placed += 1
        budget_left = budget - placed
    else:
        budget_left = budget

    if budget_left > 0 and fn_area > 0:
        fn_cc, _ = _largest_cc_seed(fn, rng)
        tgt_interior = np.logical_and(fn_cc, gt)
        cand = _interior_candidates(tgt_interior, cfg.point_thre, cfg.max_candidate_pool, rng)
        if cand.shape[0] == 0:
            ys, xs = np.where(fn_cc)
            if ys.size > 0:
                pick = int(rng.integers(0, ys.size))
                cand = np.array([[ys[pick], xs[pick]]], dtype=np.int64)
        padded = np.pad(gt.astype(np.uint8), 1, mode="constant", constant_values=0)
        dist_gt = distance_transform_edt(padded)[1:-1, 1:-1]
        for p in _pick_diverse_points(cand, min(budget_left, cand.shape[0]), dist_gt, rng):
            y, x = p[1], p[0]
            if _min_dist_to_points(y, x, existing_pos + new_xy, existing_neg) >= cfg.min_new_point_sep_px:
                new_xy.append([x, y])
                new_lab.append(1)
                budget_left -= 1
                if budget_left <= 0:
                    break

    if budget_left > 0 and fp_area > 0 and not merge_bias:
        fp_cc, _ = _largest_cc_seed(fp, rng)
        best_nid = -1
        best_ov = 0
        for nid in np.unique(labels):
            if nid == 0 or int(nid) == target_id:
                continue
            nm = labels == int(nid)
            ov = int(np.logical_and(fp_cc, nm).sum())
            if ov > best_ov:
                best_ov = ov
                best_nid = int(nid)
        if best_nid > 0 and best_ov > 2:
            ys, xs = np.where(np.logical_and(fp_cc, labels == best_nid))
            if ys.size > 0:
                pick = int(rng.integers(0, ys.size))
                y, x = int(ys[pick]), int(xs[pick])
                if _min_dist_to_points(y, x, existing_pos + new_xy, existing_neg) >= cfg.min_new_point_sep_px:
                    new_xy.append([x, y])
                    new_lab.append(0)

    return new_xy, new_lab


def build_interactive_prompt_episode(
    labels: np.ndarray,
    target_id: int,
    cfg: InteractiveSegEMConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Build a random-depth prompt episode (teacher-forced corrections from a simulated prediction).

    Returns:
        point_map: (H, W) float32, signed gaussian map
        target_mask: (H, W) uint8
        meta: includes interaction_history, difficulty, crop unused here
    """
    labels = np.asarray(labels)
    h, w = labels.shape
    tgt_mask = (labels == target_id).astype(np.uint8)
    pos_xy, pos_lab, neg_xy, neg_lab = sample_initial_prompts(labels, target_id, cfg, rng)
    history: List[Dict[str, Any]] = [
        {
            "round": 0,
            "positive_points": [tuple(p) for p in pos_xy],
            "negative_points": [tuple(p) for p in neg_xy],
        }
    ]

    pred_sim = simulate_prediction_mask(tgt_mask, labels, target_id, cfg, rng)
    all_pos = list(pos_xy)
    all_neg = list(neg_xy)
    all_plab = list(pos_lab)
    all_nlab = list(neg_lab)

    n_rounds = int(cfg.max_interaction_rounds)
    for r in range(1, n_rounds + 1):
        nx, nl = sample_correction_prompts(
            pred_sim, tgt_mask, labels, target_id, all_pos, all_neg, cfg, rng
        )
        if not nx:
            break
        history.append({"round": r, "new_points": list(zip(nx, nl))})
        for p, l in zip(nx, nl):
            if l == 1:
                all_pos.append(p)
                all_plab.append(1)
            else:
                all_neg.append(p)
                all_nlab.append(0)

    max_r = len(history) - 1
    use_r = int(rng.integers(0, max_r + 1))
    trim_pos = []
    trim_neg = []
    trim_plab = []
    trim_nlab = []
    if use_r == 0:
        trim_pos = list(pos_xy)
        trim_neg = list(neg_xy)
        trim_plab = list(pos_lab)
        trim_nlab = list(neg_lab)
    else:
        for p in pos_xy:
            trim_pos.append(p)
            trim_plab.append(1)
        for p in neg_xy:
            trim_neg.append(p)
            trim_nlab.append(0)
        extra = history[1 : use_r + 1]
        for block in extra:
            for pt, lb in block.get("new_points", []):
                if lb == 1:
                    trim_pos.append(list(pt))
                    trim_plab.append(1)
                else:
                    trim_neg.append(list(pt))
                    trim_nlab.append(0)

    points = trim_pos + trim_neg
    labs = trim_plab + trim_nlab
    point_map = gaussian_point_map(points, labs, h, w, cfg.theta)

    touch = binary_dilation(tgt_mask.astype(bool), iterations=max(1, cfg.touch_dilate))
    n_neighbors = int(len([l for l in np.unique(labels[touch]) if l != 0 and int(l) != target_id]))
    padded = np.pad(tgt_mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    edt = distance_transform_edt(padded)[1:-1, 1:-1]
    boundary_ratio = float((np.logical_and(tgt_mask.astype(bool), edt <= 2.0).sum()) / max(1, int(tgt_mask.sum())))

    meta = {
        "target_instance_id": int(target_id),
        "interaction_history": history,
        "replay_round": int(use_r),
        "n_touching_neighbors": n_neighbors,
        "difficulty": float(min(1.0, 0.15 * n_neighbors + 0.5 * boundary_ratio)),
    }
    return point_map, tgt_mask, meta


def pick_random_target_id(labels: np.ndarray, rng: np.random.Generator) -> Optional[int]:
    fg = labels[labels != 0]
    if fg.size == 0:
        return None
    return int(rng.choice(np.unique(fg)))


def build_seg_interactive_sample(
    labels: np.ndarray,
    cfg: InteractiveSegEMConfig,
    rng: Optional[np.random.Generator] = None,
    *,
    forced_target_id: Optional[int] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    """One training sample: random target + interactive prompts. Returns None if empty."""
    if rng is None:
        rng = np.random.default_rng()
    if forced_target_id is not None and np.any(labels == int(forced_target_id)):
        tid = int(forced_target_id)
    else:
        tid = pick_random_target_id(labels, rng)
    if tid is None:
        return None
    point_map, tgt_mask, meta = build_interactive_prompt_episode(labels, tid, cfg, rng)
    has_fg = bool(np.any(tgt_mask))
    if not has_fg:
        return None
    return point_map, tgt_mask, meta


__all__ = [
    "InteractiveSegEMConfig",
    "gaussian_point_map",
    "prepare_2d_instance_centric_pair",
    "sample_initial_prompts",
    "sample_correction_prompts",
    "simulate_prediction_mask",
    "build_interactive_prompt_episode",
    "build_seg_interactive_sample",
    "pick_random_target_id",
]
