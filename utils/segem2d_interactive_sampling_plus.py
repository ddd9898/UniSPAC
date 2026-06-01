from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, label as cc_label


def gaussian_point_map(
    points_pos: Sequence[Sequence[int]],
    points_lab: Sequence[int],
    h: int,
    w: int,
    theta: float,
) -> np.ndarray:
    if points_pos is None or len(points_pos) == 0:
        return np.ones((h, w), dtype=np.float32)
    total = np.zeros((h, w), dtype=np.float32)
    seen = set()
    xg, yg = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    inv_two_theta = 0.5 / float(theta)
    for n, (x_pos, y_pos) in enumerate(points_pos):
        key = (int(x_pos), int(y_pos))
        if key in seen:
            continue
        seen.add(key)
        gauss = np.exp(-(((xg - float(x_pos)) ** 2 + (yg - float(y_pos)) ** 2) * inv_two_theta))
        peak = float(np.max(gauss))
        if peak > 0.0:
            gauss *= 1.0 / peak
        total = total + gauss * (int(points_lab[n]) * 2 - 1)
    if points_lab is not None and len(points_lab) and int(np.max(points_lab)) == 0:
        total = total * 2 + 1
    return total.astype(np.float32, copy=False)


def _interior_candidates(
    target_bool: np.ndarray,
    point_thre: float,
    max_pool: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(target_bool)
    if ys.size == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros_like(target_bool, dtype=np.float32)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = target_bool[y0:y1, x0:x1]
    padded = np.pad(crop.astype(np.uint8), 1, mode="constant", constant_values=0)
    dist = distance_transform_edt(padded)[1:-1, 1:-1]
    dist_full = np.zeros_like(target_bool, dtype=np.float32)
    dist_full[y0:y1, x0:x1] = dist.astype(np.float32, copy=False)
    max_depth = float(dist.max())
    if max_depth > 0:
        d_min = max(1.0, max_depth * float(point_thre))
        cand = (dist >= d_min) & crop
        if not np.any(cand):
            cand = (dist >= 1.0) & crop
        if not np.any(cand):
            cand = crop
    else:
        cand = crop
    yy, xx = np.where(cand)
    if yy.size == 0:
        return np.zeros((0, 2), dtype=np.int64), dist_full
    pts = np.stack([yy + y0, xx + x0], axis=1).astype(np.int64)
    if pts.shape[0] > max_pool:
        keep = rng.choice(pts.shape[0], size=max_pool, replace=False)
        pts = pts[keep]
    return pts, dist_full


def _largest_cc(mask_bool: np.ndarray) -> np.ndarray:
    if not np.any(mask_bool):
        return mask_bool.astype(bool)
    lab, n_cc = cc_label(mask_bool.astype(np.uint8))
    if n_cc <= 0:
        return mask_bool.astype(bool)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return lab == int(np.argmax(counts))


def _sample_component_point(
    component_mask: np.ndarray,
    cfg: "InteractiveSegEMPlusConfig",
    rng: np.random.Generator,
    existing_pos: Sequence[Sequence[int]],
    existing_neg: Sequence[Sequence[int]],
    *,
    prefer_deep: bool,
) -> Optional[List[int]]:
    if not np.any(component_mask):
        return None
    cand, dist_full = _interior_candidates(component_mask, cfg.point_thre, cfg.max_candidate_pool, rng)
    if cand.shape[0] == 0:
        ys, xs = np.where(component_mask)
        if ys.size == 0:
            return None
        pick = int(rng.integers(0, ys.size))
        y_pos, x_pos = int(ys[pick]), int(xs[pick])
        if _min_dist_to_points(y_pos, x_pos, existing_pos, existing_neg) < cfg.min_new_point_sep_px:
            return None
        return [x_pos, y_pos]

    if prefer_deep and float(rng.random()) < cfg.one_click_center_bias:
        score = dist_full[cand[:, 0], cand[:, 1]]
        pick = int(np.argmax(score))
    else:
        pick = int(rng.integers(0, cand.shape[0]))
    y_pos, x_pos = int(cand[pick, 0]), int(cand[pick, 1])
    if _min_dist_to_points(y_pos, x_pos, existing_pos, existing_neg) < cfg.min_new_point_sep_px:
        order = np.argsort(-dist_full[cand[:, 0], cand[:, 1]])
        for alt in order.tolist():
            y_alt, x_alt = int(cand[alt, 0]), int(cand[alt, 1])
            if _min_dist_to_points(y_alt, x_alt, existing_pos, existing_neg) >= cfg.min_new_point_sep_px:
                return [x_alt, y_alt]
        return None
    return [x_pos, y_pos]


def _min_dist_to_points(
    y_pos: int,
    x_pos: int,
    pos_xy: Sequence[Sequence[int]],
    neg_xy: Sequence[Sequence[int]],
) -> float:
    if not pos_xy and not neg_xy:
        return float("inf")
    best = 1e18
    for px, py in list(pos_xy) + list(neg_xy):
        d2 = (int(px) - int(x_pos)) ** 2 + (int(py) - int(y_pos)) ** 2
        if d2 < best:
            best = d2
    return float(np.sqrt(best + 1e-6))


@dataclass
class InteractiveSegEMPlusConfig:
    max_total_points: int = 5
    point_thre: float = 0.2
    max_candidate_pool: int = 2048
    theta: float = 30.0
    min_new_point_sep_px: float = 4.0
    one_click_center_bias: float = 0.75
    merge_fp_vs_fn_weight: float = 1.35
    sim_merge_pixel_prob: float = 0.45
    sim_fn_erosion_max: int = 4
    p_sim_mode_merge: float = 0.55
    neighbor_halo_px: int = 28
    center_jitter_px: int = 10
    touch_dilate: int = 2
    max_touching_neighbors: int = 8


def plus_ic_kwargs_from_config(cfg: InteractiveSegEMPlusConfig) -> dict:
    return {
        "neighbor_halo_px": cfg.neighbor_halo_px,
        "center_jitter_px": cfg.center_jitter_px,
        "touch_dilate": cfg.touch_dilate,
        "max_touching_neighbors": cfg.max_touching_neighbors,
    }


def simulate_prediction_mask(
    target_mask: np.ndarray,
    labels: np.ndarray,
    target_id: int,
    cfg: InteractiveSegEMPlusConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    gt = target_mask.astype(bool)
    pred = gt.copy()
    touch = binary_dilation(gt, iterations=max(1, cfg.touch_dilate))
    neighbor_ids = [int(v) for v in np.unique(labels[touch]) if int(v) != 0 and int(v) != target_id]

    if float(rng.random()) < cfg.p_sim_mode_merge and neighbor_ids:
        neighbor_id = int(rng.choice(neighbor_ids))
        neighbor_mask = labels == neighbor_id
        if np.any(neighbor_mask):
            select = rng.random(neighbor_mask.shape) < cfg.sim_merge_pixel_prob
            pred = np.logical_or(pred, np.logical_and(neighbor_mask, select))

    if float(rng.random()) < (1.0 - cfg.p_sim_mode_merge) or not neighbor_ids:
        n_erode = int(rng.integers(1, cfg.sim_fn_erosion_max + 1))
        pred = binary_erosion(pred.astype(np.uint8), iterations=n_erode).astype(bool)

    return pred.astype(np.uint8)


def pick_random_target_id(labels: np.ndarray, rng: np.random.Generator) -> Optional[int]:
    fg = labels[labels != 0]
    if fg.size == 0:
        return None
    return int(rng.choice(np.unique(fg)))


def sample_one_click_prompt(
    labels: np.ndarray,
    target_id: int,
    cfg: InteractiveSegEMPlusConfig,
    rng: np.random.Generator,
) -> Optional[List[int]]:
    target = labels == target_id
    return _sample_component_point(target, cfg, rng, [], [], prefer_deep=True)


def _oracle_pick_fp_component(
    fp_mask: np.ndarray,
    labels: np.ndarray,
    target_id: int,
) -> np.ndarray:
    fp_cc = _largest_cc(fp_mask)
    best_neighbor = -1
    best_overlap = 0
    for neighbor_id in np.unique(labels):
        neighbor_id = int(neighbor_id)
        if neighbor_id == 0 or neighbor_id == target_id:
            continue
        overlap = int(np.logical_and(fp_cc, labels == neighbor_id).sum())
        if overlap > best_overlap:
            best_overlap = overlap
            best_neighbor = neighbor_id
    if best_neighbor > 0 and best_overlap > 0:
        overlap_cc = np.logical_and(fp_cc, labels == best_neighbor)
        if np.any(overlap_cc):
            return overlap_cc
    return fp_cc


def pick_next_correction_prompt(
    pred_mask: np.ndarray,
    gt_target: np.ndarray,
    labels: np.ndarray,
    target_id: int,
    existing_pos: Sequence[Sequence[int]],
    existing_neg: Sequence[Sequence[int]],
    cfg: InteractiveSegEMPlusConfig,
    rng: np.random.Generator,
) -> Tuple[Optional[List[int]], Optional[int], np.ndarray, Dict[str, Any]]:
    pred = pred_mask.astype(bool)
    gt = gt_target.astype(bool)
    fp = np.logical_and(pred, np.logical_not(gt))
    fn = np.logical_and(gt, np.logical_not(pred))
    fp_area = int(fp.sum())
    fn_area = int(fn.sum())
    meta: Dict[str, Any] = {
        "fp_area": fp_area,
        "fn_area": fn_area,
        "mode": "stop",
    }
    if fp_area <= 0 and fn_area <= 0:
        return None, None, pred.astype(np.uint8), meta

    choose_negative = fp_area > cfg.merge_fp_vs_fn_weight * max(1, fn_area)
    if choose_negative and fp_area > 0:
        component = _oracle_pick_fp_component(fp, labels, target_id)
        point = _sample_component_point(component, cfg, rng, existing_pos, existing_neg, prefer_deep=False)
        if point is not None:
            updated = np.logical_and(pred, np.logical_not(component))
            meta["mode"] = "negative_fp"
            meta["component_pixels"] = int(component.sum())
            return point, 0, updated.astype(np.uint8), meta

    if fn_area > 0:
        component = _largest_cc(fn)
        point = _sample_component_point(component, cfg, rng, existing_pos, existing_neg, prefer_deep=False)
        if point is not None:
            updated = np.logical_or(pred, component)
            meta["mode"] = "positive_fn"
            meta["component_pixels"] = int(component.sum())
            return point, 1, updated.astype(np.uint8), meta

    if fp_area > 0:
        component = _oracle_pick_fp_component(fp, labels, target_id)
        point = _sample_component_point(component, cfg, rng, existing_pos, existing_neg, prefer_deep=False)
        if point is not None:
            updated = np.logical_and(pred, np.logical_not(component))
            meta["mode"] = "negative_fp_fallback"
            meta["component_pixels"] = int(component.sum())
            return point, 0, updated.astype(np.uint8), meta

    return None, None, pred.astype(np.uint8), meta


def build_interactive_prompt_episode_plus(
    labels: np.ndarray,
    target_id: int,
    cfg: InteractiveSegEMPlusConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    labels = np.asarray(labels)
    h, w = labels.shape
    target_mask = (labels == target_id).astype(np.uint8)
    one_click = sample_one_click_prompt(labels, target_id, cfg, rng)
    if one_click is None:
        raise RuntimeError("Failed to sample initial one-click prompt for target_id=%r" % (target_id,))

    pos_xy: List[List[int]] = [one_click]
    neg_xy: List[List[int]] = []
    point_maps: List[np.ndarray] = [
        gaussian_point_map(pos_xy, [1], h, w, cfg.theta)
    ]
    history: List[Dict[str, Any]] = [
        {"round": 0, "new_points": [(tuple(one_click), 1)], "mode": "one_click_completion"}
    ]

    pred_state = simulate_prediction_mask(target_mask, labels, target_id, cfg, rng)
    pred_states: List[np.ndarray] = [pred_state.astype(np.uint8, copy=False)]

    while len(point_maps) < int(cfg.max_total_points):
        point, label_value, pred_state, step_meta = pick_next_correction_prompt(
            pred_state,
            target_mask,
            labels,
            target_id,
            pos_xy,
            neg_xy,
            cfg,
            rng,
        )
        if point is None or label_value is None:
            break
        if label_value == 1:
            pos_xy.append(point)
        else:
            neg_xy.append(point)
        all_points = pos_xy + neg_xy
        all_labels = [1] * len(pos_xy) + [0] * len(neg_xy)
        point_maps.append(gaussian_point_map(all_points, all_labels, h, w, cfg.theta))
        pred_states.append(pred_state.astype(np.uint8, copy=False))
        history.append(
            {
                "round": len(point_maps) - 1,
                "new_points": [(tuple(point), int(label_value))],
                "mode": step_meta.get("mode", "unknown"),
                "fp_area": int(step_meta.get("fp_area", 0)),
                "fn_area": int(step_meta.get("fn_area", 0)),
            }
        )

    meta = {
        "target_instance_id": int(target_id),
        "interaction_history": history,
        "n_steps": int(len(point_maps)),
        "n_positive_points": int(len(pos_xy)),
        "n_negative_points": int(len(neg_xy)),
        "pseudo_pred_states": pred_states,
    }
    return np.stack(point_maps, axis=0).astype(np.float32, copy=False), target_mask, meta


def build_seg_interactive_sample_plus(
    labels: np.ndarray,
    cfg: InteractiveSegEMPlusConfig,
    rng: Optional[np.random.Generator] = None,
    *,
    forced_target_id: Optional[int] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    if rng is None:
        rng = np.random.default_rng()
    if forced_target_id is not None and np.any(labels == int(forced_target_id)):
        target_id = int(forced_target_id)
    else:
        target_id = pick_random_target_id(labels, rng)
    if target_id is None:
        return None
    point_maps, target_mask, meta = build_interactive_prompt_episode_plus(labels, target_id, cfg, rng)
    if not np.any(target_mask):
        return None
    return point_maps, target_mask, meta


__all__ = [
    "InteractiveSegEMPlusConfig",
    "build_interactive_prompt_episode_plus",
    "build_seg_interactive_sample_plus",
    "gaussian_point_map",
    "pick_next_correction_prompt",
    "pick_random_target_id",
    "plus_ic_kwargs_from_config",
    "sample_one_click_prompt",
    "simulate_prediction_mask",
]
