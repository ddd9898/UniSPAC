import argparse
import contextlib
import logging
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

from train_ACRLSD_2d_neo import LEAVE_SPECIES_CHOICES, ModelEma, count_parameters, seed_worker, set_seed
from train_segEM2d_neo import (
    ALL_SOURCE_KEYS,
    SPECIES_TO_SOURCES,
    SegEM2dNeo,
    _all_seg_source_specs,
    _install_persistent_diagnostics,
    _weighted_elementwise_mean,
    _weighted_soft_dice_loss,
    batch_mask_iou,
    concat_source_balanced_weights,
    mask_to_boundary_affinity,
)
from utils.segem2d_interactive_sampling_plus import (
    InteractiveSegEMPlusConfig,
    build_seg_interactive_sample_plus,
    plus_ic_kwargs_from_config,
)


def build_argparser():
    parser = argparse.ArgumentParser(
        description=(
            "Train segEM2d plus with the same frozen ACRLSD-neo teacher/mask head as segEM2d-neo, "
            "but using one-click-first prompt episodes and multi-step weighted supervision."
        )
    )
    parser.add_argument("--leave-species", type=str, default="human", choices=LEAVE_SPECIES_CHOICES)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=40)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--n-val-holdout", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--cosine-eta-min-ratio", type=float, default=0.001)
    parser.add_argument("--early-stop", type=int, default=20)
    parser.add_argument("--mask-width", type=int, default=96)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--save-top-k", type=int, default=3)
    parser.add_argument("--weight-bce", type=float, default=1.0)
    parser.add_argument("--weight-dice", type=float, default=1.0)
    parser.add_argument("--weight-boundary", type=float, default=0.3)
    parser.add_argument(
        "--backbone-checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=1998)
    parser.add_argument("--save-name", type=str, default=None)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision. By default AMP is enabled.")
    parser.add_argument(
        "--no-train-balance-sources",
        dest="train_balance_sources",
        action="store_false",
        help="Turn off per-source balanced training sampling.",
    )
    parser.set_defaults(train_balance_sources=True)
    parser.add_argument(
        "--positive-aux-loss-weight",
        type=float,
        default=0.0,
        help="Per-sample boundary/aux loss weight for positive-target-instance prompt samples.",
    )
    parser.add_argument(
        "--use-teacher-lsd",
        dest="use_teacher_lsd",
        action="store_true",
        help="Use teacher LSD features/probabilities in segEM2d mask-head inputs (default: enabled).",
    )
    parser.add_argument(
        "--no-teacher-lsd",
        dest="use_teacher_lsd",
        action="store_false",
        help="Disable teacher LSD features/probabilities in segEM2d mask-head inputs.",
    )
    parser.set_defaults(use_teacher_lsd=True)
    parser.add_argument(
        "--no-prompt-guidance-prior",
        dest="use_prompt_guidance_prior",
        action="store_false",
        help="Disable prompt-conditioned propagation prior derived from teacher affinity.",
    )
    parser.add_argument(
        "--prompt-guidance-iters",
        type=int,
        default=64,
        help="Iterations for prompt-conditioned teacher-affinity propagation prior.",
    )
    parser.set_defaults(use_prompt_guidance_prior=True)

    parser.add_argument(
        "--plus-max-total-points",
        type=int,
        default=5,
        help="Maximum total prompts per episode, including the initial one-click positive point.",
    )
    parser.add_argument(
        "--plus-step-discount",
        type=float,
        default=0.55,
        help="Discount factor for multi-step anytime loss. Earlier steps receive higher weight.",
    )
    parser.add_argument(
        "--plus-step0-dice-boost",
        type=float,
        default=1.5,
        help="Extra multiplier on Dice loss for the first one-click completion step.",
    )
    parser.add_argument(
        "--plus-point-thre",
        type=float,
        default=0.2,
        help="Interior prompt band threshold for one-click/correction prompt placement.",
    )
    parser.add_argument(
        "--plus-prompt-max-candidate-pool",
        type=int,
        default=2048,
        help="Subsample candidate pixels to this many before selecting a prompt.",
    )
    parser.add_argument(
        "--plus-prompt-theta",
        type=float,
        default=30.0,
        help="Gaussian theta used to rasterize point prompts into point maps.",
    )
    parser.add_argument(
        "--plus-min-new-point-sep",
        type=float,
        default=4.0,
        help="Minimum L2 distance in pixels between a new correction point and existing prompts.",
    )
    parser.add_argument(
        "--plus-one-click-center-bias",
        type=float,
        default=0.75,
        help="Probability of using the deepest interior point for the initial one-click prompt.",
    )
    parser.add_argument(
        "--plus-merge-fp-vs-fn-weight",
        type=float,
        default=1.35,
        help="If FP area exceeds this multiple of FN area, prefer a negative merge-correction prompt.",
    )
    parser.add_argument(
        "--plus-sim-merge-pixel-prob",
        type=float,
        default=0.45,
        help="Per-neighbor-pixel Bernoulli probability when simulating an initial merge-biased pseudo prediction.",
    )
    parser.add_argument(
        "--plus-sim-fn-erosion-max",
        type=int,
        default=4,
        help="Maximum erosion iterations when simulating an initial miss-biased pseudo prediction.",
    )
    parser.add_argument(
        "--plus-p-sim-mode-merge",
        type=float,
        default=0.55,
        help="Probability of starting the offline correction rollout from a merge-biased pseudo prediction.",
    )
    parser.add_argument(
        "--plus-ic-neighbor-halo",
        type=int,
        default=28,
        help="BBox expansion (px) around target union neighbors for instance-centric crops.",
    )
    parser.add_argument(
        "--plus-ic-center-jitter",
        type=int,
        default=10,
        help="Max absolute jitter (px) applied to the instance-centric crop center.",
    )
    return parser


def plus_config_from_args(args: argparse.Namespace) -> InteractiveSegEMPlusConfig:
    return InteractiveSegEMPlusConfig(
        max_total_points=int(args.plus_max_total_points),
        point_thre=float(args.plus_point_thre),
        max_candidate_pool=int(args.plus_prompt_max_candidate_pool),
        theta=float(args.plus_prompt_theta),
        min_new_point_sep_px=float(args.plus_min_new_point_sep),
        one_click_center_bias=float(args.plus_one_click_center_bias),
        merge_fp_vs_fn_weight=float(args.plus_merge_fp_vs_fn_weight),
        sim_merge_pixel_prob=float(args.plus_sim_merge_pixel_prob),
        sim_fn_erosion_max=int(args.plus_sim_fn_erosion_max),
        p_sim_mode_merge=float(args.plus_p_sim_mode_merge),
        neighbor_halo_px=int(args.plus_ic_neighbor_halo),
        center_jitter_px=int(args.plus_ic_center_jitter),
    )


def _make_seeded_rng() -> np.random.Generator:
    return np.random.default_rng(int(np.random.randint(0, 2**31 - 1)))


class _PromptEpisodeDatasetPlus(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        *,
        positive_aux_loss_weight: float,
        plus_config: InteractiveSegEMPlusConfig,
        prompt_tries: int = 2,
        crop_resample_tries: int = 48,
        random_index_fallback_tries: int = 32,
    ):
        self.base_dataset = base_dataset
        self.positive_aux_loss_weight = float(positive_aux_loss_weight)
        self.plus_config = plus_config
        self.prompt_tries = int(prompt_tries)
        self.crop_resample_tries = int(crop_resample_tries)
        self.random_index_fallback_tries = int(random_index_fallback_tries)

    def __len__(self):
        return len(self.base_dataset)

    @staticmethod
    def _unpack_base_item(item):
        n_items = len(item)
        if n_items == 5:
            raw, labels, _point_map, _mask, affinity = item
            return raw, labels, affinity, None
        if n_items == 6:
            raw, labels, _point_map, _mask, affinity, last = item
            if isinstance(last, dict):
                return raw, labels, affinity, last
            return raw, labels, affinity, None
        if n_items == 7:
            raw, labels, _point_map, _mask, affinity, _lsd, last = item
            meta = last if isinstance(last, dict) else None
            return raw, labels, affinity, meta
        raise ValueError("Unexpected EM2D __getitem__ length: %d" % (n_items,))

    def _try_episode_pack(self, raw, labels, affinity, crop_meta=None):
        if labels is None or not np.any(labels):
            return None
        forced = None
        if crop_meta and crop_meta.get("centric_target_id") is not None:
            forced = int(crop_meta["centric_target_id"])
        for _ in range(self.prompt_tries):
            rng = _make_seeded_rng()
            out = build_seg_interactive_sample_plus(
                labels,
                self.plus_config,
                rng=rng,
                forced_target_id=forced,
            )
            if out is None:
                continue
            point_maps, mask, meta = out
            max_steps = int(self.plus_config.max_total_points)
            padded_maps = np.zeros((max_steps, point_maps.shape[1], point_maps.shape[2]), dtype=np.float32)
            valid_steps = np.zeros((max_steps,), dtype=np.float32)
            n_steps = min(max_steps, int(point_maps.shape[0]))
            padded_maps[:n_steps] = point_maps[:n_steps]
            valid_steps[:n_steps] = 1.0
            return (
                raw,
                labels,
                padded_maps,
                mask,
                affinity,
                np.float32(1.0),
                np.float32(self.positive_aux_loss_weight),
                valid_steps,
                meta,
            )
        return None

    def __getitem__(self, idx: int):
        n_items = len(self.base_dataset)
        for _ in range(self.crop_resample_tries):
            raw, labels, affinity, crop_meta = self._unpack_base_item(self.base_dataset[idx])
            pack = self._try_episode_pack(raw, labels, affinity, crop_meta)
            if pack is not None:
                return pack
        for _ in range(self.random_index_fallback_tries):
            j = int(np.random.randint(0, n_items))
            raw, labels, affinity, crop_meta = self._unpack_base_item(self.base_dataset[j])
            pack = self._try_episode_pack(raw, labels, affinity, crop_meta)
            if pack is not None:
                return pack
        raise RuntimeError(
            "segEM2d plus: could not sample a prompt episode (idx=%r, len=%d). "
            "Try more data or increase crop_resample_tries / random_index_fallback_tries."
            % (idx, n_items)
        )


def collate_fn_seg_prompt_plus_train(batch):
    raw = np.array([item[0] for item in batch]).astype(np.float32)
    labels = np.array([item[1] for item in batch]).astype(np.int32)
    point_maps = np.array([item[2] for item in batch]).astype(np.float32)
    mask = np.array([item[3] for item in batch]).astype(np.uint8)
    affinity = np.array([item[4] for item in batch]).astype(np.float32)
    mask_loss_weight = np.array([item[5] for item in batch]).astype(np.float32)
    aux_loss_weight = np.array([item[6] for item in batch]).astype(np.float32)
    step_valid = np.array([item[7] for item in batch]).astype(np.float32)
    return raw, labels, point_maps, mask, affinity, mask_loss_weight, aux_loss_weight, step_valid


def _concat_seg_pool_for_keys_plus(
    allowed_keys: frozenset,
    crop_size: int,
    *,
    split: str,
    n_val: int,
    augment,
    tag: str,
    positive_aux_loss_weight: float,
    eval_holdout: bool,
    plus_config: InteractiveSegEMPlusConfig,
    segem_ic_kwargs: dict,
):
    parts = []
    for name, factory in _all_seg_source_specs(
        crop_size,
        split=split,
        n_val=n_val,
        augment=augment,
        eval_holdout=eval_holdout,
    ):
        if name not in allowed_keys:
            continue
        try:
            ds = factory()
        except Exception as exc:
            logging.warning("Skipping dataset %s (%s): %s", name, tag, exc)
            continue
        if len(ds) == 0:
            logging.warning("Skipping empty dataset %s (%s)", name, tag)
            continue
        ds.crop_mode = "instance_centric"
        ds.extra_item_meta = True
        ds.segem_crop_kwargs.update(dict(segem_ic_kwargs))
        ds = _PromptEpisodeDatasetPlus(
            ds,
            positive_aux_loss_weight=positive_aux_loss_weight,
            plus_config=plus_config,
        )
        parts.append(ds)
        logging.info("Loaded %s (%s): %d samples", name, tag, len(ds))
    if not parts:
        raise RuntimeError(
            "No datasets could be loaded for keys {} ({}); check paths under ./data/".format(
                sorted(allowed_keys), tag
            )
        )
    return ConcatDataset(parts)


def build_train_val_pool_leave_one_species_plus(
    leave_species: str,
    crop_size: int,
    *,
    split: str,
    n_val_holdout: int = 16,
    augment=None,
    positive_aux_loss_weight: float = 0.0,
    plus_config: InteractiveSegEMPlusConfig,
    segem_ic_kwargs: dict,
):
    held = SPECIES_TO_SOURCES[leave_species]
    allowed = frozenset(ALL_SOURCE_KEYS - held)
    return _concat_seg_pool_for_keys_plus(
        allowed,
        crop_size,
        split=split,
        n_val=n_val_holdout,
        augment=augment,
        tag="train_val leave_out={}".format(leave_species),
        positive_aux_loss_weight=positive_aux_loss_weight,
        eval_holdout=False,
        plus_config=plus_config,
        segem_ic_kwargs=segem_ic_kwargs,
    )


def build_test_pool_leave_one_species_plus(
    leave_species: str,
    crop_size: int,
    *,
    positive_aux_loss_weight: float = 0.0,
    plus_config: InteractiveSegEMPlusConfig,
    segem_ic_kwargs: dict,
):
    allowed = SPECIES_TO_SOURCES[leave_species]
    return _concat_seg_pool_for_keys_plus(
        allowed,
        crop_size,
        split="train",
        n_val=0,
        augment=False,
        tag="test holdout species={}".format(leave_species),
        positive_aux_loss_weight=positive_aux_loss_weight,
        eval_holdout=True,
        plus_config=plus_config,
        segem_ic_kwargs=segem_ic_kwargs,
    )


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_prob, target):
        batch_size = target.size(0)
        pred_flat = pred_prob.reshape(batch_size, -1)
        target_flat = target.reshape(batch_size, -1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        score = (2.0 * intersection + self.smooth) / (
            pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth
        )
        return 1.0 - score.mean()


def _loss_from_prediction(
    pred_prob: torch.Tensor,
    pred_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    gt_affinity: torch.Tensor,
    mask_loss_weight: torch.Tensor,
    aux_loss_weight: torch.Tensor,
    *,
    weight_bce: float,
    weight_dice: float,
    weight_boundary: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    loss_bce = _weighted_elementwise_mean(
        F.binary_cross_entropy_with_logits(pred_logits, gt_mask, reduction="none"),
        mask_loss_weight,
    )
    loss_dice = _weighted_soft_dice_loss(pred_prob, gt_mask, mask_loss_weight)
    pred_boundary = mask_to_boundary_affinity(pred_prob)
    loss_boundary = _weighted_elementwise_mean(
        F.smooth_l1_loss(pred_boundary, gt_affinity, reduction="none"),
        aux_loss_weight,
    )
    total = weight_bce * loss_bce + weight_dice * loss_dice + weight_boundary * loss_boundary
    return total, loss_bce, loss_dice, loss_boundary


def prompt_model_step_plus(
    model,
    bce_loss_fn,
    dice_loss_fn,
    optimizer,
    raw,
    point_maps,
    gt_mask,
    gt_affinity,
    mask_loss_weight,
    aux_loss_weight,
    step_valid,
    device,
    *,
    weight_bce: float,
    weight_dice: float,
    weight_boundary: float,
    step_discount: float,
    step0_dice_boost: float,
    train_step=True,
    scheduler=None,
    scaler=None,
    amp_enabled=False,
    grad_clip_norm=None,
):
    del bce_loss_fn, dice_loss_fn
    if train_step:
        optimizer.zero_grad(set_to_none=True)

    if gt_mask.ndim == 3:
        gt_mask = gt_mask.unsqueeze(1)
    if raw.ndim == 3:
        raw = raw.unsqueeze(1)
    mask_loss_weight = mask_loss_weight.reshape(-1)
    aux_loss_weight = aux_loss_weight.reshape(-1)
    step_valid = step_valid.to(dtype=torch.float32)

    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
        teacher = model._teacher_forward(raw)
        step_probs = []
        step_logits = []
        discounted_total = raw.new_zeros(())
        norm = 0.0
        step0_loss = None
        step0_bce = None
        step0_dice = None
        step0_boundary = None

        n_steps = int(point_maps.shape[1])
        for step_idx in range(n_steps):
            valid_step = step_valid[:, step_idx]
            if float(valid_step.max().detach().item()) <= 0.0:
                break
            x_prompt = point_maps[:, step_idx]
            if x_prompt.ndim == 3:
                x_prompt = x_prompt.unsqueeze(1)
            prompt_logits = model.mask_head(model._build_mask_input(raw, x_prompt, teacher))
            prompt_prob = torch.sigmoid(prompt_logits)
            step_mask_weight = mask_loss_weight * valid_step
            step_aux_weight = aux_loss_weight * valid_step
            dice_weight_this_step = weight_dice * (float(step0_dice_boost) if step_idx == 0 else 1.0)
            step_loss, loss_bce, loss_dice, loss_boundary = _loss_from_prediction(
                prompt_prob,
                prompt_logits,
                gt_mask,
                gt_affinity,
                step_mask_weight,
                step_aux_weight,
                weight_bce=weight_bce,
                weight_dice=dice_weight_this_step,
                weight_boundary=weight_boundary,
            )
            discount = float(step_discount) ** step_idx
            discounted_total = discounted_total + step_loss * discount
            norm += discount
            step_probs.append(prompt_prob)
            step_logits.append(prompt_logits)
            if step_idx == 0:
                step0_loss = step_loss
                step0_bce = loss_bce
                step0_dice = loss_dice
                step0_boundary = loss_boundary

        if not step_probs:
            raise RuntimeError("segEM2d plus: no valid prompt steps in batch.")

        loss_value = discounted_total / max(norm, 1e-6)
        stacked_probs = torch.stack(step_probs, dim=1)
        batch_size = stacked_probs.shape[0]
        h_size, w_size = stacked_probs.shape[-2], stacked_probs.shape[-1]
        valid_counts = step_valid.sum(dim=1).clamp_min(1).long()
        gather_idx = (valid_counts - 1).view(batch_size, 1, 1, 1, 1).expand(-1, 1, 1, h_size, w_size)
        final_prob = stacked_probs.gather(1, gather_idx).squeeze(1)
        final_logits = torch.logit(final_prob.clamp(1e-4, 1.0 - 1e-4))
        final_loss, final_bce, final_dice, final_boundary = _loss_from_prediction(
            final_prob,
            final_logits,
            gt_mask,
            gt_affinity,
            mask_loss_weight,
            aux_loss_weight,
            weight_bce=weight_bce,
            weight_dice=weight_dice,
            weight_boundary=weight_boundary,
        )

    if train_step:
        if scaler is not None and amp_enabled:
            scaler.scale(loss_value).backward()
            scaler.unscale_(optimizer)
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    grad_clip_norm,
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_value.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    grad_clip_norm,
                )
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

    return loss_value, {
        "prompt_loss": loss_value.detach(),
        "loss_step0": step0_loss.detach(),
        "loss_step_last": final_loss.detach(),
        "loss_bce_step0": step0_bce.detach(),
        "loss_dice_step0": step0_dice.detach(),
        "loss_boundary_step0": step0_boundary.detach(),
        "loss_bce_step_last": final_bce.detach(),
        "loss_dice_step_last": final_dice.detach(),
        "loss_boundary_step_last": final_boundary.detach(),
        "pred_mask_step0": step_probs[0].detach(),
        "pred_mask_last": final_prob.detach(),
        "teacher_affinity": teacher["affinity_prob"].detach(),
        "mean_steps": valid_counts.to(dtype=torch.float32).mean().detach(),
    }


if __name__ == "__main__":
    args = build_argparser().parse_args()
    leave_species = args.leave_species
    backbone_checkpoint = args.backbone_checkpoint or (
        "./output/checkpoints/ACRLSD_2D_leaveout_{}_holdoutVal{}_neo_Best_in_val.model".format(
            leave_species, args.n_val_holdout
        )
    )
    save_name = args.save_name or (
        "segEM2d_plus_leaveout_{}_holdoutVal{}_neo_wb{}_wd{}_wbd{}_mtp{}_sd{}".format(
            leave_species,
            args.n_val_holdout,
            args.weight_bce,
            args.weight_dice,
            args.weight_boundary,
            args.plus_max_total_points,
            args.plus_step_discount,
        )
    )
    _install_persistent_diagnostics(save_name)

    set_seed(args.seed)
    plus_cfg = plus_config_from_args(args)
    segem_ic_extra = plus_ic_kwargs_from_config(plus_cfg)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    amp_enabled = (device.type == "cuda") and (not args.no_amp)

    model = SegEM2dNeo(
        device=device,
        backbone_checkpoint=backbone_checkpoint,
        mask_width=args.mask_width,
        use_teacher_lsd=args.use_teacher_lsd,
        use_prompt_guidance_prior=args.use_prompt_guidance_prior,
        prompt_guidance_iters=args.prompt_guidance_iters,
    ).to(device)
    ema = ModelEma(model.mask_head, decay=args.ema_decay)

    train_dataset = build_train_val_pool_leave_one_species_plus(
        leave_species,
        crop_size=args.crop_size,
        split="train",
        n_val_holdout=args.n_val_holdout,
        augment=True,
        positive_aux_loss_weight=args.positive_aux_loss_weight,
        plus_config=plus_cfg,
        segem_ic_kwargs=segem_ic_extra,
    )
    val_dataset = build_train_val_pool_leave_one_species_plus(
        leave_species,
        crop_size=args.crop_size,
        split="val",
        n_val_holdout=args.n_val_holdout,
        augment=False,
        positive_aux_loss_weight=args.positive_aux_loss_weight,
        plus_config=plus_cfg,
        segem_ic_kwargs=segem_ic_extra,
    )
    test_dataset = build_test_pool_leave_one_species_plus(
        leave_species,
        args.crop_size,
        positive_aux_loss_weight=args.positive_aux_loss_weight,
        plus_config=plus_cfg,
        segem_ic_kwargs=segem_ic_extra,
    )
    train_gen = torch.Generator().manual_seed(args.seed + 7)
    val_gen = torch.Generator().manual_seed(args.seed + 8)
    test_gen = torch.Generator().manual_seed(args.seed + 9)

    train_sampler = None
    train_shuffle = True
    if args.train_balance_sources and isinstance(train_dataset, ConcatDataset):
        bw = concat_source_balanced_weights(train_dataset)
        train_sampler = WeightedRandomSampler(
            bw,
            num_samples=len(train_dataset),
            replacement=True,
            generator=train_gen,
        )
        train_shuffle = False
        logging.info(
            "Train sampling: per-source balanced (WeightedRandomSampler, num_samples=%d, %d sources)",
            len(train_dataset),
            len(train_dataset.datasets),
        )
    elif args.train_balance_sources and not isinstance(train_dataset, ConcatDataset):
        logging.warning("train_balance_sources=True but train_dataset is not ConcatDataset; using shuffle.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=collate_fn_seg_prompt_plus_train,
        generator=train_gen if train_sampler is None else None,
        worker_init_fn=seed_worker,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn_seg_prompt_plus_train,
        generator=val_gen,
        worker_init_fn=seed_worker,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn_seg_prompt_plus_train,
        generator=test_gen,
        worker_init_fn=seed_worker,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    steps_per_epoch = len(train_loader)
    warmup_steps = max(1, args.warmup_epochs * steps_per_epoch)
    max_train_steps = max(1, args.epochs * steps_per_epoch)

    def _lr_lambda(last_epoch: int):
        if last_epoch < warmup_steps:
            return float(last_epoch + 1) / float(warmup_steps)
        t_step = last_epoch - warmup_steps
        total_after_warmup = max(1, max_train_steps - warmup_steps)
        progress = min(float(t_step) / float(total_after_warmup), 1.0)
        cos_part = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.cosine_eta_min_ratio + (1.0 - args.cosine_eta_min_ratio) * cos_part

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    bce_loss_fn = nn.BCEWithLogitsLoss().to(device)
    dice_loss_fn = SoftDiceLoss().to(device)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logfile = "./output/log/log_{}.txt".format(save_name)
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    fh = logging.FileHandler(logfile, mode="a")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logging.info(
        """Starting plus training:
    leave_species:        %s (test sources: %s)
    training_epochs:      %s
    Train samples:        %d
    Train sampling:       %s
    Val samples:          %d
    Test samples:         %d
    Holdout slices:       %d
    Batch size:           %s
    Val batch size:       %s
    Learning rate:        %s
    Optimizer:            AdamW (weight_decay=%s)
    Loss weights:         BCE=%s | Dice=%s | Boundary=%s
    Step weighting:       discount=%s | step0_dice_boost=%s | max_points=%s
    LR schedule:          linear warmup %s epochs (~%s steps) + cosine to %.4f * base lr
    EMA decay:            %s
    Grad clip norm:       %s
    Save top-k ckpts:     %s
    AMP enabled:          %s
    Backbone checkpoint:  %s
    Mask width:           %s
    Teacher LSD in mask:  %s
    Prompt guide prior:   %s (iters=%s)
    Plus prompt config:   theta=%s | point_thre=%s | max_pool=%s | min_sep=%s | one_click_bias=%s
    Plus correction cfg:  merge_fp_vs_fn=%s | sim_merge_prob=%s | sim_fn_erosion_max=%s
    Positive aux weight:  %s
    Trainable params (M): %.2f
    num_workers:          %s
    Device:               %s
    """,
        leave_species,
        ", ".join(sorted(SPECIES_TO_SOURCES[leave_species])),
        args.epochs,
        len(train_dataset),
        (
            "balanced per source (%d chunks, replacement)" % len(train_dataset.datasets)
            if train_sampler is not None
            else "uniform shuffle (all indices equal weight)"
        ),
        len(val_dataset),
        len(test_dataset),
        args.n_val_holdout,
        args.batch_size,
        args.val_batch_size,
        args.learning_rate,
        args.weight_decay,
        args.weight_bce,
        args.weight_dice,
        args.weight_boundary,
        args.plus_step_discount,
        args.plus_step0_dice_boost,
        args.plus_max_total_points,
        args.warmup_epochs,
        warmup_steps,
        args.cosine_eta_min_ratio,
        args.ema_decay,
        args.grad_clip_norm,
        args.save_top_k,
        amp_enabled,
        backbone_checkpoint,
        args.mask_width,
        model.use_teacher_lsd,
        model.use_prompt_guidance_prior,
        model.prompt_guidance_iters,
        args.plus_prompt_theta,
        args.plus_point_thre,
        args.plus_prompt_max_candidate_pool,
        args.plus_min_new_point_sep,
        args.plus_one_click_center_bias,
        args.plus_merge_fp_vs_fn_weight,
        args.plus_sim_merge_pixel_prob,
        args.plus_sim_fn_erosion_max,
        args.positive_aux_loss_weight,
        count_parameters(model) / 1e6,
        args.num_workers,
        device.type,
    )

    model.train()
    bce_loss_fn.train()
    dice_loss_fn.train()
    epoch = 0
    best_val_loss = float("inf")
    best_epoch = 0
    best_val_metrics = None
    best_test_metrics = None
    no_improve_count = 0
    ckpt_path = "./output/checkpoints/{}_Best_in_val.model".format(save_name)

    def run_eval_loader(loader, use_ema=True):
        model.eval()
        acc_total = []
        acc_step0 = []
        acc_last = []
        acc_iou_step0 = []
        acc_iou_last = []
        acc_mean_steps = []
        weight_scope = ema.apply_to(model.mask_head) if use_ema else contextlib.nullcontext()
        with weight_scope:
            for raw, labels, point_maps, mask, gt_affinity, mask_loss_weight, aux_loss_weight, step_valid in loader:
                del labels
                raw = torch.as_tensor(raw, dtype=torch.float32, device=device)
                point_maps = torch.as_tensor(point_maps, dtype=torch.float32, device=device)
                mask = torch.as_tensor(mask, dtype=torch.float32, device=device)
                gt_affinity = torch.as_tensor(gt_affinity, dtype=torch.float32, device=device)
                mask_loss_weight = torch.as_tensor(mask_loss_weight, dtype=torch.float32, device=device)
                aux_loss_weight = torch.as_tensor(aux_loss_weight, dtype=torch.float32, device=device)
                step_valid = torch.as_tensor(step_valid, dtype=torch.float32, device=device)

                with torch.no_grad():
                    loss_value, stats = prompt_model_step_plus(
                        model,
                        bce_loss_fn,
                        dice_loss_fn,
                        optimizer,
                        raw,
                        point_maps,
                        mask,
                        gt_affinity,
                        mask_loss_weight,
                        aux_loss_weight,
                        step_valid,
                        device,
                        weight_bce=args.weight_bce,
                        weight_dice=args.weight_dice,
                        weight_boundary=args.weight_boundary,
                        step_discount=args.plus_step_discount,
                        step0_dice_boost=args.plus_step0_dice_boost,
                        train_step=False,
                        scheduler=None,
                        scaler=None,
                        amp_enabled=amp_enabled,
                        grad_clip_norm=None,
                    )
                acc_total.append(float(loss_value.detach().cpu().item()))
                acc_step0.append(float(stats["loss_step0"].cpu().item()))
                acc_last.append(float(stats["loss_step_last"].cpu().item()))
                acc_iou_step0.append(float(batch_mask_iou(stats["pred_mask_step0"], mask).mean().cpu().item()))
                acc_iou_last.append(float(batch_mask_iou(stats["pred_mask_last"], mask).mean().cpu().item()))
                acc_mean_steps.append(float(stats["mean_steps"].cpu().item()))

        return {
            "loss": float(np.mean(acc_total)) if acc_total else float("nan"),
            "loss_step0": float(np.mean(acc_step0)) if acc_step0 else float("nan"),
            "loss_step_last": float(np.mean(acc_last)) if acc_last else float("nan"),
            "mean_iou_step0": float(np.mean(acc_iou_step0)) if acc_iou_step0 else float("nan"),
            "mean_iou_step_last": float(np.mean(acc_iou_last)) if acc_iou_last else float("nan"),
            "mean_steps": float(np.mean(acc_mean_steps)) if acc_mean_steps else float("nan"),
        }

    with tqdm(total=args.epochs) as pbar:
        while epoch < args.epochs:
            model.train()
            train_total = []
            train_step0 = []
            train_last = []
            train_iou_step0 = []
            train_iou_last = []
            train_mean_steps = []

            for raw, labels, point_maps, mask, gt_affinity, mask_loss_weight, aux_loss_weight, step_valid in train_loader:
                del labels
                raw = torch.as_tensor(raw, dtype=torch.float32, device=device)
                point_maps = torch.as_tensor(point_maps, dtype=torch.float32, device=device)
                mask = torch.as_tensor(mask, dtype=torch.float32, device=device)
                gt_affinity = torch.as_tensor(gt_affinity, dtype=torch.float32, device=device)
                mask_loss_weight = torch.as_tensor(mask_loss_weight, dtype=torch.float32, device=device)
                aux_loss_weight = torch.as_tensor(aux_loss_weight, dtype=torch.float32, device=device)
                step_valid = torch.as_tensor(step_valid, dtype=torch.float32, device=device)

                loss_value, stats = prompt_model_step_plus(
                    model,
                    bce_loss_fn,
                    dice_loss_fn,
                    optimizer,
                    raw,
                    point_maps,
                    mask,
                    gt_affinity,
                    mask_loss_weight,
                    aux_loss_weight,
                    step_valid,
                    device,
                    weight_bce=args.weight_bce,
                    weight_dice=args.weight_dice,
                    weight_boundary=args.weight_boundary,
                    step_discount=args.plus_step_discount,
                    step0_dice_boost=args.plus_step0_dice_boost,
                    train_step=True,
                    scheduler=scheduler,
                    scaler=scaler,
                    amp_enabled=amp_enabled,
                    grad_clip_norm=args.grad_clip_norm,
                )
                ema.update(model.mask_head)
                train_total.append(float(loss_value.detach().cpu().item()))
                train_step0.append(float(stats["loss_step0"].cpu().item()))
                train_last.append(float(stats["loss_step_last"].cpu().item()))
                train_iou_step0.append(float(batch_mask_iou(stats["pred_mask_step0"], mask).mean().cpu().item()))
                train_iou_last.append(float(batch_mask_iou(stats["pred_mask_last"], mask).mean().cpu().item()))
                train_mean_steps.append(float(stats["mean_steps"].cpu().item()))

            epoch += 1
            pbar.update(1)

            train_metrics = {
                "loss": float(np.mean(train_total)) if train_total else float("nan"),
                "loss_step0": float(np.mean(train_step0)) if train_step0 else float("nan"),
                "loss_step_last": float(np.mean(train_last)) if train_last else float("nan"),
                "mean_iou_step0": float(np.mean(train_iou_step0)) if train_iou_step0 else float("nan"),
                "mean_iou_step_last": float(np.mean(train_iou_last)) if train_iou_last else float("nan"),
                "mean_steps": float(np.mean(train_mean_steps)) if train_mean_steps else float("nan"),
            }
            val_metrics = run_eval_loader(val_loader, use_ema=True)
            test_metrics = run_eval_loader(test_loader, use_ema=True)
            current_lr = optimizer.param_groups[0]["lr"]

            improved = val_metrics["loss"] < best_val_loss
            if improved:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                best_val_metrics = dict(val_metrics)
                best_test_metrics = dict(test_metrics)
                os.makedirs("./output/checkpoints", exist_ok=True)
                ckpt_state = {
                    "epoch": epoch,
                    "mask_head_state_dict": model.mask_head.state_dict(),
                    "ema_state_dict": ema.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "learning_rate": current_lr,
                    "backbone_checkpoint": backbone_checkpoint,
                    "backbone_config": model.backbone_config,
                    "plus_config": vars(args),
                    "config": vars(args),
                }
                torch.save(ckpt_state, ckpt_path)
                no_improve_count = 0
            else:
                no_improve_count += 1

            logging.info(
                "Epoch %d: train(total=%.6f step0=%.6f last=%.6f iou0=%.6f iou_last=%.6f steps=%.3f) | "
                "val(ema,total=%.6f step0=%.6f last=%.6f iou0=%.6f iou_last=%.6f steps=%.3f) | "
                "test(ema,total=%.6f step0=%.6f last=%.6f iou0=%.6f iou_last=%.6f steps=%.3f, leave_species=%s) | "
                "best_val=%.6f@%d | lr = %.8f%s",
                epoch,
                train_metrics["loss"],
                train_metrics["loss_step0"],
                train_metrics["loss_step_last"],
                train_metrics["mean_iou_step0"],
                train_metrics["mean_iou_step_last"],
                train_metrics["mean_steps"],
                val_metrics["loss"],
                val_metrics["loss_step0"],
                val_metrics["loss_step_last"],
                val_metrics["mean_iou_step0"],
                val_metrics["mean_iou_step_last"],
                val_metrics["mean_steps"],
                test_metrics["loss"],
                test_metrics["loss_step0"],
                test_metrics["loss_step_last"],
                test_metrics["mean_iou_step0"],
                test_metrics["mean_iou_step_last"],
                test_metrics["mean_steps"],
                leave_species,
                best_val_loss,
                best_epoch,
                current_lr,
                " | saved {}".format(ckpt_path) if improved else "",
            )

            if no_improve_count >= args.early_stop:
                logging.info("Early stop!")
                break

    logging.info(
        "Best checkpoint: val(total=%.6f step0=%.6f last=%.6f iou0=%.6f iou_last=%.6f) @ epoch %d | "
        "test(total=%.6f step0=%.6f last=%.6f iou0=%.6f iou_last=%.6f) | checkpoint=%s",
        best_val_loss,
        float(best_val_metrics["loss_step0"]) if best_val_metrics else float("nan"),
        float(best_val_metrics["loss_step_last"]) if best_val_metrics else float("nan"),
        float(best_val_metrics["mean_iou_step0"]) if best_val_metrics else float("nan"),
        float(best_val_metrics["mean_iou_step_last"]) if best_val_metrics else float("nan"),
        best_epoch,
        float(best_test_metrics["loss"]) if best_test_metrics else float("nan"),
        float(best_test_metrics["loss_step0"]) if best_test_metrics else float("nan"),
        float(best_test_metrics["loss_step_last"]) if best_test_metrics else float("nan"),
        float(best_test_metrics["mean_iou_step0"]) if best_test_metrics else float("nan"),
        float(best_test_metrics["mean_iou_step_last"]) if best_test_metrics else float("nan"),
        ckpt_path,
    )
