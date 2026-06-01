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

from train_ACRLSD_2d_neo import (
    ACRLSDneo,
    LEAVE_SPECIES_CHOICES,
    ModelEma,
    ResidualBlock,
    SPECIES_TO_SOURCES,
    SqueezeExcite,
    TaskHead,
    count_parameters,
    seed_worker,
    set_seed,
    _install_persistent_diagnostics,
)
from utils.dataloader import (
    ZEBRA_PATCH_FILES,
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
    Dataset_2D_VNC_Train,
    Dataset_2D_zebrafinch_Train_CL,
)
from utils.segem2d_interactive_sampling import (
    InteractiveSegEMConfig,
    build_seg_interactive_sample,
)


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Train segEM2d neo with a fully frozen ACRLSD-neo teacher and a stronger mask head."
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
        help=(
            "Turn off per-source balanced training sampling. Default: each ConcatDataset source (hemi, cremi, …) "
            "gets equal total draw weight so slice-heavy volumes do not dominate—favors leave-one-species generalization."
        ),
    )
    parser.set_defaults(train_balance_sources=True)
    parser.add_argument(
        "--positive-aux-loss-weight",
        type=float,
        default=0.0,
        help="Per-sample boundary/aux loss weight for positive-target-instance prompt samples.",
    )
    parser.add_argument(
        "--positive-point-thre",
        type=float,
        default=0.2,
        help="Interior prompt band: EDT distance >= max(max_depth*ratio, 1px) inside the target mask.",
    )
    parser.add_argument(
        "--positive-point-grid-stride",
        type=int,
        default=16,
        help="Unused (kept for CLI compatibility). Prompts are drawn randomly from interior pixels.",
    )
    parser.add_argument(
        "--positive-point-count",
        type=int,
        default=8,
        help="Max positive clicks when the interactive sampler draws 4+ initial positives.",
    )
    parser.add_argument(
        "--prompt-max-candidate-pool",
        type=int,
        default=2048,
        help="Subsample interior candidates to this many before drawing points (speed for huge instances).",
    )
    parser.add_argument(
        "--prompt-theta",
        type=float,
        default=30.0,
        help="Gaussian theta used to rasterize positive prompt points into point maps.",
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
        "--segem-positive-count-probs",
        type=str,
        default="0.5,0.3,0.15,0.05",
        help="Four comma-separated probabilities for sampling 1 / 2 / 3 / 4+ initial positive clicks.",
    )
    parser.add_argument(
        "--segem-p-initial-negative",
        type=float,
        default=0.35,
        help="Probability of adding at least one initial negative click on a touching neighbor.",
    )
    parser.add_argument(
        "--segem-max-initial-negatives",
        type=int,
        default=2,
        help="Cap on distinct neighbor instances used for initial negative clicks.",
    )
    parser.add_argument(
        "--segem-max-interaction-rounds",
        type=int,
        default=3,
        help="Maximum synthetic correction rounds after the initial prompt.",
    )
    parser.add_argument(
        "--segem-max-correction-points",
        type=int,
        default=2,
        help="Max new points sampled per correction round.",
    )
    parser.add_argument(
        "--segem-merge-fp-vs-fn-weight",
        type=float,
        default=1.35,
        help="If FP area > this × FN area, prioritize merge-aware negative corrections.",
    )
    parser.add_argument(
        "--segem-sim-merge-pixel-prob",
        type=float,
        default=0.45,
        help="Bernoulli probability per neighbor pixel when simulating a merge error.",
    )
    parser.add_argument(
        "--segem-sim-fn-erosion-max",
        type=int,
        default=4,
        help="Max binary-erosion iterations when simulating false-negative (miss) errors.",
    )
    parser.add_argument(
        "--segem-p-sim-mode-merge",
        type=float,
        default=0.55,
        help="Probability of choosing merge-biased vs miss-biased simulation for synthetic prediction.",
    )
    parser.add_argument(
        "--segem-min-new-point-sep",
        type=float,
        default=4.0,
        help="Minimum L2 distance (pixels) when adding a new click away from existing prompts.",
    )
    parser.add_argument(
        "--segem-ic-neighbor-halo",
        type=int,
        default=28,
        help="BBox expansion (px) around target ∪ neighbors for instance-centric crops.",
    )
    parser.add_argument(
        "--segem-ic-center-jitter",
        type=int,
        default=10,
        help="Max absolute jitter (px) applied to the instance-centric crop center.",
    )
    return parser


def _parse_four_probs(s: str) -> tuple[float, float, float, float]:
    parts = [float(x.strip()) for x in str(s).split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Expected four comma-separated probabilities, got {!r}".format(s))
    if any(p < 0 for p in parts):
        raise argparse.ArgumentTypeError("Probabilities must be non-negative")
    return (parts[0], parts[1], parts[2], parts[3])


def segem_ic_kwargs_from_config(cfg: InteractiveSegEMConfig) -> dict:
    return {
        "neighbor_halo_px": cfg.neighbor_halo_px,
        "center_jitter_px": cfg.center_jitter_px,
        "touch_dilate": cfg.touch_dilate,
        "max_touching_neighbors": cfg.max_touching_neighbors,
    }


def interactive_seg_config_from_args(args: argparse.Namespace) -> InteractiveSegEMConfig:
    probs = _parse_four_probs(args.segem_positive_count_probs)
    return InteractiveSegEMConfig(
        positive_count_probs=probs,
        max_positive_points=int(args.positive_point_count),
        point_thre=float(args.positive_point_thre),
        max_candidate_pool=int(args.prompt_max_candidate_pool),
        theta=float(args.prompt_theta),
        p_initial_negative=float(args.segem_p_initial_negative),
        max_initial_negatives=int(args.segem_max_initial_negatives),
        max_interaction_rounds=int(args.segem_max_interaction_rounds),
        max_correction_points_per_round=int(args.segem_max_correction_points),
        merge_fp_vs_fn_weight=float(args.segem_merge_fp_vs_fn_weight),
        sim_merge_pixel_prob=float(args.segem_sim_merge_pixel_prob),
        sim_fn_erosion_max=int(args.segem_sim_fn_erosion_max),
        p_sim_mode_merge=float(args.segem_p_sim_mode_merge),
        min_new_point_sep_px=float(args.segem_min_new_point_sep),
        neighbor_halo_px=int(args.segem_ic_neighbor_halo),
        center_jitter_px=int(args.segem_ic_center_jitter),
    )


def _strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def load_frozen_acrlsd_neo(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        backbone_config = checkpoint.get("config", {})
        state_dict = checkpoint.get("ema_state_dict") or checkpoint.get("model_state_dict") or checkpoint
    else:
        backbone_config = {}
        state_dict = checkpoint

    state_dict = _strip_module_prefix(state_dict)
    base_width = int(backbone_config.get("base_width", 32))
    bottleneck_channels = int(backbone_config.get("bottleneck_channels", 384))
    fusion_width = int(backbone_config.get("fusion_width", 64))

    model = ACRLSDneo(
        in_channels=1,
        base_width=base_width,
        encoder_widths=(base_width, base_width * 2, base_width * 4, base_width * 8),
        bottleneck_channels=bottleneck_channels,
        fusion_width=fusion_width,
        detach_lsd_for_affinity=True,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, backbone_config


def _segem_mask_in_channels(
    fusion_width: int,
    use_teacher_lsd: bool,
    use_prompt_guidance_prior: bool = False,
) -> int:
    channels = 1 + 1 + 2 + fusion_width * 2
    if use_teacher_lsd:
        channels += 6 + fusion_width
    if use_prompt_guidance_prior:
        channels += 1
    return channels


def _infer_use_teacher_lsd(
    backbone_config: dict,
    mask_head_state: dict | None,
    config: dict | None = None,
    *,
    use_prompt_guidance_prior: bool | None = None,
    default: bool = True,
) -> bool:
    if config is not None and "use_teacher_lsd" in config:
        return bool(config["use_teacher_lsd"])
    if not mask_head_state:
        return default

    fusion_width = int(backbone_config.get("fusion_width", 64))
    guidance_choices = (
        (bool(use_prompt_guidance_prior),)
        if use_prompt_guidance_prior is not None
        else (False, True)
    )
    true_channels = {_segem_mask_in_channels(fusion_width, True, flag) for flag in guidance_choices}
    false_channels = {_segem_mask_in_channels(fusion_width, False, flag) for flag in guidance_choices}
    for key in ("fuse.0.conv1.conv.weight", "module.fuse.0.conv1.conv.weight"):
        weight = mask_head_state.get(key)
        if weight is not None and getattr(weight, "ndim", 0) == 4:
            in_channels = int(weight.shape[1])
            if in_channels in true_channels:
                return True
            if in_channels in false_channels:
                return False

    for weight in mask_head_state.values():
        if getattr(weight, "ndim", 0) == 4:
            in_channels = int(weight.shape[1])
            if in_channels in true_channels:
                return True
            if in_channels in false_channels:
                return False
    return default


def _infer_prompt_guidance_prior(
    backbone_config: dict,
    mask_head_state: dict | None,
    config: dict | None = None,
    *,
    default: bool = False,
) -> bool:
    if config is not None and "use_prompt_guidance_prior" in config:
        return bool(config["use_prompt_guidance_prior"])
    if not mask_head_state:
        return default

    fusion_width = int(backbone_config.get("fusion_width", 64))
    candidate_channels = {
        _segem_mask_in_channels(fusion_width, True, False): False,
        _segem_mask_in_channels(fusion_width, False, False): False,
        _segem_mask_in_channels(fusion_width, True, True): True,
        _segem_mask_in_channels(fusion_width, False, True): True,
    }
    for key in ("fuse.0.conv1.conv.weight", "module.fuse.0.conv1.conv.weight"):
        weight = mask_head_state.get(key)
        if weight is not None and getattr(weight, "ndim", 0) == 4:
            return candidate_channels.get(int(weight.shape[1]), default)
    for weight in mask_head_state.values():
        if getattr(weight, "ndim", 0) == 4:
            return candidate_channels.get(int(weight.shape[1]), default)
    return default


ALL_SOURCE_KEYS = frozenset().union(*SPECIES_TO_SOURCES.values())
ISOTROPIC_2D_SOURCES = frozenset({"hemi", "fib25"})


class _PromptModeDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        *,
        positive_aux_loss_weight: float = 0.0,
        seg_config: InteractiveSegEMConfig,
        prompt_tries: int = 2,
        crop_resample_tries: int = 48,
        random_index_fallback_tries: int = 32,
    ):
        self.base_dataset = base_dataset
        self.positive_aux_loss_weight = float(positive_aux_loss_weight)
        self.seg_config = seg_config
        self.prompt_tries = int(prompt_tries)
        self.crop_resample_tries = int(crop_resample_tries)
        self.random_index_fallback_tries = int(random_index_fallback_tries)

    def __len__(self):
        return len(self.base_dataset)

    @staticmethod
    def _unpack_base_item(item):
        n = len(item)
        if n == 5:
            raw, labels, _point_map, _mask, affinity = item
            return raw, labels, affinity, None
        if n == 6:
            raw, labels, _point_map, _mask, affinity, last = item
            if isinstance(last, dict):
                return raw, labels, affinity, last
            return raw, labels, affinity, None
        if n == 7:
            raw, labels, _point_map, _mask, affinity, _lsd, last = item
            meta = last if isinstance(last, dict) else None
            return raw, labels, affinity, meta
        raise ValueError("Unexpected EM2D __getitem__ length: %d" % (n,))

    def _try_positive_target_pack(self, raw, labels, affinity, crop_meta=None):
        if labels is None or not np.any(labels):
            return None
        forced = None
        if crop_meta and crop_meta.get("centric_target_id") is not None:
            forced = int(crop_meta["centric_target_id"])
        for _ in range(self.prompt_tries):
            rng = np.random.default_rng()
            out = build_seg_interactive_sample(
                labels,
                self.seg_config,
                rng=rng,
                forced_target_id=forced,
            )
            if out is None:
                continue
            point_map, mask, _meta = out
            return (
                raw,
                labels,
                point_map,
                mask,
                affinity,
                np.float32(1.0),
                np.float32(self.positive_aux_loss_weight),
            )
        return None

    def __getitem__(self, idx: int):
        n = len(self.base_dataset)
        for _ in range(self.crop_resample_tries):
            raw, labels, affinity, crop_meta = self._unpack_base_item(self.base_dataset[idx])
            pack = self._try_positive_target_pack(raw, labels, affinity, crop_meta)
            if pack is not None:
                return pack
        for _ in range(self.random_index_fallback_tries):
            j = int(np.random.randint(0, n))
            raw, labels, affinity, crop_meta = self._unpack_base_item(self.base_dataset[j])
            pack = self._try_positive_target_pack(raw, labels, affinity, crop_meta)
            if pack is not None:
                return pack
        raise RuntimeError(
            "segEM2d: could not sample a positive-target prompt (idx=%r, len=%d). "
            "Try more data or increase crop_resample_tries / random_index_fallback_tries."
            % (idx, n)
        )


def concat_source_balanced_weights(concat_ds: ConcatDataset) -> torch.Tensor:
    """Per-index weights so each sub-dataset in ``concat_ds`` has the same aggregate sampling mass."""
    weights: list[float] = []
    for sub in concat_ds.datasets:
        n = len(sub)
        if n <= 0:
            continue
        w = 1.0 / float(n)
        weights.extend([w] * n)
    if len(weights) != len(concat_ds):
        raise RuntimeError(
            "Balanced weight length mismatch: got %d weights for ConcatDataset len %d"
            % (len(weights), len(concat_ds))
        )
    return torch.DoubleTensor(weights)


def collate_fn_seg_prompt_train(batch):
    raw = np.array([item[0] for item in batch]).astype(np.float32)
    labels = np.array([item[1] for item in batch]).astype(np.int32)
    point_map = np.array([item[2] for item in batch]).astype(np.float32)
    mask = np.array([item[3] for item in batch]).astype(np.uint8)
    affinity = np.array([item[4] for item in batch]).astype(np.float32)
    mask_loss_weight = np.array([item[5] for item in batch]).astype(np.float32)
    aux_loss_weight = np.array([item[6] for item in batch]).astype(np.float32)
    return raw, labels, point_map, mask, affinity, mask_loss_weight, aux_loss_weight


def _weighted_elementwise_mean(loss_map: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    view_shape = (sample_weight.shape[0],) + (1,) * (loss_map.ndim - 1)
    weights = sample_weight.view(view_shape)
    denom = weights.expand_as(loss_map).sum().clamp_min(1.0)
    return (loss_map * weights).sum() / denom


def _weighted_soft_dice_loss(pred_prob: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    batch_size = target.size(0)
    pred_flat = pred_prob.reshape(batch_size, -1)
    target_flat = target.reshape(batch_size, -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    score = (2.0 * intersection + 1.0) / (
        pred_flat.sum(dim=1) + target_flat.sum(dim=1) + 1.0
    )
    loss_per_sample = 1.0 - score
    return (loss_per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)


def _use_xz_yz_for_source(source_name: str, *, split: str, eval_holdout: bool) -> bool:
    if eval_holdout:
        return False
    return source_name in ISOTROPIC_2D_SOURCES


def _common_seg_kwargs(crop_size: int, *, split: str, n_val: int, augment, require_xz_yz: bool):
    return dict(
        split=split,
        crop_size=crop_size,
        # segEM2d supervises mask + boundary affinity only; no LSD ground truth is requested.
        require_lsd=False,
        require_xz_yz=require_xz_yz,
        n_val=n_val,
        augment=augment,
    )


def _all_seg_source_specs(crop_size: int, *, split: str, n_val: int, augment, eval_holdout: bool):
    return [
        (
            "hemi",
            lambda: Dataset_2D_hemi_Train(
                data_dir="./data/funke/hemi/training/",
                **_common_seg_kwargs(
                    crop_size,
                    split=split,
                    n_val=n_val,
                    augment=augment,
                    require_xz_yz=_use_xz_yz_for_source("hemi", split=split, eval_holdout=eval_holdout),
                ),
            ),
        ),
        (
            "fib25",
            lambda: Dataset_2D_fib25_Train(
                data_dir="./data/funke/fib25/training/",
                **_common_seg_kwargs(
                    crop_size,
                    split=split,
                    n_val=n_val,
                    augment=augment,
                    require_xz_yz=_use_xz_yz_for_source("fib25", split=split, eval_holdout=eval_holdout),
                ),
            ),
        ),
        (
            "cremi",
            lambda: Dataset_2D_cremi_Train(
                data_dir="./data/CREMI/",
                **_common_seg_kwargs(
                    crop_size,
                    split=split,
                    n_val=n_val,
                    augment=augment,
                    require_xz_yz=_use_xz_yz_for_source("cremi", split=split, eval_holdout=eval_holdout),
                ),
            ),
        ),
        (
            "vnc",
            lambda: Dataset_2D_VNC_Train(
                data_dir="./data/groundtruth-drosophila-vnc-master/stack1/",
                **_common_seg_kwargs(
                    crop_size,
                    split=split,
                    n_val=n_val,
                    augment=augment,
                    require_xz_yz=_use_xz_yz_for_source("vnc", split=split, eval_holdout=eval_holdout),
                ),
            ),
        ),
        (
            "isbi2012",
            lambda: Dataset_2D_isbi2012_Train(
                data_dir="./data/ISBI-2012/",
                **_common_seg_kwargs(
                    crop_size,
                    split=split,
                    n_val=n_val,
                    augment=augment,
                    require_xz_yz=_use_xz_yz_for_source("isbi2012", split=split, eval_holdout=eval_holdout),
                ),
            ),
        ),
        (
            "ac3",
            lambda: Dataset_2D_ac3_Train(
                data_dir="./data/AC3/",
                **_common_seg_kwargs(
                    crop_size,
                    split=split,
                    n_val=n_val,
                    augment=augment,
                    require_xz_yz=_use_xz_yz_for_source("ac3", split=split, eval_holdout=eval_holdout),
                ),
            ),
        ),
        (
            "ac4",
            lambda: Dataset_2D_ac4_Train(
                data_dir="./data/AC4/",
                **_common_seg_kwargs(
                    crop_size,
                    split=split,
                    n_val=n_val,
                    augment=augment,
                    require_xz_yz=_use_xz_yz_for_source("ac4", split=split, eval_holdout=eval_holdout),
                ),
            ),
        ),
        (
            "basil",
            lambda: Dataset_2D_basil_Train(
                data_dir="./data/MICrONS/Neuron_zarr/basil/",
                **_common_seg_kwargs(
                    crop_size,
                    split=split,
                    n_val=n_val,
                    augment=augment,
                    require_xz_yz=_use_xz_yz_for_source("basil", split=split, eval_holdout=eval_holdout),
                ),
            ),
        ),
        (
            "minnie",
            lambda: Dataset_2D_minnie_Train(
                data_dir="./data/MICrONS/Neuron_zarr/minnie/",
                **_common_seg_kwargs(
                    crop_size,
                    split=split,
                    n_val=n_val,
                    augment=augment,
                    require_xz_yz=_use_xz_yz_for_source("minnie", split=split, eval_holdout=eval_holdout),
                ),
            ),
        ),
        (
            "pinky",
            lambda: Dataset_2D_pinky_Train(
                data_dir="./data/MICrONS/Neuron_zarr/pinky/",
                **_common_seg_kwargs(
                    crop_size,
                    split=split,
                    n_val=n_val,
                    augment=augment,
                    require_xz_yz=_use_xz_yz_for_source("pinky", split=split, eval_holdout=eval_holdout),
                ),
            ),
        ),
        (
            "axonem_m",
            lambda: Dataset_2D_axonem_m_Train(
                data_dir="./data/AxonEM/EM30-M-axon-train-9vol/",
                **_common_seg_kwargs(
                    crop_size,
                    split=split,
                    n_val=n_val,
                    augment=augment,
                    require_xz_yz=_use_xz_yz_for_source("axonem_m", split=split, eval_holdout=eval_holdout),
                ),
            ),
        ),
        (
            "axonem_h",
            lambda: Dataset_2D_axonem_h_Train(
                data_dir="./data/AxonEM/EM30-H-axon-train-9vol/",
                **_common_seg_kwargs(
                    crop_size,
                    split=split,
                    n_val=n_val,
                    augment=augment,
                    require_xz_yz=_use_xz_yz_for_source("axonem_h", split=split, eval_holdout=eval_holdout),
                ),
            ),
        ),
        (
            "zebrafinch",
            lambda: Dataset_2D_zebrafinch_Train_CL(
                data_dir="./data/funke/zebrafinch/training/",
                data_idxs=tuple(range(len(ZEBRA_PATCH_FILES))),
                n_val=n_val,
                split=split,
                crop_size=crop_size,
                require_lsd=False,
                require_xz_yz=_use_xz_yz_for_source("zebrafinch", split=split, eval_holdout=eval_holdout),
                augment=augment,
            ),
        ),
    ]


def _concat_seg_pool_for_keys(
    allowed_keys: frozenset,
    crop_size: int,
    *,
    split: str,
    n_val: int,
    augment,
    tag: str,
    positive_aux_loss_weight: float,
    eval_holdout: bool,
    seg_config: InteractiveSegEMConfig,
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
        ds = _PromptModeDataset(
            ds,
            positive_aux_loss_weight=positive_aux_loss_weight,
            seg_config=seg_config,
        )
        parts.append(ds)
        logging.info(
            "Loaded %s (%s): %d samples [%s]",
            name,
            tag,
            len(ds),
            "xy+xz+yz" if _use_xz_yz_for_source(name, split=split, eval_holdout=eval_holdout) else "xy-only",
        )
    if not parts:
        raise RuntimeError(
            "No datasets could be loaded for keys {} ({}); check paths under ./data/".format(
                sorted(allowed_keys), tag
            )
        )
    return ConcatDataset(parts)


def build_train_val_pool_leave_one_species(
    leave_species: str,
    crop_size: int,
    *,
    split: str,
    n_val_holdout: int = 16,
    augment=None,
    positive_aux_loss_weight: float = 0.0,
    seg_config: InteractiveSegEMConfig,
    segem_ic_kwargs: dict,
):
    held = SPECIES_TO_SOURCES[leave_species]
    allowed = frozenset(ALL_SOURCE_KEYS - held)
    return _concat_seg_pool_for_keys(
        allowed,
        crop_size,
        split=split,
        n_val=n_val_holdout,
        augment=augment,
        tag="train_val leave_out={}".format(leave_species),
        positive_aux_loss_weight=positive_aux_loss_weight,
        eval_holdout=False,
        seg_config=seg_config,
        segem_ic_kwargs=segem_ic_kwargs,
    )


def build_test_pool_leave_one_species(
    leave_species: str,
    crop_size: int,
    *,
    positive_aux_loss_weight: float = 0.0,
    seg_config: InteractiveSegEMConfig,
    segem_ic_kwargs: dict,
):
    allowed = SPECIES_TO_SOURCES[leave_species]
    return _concat_seg_pool_for_keys(
        allowed,
        crop_size,
        split="train",
        n_val=0,
        augment=False,
        tag="test holdout species={}".format(leave_species),
        positive_aux_loss_weight=positive_aux_loss_weight,
        eval_holdout=True,
        seg_config=seg_config,
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


def mask_to_boundary_affinity(mask_prob):
    padded = F.pad(mask_prob, (0, 1, 0, 1), mode="replicate")
    diff_x = torch.abs(mask_prob - padded[:, :, 1:, :-1])
    diff_y = torch.abs(mask_prob - padded[:, :, :-1, 1:])
    return torch.cat([diff_x, diff_y], dim=1)


class SegMaskHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.fuse = nn.Sequential(
            ResidualBlock(in_channels, hidden_channels, dropout=0.05, use_se=True),
            ResidualBlock(hidden_channels, hidden_channels, dropout=0.05, use_se=False),
            SqueezeExcite(hidden_channels),
            ResidualBlock(hidden_channels, hidden_channels, dropout=0.03, use_se=False),
        )
        self.head = TaskHead(hidden_channels, 1, hidden_channels=hidden_channels, dropout=0.02)

    def forward(self, x):
        x = self.fuse(x)
        return self.head(x)


class SegEM2dNeo(nn.Module):
    def __init__(
        self,
        device: torch.device,
        backbone_checkpoint: str,
        mask_width: int,
        use_teacher_lsd: bool = True,
        use_prompt_guidance_prior: bool = True,
        prompt_guidance_iters: int = 64,
    ):
        super().__init__()
        self.model_affinity, self.backbone_config = load_frozen_acrlsd_neo(backbone_checkpoint, device)
        fusion_width = int(self.backbone_config.get("fusion_width", 64))
        self.use_teacher_lsd = bool(use_teacher_lsd)
        self.use_prompt_guidance_prior = bool(use_prompt_guidance_prior)
        self.prompt_guidance_iters = int(max(1, prompt_guidance_iters))
        mask_in_channels = _segem_mask_in_channels(
            fusion_width,
            self.use_teacher_lsd,
            self.use_prompt_guidance_prior,
        )
        self.mask_head = SegMaskHead(mask_in_channels, hidden_channels=mask_width)

    def train(self, mode: bool = True):
        super().train(mode)
        # Teacher remains completely frozen and always stays in eval mode.
        self.model_affinity.eval()
        return self

    def _teacher_forward(self, x_raw):
        with torch.no_grad():
            teacher = self.model_affinity.forward_with_intermediates(x_raw)
            affinity_logits = teacher["affinity_logits"]
            affinity_prob = torch.sigmoid(affinity_logits)

        return {
            "encoder_fused": teacher["encoder_fused"],
            "lsd_feat": teacher["lsd_feat"],
            "affinity_feat": teacher["affinity_feat"],
            "lsd_logits": teacher["lsd_logits"],
            "lsd_prob": teacher["lsd_prob"],
            "affinity_logits": affinity_logits,
            "affinity_prob": affinity_prob,
        }

    @staticmethod
    def _propagate_prompt_seed(seed: torch.Tensor, merge_x: torch.Tensor, merge_y: torch.Tensor, n_iters: int) -> torch.Tensor:
        score = seed.clamp(0.0, 1.0)
        for _ in range(int(max(1, n_iters))):
            left_to_right = F.pad(score[:, :, :, :-1] * merge_x[:, :, :, :-1], (1, 0, 0, 0))
            right_to_left = F.pad(score[:, :, :, 1:] * merge_x[:, :, :, :-1], (0, 1, 0, 0))
            top_to_bottom = F.pad(score[:, :, :-1, :] * merge_y[:, :, :-1, :], (0, 0, 1, 0))
            bottom_to_top = F.pad(score[:, :, 1:, :] * merge_y[:, :, :-1, :], (0, 0, 0, 1))
            score = torch.maximum(
                score,
                torch.maximum(
                    torch.maximum(left_to_right, right_to_left),
                    torch.maximum(top_to_bottom, bottom_to_top),
                ),
            )
        return score

    def _prompt_guidance_prior(self, x_prompt: torch.Tensor, teacher: dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.use_prompt_guidance_prior:
            return x_prompt.new_zeros((x_prompt.shape[0], 1, x_prompt.shape[2], x_prompt.shape[3]))

        pos_seed = torch.clamp((x_prompt - 0.70) / 0.30, min=0.0, max=1.0)
        neg_seed = torch.clamp((-x_prompt - 0.20) / 0.80, min=0.0, max=1.0)

        merge_x = 1.0 - teacher["affinity_prob"][:, 0:1].detach()
        merge_y = 1.0 - teacher["affinity_prob"][:, 1:2].detach()
        lsd_support = torch.mean(teacher["lsd_prob"].detach(), dim=1, keepdim=True)
        edge_lsd_x = torch.sqrt(
            torch.clamp(lsd_support[:, :, :, :-1] * lsd_support[:, :, :, 1:], min=0.0, max=1.0)
        )
        edge_lsd_y = torch.sqrt(
            torch.clamp(lsd_support[:, :, :-1, :] * lsd_support[:, :, 1:, :], min=0.0, max=1.0)
        )
        merge_x = merge_x.clone()
        merge_y = merge_y.clone()
        merge_x[:, :, :, :-1] = merge_x[:, :, :, :-1] * edge_lsd_x
        merge_y[:, :, :-1, :] = merge_y[:, :, :-1, :] * edge_lsd_y

        aff_interior = torch.clamp(0.5 * (merge_x + merge_y), min=0.0, max=1.0)
        interior = torch.clamp(0.65 * aff_interior + 0.35 * lsd_support, min=0.0, max=1.0)

        pos_reach = self._propagate_prompt_seed(pos_seed, merge_x, merge_y, self.prompt_guidance_iters)
        neg_reach = self._propagate_prompt_seed(neg_seed, merge_x, merge_y, max(8, self.prompt_guidance_iters // 2))
        prior = torch.clamp(pos_reach - 0.85 * neg_reach, min=0.0, max=1.0)
        return prior * interior

    def _build_mask_input(self, x_raw: torch.Tensor, x_prompt: torch.Tensor, teacher: dict[str, torch.Tensor]) -> torch.Tensor:
        mask_parts = [x_raw, x_prompt, teacher["affinity_prob"], teacher["encoder_fused"], teacher["affinity_feat"]]
        if self.use_teacher_lsd:
            mask_parts.insert(2, teacher["lsd_prob"])
            mask_parts.insert(5, teacher["lsd_feat"])
        if self.use_prompt_guidance_prior:
            mask_parts.append(self._prompt_guidance_prior(x_prompt, teacher))
        return torch.cat(mask_parts, dim=1)

    def forward_prompt(self, x_raw, x_prompt):
        if x_raw.ndim == 3:
            x_raw = x_raw.unsqueeze(1)
        if x_prompt.ndim == 3:
            x_prompt = x_prompt.unsqueeze(1)
        teacher = self._teacher_forward(x_raw)
        prompt_logits = self.mask_head(self._build_mask_input(x_raw, x_prompt, teacher))
        return prompt_logits, teacher

    def forward(self, x_raw, x_prompt):
        return self.forward_prompt(x_raw, x_prompt)


def prompt_model_step(
    model,
    bce_loss_fn,
    dice_loss_fn,
    optimizer,
    raw,
    point_map,
    gt_mask,
    gt_affinity,
    mask_loss_weight,
    aux_loss_weight,
    device,
    *,
    weight_bce: float,
    weight_dice: float,
    weight_boundary: float,
    train_step=True,
    scheduler=None,
    scaler=None,
    amp_enabled=False,
    grad_clip_norm=None,
):
    if train_step:
        optimizer.zero_grad(set_to_none=True)

    if gt_mask.ndim == 3:
        gt_mask = gt_mask.unsqueeze(1)
    mask_loss_weight = mask_loss_weight.reshape(-1)
    aux_loss_weight = aux_loss_weight.reshape(-1)

    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
        prompt_logits, teacher = model.forward_prompt(raw, point_map)
        prompt_prob = torch.sigmoid(prompt_logits)
        loss_bce = _weighted_elementwise_mean(
            F.binary_cross_entropy_with_logits(prompt_logits, gt_mask, reduction="none"),
            mask_loss_weight,
        )
        loss_dice = _weighted_soft_dice_loss(prompt_prob, gt_mask, mask_loss_weight)
        pred_boundary = mask_to_boundary_affinity(prompt_prob)
        loss_boundary = _weighted_elementwise_mean(
            F.smooth_l1_loss(pred_boundary, gt_affinity, reduction="none"),
            aux_loss_weight,
        )
        prompt_loss = weight_bce * loss_bce + weight_dice * loss_dice + weight_boundary * loss_boundary
        loss_value = prompt_loss

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
        "prompt_loss": prompt_loss.detach(),
        "loss_bce": loss_bce.detach(),
        "loss_dice": loss_dice.detach(),
        "loss_boundary": loss_boundary.detach(),
        "pred_mask": prompt_prob.detach(),
        "teacher_affinity": teacher["affinity_prob"].detach(),
    }


def batch_mask_iou(pred_prob: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    if gt_mask.ndim == 3:
        gt_mask = gt_mask.unsqueeze(1)
    pred_bin = pred_prob >= float(threshold)
    gt_bin = gt_mask >= 0.5
    intersection = torch.logical_and(pred_bin, gt_bin).flatten(1).sum(dim=1).to(dtype=torch.float32)
    union = torch.logical_or(pred_bin, gt_bin).flatten(1).sum(dim=1).to(dtype=torch.float32)
    return torch.where(union > 0, intersection / union, torch.ones_like(union))


if __name__ == "__main__":
    args = build_argparser().parse_args()
    leave_species = args.leave_species
    backbone_checkpoint = args.backbone_checkpoint or (
        "./output/checkpoints/ACRLSD_2D_leaveout_{}_holdoutVal{}_neo_Best_in_val.model".format(
            leave_species, args.n_val_holdout
        )
    )

    save_name = args.save_name or "segEM2d_leaveout_{}_holdoutVal{}_neo_wb{}_wd{}_wbd{}".format(
        leave_species,
        args.n_val_holdout,
        args.weight_bce,
        args.weight_dice,
        args.weight_boundary,
    )
    _install_persistent_diagnostics(save_name)

    set_seed(args.seed)

    interactive_cfg = interactive_seg_config_from_args(args)
    segem_ic_extra = segem_ic_kwargs_from_config(interactive_cfg)

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

    train_dataset = build_train_val_pool_leave_one_species(
        leave_species,
        crop_size=args.crop_size,
        split="train",
        n_val_holdout=args.n_val_holdout,
        augment=True,
        positive_aux_loss_weight=args.positive_aux_loss_weight,
        seg_config=interactive_cfg,
        segem_ic_kwargs=segem_ic_extra,
    )
    val_dataset = build_train_val_pool_leave_one_species(
        leave_species,
        crop_size=args.crop_size,
        split="val",
        n_val_holdout=args.n_val_holdout,
        augment=False,
        positive_aux_loss_weight=args.positive_aux_loss_weight,
        seg_config=interactive_cfg,
        segem_ic_kwargs=segem_ic_extra,
    )
    test_dataset = build_test_pool_leave_one_species(
        leave_species,
        args.crop_size,
        positive_aux_loss_weight=args.positive_aux_loss_weight,
        seg_config=interactive_cfg,
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
        collate_fn=collate_fn_seg_prompt_train,
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
        collate_fn=collate_fn_seg_prompt_train,
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
        collate_fn=collate_fn_seg_prompt_train,
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
        t = last_epoch - warmup_steps
        total_after_warmup = max(1, max_train_steps - warmup_steps)
        progress = min(float(t) / float(total_after_warmup), 1.0)
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
        """Starting training:
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
    LR schedule:          linear warmup %s epochs (~%s steps) + cosine to %.4f * base lr
    EMA decay:            %s
    Grad clip norm:       %s
    Save top-k ckpts:     %s
    AMP enabled:          %s
    Backbone checkpoint:  %s
    Backbone config:      %s
    Mask width:           %s
    Teacher LSD in mask:  %s
    Prompt guide prior:   %s (iters=%s)
    Prompts / crop:       %s
    Interactive detail:   pos_probs=%s | max_pos=%s | cand_pool=%s | interior_thre=%s | theta=%s | rounds=%s
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
        args.warmup_epochs,
        warmup_steps,
        args.cosine_eta_min_ratio,
        args.ema_decay,
        args.grad_clip_norm,
        args.save_top_k,
        amp_enabled,
        backbone_checkpoint,
        model.backbone_config if model.backbone_config else "defaults(base=32,bottleneck=384,fusion=64)",
        args.mask_width,
        model.use_teacher_lsd,
        model.use_prompt_guidance_prior,
        model.prompt_guidance_iters,
        "instance-centric crop, pos/neg Gaussians, multi-round correction replay",
        args.segem_positive_count_probs,
        args.positive_point_count,
        args.prompt_max_candidate_pool,
        args.positive_point_thre,
        args.prompt_theta,
        args.segem_max_interaction_rounds,
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
        acc_bce = []
        acc_dice = []
        acc_boundary = []
        acc_iou = []
        weight_scope = ema.apply_to(model.mask_head) if use_ema else contextlib.nullcontext()
        with weight_scope:
            for raw, labels, point_map, mask, gt_affinity, mask_loss_weight, aux_loss_weight in loader:
                raw = torch.as_tensor(raw, dtype=torch.float32, device=device)
                point_map = torch.as_tensor(point_map, dtype=torch.float32, device=device)
                mask = torch.as_tensor(mask, dtype=torch.float32, device=device)
                gt_affinity = torch.as_tensor(gt_affinity, dtype=torch.float32, device=device)
                mask_loss_weight = torch.as_tensor(mask_loss_weight, dtype=torch.float32, device=device)
                aux_loss_weight = torch.as_tensor(aux_loss_weight, dtype=torch.float32, device=device)

                with torch.no_grad():
                    loss_value, stats = prompt_model_step(
                        model,
                        bce_loss_fn,
                        dice_loss_fn,
                        optimizer,
                        raw,
                        point_map,
                        mask,
                        gt_affinity,
                        mask_loss_weight,
                        aux_loss_weight,
                        device,
                        weight_bce=args.weight_bce,
                        weight_dice=args.weight_dice,
                        weight_boundary=args.weight_boundary,
                        train_step=False,
                        scheduler=None,
                        scaler=None,
                        amp_enabled=amp_enabled,
                        grad_clip_norm=None,
                    )
                acc_total.append(float(loss_value.detach().cpu().item()))
                acc_bce.append(float(stats["loss_bce"].cpu().item()))
                acc_dice.append(float(stats["loss_dice"].cpu().item()))
                acc_boundary.append(float(stats["loss_boundary"].cpu().item()))
                batch_iou = batch_mask_iou(stats["pred_mask"], mask).mean().cpu().item()
                acc_iou.append(float(batch_iou))

        return {
            "loss": float(np.mean(acc_total)) if acc_total else float("nan"),
            "loss_bce": float(np.mean(acc_bce)) if acc_bce else float("nan"),
            "loss_dice": float(np.mean(acc_dice)) if acc_dice else float("nan"),
            "loss_boundary": float(np.mean(acc_boundary)) if acc_boundary else float("nan"),
            "mean_iou": float(np.mean(acc_iou)) if acc_iou else float("nan"),
        }

    with tqdm(total=args.epochs) as pbar:
        while epoch < args.epochs:
            model.train()
            train_total = []
            train_prompt_total = []
            train_bce = []
            train_dice = []
            train_boundary = []
            train_iou = []

            for raw, labels, point_map, mask, gt_affinity, mask_loss_weight, aux_loss_weight in train_loader:
                raw = torch.as_tensor(raw, dtype=torch.float32, device=device)
                point_map = torch.as_tensor(point_map, dtype=torch.float32, device=device)
                mask = torch.as_tensor(mask, dtype=torch.float32, device=device)
                gt_affinity = torch.as_tensor(gt_affinity, dtype=torch.float32, device=device)
                mask_loss_weight = torch.as_tensor(mask_loss_weight, dtype=torch.float32, device=device)
                aux_loss_weight = torch.as_tensor(aux_loss_weight, dtype=torch.float32, device=device)

                loss_value, stats = prompt_model_step(
                    model,
                    bce_loss_fn,
                    dice_loss_fn,
                    optimizer,
                    raw,
                    point_map,
                    mask,
                    gt_affinity,
                    mask_loss_weight,
                    aux_loss_weight,
                    device,
                    weight_bce=args.weight_bce,
                    weight_dice=args.weight_dice,
                    weight_boundary=args.weight_boundary,
                    train_step=True,
                    scheduler=scheduler,
                    scaler=scaler,
                    amp_enabled=amp_enabled,
                    grad_clip_norm=args.grad_clip_norm,
                )
                ema.update(model.mask_head)
                train_total.append(float(loss_value.detach().cpu().item()))
                train_prompt_total.append(float(stats["prompt_loss"].cpu().item()))
                train_bce.append(float(stats["loss_bce"].cpu().item()))
                train_dice.append(float(stats["loss_dice"].cpu().item()))
                train_boundary.append(float(stats["loss_boundary"].cpu().item()))
                train_iou.append(float(batch_mask_iou(stats["pred_mask"], mask).mean().cpu().item()))

            epoch += 1
            pbar.update(1)

            train_metrics = {
                "loss": float(np.mean(train_total)) if train_total else float("nan"),
                "prompt_loss": float(np.mean(train_prompt_total)) if train_prompt_total else float("nan"),
                "loss_bce": float(np.mean(train_bce)) if train_bce else float("nan"),
                "loss_dice": float(np.mean(train_dice)) if train_dice else float("nan"),
                "loss_boundary": float(np.mean(train_boundary)) if train_boundary else float("nan"),
                "mean_iou": float(np.mean(train_iou)) if train_iou else float("nan"),
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
                    "config": vars(args),
                }
                torch.save(ckpt_state, ckpt_path)
                no_improve_count = 0
            else:
                no_improve_count += 1

            logging.info(
                "Epoch %d: train(total=%.6f bce=%.6f dice=%.6f boundary=%.6f iou=%.6f) | "
                "val(ema,total=%.6f bce=%.6f dice=%.6f boundary=%.6f iou=%.6f) | "
                "test(ema,total=%.6f bce=%.6f dice=%.6f boundary=%.6f iou=%.6f, leave_species=%s) | "
                "best_val=%.6f@%d | lr = %.8f%s",
                epoch,
                train_metrics["loss"],
                train_metrics["loss_bce"],
                train_metrics["loss_dice"],
                train_metrics["loss_boundary"],
                train_metrics["mean_iou"],
                val_metrics["loss"],
                val_metrics["loss_bce"],
                val_metrics["loss_dice"],
                val_metrics["loss_boundary"],
                val_metrics["mean_iou"],
                test_metrics["loss"],
                test_metrics["loss_bce"],
                test_metrics["loss_dice"],
                test_metrics["loss_boundary"],
                test_metrics["mean_iou"],
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
        "Best checkpoint: val(total=%.6f bce=%.6f dice=%.6f boundary=%.6f iou=%.6f) @ epoch %d | "
        "test(total=%.6f bce=%.6f dice=%.6f boundary=%.6f iou=%.6f) | checkpoint=%s",
        best_val_loss,
        float(best_val_metrics["loss_bce"]) if best_val_metrics else float("nan"),
        float(best_val_metrics["loss_dice"]) if best_val_metrics else float("nan"),
        float(best_val_metrics["loss_boundary"]) if best_val_metrics else float("nan"),
        float(best_val_metrics["mean_iou"]) if best_val_metrics else float("nan"),
        best_epoch,
        float(best_test_metrics["loss"]) if best_test_metrics else float("nan"),
        float(best_test_metrics["loss_bce"]) if best_test_metrics else float("nan"),
        float(best_test_metrics["loss_dice"]) if best_test_metrics else float("nan"),
        float(best_test_metrics["loss_boundary"]) if best_test_metrics else float("nan"),
        float(best_test_metrics["mean_iou"]) if best_test_metrics else float("nan"),
        ckpt_path,
    )
