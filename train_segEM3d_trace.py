import argparse
import contextlib
import logging
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm

from train_ACRLSD_3d_neo import (
    ACRLSDneo3D,
    LEAVE_SPECIES_CHOICES,
    ModelEma,
    ResidualBlock3d,
    SPECIES_TO_SOURCES,
    SqueezeExcite3d,
    TaskHead3d,
    count_parameters,
    seed_worker,
    set_seed,
    _install_persistent_diagnostics,
)
from utils.dataloader import (
    ZEBRA_PATCH_FILES,
    Dataset_3D_VNC_Train,
    Dataset_3D_ac3_Train,
    Dataset_3D_ac4_Train,
    Dataset_3D_axonem_h_Train,
    Dataset_3D_axonem_m_Train,
    Dataset_3D_basil_Train,
    Dataset_3D_cremi_Train,
    Dataset_3D_fib25_Train,
    Dataset_3D_hemi_Train,
    Dataset_3D_isbi2012_Train,
    Dataset_3D_minnie_Train,
    Dataset_3D_pinky_Train,
    Dataset_3D_zebrafinch_Train_CL,
    collate_fn_3D_hemi_Train,
)


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Train segEM3d trace neo with a frozen ACRLSD-neo 3D teacher and leave-species-out splits."
    )
    parser.add_argument("--leave-species", type=str, default="human", choices=LEAVE_SPECIES_CHOICES)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--num-slices", type=int, default=8)
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
    parser.add_argument("--backbone-checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1998)
    parser.add_argument("--save-name", type=str, default=None)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision. By default AMP is enabled.")
    return parser


def _strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    prefixes = ("module.", "_orig_mod.")
    stripped = dict(state_dict)
    changed = True
    while changed and stripped:
        changed = False
        for prefix in prefixes:
            if all(key.startswith(prefix) for key in stripped):
                stripped = {key[len(prefix) :]: value for key, value in stripped.items()}
                changed = True
    return stripped


def load_frozen_acrlsd_neo3d(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        backbone_config = checkpoint.get("config", {})
        state_dict = checkpoint.get("ema_state_dict") or checkpoint.get("model_state_dict") or checkpoint
    else:
        backbone_config = {}
        state_dict = checkpoint

    state_dict = _strip_module_prefix(state_dict)
    base_width = int(backbone_config.get("base_width", 16))
    bottleneck_channels = int(backbone_config.get("bottleneck_channels", 160))
    fusion_width = int(backbone_config.get("fusion_width", 32))

    model = ACRLSDneo3D(
        in_channels=1,
        base_width=base_width,
        encoder_widths=(base_width, base_width * 2, base_width * 4),
        bottleneck_channels=bottleneck_channels,
        fusion_width=fusion_width,
        lsd_channels=10,
        affinity_channels=3,
        detach_lsd_for_affinity=True,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, backbone_config


def resolve_backbone_checkpoint(args) -> str:
    if args.backbone_checkpoint:
        return args.backbone_checkpoint

    candidate_paths = [
        "./output/checkpoints/ACRLSD_3D_leaveout_{}_holdoutVal{}_neo_preLSD_Best_in_val.model".format(
            args.leave_species, args.n_val_holdout
        ),
        "./output/checkpoints/ACRLSD_3D_leaveout_{}_holdoutVal{}_neo_Best_in_val.model".format(
            args.leave_species, args.n_val_holdout
        ),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Could not find a 3D ACRLSD teacher checkpoint. Tried: {}".format(", ".join(candidate_paths))
    )


ALL_SOURCE_KEYS = frozenset().union(*SPECIES_TO_SOURCES.values())


def _all_seg_source_specs(
    crop_size: int,
    num_slices: int,
    *,
    split: str,
    n_val: int,
    augment,
):
    kw = dict(
        split=split,
        crop_size=crop_size,
        num_slices=num_slices,
        require_lsd=False,
        require_xz_yz=True,
        n_val=n_val,
        augment=augment,
    )
    return [
        ("hemi", lambda: Dataset_3D_hemi_Train(data_dir="./data/funke/hemi/training/", **kw)),
        ("fib25", lambda: Dataset_3D_fib25_Train(data_dir="./data/funke/fib25/training/", **kw)),
        ("cremi", lambda: Dataset_3D_cremi_Train(data_dir="./data/CREMI/", **kw)),
        ("vnc", lambda: Dataset_3D_VNC_Train(data_dir="./data/groundtruth-drosophila-vnc-master/stack1/", **kw)),
        ("isbi2012", lambda: Dataset_3D_isbi2012_Train(data_dir="./data/ISBI-2012/", **kw)),
        ("ac3", lambda: Dataset_3D_ac3_Train(data_dir="./data/AC3/", **kw)),
        ("ac4", lambda: Dataset_3D_ac4_Train(data_dir="./data/AC4/", **kw)),
        ("basil", lambda: Dataset_3D_basil_Train(data_dir="./data/MICrONS/Neuron_zarr/basil/", **kw)),
        ("minnie", lambda: Dataset_3D_minnie_Train(data_dir="./data/MICrONS/Neuron_zarr/minnie/", **kw)),
        ("pinky", lambda: Dataset_3D_pinky_Train(data_dir="./data/MICrONS/Neuron_zarr/pinky/", **kw)),
        ("axonem_m", lambda: Dataset_3D_axonem_m_Train(data_dir="./data/AxonEM/EM30-M-axon-train-9vol/", **kw)),
        ("axonem_h", lambda: Dataset_3D_axonem_h_Train(data_dir="./data/AxonEM/EM30-H-axon-train-9vol/", **kw)),
        (
            "zebrafinch",
            lambda: Dataset_3D_zebrafinch_Train_CL(
                data_dir="./data/funke/zebrafinch/training/",
                data_idxs=tuple(range(len(ZEBRA_PATCH_FILES))),
                **kw,
            ),
        ),
    ]


def _concat_seg_pool_for_keys(
    allowed_keys: frozenset,
    crop_size: int,
    num_slices: int,
    *,
    split: str,
    n_val: int,
    augment,
    tag: str,
):
    parts = []
    for name, factory in _all_seg_source_specs(crop_size, num_slices, split=split, n_val=n_val, augment=augment):
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
        parts.append(ds)
        logging.info("Loaded %s (%s): %d samples", name, tag, len(ds))
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
    num_slices: int,
    *,
    split: str,
    n_val_holdout: int = 16,
    augment=None,
):
    allowed = frozenset(ALL_SOURCE_KEYS - SPECIES_TO_SOURCES[leave_species])
    return _concat_seg_pool_for_keys(
        allowed,
        crop_size=crop_size,
        num_slices=num_slices,
        split=split,
        n_val=n_val_holdout,
        augment=augment,
        tag="train_val leave_out={}".format(leave_species),
    )


def build_test_pool_leave_one_species(leave_species: str, crop_size: int, num_slices: int):
    allowed = SPECIES_TO_SOURCES[leave_species]
    return _concat_seg_pool_for_keys(
        allowed,
        crop_size=crop_size,
        num_slices=num_slices,
        split="train",
        n_val=0,
        augment=False,
        tag="test holdout species={}".format(leave_species),
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


def mask_to_boundary_affinity_3d(mask_prob: torch.Tensor) -> torch.Tensor:
    padded = F.pad(mask_prob, (0, 1, 0, 1, 0, 1), mode="replicate")
    diff_x = torch.abs(mask_prob - padded[:, :, 1:, :-1, :-1])
    diff_y = torch.abs(mask_prob - padded[:, :, :-1, 1:, :-1])
    diff_z = torch.abs(mask_prob - padded[:, :, :-1, :-1, 1:])
    return torch.cat([diff_x, diff_y, diff_z], dim=1)


class SegMaskHead3d(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.fuse = nn.Sequential(
            ResidualBlock3d(in_channels, hidden_channels, dropout=0.05, use_se=True),
            ResidualBlock3d(hidden_channels, hidden_channels, dropout=0.05, use_se=False),
            SqueezeExcite3d(hidden_channels),
            ResidualBlock3d(hidden_channels, hidden_channels, dropout=0.03, use_se=False),
        )
        self.head = TaskHead3d(hidden_channels, 1, hidden_channels=hidden_channels, dropout=0.02)

    def forward(self, x):
        x = self.fuse(x)
        return self.head(x)


class SegEM3dTraceNeo(nn.Module):
    def __init__(self, device: torch.device, backbone_checkpoint: str, mask_width: int):
        super().__init__()
        self.model_affinity, self.backbone_config = load_frozen_acrlsd_neo3d(backbone_checkpoint, device)
        fusion_width = int(self.backbone_config.get("fusion_width", 32))
        mask_in_channels = 1 + 1 + 10 + 3 + fusion_width * 3
        self.mask_head = SegMaskHead3d(mask_in_channels, hidden_channels=mask_width)

    def train(self, mode: bool = True):
        super().train(mode)
        self.model_affinity.eval()
        return self

    def _teacher_forward(self, x_raw):
        with torch.no_grad():
            bottleneck, skips = self.model_affinity._encode(x_raw)

            _lsd_last, lsd_scales = self.model_affinity.lsd_decoder(bottleneck, skips)
            affinity_last, affinity_scales = self.model_affinity.affinity_decoder(bottleneck, skips)

            encoder_fused = self.model_affinity.encoder_fusion(list(reversed(skips)))
            lsd_fused = self.model_affinity.lsd_scale_fusion(lsd_scales)
            lsd_feat = self.model_affinity.lsd_refine(torch.cat([lsd_fused, encoder_fused], dim=1))
            lsd_logits = self.model_affinity.lsd_head(lsd_feat)
            lsd_prob = torch.sigmoid(lsd_logits)

            affinity_fused = self.model_affinity.affinity_scale_fusion(affinity_scales)
            lsd_for_affinity = lsd_feat.detach() if self.model_affinity.detach_lsd_for_affinity else lsd_feat
            prob_for_affinity = lsd_prob.detach() if self.model_affinity.detach_lsd_for_affinity else lsd_prob

            if affinity_last.shape[-3:] != lsd_feat.shape[-3:]:
                affinity_last = F.interpolate(
                    affinity_last,
                    size=lsd_feat.shape[-3:],
                    mode="trilinear",
                    align_corners=False,
                )

            affinity_input = torch.cat(
                [affinity_fused, affinity_last, encoder_fused, x_raw, lsd_for_affinity, prob_for_affinity],
                dim=1,
            )
            affinity_feat = self.model_affinity.affinity_refine(affinity_input)
            affinity_logits = self.model_affinity.affinity_head(affinity_feat)
            affinity_prob = torch.sigmoid(affinity_logits)

        return {
            "encoder_fused": encoder_fused,
            "lsd_feat": lsd_feat,
            "affinity_feat": affinity_feat,
            "lsd_logits": lsd_logits,
            "lsd_prob": lsd_prob,
            "affinity_logits": affinity_logits,
            "affinity_prob": affinity_prob,
        }

    def forward(self, x_raw, gt_mask2d_slice0):
        if x_raw.ndim == 4:
            x_raw = x_raw.unsqueeze(1)
        if gt_mask2d_slice0.ndim == 4:
            gt_mask2d_slice0 = gt_mask2d_slice0.squeeze(1)

        teacher = self._teacher_forward(x_raw)
        seed_volume = torch.zeros_like(x_raw)
        seed_volume[:, 0, :, :, 0] = gt_mask2d_slice0

        mask_input = torch.cat(
            [
                x_raw,
                seed_volume,
                teacher["lsd_prob"],
                teacher["affinity_prob"],
                teacher["encoder_fused"],
                teacher["lsd_feat"],
                teacher["affinity_feat"],
            ],
            dim=1,
        )
        mask_logits = self.mask_head(mask_input)
        return mask_logits, teacher


def model_step(
    model,
    bce_loss_fn,
    dice_loss_fn,
    optimizer,
    raw,
    gt_mask,
    gt_affinity,
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

    if raw.ndim == 4:
        raw = raw.unsqueeze(1)
    if gt_mask.ndim == 4:
        gt_mask = gt_mask.unsqueeze(1)

    seed_slice0 = gt_mask[:, 0, :, :, 0]
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
        mask_logits, _teacher = model(raw, seed_slice0)
        mask_prob = torch.sigmoid(mask_logits)
        loss_bce = bce_loss_fn(mask_logits, gt_mask)
        loss_dice = dice_loss_fn(mask_prob, gt_mask)
        pred_boundary = mask_to_boundary_affinity_3d(mask_prob)
        loss_boundary = F.smooth_l1_loss(pred_boundary, gt_affinity)
        loss_value = weight_bce * loss_bce + weight_dice * loss_dice + weight_boundary * loss_boundary

    if train_step:
        if scaler is not None and amp_enabled:
            scaler.scale(loss_value).backward()
            scaler.unscale_(optimizer)
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.mask_head.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_value.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.mask_head.parameters(), grad_clip_norm)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

    return loss_value, {
        "loss_bce": loss_bce.detach(),
        "loss_dice": loss_dice.detach(),
        "loss_boundary": loss_boundary.detach(),
        "pred_mask": mask_prob.detach(),
    }


if __name__ == "__main__":
    args = build_argparser().parse_args()
    leave_species = args.leave_species
    backbone_checkpoint = resolve_backbone_checkpoint(args)
    save_name = args.save_name or "segEM3d_trace_leaveout_{}_holdoutVal{}_neo_wb{}_wd{}_wbd{}".format(
        leave_species,
        args.n_val_holdout,
        args.weight_bce,
        args.weight_dice,
        args.weight_boundary,
    )
    _install_persistent_diagnostics(save_name)

    set_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    amp_enabled = (device.type == "cuda") and (not args.no_amp)

    model = SegEM3dTraceNeo(
        device=device,
        backbone_checkpoint=backbone_checkpoint,
        mask_width=args.mask_width,
    ).to(device)
    ema = ModelEma(model.mask_head, decay=args.ema_decay)

    train_dataset = build_train_val_pool_leave_one_species(
        leave_species,
        crop_size=args.crop_size,
        num_slices=args.num_slices,
        split="train",
        n_val_holdout=args.n_val_holdout,
        augment=True,
    )
    val_dataset = build_train_val_pool_leave_one_species(
        leave_species,
        crop_size=args.crop_size,
        num_slices=args.num_slices,
        split="val",
        n_val_holdout=args.n_val_holdout,
        augment=False,
    )
    test_dataset = build_test_pool_leave_one_species(
        leave_species,
        args.crop_size,
        args.num_slices,
    )

    train_gen = torch.Generator().manual_seed(args.seed + 7)
    val_gen = torch.Generator().manual_seed(args.seed + 8)
    test_gen = torch.Generator().manual_seed(args.seed + 9)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=collate_fn_3D_hemi_Train,
        generator=train_gen,
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
        collate_fn=collate_fn_3D_hemi_Train,
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
        collate_fn=collate_fn_3D_hemi_Train,
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
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logging.info(
        "Diagnostics: uncaught exceptions -> ./output/log/crash_%s.log | stderr copy -> ./output/log/stderr_%s.log",
        save_name,
        save_name,
    )
    logging.info(
        """Starting training:
    leave_species:        %s (test sources: %s)
    training_epochs:      %s
    Train samples:        %d
    Val samples:          %d
    Test samples:         %d
    Holdout slices:       %d
    Batch size:           %s
    Val batch size:       %s
    Crop size:            %s
    Num slices:           %s
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
    Trainable params (M): %.2f
    num_workers:          %s
    Device:               %s
    """,
        leave_species,
        ", ".join(sorted(SPECIES_TO_SOURCES[leave_species])),
        args.epochs,
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
        args.n_val_holdout,
        args.batch_size,
        args.val_batch_size,
        args.crop_size,
        args.num_slices,
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
        model.backbone_config if model.backbone_config else "defaults(base=16,bottleneck=160,fusion=32)",
        args.mask_width,
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
    no_improve_count = 0
    ranked_checkpoints = []

    def run_eval_loader(loader, use_ema=True):
        model.eval()
        acc_total = []
        acc_bce = []
        acc_dice = []
        acc_boundary = []
        weight_scope = ema.apply_to(model.mask_head) if use_ema else contextlib.nullcontext()
        with weight_scope:
            for raw, labels, mask_3d, gt_affinity, point_map in loader:
                raw = torch.as_tensor(raw, dtype=torch.float32, device=device)
                mask_3d = torch.as_tensor(mask_3d, dtype=torch.float32, device=device)
                gt_affinity = torch.as_tensor(gt_affinity, dtype=torch.float32, device=device)
                with torch.no_grad():
                    loss_value, stats = model_step(
                        model,
                        bce_loss_fn,
                        dice_loss_fn,
                        optimizer,
                        raw,
                        mask_3d,
                        gt_affinity,
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

        return {
            "loss": float(np.mean(acc_total)) if acc_total else float("nan"),
            "loss_bce": float(np.mean(acc_bce)) if acc_bce else float("nan"),
            "loss_dice": float(np.mean(acc_dice)) if acc_dice else float("nan"),
            "loss_boundary": float(np.mean(acc_boundary)) if acc_boundary else float("nan"),
        }

    with tqdm(total=args.epochs) as pbar:
        while epoch < args.epochs:
            model.train()
            train_total = []
            train_bce = []
            train_dice = []
            train_boundary = []

            for raw, labels, mask_3d, gt_affinity, point_map in train_loader:
                raw = torch.as_tensor(raw, dtype=torch.float32, device=device)
                mask_3d = torch.as_tensor(mask_3d, dtype=torch.float32, device=device)
                gt_affinity = torch.as_tensor(gt_affinity, dtype=torch.float32, device=device)

                loss_value, stats = model_step(
                    model,
                    bce_loss_fn,
                    dice_loss_fn,
                    optimizer,
                    raw,
                    mask_3d,
                    gt_affinity,
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
                train_bce.append(float(stats["loss_bce"].cpu().item()))
                train_dice.append(float(stats["loss_dice"].cpu().item()))
                train_boundary.append(float(stats["loss_boundary"].cpu().item()))

            epoch += 1
            pbar.update(1)

            train_metrics = {
                "loss": float(np.mean(train_total)) if train_total else float("nan"),
                "loss_bce": float(np.mean(train_bce)) if train_bce else float("nan"),
                "loss_dice": float(np.mean(train_dice)) if train_dice else float("nan"),
                "loss_boundary": float(np.mean(train_boundary)) if train_boundary else float("nan"),
            }
            val_metrics = run_eval_loader(val_loader, use_ema=True)
            test_metrics = run_eval_loader(test_loader, use_ema=True)
            current_lr = optimizer.param_groups[0]["lr"]

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
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
                ckpt_path = "./output/checkpoints/{}_Best_in_val.model".format(save_name)
                torch.save(ckpt_state, ckpt_path)

                if args.save_top_k > 0:
                    ranked_ckpt_path = "./output/checkpoints/{}_epoch{:03d}_val{:.6f}.model".format(
                        save_name,
                        epoch,
                        val_metrics["loss"],
                    )
                    torch.save(ckpt_state, ranked_ckpt_path)
                    ranked_checkpoints.append(
                        {
                            "val_loss": val_metrics["loss"],
                            "epoch": epoch,
                            "path": ranked_ckpt_path,
                        }
                    )
                    ranked_checkpoints.sort(key=lambda item: (item["val_loss"], item["epoch"]))
                    while len(ranked_checkpoints) > args.save_top_k:
                        stale_ckpt = ranked_checkpoints.pop()
                        if os.path.exists(stale_ckpt["path"]):
                            os.remove(stale_ckpt["path"])

                logging.info(
                    "Epoch %d: train = %.6f (bce=%.6f, dice=%.6f, boundary=%.6f) | "
                    "val(ema) = %.6f (bce=%.6f, dice=%.6f, boundary=%.6f) -> saved %s | "
                    "test(ema, leave_species=%s) = %.6f (bce=%.6f, dice=%.6f, boundary=%.6f) | lr = %.8f",
                    epoch,
                    train_metrics["loss"],
                    train_metrics["loss_bce"],
                    train_metrics["loss_dice"],
                    train_metrics["loss_boundary"],
                    val_metrics["loss"],
                    val_metrics["loss_bce"],
                    val_metrics["loss_dice"],
                    val_metrics["loss_boundary"],
                    ckpt_path,
                    leave_species,
                    test_metrics["loss"],
                    test_metrics["loss_bce"],
                    test_metrics["loss_dice"],
                    test_metrics["loss_boundary"],
                    current_lr,
                )
                no_improve_count = 0
            else:
                no_improve_count += 1
                logging.info(
                    "Epoch %d: train = %.6f (bce=%.6f, dice=%.6f, boundary=%.6f) | "
                    "val(ema) = %.6f (bce=%.6f, dice=%.6f, boundary=%.6f), best_val = %.6f @ epoch %d | "
                    "test(ema, leave_species=%s) = %.6f (bce=%.6f, dice=%.6f, boundary=%.6f) | lr = %.8f",
                    epoch,
                    train_metrics["loss"],
                    train_metrics["loss_bce"],
                    train_metrics["loss_dice"],
                    train_metrics["loss_boundary"],
                    val_metrics["loss"],
                    val_metrics["loss_bce"],
                    val_metrics["loss_dice"],
                    val_metrics["loss_boundary"],
                    best_val_loss,
                    best_epoch,
                    leave_species,
                    test_metrics["loss"],
                    test_metrics["loss_bce"],
                    test_metrics["loss_dice"],
                    test_metrics["loss_boundary"],
                    current_lr,
                )

            if no_improve_count >= args.early_stop:
                logging.info("Early stop!")
                break

    if ranked_checkpoints:
        ranked_summary = ", ".join(
            "epoch {}: {:.6f}".format(item["epoch"], item["val_loss"]) for item in ranked_checkpoints
        )
        logging.info("Top-k checkpoints kept: %s", ranked_summary)
