import argparse
import contextlib
import logging
import math
import os
import random
import sys
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm

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


def _install_persistent_diagnostics(save_name: str, log_dir: str = "./output/log") -> None:
    """Write uncaught tracebacks + stderr to files so failures survive lost SSH sessions."""
    os.makedirs(log_dir, exist_ok=True)
    crash_path = os.path.join(log_dir, "crash_{}.log".format(save_name))
    stderr_path = os.path.join(log_dir, "stderr_{}.log".format(save_name))

    def _excepthook(exc_type, exc_value, exc_tb):
        try:
            with open(crash_path, "a", encoding="utf-8") as f:
                f.write("\n" + "=" * 72 + "\n")
                traceback.print_exception(exc_type, exc_value, exc_tb, file=f)
                f.flush()
        except Exception:
            pass
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _excepthook

    _install_persistent_diagnostics._stderr_file = open(  # type: ignore[attr-defined]
        stderr_path, "a", encoding="utf-8", buffering=1
    )

    class _TeeStderr:
        def write(self, data):
            sys.__stderr__.write(data)
            try:
                _install_persistent_diagnostics._stderr_file.write(data)  # type: ignore[attr-defined]
            except Exception:
                pass

        def flush(self):
            sys.__stderr__.flush()
            try:
                _install_persistent_diagnostics._stderr_file.flush()  # type: ignore[attr-defined]
            except Exception:
                pass

        def fileno(self):
            return sys.__stderr__.fileno()

    sys.stderr = _TeeStderr()  # type: ignore[misc]


def set_seed(seed=1998):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _num_groups(num_channels: int, max_groups: int = 8) -> int:
    groups = min(max_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return max(1, groups)


class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.0):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.act = nn.SiLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        scale = self.pool(x)
        scale = self.fc1(scale)
        scale = self.act(scale)
        scale = self.fc2(scale)
        scale = self.gate(scale)
        return x * scale


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0, use_se: bool = False):
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, kernel_size=3, stride=1, dropout=0.0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.se = SqueezeExcite(out_channels) if use_se else nn.Identity()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.se(x)
        x = self.drop(x)
        x = x + residual
        x = self.act(x)
        return x


class TaskHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = None, dropout: float = 0.0):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels
        self.block = ResidualBlock(in_channels, hidden_channels, dropout=dropout, use_se=False)
        self.proj = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.block(x)
        return self.proj(x)


class ModelEma:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        for k, v in model.state_dict().items():
            self.shadow[k] = v.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, model_v in model.state_dict().items():
            shadow_v = self.shadow[k]
            model_v = model_v.detach()
            if torch.is_floating_point(model_v):
                shadow_v.mul_(self.decay).add_(model_v, alpha=1.0 - self.decay)
            else:
                shadow_v.copy_(model_v)

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    @contextlib.contextmanager
    def apply_to(self, model: nn.Module):
        backup = {}
        try:
            for k, v in model.state_dict().items():
                backup[k] = v.detach().clone()
            model.load_state_dict(self.shadow, strict=True)
            yield
        finally:
            model.load_state_dict(backup, strict=True)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


SPECIES_TO_SOURCES = {
    "drosophila": frozenset({"hemi", "fib25", "cremi", "vnc", "isbi2012"}),
    "mouse": frozenset({"ac3", "ac4", "basil", "minnie", "pinky", "axonem_m"}),
    "human": frozenset({"axonem_h"}),
    "zebrafinch": frozenset({"zebrafinch"}),
}
ALL_SOURCE_KEYS = frozenset().union(*SPECIES_TO_SOURCES.values())
LEAVE_SPECIES_CHOICES = tuple(sorted(SPECIES_TO_SOURCES.keys()))
ISOTROPIC_2D_SOURCES = frozenset({"hemi", "fib25"})


def _use_xz_yz_for_source(source_name: str, *, split: str, eval_holdout: bool) -> bool:
    if eval_holdout:
        return False
    return source_name in ISOTROPIC_2D_SOURCES


def _common_source_kwargs(crop_size: int, *, split: str, n_val: int, augment, require_xz_yz: bool):
    return dict(
        split=split,
        crop_size=crop_size,
        require_lsd=True,
        require_xz_yz=require_xz_yz,
        n_val=n_val,
        augment=augment,
    )


def _all_source_specs(
    crop_size: int,
    *,
    split: str,
    n_val: int,
    augment,
    eval_holdout: bool,
    lsd_cache_dir: str,
):
    return [
        (
            "hemi",
            lambda: Dataset_2D_hemi_Train(
                data_dir="./data/funke/hemi/training/",
                **_common_source_kwargs(
                    crop_size,
                    split=split,
                    n_val=n_val,
                    augment=augment,
                    require_xz_yz=_use_xz_yz_for_source("hemi", split=split, eval_holdout=eval_holdout),
                ),
                lsd_cache_dir=lsd_cache_dir,
            ),
        ),
        (
            "fib25",
            lambda: Dataset_2D_fib25_Train(
                data_dir="./data/funke/fib25/training/",
                **_common_source_kwargs(
                    crop_size,
                    split=split,
                    n_val=n_val,
                    augment=augment,
                    require_xz_yz=_use_xz_yz_for_source("fib25", split=split, eval_holdout=eval_holdout),
                ),
                lsd_cache_dir=lsd_cache_dir,
            ),
        ),
        (
            "cremi",
            lambda: Dataset_2D_cremi_Train(
                data_dir="./data/CREMI/",
                **_common_source_kwargs(
                    crop_size,
                    split=split,
                    n_val=n_val,
                    augment=augment,
                    require_xz_yz=_use_xz_yz_for_source("cremi", split=split, eval_holdout=eval_holdout),
                ),
                lsd_cache_dir=lsd_cache_dir,
            ),
        ),
        (
            "vnc",
            lambda: Dataset_2D_VNC_Train(
                data_dir="./data/groundtruth-drosophila-vnc-master/stack1/",
                **_common_source_kwargs(
                    crop_size,
                    split=split,
                    n_val=n_val,
                    augment=augment,
                    require_xz_yz=_use_xz_yz_for_source("vnc", split=split, eval_holdout=eval_holdout),
                ),
                lsd_cache_dir=lsd_cache_dir,
            ),
        ),
        (
            "isbi2012",
            lambda: Dataset_2D_isbi2012_Train(
                data_dir="./data/ISBI-2012/",
                **_common_source_kwargs(
                    crop_size,
                    split=split,
                    n_val=n_val,
                    augment=augment,
                    require_xz_yz=_use_xz_yz_for_source("isbi2012", split=split, eval_holdout=eval_holdout),
                ),
                lsd_cache_dir=lsd_cache_dir,
            ),
        ),
        (
            "ac3",
            lambda: Dataset_2D_ac3_Train(
                data_dir="./data/AC3/",
                **_common_source_kwargs(
                    crop_size,
                    split=split,
                    n_val=n_val,
                    augment=augment,
                    require_xz_yz=_use_xz_yz_for_source("ac3", split=split, eval_holdout=eval_holdout),
                ),
                lsd_cache_dir=lsd_cache_dir,
            ),
        ),
        (
            "ac4",
            lambda: Dataset_2D_ac4_Train(
                data_dir="./data/AC4/",
                **_common_source_kwargs(
                    crop_size,
                    split=split,
                    n_val=n_val,
                    augment=augment,
                    require_xz_yz=_use_xz_yz_for_source("ac4", split=split, eval_holdout=eval_holdout),
                ),
                lsd_cache_dir=lsd_cache_dir,
            ),
        ),
        (
            "basil",
            lambda: Dataset_2D_basil_Train(
                data_dir="./data/MICrONS/Neuron_zarr/basil/",
                **_common_source_kwargs(
                    crop_size,
                    split=split,
                    n_val=n_val,
                    augment=augment,
                    require_xz_yz=_use_xz_yz_for_source("basil", split=split, eval_holdout=eval_holdout),
                ),
                lsd_cache_dir=lsd_cache_dir,
            ),
        ),
        (
            "minnie",
            lambda: Dataset_2D_minnie_Train(
                data_dir="./data/MICrONS/Neuron_zarr/minnie/",
                **_common_source_kwargs(
                    crop_size,
                    split=split,
                    n_val=n_val,
                    augment=augment,
                    require_xz_yz=_use_xz_yz_for_source("minnie", split=split, eval_holdout=eval_holdout),
                ),
                lsd_cache_dir=lsd_cache_dir,
            ),
        ),
        (
            "pinky",
            lambda: Dataset_2D_pinky_Train(
                data_dir="./data/MICrONS/Neuron_zarr/pinky/",
                **_common_source_kwargs(
                    crop_size,
                    split=split,
                    n_val=n_val,
                    augment=augment,
                    require_xz_yz=_use_xz_yz_for_source("pinky", split=split, eval_holdout=eval_holdout),
                ),
                lsd_cache_dir=lsd_cache_dir,
            ),
        ),
        (
            "axonem_m",
            lambda: Dataset_2D_axonem_m_Train(
                data_dir="./data/AxonEM/EM30-M-axon-train-9vol/",
                **_common_source_kwargs(
                    crop_size,
                    split=split,
                    n_val=n_val,
                    augment=augment,
                    require_xz_yz=_use_xz_yz_for_source("axonem_m", split=split, eval_holdout=eval_holdout),
                ),
                lsd_cache_dir=lsd_cache_dir,
            ),
        ),
        (
            "axonem_h",
            lambda: Dataset_2D_axonem_h_Train(
                data_dir="./data/AxonEM/EM30-H-axon-train-9vol/",
                **_common_source_kwargs(
                    crop_size,
                    split=split,
                    n_val=n_val,
                    augment=augment,
                    require_xz_yz=_use_xz_yz_for_source("axonem_h", split=split, eval_holdout=eval_holdout),
                ),
                lsd_cache_dir=lsd_cache_dir,
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
                require_lsd=True,
                require_xz_yz=_use_xz_yz_for_source("zebrafinch", split=split, eval_holdout=eval_holdout),
                augment=augment,
                lsd_cache_dir=lsd_cache_dir,
            ),
        ),
    ]


def _concat_pool_for_keys(
    allowed_keys: frozenset,
    crop_size: int,
    *,
    split: str,
    n_val: int,
    augment,
    tag: str,
    eval_holdout: bool,
    lsd_cache_dir: str,
):
    parts = []
    for name, factory in _all_source_specs(
        crop_size,
        split=split,
        n_val=n_val,
        augment=augment,
        eval_holdout=eval_holdout,
        lsd_cache_dir=lsd_cache_dir,
    ):
        if name not in allowed_keys:
            continue
        try:
            ds = factory()
        except Exception as exc:
            raise RuntimeError("Failed to load dataset {} ({}): {}".format(name, tag, exc)) from exc
        if len(ds) == 0:
            raise RuntimeError("Loaded empty dataset {} ({})".format(name, tag))
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


def collate_fn_2d_live_lsd_minimal(batch):
    raw = torch.from_numpy(np.stack([item[0] for item in batch], axis=0).astype(np.float32, copy=False))
    affinity = torch.from_numpy(np.stack([item[1] for item in batch], axis=0).astype(np.float32, copy=False))
    lsds = torch.from_numpy(np.stack([item[2] for item in batch], axis=0).astype(np.float32, copy=False))
    return raw, affinity, lsds


def set_concat_dataset_attr(dataset, attr_name: str, value) -> None:
    if isinstance(dataset, ConcatDataset):
        for part in dataset.datasets:
            set_concat_dataset_attr(part, attr_name, value)
        return
    setattr(dataset, attr_name, value)


def _validate_leave_species(leave_species: str) -> None:
    if leave_species not in SPECIES_TO_SOURCES:
        raise ValueError("leave_species must be one of {}, got {!r}".format(LEAVE_SPECIES_CHOICES, leave_species))


def _filtered_sources_for_species(leave_species: str) -> frozenset:
    _validate_leave_species(leave_species)
    return SPECIES_TO_SOURCES[leave_species]


def build_train_val_pool_leave_one_species(
    leave_species: str,
    crop_size: int,
    *,
    split: str,
    n_val_holdout: int = 16,
    augment=None,
    lsd_cache_dir: str = os.path.abspath("./LSD_cache"),
):
    _validate_leave_species(leave_species)
    allowed = frozenset(ALL_SOURCE_KEYS - SPECIES_TO_SOURCES[leave_species])
    return _concat_pool_for_keys(
        allowed,
        crop_size=crop_size,
        split=split,
        n_val=n_val_holdout,
        augment=augment,
        tag="train_val leave_out={}".format(leave_species),
        eval_holdout=False,
        lsd_cache_dir=lsd_cache_dir,
    )


def build_test_pool_leave_one_species(
    leave_species: str, crop_size: int, *, lsd_cache_dir: str = os.path.abspath("./LSD_cache")
):
    allowed = _filtered_sources_for_species(leave_species)
    return _concat_pool_for_keys(
        allowed,
        crop_size,
        split="train",
        n_val=0,
        augment=False,
        tag="test holdout species={}".format(leave_species),
        eval_holdout=True,
        lsd_cache_dir=lsd_cache_dir,
    )


class ConvGNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.block1 = ResidualBlock(out_channels + skip_channels, out_channels, dropout=dropout, use_se=False)
        self.block2 = ResidualBlock(out_channels, out_channels, dropout=dropout, use_se=False)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.proj(x)
        x = torch.cat([skip, x], dim=1)
        x = self.block1(x)
        x = self.block2(x)
        return x


class MultiScaleFusion(nn.Module):
    """Project multi-resolution feature maps to a common width and fuse at full resolution."""

    def __init__(self, in_channels_list, out_channels: int, dropout: float = 0.0, use_se: bool = True):
        super().__init__()
        self.projs = nn.ModuleList([ConvGNAct(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list])
        fused_channels = out_channels * len(in_channels_list)
        self.refine = nn.Sequential(
            ResidualBlock(fused_channels, out_channels, dropout=dropout, use_se=use_se),
            ResidualBlock(out_channels, out_channels, dropout=dropout, use_se=False),
        )

    def forward(self, features):
        if not features:
            raise ValueError("features must be non-empty")
        target_size = features[-1].shape[-2:]
        projected = []
        for feat, proj in zip(features, self.projs):
            feat = proj(feat)
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            projected.append(feat)
        fused = torch.cat(projected, dim=1)
        return self.refine(fused)


class FusionDecoder(nn.Module):
    def __init__(self, bottleneck_channels: int, skip_channels, decoder_dropouts):
        super().__init__()
        in_channels = bottleneck_channels
        self.blocks = nn.ModuleList()
        for skip_ch, dropout in zip(reversed(skip_channels), decoder_dropouts):
            self.blocks.append(UpBlock(in_channels, skip_ch, skip_ch, dropout=dropout))
            in_channels = skip_ch

    def forward(self, x, skips):
        outputs = []
        for block, skip in zip(self.blocks, reversed(skips)):
            x = block(x, skip)
            outputs.append(x)
        return x, outputs


class ACRLSDneo(nn.Module):
    """
    Shared encoder + dual decoders + multi-scale feature fusion.

    LSD and affinity each decode from the same encoder bottleneck. Final prediction uses:
    - decoder multi-scale fusion
    - encoder skip multi-scale fusion
    - affinity conditioned on fused LSD features and probabilities
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_width: int = 32,
        encoder_widths=(32, 64, 128, 256),
        bottleneck_channels: int = 384,
        fusion_width: int = 48,
        detach_lsd_for_affinity: bool = True,
    ):
        super().__init__()
        if encoder_widths[0] != base_width:
            raise ValueError("encoder_widths[0] must equal base_width")

        self.detach_lsd_for_affinity = detach_lsd_for_affinity

        self.stem = nn.Sequential(
            ResidualBlock(in_channels, base_width, dropout=0.0, use_se=False),
            ResidualBlock(base_width, base_width, dropout=0.0, use_se=False),
        )

        encoder_dropouts = [0.0, 0.03, 0.05]
        self.encoder_stages = nn.ModuleList()
        in_ch = base_width
        for out_ch, dropout in zip(encoder_widths[1:], encoder_dropouts):
            self.encoder_stages.append(
                nn.Sequential(
                    DownsampleBlock(in_ch, out_ch),
                    ResidualBlock(out_ch, out_ch, dropout=dropout, use_se=False),
                    ResidualBlock(out_ch, out_ch, dropout=dropout, use_se=False),
                )
            )
            in_ch = out_ch

        self.bottleneck = nn.Sequential(
            DownsampleBlock(encoder_widths[-1], bottleneck_channels),
            ResidualBlock(bottleneck_channels, bottleneck_channels, dropout=0.10, use_se=True),
            ResidualBlock(bottleneck_channels, bottleneck_channels, dropout=0.10, use_se=True),
        )

        decoder_dropouts = [0.08, 0.05, 0.03, 0.0]
        self.lsd_decoder = FusionDecoder(bottleneck_channels, encoder_widths, decoder_dropouts)
        self.affinity_decoder = FusionDecoder(bottleneck_channels, encoder_widths, decoder_dropouts)

        decoder_scale_channels = list(reversed(encoder_widths))
        encoder_scale_channels = list(reversed(encoder_widths))

        self.encoder_fusion = MultiScaleFusion(encoder_scale_channels, fusion_width, dropout=0.03, use_se=False)
        self.lsd_scale_fusion = MultiScaleFusion(decoder_scale_channels, fusion_width, dropout=0.05, use_se=True)
        self.affinity_scale_fusion = MultiScaleFusion(decoder_scale_channels, fusion_width, dropout=0.05, use_se=True)

        self.lsd_refine = nn.Sequential(
            ResidualBlock(fusion_width * 2, fusion_width, dropout=0.03, use_se=True),
            SqueezeExcite(fusion_width),
        )
        self.lsd_head = TaskHead(fusion_width, 6, hidden_channels=fusion_width, dropout=0.02)

        affinity_in_channels = fusion_width * 3 + base_width + in_channels + 6
        self.affinity_refine = nn.Sequential(
            ResidualBlock(affinity_in_channels, fusion_width, dropout=0.05, use_se=True),
            ResidualBlock(fusion_width, fusion_width, dropout=0.03, use_se=False),
            SqueezeExcite(fusion_width),
        )
        self.affinity_head = TaskHead(fusion_width, 2, hidden_channels=fusion_width, dropout=0.02)

    def _encode(self, x):
        skips = []
        x = self.stem(x)
        skips.append(x)
        for stage in self.encoder_stages:
            x = stage(x)
            skips.append(x)
        x = self.bottleneck(x)
        return x, skips

    def forward_with_intermediates(self, x):
        """Run the exact ACRLSD forward path and expose intermediate tensors."""
        bottleneck, skips = self._encode(x)

        lsd_last, lsd_scales = self.lsd_decoder(bottleneck, skips)
        affinity_last, affinity_scales = self.affinity_decoder(bottleneck, skips)

        encoder_fused = self.encoder_fusion(list(reversed(skips)))
        lsd_fused = self.lsd_scale_fusion(lsd_scales)
        lsd_feat = self.lsd_refine(torch.cat([lsd_fused, encoder_fused], dim=1))
        lsd_logits = self.lsd_head(lsd_feat)
        lsd_prob = torch.sigmoid(lsd_logits)

        affinity_fused = self.affinity_scale_fusion(affinity_scales)
        lsd_for_affinity = lsd_feat.detach() if self.detach_lsd_for_affinity else lsd_feat
        prob_for_affinity = lsd_prob.detach() if self.detach_lsd_for_affinity else lsd_prob

        if affinity_last.shape[-2:] != lsd_feat.shape[-2:]:
            affinity_last = F.interpolate(affinity_last, size=lsd_feat.shape[-2:], mode="bilinear", align_corners=False)

        affinity_input = torch.cat(
            [affinity_fused, affinity_last, encoder_fused, x, lsd_for_affinity, prob_for_affinity], dim=1
        )
        affinity_feat = self.affinity_refine(affinity_input)
        affinity_logits = self.affinity_head(affinity_feat)

        return {
            "encoder_fused": encoder_fused,
            "lsd_last": lsd_last,
            "lsd_scales": lsd_scales,
            "lsd_feat": lsd_feat,
            "lsd_logits": lsd_logits,
            "lsd_prob": lsd_prob,
            "affinity_last": affinity_last,
            "affinity_scales": affinity_scales,
            "affinity_feat": affinity_feat,
            "affinity_logits": affinity_logits,
        }

    def forward(self, x):
        outputs = self.forward_with_intermediates(x)
        return outputs["lsd_logits"], outputs["affinity_logits"]


def model_step(
    model,
    lsd_loss_fn,
    affinity_loss_fn,
    optimizer,
    raw,
    gt_lsds,
    gt_affinity,
    activation,
    device,
    *,
    train_step=True,
    scheduler=None,
    scaler=None,
    amp_enabled=False,
    grad_clip_norm=None,
):
    if train_step:
        optimizer.zero_grad(set_to_none=True)

    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
        lsd_logits, affinity_logits = model(raw)
        lsd_output = activation(lsd_logits)
        affinity_output = activation(affinity_logits)
        loss_lsd = lsd_loss_fn(lsd_output, gt_lsds)
        loss_affinity = affinity_loss_fn(affinity_output, gt_affinity)
        loss_value = loss_lsd + loss_affinity

    if train_step:
        if scaler is not None and amp_enabled:
            scaler.scale(loss_value).backward()
            scaler.unscale_(optimizer)
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_value.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

    return loss_value, {
        "loss_lsd": loss_lsd.detach(),
        "loss_affinity": loss_affinity.detach(),
        "pred_lsds": lsd_output,
        "lsds_logits": lsd_logits,
        "pred_affinity": affinity_output,
        "affinity_logits": affinity_logits,
    }


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Train ACRLSD 2D neo with shared encoder, dual decoders, and multi-scale feature fusion."
    )
    parser.add_argument("--leave-species", type=str, default="human", choices=LEAVE_SPECIES_CHOICES)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=40)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--n-val-holdout", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--cosine-eta-min-ratio", type=float, default=0.001)
    parser.add_argument("--early-stop", type=int, default=15)
    parser.add_argument("--base-width", type=int, default=32)
    parser.add_argument("--bottleneck-channels", type=int, default=384)
    parser.add_argument("--fusion-width", type=int, default=64)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument(
        "--save-top-k",
        type=int,
        default=3,
        help="Keep the best K epoch-specific checkpoints ranked by val loss in addition to the rolling best checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=1998)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision. By default AMP is enabled.")
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    leave_species = args.leave_species

    training_epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    val_batch_size = args.val_batch_size
    num_workers = args.num_workers
    crop_size = args.crop_size
    n_val_holdout = args.n_val_holdout
    weight_decay = args.weight_decay
    warmup_epochs = args.warmup_epochs
    cosine_eta_min_ratio = args.cosine_eta_min_ratio
    early_stop_count = args.early_stop
    grad_clip_norm = args.grad_clip_norm
    ema_decay = args.ema_decay
    save_top_k = max(0, args.save_top_k)

    save_name = "ACRLSD_2D_leaveout_{}_holdoutVal{}_neo".format(leave_species, n_val_holdout)
    _install_persistent_diagnostics(save_name)

    set_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    amp_enabled = (device.type == "cuda") and (not args.no_amp)

    model = ACRLSDneo(
        in_channels=1,
        base_width=args.base_width,
        encoder_widths=(args.base_width, args.base_width * 2, args.base_width * 4, args.base_width * 8),
        bottleneck_channels=args.bottleneck_channels,
        fusion_width=args.fusion_width,
        detach_lsd_for_affinity=True,
    ).to(device)
    ema = ModelEma(model, decay=ema_decay)

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
        "Diagnostics: uncaught exceptions -> ./output/log/crash_%s.log | stderr copy -> ./output/log/stderr_%s.log",
        save_name,
        save_name,
    )
    train_dataset = build_train_val_pool_leave_one_species(
        leave_species,
        crop_size=crop_size,
        split="train",
        n_val_holdout=n_val_holdout,
        augment=True,
    )
    val_dataset = build_train_val_pool_leave_one_species(
        leave_species,
        crop_size=crop_size,
        split="val",
        n_val_holdout=n_val_holdout,
        augment=False,
    )
    test_dataset = build_test_pool_leave_one_species(
        leave_species,
        crop_size,
    )

    set_concat_dataset_attr(train_dataset, "minimal_output", True)
    set_concat_dataset_attr(val_dataset, "minimal_output", True)
    set_concat_dataset_attr(test_dataset, "minimal_output", True)

    train_gen = torch.Generator().manual_seed(args.seed + 7)
    val_gen = torch.Generator().manual_seed(args.seed + 8)
    test_gen = torch.Generator().manual_seed(args.seed + 9)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=collate_fn_2d_live_lsd_minimal,
        generator=train_gen,
        worker_init_fn=seed_worker,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn_2d_live_lsd_minimal,
        generator=val_gen,
        worker_init_fn=seed_worker,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn_2d_live_lsd_minimal,
        generator=test_gen,
        worker_init_fn=seed_worker,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    steps_per_epoch = len(train_loader)
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    max_train_steps = max(1, training_epochs * steps_per_epoch)

    def _lr_lambda(last_epoch: int):
        if last_epoch < warmup_steps:
            return float(last_epoch + 1) / float(warmup_steps)
        t = last_epoch - warmup_steps
        T = max(1, max_train_steps - warmup_steps)
        progress = min(float(t) / float(T), 1.0)
        cos_part = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cosine_eta_min_ratio + (1.0 - cosine_eta_min_ratio) * cos_part

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    activation = torch.nn.Sigmoid()
    lsd_loss_fn = torch.nn.MSELoss().to(device)
    affinity_loss_fn = torch.nn.MSELoss().to(device)

    logging.info(
        """Starting training:
    leave_species:         %s (test sources: %s)
    training_epochs:       %s
    Train samples:         %d
    Val samples:           %d
    Test samples:          %d
    Holdout slices:        %d
    Batch size:            %s
    Val/Test batch size:   %s
    Learning rate:         %s
    Optimizer:             AdamW (weight_decay=%s)
    Losses:                LSD=%s | Affinity=%s
    LR schedule:           linear warmup %s epochs (~%s steps) + cosine to %.4f * base lr
    EMA decay:             %s
    Grad clip norm:        %s
    Save top-k ckpts:      %s
    AMP enabled:           %s
    Base width:            %s
    Bottleneck channels:   %s
    Fusion width:          %s
    Parameters (M):        %.2f
    num_workers:           %s
    Device:                %s
    """,
        leave_species,
        ", ".join(sorted(_filtered_sources_for_species(leave_species))),
        training_epochs,
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
        n_val_holdout,
        batch_size,
        val_batch_size,
        learning_rate,
        weight_decay,
        "MSE(sigmoid(lsd_logits), gt_lsds)",
        "MSELoss(sigmoid(affinity_logits), gt_affinity)",
        warmup_epochs,
        warmup_steps,
        cosine_eta_min_ratio,
        ema_decay,
        grad_clip_norm,
        save_top_k,
        amp_enabled,
        args.base_width,
        args.bottleneck_channels,
        args.fusion_width,
        count_parameters(model) / 1e6,
        num_workers,
        device.type,
    )

    model.train()
    lsd_loss_fn.train()
    affinity_loss_fn.train()
    epoch = 0
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve_count = 0
    ranked_checkpoints = []

    def run_eval_loader(loader, use_ema=True):
        model.eval()
        acc_loss = []
        weight_scope = ema.apply_to(model) if use_ema else contextlib.nullcontext()
        with weight_scope:
            for raw, gt_affinity, gt_lsds in loader:
                raw = raw.to(device=device, dtype=torch.float32, non_blocking=pin_memory)
                gt_lsds = gt_lsds.to(device=device, dtype=torch.float32, non_blocking=pin_memory)
                gt_affinity = gt_affinity.to(device=device, dtype=torch.float32, non_blocking=pin_memory)
                with torch.no_grad():
                    loss_value, _ = model_step(
                        model,
                        lsd_loss_fn,
                        affinity_loss_fn,
                        optimizer,
                        raw,
                        gt_lsds,
                        gt_affinity,
                        activation,
                        device,
                        train_step=False,
                        scheduler=None,
                        scaler=None,
                        amp_enabled=amp_enabled,
                        grad_clip_norm=None,
                    )
                acc_loss.append(float(loss_value.detach().cpu().item()))
        return float(np.mean(acc_loss)) if acc_loss else float("nan")

    with tqdm(total=training_epochs) as pbar:
        while epoch < training_epochs:
            model.train()
            train_losses = []

            for raw, gt_affinity, gt_lsds in train_loader:
                raw = raw.to(device=device, dtype=torch.float32, non_blocking=pin_memory)
                gt_lsds = gt_lsds.to(device=device, dtype=torch.float32, non_blocking=pin_memory)
                gt_affinity = gt_affinity.to(device=device, dtype=torch.float32, non_blocking=pin_memory)

                loss_value, _ = model_step(
                    model,
                    lsd_loss_fn,
                    affinity_loss_fn,
                    optimizer,
                    raw,
                    gt_lsds,
                    gt_affinity,
                    activation,
                    device,
                    train_step=True,
                    scheduler=scheduler,
                    scaler=scaler,
                    amp_enabled=amp_enabled,
                    grad_clip_norm=grad_clip_norm,
                )
                ema.update(model)
                train_losses.append(float(loss_value.detach().cpu().item()))

            epoch += 1
            pbar.update(1)

            train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
            val_loss = run_eval_loader(val_loader, use_ema=True)
            test_loss = run_eval_loader(test_loader, use_ema=True)
            current_lr = optimizer.param_groups[0]["lr"]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                os.makedirs("./output/checkpoints", exist_ok=True)
                ckpt_state = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "test_loss": test_loss,
                    "learning_rate": current_lr,
                    "config": vars(args),
                }
                ckpt_path = "./output/checkpoints/{}_Best_in_val.model".format(save_name)
                torch.save(ckpt_state, ckpt_path)

                if save_top_k > 0:
                    ranked_ckpt_path = "./output/checkpoints/{}_epoch{:03d}_val{:.6f}.model".format(
                        save_name,
                        epoch,
                        val_loss,
                    )
                    torch.save(ckpt_state, ranked_ckpt_path)
                    ranked_checkpoints.append(
                        {
                            "val_loss": val_loss,
                            "epoch": epoch,
                            "path": ranked_ckpt_path,
                        }
                    )
                    ranked_checkpoints.sort(key=lambda item: (item["val_loss"], item["epoch"]))
                    while len(ranked_checkpoints) > save_top_k:
                        stale_ckpt = ranked_checkpoints.pop()
                        if os.path.exists(stale_ckpt["path"]):
                            os.remove(stale_ckpt["path"])
                logging.info(
                    "Epoch %d: train = %.6f | val(ema) = %.6f -> saved %s | test(ema, leave_species=%s) = %.6f | lr = %.8f | kept_top_k = %d",
                    epoch,
                    train_loss,
                    val_loss,
                    ckpt_path,
                    leave_species,
                    test_loss,
                    current_lr,
                    min(len(ranked_checkpoints), save_top_k),
                )
                no_improve_count = 0
            else:
                no_improve_count += 1
                logging.info(
                    "Epoch %d: train = %.6f | val(ema) = %.6f (no improvement), best_val = %.6f @ epoch %d | test(ema) = %.6f | lr = %.8f",
                    epoch,
                    train_loss,
                    val_loss,
                    best_val_loss,
                    best_epoch,
                    test_loss,
                    current_lr,
                )

            if no_improve_count >= early_stop_count:
                logging.info("Early stop!")
                break

    if ranked_checkpoints:
        ranked_summary = ", ".join(
            "epoch {}: {:.6f}".format(item["epoch"], item["val_loss"]) for item in ranked_checkpoints
        )
        logging.info("Retained top-%d val checkpoints: %s", len(ranked_checkpoints), ranked_summary)
