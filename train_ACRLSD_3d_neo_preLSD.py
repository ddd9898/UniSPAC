import contextlib
import logging
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler, Subset
from tqdm.auto import tqdm

from train_ACRLSD_3d_neo import (
    ACRLSDneo3D,
    ALL_SOURCE_KEYS,
    LEAVE_SPECIES_CHOICES,
    ModelEma,
    SPECIES_TO_SOURCES,
    _filtered_sources_for_species,
    _install_persistent_diagnostics,
    _validate_leave_species,
    build_argparser as _base_build_argparser,
    count_parameters,
    model_step,
    seed_worker,
    set_seed,
)
from utils.dataloader_preLSD import (
    DEFAULT_LSD_CACHE_DIR,
    ZEBRA_PATCH_FILES,
    set_getitem_profile,
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
)


_FORK_SHARED_STATE = {}


def build_argparser():
    parser = _base_build_argparser()
    parser.description = (
        "Train ACRLSD 3D neo with leave-species-out splits using precomputed LSD caches."
    )
    parser.set_defaults(batch_size=8, num_workers=1)
    parser.add_argument(
        "--lsd-cache-dir",
        type=str,
        default=DEFAULT_LSD_CACHE_DIR,
        help="Directory with precomputed LSD (.npy from process_lsd.py or .zarr from tran_lsd_to_zarr.py)",
    )
    parser.add_argument(
        "--lsd-cache-storage",
        type=str,
        choices=("zarr", "npy"),
        default="zarr",
        help="On-disk LSD format: chunked zarr (recommended) or legacy numpy .npy",
    )
    parser.add_argument(
        "--no-preload-lsd-to-ram",
        action="store_true",
        help=(
            "Disable eager LSD preload. By default this script preloads LSD caches into RAM "
            "because that has been faster on the current storage setup."
        ),
    )
    parser.add_argument(
        "--profile-data",
        type=int,
        default=0,
        metavar="N",
        help=(
            "If N>0: log per-stage __getitem__ timings and first batches' data_wait / tensor H2D / model_step. "
            "Lines are written to sys.stderr, which train_ACRLSD_3d_neo_preLSD tees to "
            "./output/log/stderr_<run>.log (after _install_persistent_diagnostics). "
            "With num_workers=1 data profiling is emitted by the worker process; 0=off."
        ),
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help=(
            "Single-node GPU count. If >1, uses fork-based DDP so RAM-preloaded LSD is "
            "loaded once in the parent and shared read-only across ranks."
        ),
    )
    parser.add_argument(
        "--ddp-port",
        type=int,
        default=29500,
        help="Single-node DDP TCP port used when --num-gpus > 1.",
    )
    return parser


def _all_source_specs(
    crop_size: int,
    num_slices: int,
    *,
    split: str,
    n_val: int,
    augment,
    lsd_cache_dir: str,
    lsd_storage: str = "zarr",
    preload_lsd_to_ram: bool = False,
):
    kw = dict(
        split=split,
        crop_size=crop_size,
        num_slices=num_slices,
        require_lsd=True,
        require_xz_yz=True,
        n_val=n_val,
        lsd_cache_dir=lsd_cache_dir,
        lsd_storage=lsd_storage,
        augment=augment,
        preload_lsd_to_ram=preload_lsd_to_ram,
        # neo preLSD path only uses raw / gt_affinity / gt_lsds; skip SAM-style prompts.
        need_point_map=False,
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
                **kw
            ),
        ),
    ]


def _concat_pool_for_keys(
    allowed_keys: frozenset,
    crop_size: int,
    num_slices: int,
    *,
    split: str,
    n_val: int,
    augment,
    tag: str,
    lsd_cache_dir: str,
    lsd_storage: str = "zarr",
    preload_lsd_to_ram: bool = False,
):
    parts = []
    for name, factory in _all_source_specs(
        crop_size,
        num_slices,
        split=split,
        n_val=n_val,
        augment=augment,
        lsd_cache_dir=lsd_cache_dir,
        lsd_storage=lsd_storage,
        preload_lsd_to_ram=preload_lsd_to_ram,
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
        parts.append(ds)
        logging.info("Loaded %s (%s): %d samples", name, tag, len(ds))
    if not parts:
        raise RuntimeError(
            "No datasets could be loaded for keys {} ({}); check paths under ./data/ and LSD cache under {}.".format(
                sorted(allowed_keys), tag, lsd_cache_dir
            )
        )
    return ConcatDataset(parts)


def collate_fn_3d_prelsd_minimal(batch):
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


def build_train_val_pool_leave_one_species(
    leave_species: str,
    crop_size: int,
    num_slices: int,
    *,
    split: str,
    n_val_holdout: int = 16,
    augment=None,
    lsd_cache_dir: str = DEFAULT_LSD_CACHE_DIR,
    lsd_storage: str = "zarr",
    preload_lsd_to_ram: bool = False,
):
    _validate_leave_species(leave_species)
    allowed = frozenset(ALL_SOURCE_KEYS - SPECIES_TO_SOURCES[leave_species])
    return _concat_pool_for_keys(
        allowed,
        crop_size=crop_size,
        num_slices=num_slices,
        split=split,
        n_val=n_val_holdout,
        augment=augment,
        tag="train_val leave_out={}".format(leave_species),
        lsd_cache_dir=lsd_cache_dir,
        lsd_storage=lsd_storage,
        preload_lsd_to_ram=preload_lsd_to_ram,
    )


def build_test_pool_leave_one_species(
    leave_species: str,
    crop_size: int,
    num_slices: int,
    *,
    lsd_cache_dir: str = DEFAULT_LSD_CACHE_DIR,
    lsd_storage: str = "zarr",
    preload_lsd_to_ram: bool = False,
):
    allowed = _filtered_sources_for_species(leave_species)
    return _concat_pool_for_keys(
        allowed,
        crop_size=crop_size,
        num_slices=num_slices,
        split="train",
        n_val=0,
        augment=False,
        tag="test holdout species={}".format(leave_species),
        lsd_cache_dir=lsd_cache_dir,
        lsd_storage=lsd_storage,
        preload_lsd_to_ram=preload_lsd_to_ram,
    )


def is_main_process(rank: int) -> bool:
    return rank == 0


def setup_root_logger(save_name: str, rank: int) -> logging.Logger:
    logger = logging.getLogger()
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    logger.setLevel(logging.INFO if is_main_process(rank) else logging.WARNING)
    if not is_main_process(rank):
        return logger

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
    return logger


def build_prelsd_datasets(
    *,
    leave_species: str,
    crop_size: int,
    num_slices: int,
    n_val_holdout: int,
    lsd_cache_dir: str,
    lsd_storage: str,
    preload_lsd_to_ram: bool,
):
    train_dataset = build_train_val_pool_leave_one_species(
        leave_species,
        crop_size=crop_size,
        num_slices=num_slices,
        split="train",
        n_val_holdout=n_val_holdout,
        augment=True,
        lsd_cache_dir=lsd_cache_dir,
        lsd_storage=lsd_storage,
        preload_lsd_to_ram=preload_lsd_to_ram,
    )
    val_dataset = build_train_val_pool_leave_one_species(
        leave_species,
        crop_size=crop_size,
        num_slices=num_slices,
        split="val",
        n_val_holdout=n_val_holdout,
        augment=False,
        lsd_cache_dir=lsd_cache_dir,
        lsd_storage=lsd_storage,
        preload_lsd_to_ram=preload_lsd_to_ram,
    )
    test_dataset = build_test_pool_leave_one_species(
        leave_species,
        crop_size,
        num_slices,
        lsd_cache_dir=lsd_cache_dir,
        lsd_storage=lsd_storage,
        preload_lsd_to_ram=preload_lsd_to_ram,
    )
    set_concat_dataset_attr(train_dataset, "minimal_output", True)
    set_concat_dataset_attr(val_dataset, "minimal_output", True)
    set_concat_dataset_attr(test_dataset, "minimal_output", True)
    return train_dataset, val_dataset, test_dataset


def shard_eval_dataset(dataset, rank: int, world_size: int):
    if world_size <= 1:
        return dataset
    return Subset(dataset, range(rank, len(dataset), world_size))


def build_loaders(
    *,
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size: int,
    val_batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    rank: int,
    world_size: int,
):
    train_gen = torch.Generator().manual_seed(seed + 7 + rank)
    val_gen = torch.Generator().manual_seed(seed + 8 + rank)
    test_gen = torch.Generator().manual_seed(seed + 9 + rank)

    train_sampler = None
    train_shuffle = True
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        train_shuffle = False

    val_dataset_rank = shard_eval_dataset(val_dataset, rank, world_size)
    test_dataset_rank = shard_eval_dataset(test_dataset, rank, world_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=collate_fn_3d_prelsd_minimal,
        generator=train_gen,
        worker_init_fn=seed_worker,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset_rank,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn_3d_prelsd_minimal,
        generator=val_gen,
        worker_init_fn=seed_worker,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_dataset_rank,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn_3d_prelsd_minimal,
        generator=test_gen,
        worker_init_fn=seed_worker,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return train_loader, val_loader, test_loader, train_sampler


def distributed_weighted_mean(loss_sum: float, sample_count: int, device, world_size: int) -> float:
    if world_size <= 1:
        return float(loss_sum / sample_count) if sample_count else float("nan")
    stats = torch.tensor([loss_sum, float(sample_count)], device=device, dtype=torch.float64)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    total_count = int(stats[1].item())
    if total_count <= 0:
        return float("nan")
    return float(stats[0].item() / total_count)


def train_worker(rank: int, world_size: int, args, save_name: str) -> None:
    multi_gpu = world_size > 1
    if multi_gpu:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(args.ddp_port)
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:{}".format(args.ddp_port),
            world_size=world_size,
            rank=rank,
        )

    if is_main_process(rank):
        _install_persistent_diagnostics(save_name)
    setup_root_logger(save_name, rank)
    set_getitem_profile(args.profile_data if is_main_process(rank) else 0)
    set_seed(args.seed)

    leave_species = args.leave_species
    training_epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    val_batch_size = args.val_batch_size
    num_workers = args.num_workers
    crop_size = args.crop_size
    num_slices = args.num_slices
    n_val_holdout = args.n_val_holdout
    weight_decay = args.weight_decay
    warmup_epochs = args.warmup_epochs
    cosine_eta_min_ratio = args.cosine_eta_min_ratio
    early_stop_count = args.early_stop
    grad_clip_norm = args.grad_clip_norm
    ema_decay = args.ema_decay
    save_top_k = max(0, args.save_top_k)
    preload_lsd_to_ram = multi_gpu or (not bool(args.no_preload_lsd_to_ram))

    device = torch.device("cuda:{}".format(rank) if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    amp_enabled = (device.type == "cuda") and (not args.no_amp)

    if multi_gpu:
        train_dataset = _FORK_SHARED_STATE["train_dataset"]
        val_dataset = _FORK_SHARED_STATE["val_dataset"]
        test_dataset = _FORK_SHARED_STATE["test_dataset"]
    else:
        train_dataset, val_dataset, test_dataset = build_prelsd_datasets(
            leave_species=leave_species,
            crop_size=crop_size,
            num_slices=num_slices,
            n_val_holdout=n_val_holdout,
            lsd_cache_dir=args.lsd_cache_dir,
            lsd_storage=args.lsd_cache_storage,
            preload_lsd_to_ram=preload_lsd_to_ram,
        )

    if is_main_process(rank):
        logging.info(
            "Diagnostics: uncaught exceptions -> ./output/log/crash_%s.log | stderr copy -> ./output/log/stderr_%s.log",
            save_name,
            save_name,
        )
        if multi_gpu:
            logging.info(
                "Using single-node fork-based DDP across %d GPU(s); parent-preloaded LSD is inherited read-only by all ranks.",
                world_size,
            )
        if args.profile_data > 0:
            logging.info(
                "DataLoader timing (--profile-data=%d) is printed to stderr → ./output/log/stderr_%s.log",
                args.profile_data,
                save_name,
            )

    train_loader, val_loader, test_loader, train_sampler = build_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
    )

    model_raw = ACRLSDneo3D(
        in_channels=1,
        base_width=args.base_width,
        encoder_widths=(args.base_width, args.base_width * 2, args.base_width * 4),
        bottleneck_channels=args.bottleneck_channels,
        fusion_width=args.fusion_width,
        lsd_channels=10,
        affinity_channels=3,
        detach_lsd_for_affinity=True,
    ).to(device)
    if hasattr(torch, "compile"):
        model_raw = torch.compile(model_raw)
    model = DDP(model_raw, device_ids=[rank], output_device=rank) if multi_gpu and device.type == "cuda" else model_raw
    ema = ModelEma(model_raw, decay=ema_decay)

    steps_per_epoch = len(train_loader)
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    max_train_steps = max(1, training_epochs * steps_per_epoch)

    def _lr_lambda(last_epoch: int):
        if last_epoch < warmup_steps:
            return float(last_epoch + 1) / float(warmup_steps)
        t = last_epoch - warmup_steps
        total = max(1, max_train_steps - warmup_steps)
        progress = min(float(t) / float(total), 1.0)
        cos_part = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cosine_eta_min_ratio + (1.0 - cosine_eta_min_ratio) * cos_part

    optimizer = torch.optim.AdamW(model_raw.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    activation = torch.nn.Sigmoid()
    lsd_loss_fn = torch.nn.MSELoss().to(device)
    affinity_loss_fn = torch.nn.MSELoss().to(device)

    if is_main_process(rank):
        logging.info(
            """Starting training:
    leave_species:         %s (test sources: %s)
    training_epochs:       %s
    Train samples:         %d
    Val samples:           %d
    Test samples:          %d
    Holdout slices:        %d
    Batch size / rank:     %s
    Global batch size:     %s
    Val/Test batch size:   %s
    Crop size:             %s
    Num slices:            %s
    LSD cache dir:         %s
    LSD cache storage:     %s
    Preload LSD to RAM:    %s
    Optimizer:             AdamW (weight_decay=%s)
    LR schedule:           linear warmup %s epochs (~%s steps/rank) + cosine to %.4f * base lr
    EMA decay:             %s
    Grad clip norm:        %s
    Save top-k ckpts:      %s
    AMP enabled:           %s
    Base width:            %s
    Bottleneck channels:   %s
    Fusion width:          %s
    Parameters (M):        %.2f
    num_workers / rank:    %s
    num_gpus:              %s
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
            batch_size * world_size,
            val_batch_size,
            crop_size,
            num_slices,
            args.lsd_cache_dir,
            args.lsd_cache_storage,
            preload_lsd_to_ram,
            weight_decay,
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
            count_parameters(model_raw) / 1e6,
            num_workers,
            world_size,
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
        local_loss_sum = 0.0
        local_count = 0
        weight_scope = ema.apply_to(model_raw) if use_ema else contextlib.nullcontext()
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
                bs = int(raw.shape[0])
                local_loss_sum += float(loss_value.detach().cpu().item()) * bs
                local_count += bs
        return distributed_weighted_mean(local_loss_sum, local_count, device, world_size)

    with tqdm(total=training_epochs, disable=not is_main_process(rank)) as pbar:
        while epoch < training_epochs:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()
            local_train_loss_sum = 0.0
            local_train_count = 0

            profile_batches = min(32, args.profile_data) if args.profile_data and is_main_process(rank) else 0
            batch_i = 0
            t_after_step = time.perf_counter()
            for raw, gt_affinity, gt_lsds in train_loader:
                t_after_data = time.perf_counter()
                if profile_batches and batch_i < profile_batches:
                    print(
                        "[data_profile main pid=%d] batch=%d data_wait=%.3fs"
                        % (os.getpid(), batch_i, t_after_data - t_after_step),
                        file=sys.stderr,
                        flush=True,
                    )

                t_h2d0 = time.perf_counter()
                raw = raw.to(device=device, dtype=torch.float32, non_blocking=pin_memory)
                gt_lsds = gt_lsds.to(device=device, dtype=torch.float32, non_blocking=pin_memory)
                gt_affinity = gt_affinity.to(device=device, dtype=torch.float32, non_blocking=pin_memory)
                t_h2d1 = time.perf_counter()

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
                t_after_step = time.perf_counter()
                if profile_batches and batch_i < profile_batches:
                    print(
                        "[data_profile main pid=%d] batch=%d tensor_cpu_to_dev=%.3fs model_step=%.3fs"
                        % (
                            os.getpid(),
                            batch_i,
                            t_h2d1 - t_h2d0,
                            t_after_step - t_h2d1,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                batch_i += 1

                ema.update(model_raw)
                bs = int(raw.shape[0])
                local_train_loss_sum += float(loss_value.detach().cpu().item()) * bs
                local_train_count += bs

            epoch += 1
            if is_main_process(rank):
                pbar.update(1)

            train_loss = distributed_weighted_mean(local_train_loss_sum, local_train_count, device, world_size)
            val_loss = run_eval_loader(val_loader, use_ema=True)
            current_lr = optimizer.param_groups[0]["lr"]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                test_loss = run_eval_loader(test_loader, use_ema=True)
                if is_main_process(rank):
                    os.makedirs("./output/checkpoints", exist_ok=True)
                    ckpt_state = {
                        "epoch": epoch,
                        "model_state_dict": model_raw.state_dict(),
                        "ema_state_dict": ema.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_val_loss": best_val_loss,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "test_loss": test_loss,
                        "learning_rate": current_lr,
                        "config": vars(args),
                        "world_size": world_size,
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
                if is_main_process(rank):
                    logging.info(
                        "Epoch %d: train = %.6f | val(ema) = %.6f (no improvement), best_val = %.6f @ epoch %d | lr = %.8f",
                        epoch,
                        train_loss,
                        val_loss,
                        best_val_loss,
                        best_epoch,
                        current_lr,
                    )

            if no_improve_count >= early_stop_count:
                if is_main_process(rank):
                    logging.info("Early stop!")
                break

    if is_main_process(rank) and ranked_checkpoints:
        ranked_summary = ", ".join(
            "epoch {}: {:.6f}".format(item["epoch"], item["val_loss"]) for item in ranked_checkpoints
        )
        logging.info("Retained top-%d val checkpoints: %s", len(ranked_checkpoints), ranked_summary)

    if multi_gpu:
        dist.destroy_process_group()


def launch_training(args) -> None:
    world_size = max(1, int(args.num_gpus))
    save_name = "ACRLSD_3D_leaveout_{}_holdoutVal{}_ns{}_neo_preLSD".format(
        args.leave_species, args.n_val_holdout, args.num_slices
    )

    if world_size > 1:
        print(
            "[launch] building shared preloaded datasets once in parent for {} GPU(s)".format(world_size),
            file=sys.stderr,
            flush=True,
        )
        train_dataset, val_dataset, test_dataset = build_prelsd_datasets(
            leave_species=args.leave_species,
            crop_size=args.crop_size,
            num_slices=args.num_slices,
            n_val_holdout=args.n_val_holdout,
            lsd_cache_dir=args.lsd_cache_dir,
            lsd_storage=args.lsd_cache_storage,
            preload_lsd_to_ram=True,
        )
        _FORK_SHARED_STATE["train_dataset"] = train_dataset
        _FORK_SHARED_STATE["val_dataset"] = val_dataset
        _FORK_SHARED_STATE["test_dataset"] = test_dataset
        mp.start_processes(
            train_worker,
            args=(world_size, args, save_name),
            nprocs=world_size,
            join=True,
            start_method="fork",
        )
        return

    train_worker(0, 1, args, save_name)


if __name__ == "__main__":
    launch_training(build_argparser().parse_args())
