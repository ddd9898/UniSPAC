# UniSPAC

This repository contains the PyTorch implementation used for UniSPAC experiments on  
promptable electron microscopy (EM) instance segmentation across species. The main  
reproduction workflow follows leave-one-species-out training on four held-out species:  
human, drosophila, mouse, and zebrafinch.

## Repository Overview

- `train_ACRLSD_2d_neo.py`: train the 2D ACRLSDneo backbone.
- `train_segEM2d_plus.py`: train the 2D promptable UniSPAC head with multi-step prompt supervision.
- `train_ACRLSD_3d_neo_preLSD.py`: train the 3D ACRLSDneo backbone using precomputed LSD caches.
- `train_segEM3d_trace.py`: train the 3D UniSPAC trace model with a frozen 3D ACRLSDneo teacher.
- `test_segEM2d_neo.py`: evaluate 2D promptable segmentation on fixed prompts.
- `process_lsd.py`: precompute full-volume LSD targets.
- `tran_lsd_to_zarr.py`: convert LSD `.npy` caches to chunked `.zarr` stores.
- `fix_2d_prompts.py`: export deterministic 2D test slices and prompts.
- `utils/`: dataset loading, augmentation, affinity targets, LSD targets, and prompt sampling utilities.

## Installation

Create a Python environment with PyTorch and install the repository requirements:

```bash
conda create -n unispac python=3.10
conda activate unispac

# Install a PyTorch build matching your CUDA/runtime first.
# See https://pytorch.org/get-started/locally/

pip install -r requirements.txt
pip install tqdm tifffile h5py matplotlib numcodecs
```

The full experiments require CUDA GPUs, large host memory, and high-throughput storage.
The 3D precomputed-LSD training script supports single-node multi-GPU training via
`--num-gpus`.

## Data and model availability

All files required to reproduce the reported experiments (datasets, trained  
checkpoints, precomputed LSD caches, and fixed-prompt evaluation assets) are provided via  
the project [OneDrive folder](https://1drv.ms/f/c/88a3ba3c5aa53eeb/IgB7ui3_ZFZ_Q4GpuLhe1umHART24jjCFeUzyIq2qZyDmJg?e=baCPvi).

The folder contains:

- `data.zip`: full data for all four species. Unzip and place under `./data/` (see the
expected dataset roots in **Data Preparation**).
- `checkpoints/`: trained model weights used in the paper experiments.
- `LSD_cache.zip`: precomputed LSD targets produced by `process_lsd.py` and
`tran_lsd_to_zarr.py`. 
- `compare_process.zip`: fixed 2D prompt assets for evaluation, produced by
`fix_2d_prompts.py`.

After downloading, the repository root is expected to contain (at minimum) the following
layout:

```text
./data/                 # unzip data.zip here
./LSD_cache/            # unzip LSD_cache.zip here (optional if recomputing)
./compare/processed/    # unzip compare_process.zip here (or regenerate via fix_2d_prompts.py)
./output/checkpoints/   # copy/symlink trained weights here (or train from scratch)
```

## Data Preparation

Place all datasets under `./data/` using the paths expected by the dataloaders. The
leave-one-species-out splits are defined in the training scripts:

- human: `axonem_h`
- drosophila: `hemi`, `fib25`, `cremi`, `vnc`, `isbi2012`
- mouse: `ac3`, `ac4`, `basil`, `minnie`, `pinky`, `axonem_m`
- zebrafinch: `zebrafinch`

Expected dataset roots include:

```text
data/funke/hemi/training/
data/funke/fib25/training/
data/funke/zebrafinch/training/
data/CREMI/
data/groundtruth-drosophila-vnc-master/stack1/
data/ISBI-2012/
data/AC3/
data/AC4/
data/MICrONS/Neuron_zarr/{basil,minnie,pinky}/
data/AxonEM/EM30-H-axon-train-9vol/
data/AxonEM/EM30-M-axon-train-9vol/
```

## LSD Precomputation

The ACRLSDneo experiments use precomputed LSD targets. Generate the  
full-volume cache once, then convert it to chunked Zarr for faster training I/O:

```bash
python process_lsd.py --num-workers 1 --native-threads 1 \
  --cache-dir /path/to/LSD_cache_npy

python tran_lsd_to_zarr.py \
  --input-dir /path/to/LSD_cache_npy \
  --skip-existing \
  --jobs 2 \
  --output-dir ./LSD_cache
```

By default, the ACRLSDneo code reads `./LSD_cache`.

## Reproducing Main Experiments

Run each stage for all four held-out species. The commands below reproduce the main
leave-one-species-out workflow used by the paper experiments.

### ACRLSDneo-2D

```bash
python train_ACRLSD_2d_neo.py --leave-species human
python train_ACRLSD_2d_neo.py --leave-species drosophila
python train_ACRLSD_2d_neo.py --leave-species mouse
python train_ACRLSD_2d_neo.py --leave-species zebrafinch
```

### UniSPAC-2D

```bash
python train_segEM2d_plus.py --leave-species human
python train_segEM2d_plus.py --leave-species drosophila
python train_segEM2d_plus.py --leave-species mouse
python train_segEM2d_plus.py --leave-species zebrafinch
```

The UniSPAC-2D stage expects the matching ACRLSDneo-2D checkpoint in `./output/checkpoints/`,
or an explicit `--backbone-checkpoint`.

### ACRLSDneo-3D With Precomputed LSD

```bash
python train_ACRLSD_3d_neo_preLSD.py --leave-species human --num-gpus 2 --lsd-cache-dir ./LSD_cache
python train_ACRLSD_3d_neo_preLSD.py --leave-species drosophila --num-gpus 2 --lsd-cache-dir ./LSD_cache
python train_ACRLSD_3d_neo_preLSD.py --leave-species mouse --num-gpus 2 --lsd-cache-dir ./LSD_cache
python train_ACRLSD_3d_neo_preLSD.py --leave-species zebrafinch --num-gpus 2 --lsd-cache-dir ./LSD_cache
```

### UniSPAC-3D

```bash
python train_segEM3d_trace.py --leave-species human
python train_segEM3d_trace.py --leave-species drosophila
python train_segEM3d_trace.py --leave-species mouse
python train_segEM3d_trace.py --leave-species zebrafinch
```

If the ACRLSDneo-3D checkpoint name differs from the auto-discovery pattern, provide it  
with `--backbone-checkpoint`.

## Fixed-Prompt Evaluation

Before evaluating 2D promptable segmentation, export deterministic test sections and
prompts:

```bash
python fix_2d_prompts.py --species human --seed 1998
python fix_2d_prompts.py --species mouse --seed 1998
python fix_2d_prompts.py --species drosophila --seed 1998
python fix_2d_prompts.py --species zebrafinch --seed 1998
```

Then run evaluation with the corresponding UniSPAC-2D checkpoint:

```bash
python test_segEM2d_neo.py --species human \
  --output ./output/segEM2d-plus_eval \
  --checkpoint /path/to/segEM2d_plus_human.model

python test_segEM2d_neo.py --species drosophila \
  --output ./output/segEM2d-plus_eval \
  --checkpoint /path/to/segEM2d_plus_drosophila.model

python test_segEM2d_neo.py --species mouse \
  --output ./output/segEM2d-plus_eval \
  --checkpoint /path/to/segEM2d_plus_mouse.model

python test_segEM2d_neo.py --species zebrafinch \
  --output ./output/segEM2d-plus_eval \
  --checkpoint /path/to/segEM2d_plus_zebrafinch.model
```

The fixed-prompt export writes to `./compare/processed/` by default. Evaluation results
are written to `./output/segEM2d-plus_eval/`.



## Reproducibility notes

- **Using pre-trained weights**: download `checkpoints/` from the OneDrive folder and place
the `.model` files under `./output/checkpoints/`. You can then run evaluation directly
(see **Fixed-Prompt Evaluation**) without re-training.
- **Training from scratch**: you can either download `LSD_cache.zip` and unzip it to
`./LSD_cache/`, or recompute it locally using `process_lsd.py` and `tran_lsd_to_zarr.py`
(see **LSD Precomputation**). The ACRLSDneo-3D preLSD training script additionally  
supports passing the cache location via `--lsd-cache-dir`.
- **Fixed prompts for fair comparison**: to reproduce the exact deterministic fixed-point
prompts used in evaluation, use the provided `compare_process.zip` and unzip to
`./compare/processed/`, or regenerate them with `fix_2d_prompts.py`.

## Citation

If you use this code, please cite the UniSPAC paper. The citation entry will be updated
after publication.

```bibtex
@article{unispac,
  title   = {UniSPAC},
  author  = {TBD},
  journal = {TBD},
  year    = {TBD}
}
```

