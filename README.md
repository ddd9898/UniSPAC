# UniSPAC

This repository contains the PyTorch implementation used for UniSPAC experiments on
promptable electron microscopy (EM) instance segmentation across species. The main
reproduction workflow follows leave-one-species-out training on four held-out species:
human, drosophila, mouse, and zebrafinch.

The code includes 2D and 3D ACRLSD pretraining, prompt-conditioned segEM training,
fixed-prompt evaluation, and utilities for precomputing local shape descriptor (LSD)
targets.

## Repository Overview

- `train_ACRLSD_2d_neo.py`: train the 2D ACRLSD backbone.
- `train_segEM2d_plus.py`: train the 2D promptable segEM head with multi-step prompt supervision.
- `train_ACRLSD_3d_neo_preLSD.py`: train the 3D ACRLSD backbone using precomputed LSD caches.
- `train_segEM3d_trace.py`: train the 3D trace model with a frozen 3D ACRLSD teacher.
- `test_segEM2d_neo.py`: evaluate 2D promptable segmentation on fixed prompts.
- `process_lsd.py`: precompute full-volume LSD targets.
- `tran_lsd_to_zarr.py`: convert LSD `.npy` caches to chunked `.zarr` stores.
- `compare/ori2section_v2.py`: export deterministic 2D test slices and prompts.
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

Large datasets, LSD caches, checkpoints, logs, and evaluation outputs are not intended
to be committed to git.

## LSD Precomputation

The 2D ACRLSD and 3D preLSD experiments use precomputed LSD targets. Generate the
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

By default, the 2D ACRLSD code reads `./LSD_cache`. For 3D preLSD training, pass
`--lsd-cache-dir ./LSD_cache` if the cache is stored in the repository root.

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

## Outputs

Training checkpoints are saved under `./output/checkpoints/`, and logs are saved under  
`./output/log/`. Checkpoint filenames encode the held-out species and key hyperparameters.

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

