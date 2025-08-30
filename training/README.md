# UniSPAC: Training Tutorial


***
### Training data preparation

Move all training data from the `../data`  directory to the current training folder.
```shell
mkdir data
mv -r ../data/funke ./data/
```

### Model Training

```shell
python train_ACRLSD_2d.py 
python train_segEM2d_wloss.py
python train_ACRLSD_3d.py
python train_segEM3d_trace.py
```

### Online Learning
```shell
python train_segEM2d_CL_avalanche.py  --strategy 'Cumulative'
python train_segEM2d_CL_avalanche.py  --strategy 'AGEM'
python train_segEM2d_CL_avalanche.py  --strategy 'GEM'
python train_segEM2d_CL_avalanche.py  --strategy 'Replay'
python train_segEM2d_CL_avalanche.py  --strategy 'Naive'
python train_segEM2d_CL_avalanche.py  --strategy 'EWC'
python train_segEM2d_CL_avalanche.py  --strategy 'LwF'
python train_segEM2d_CL_avalanche.py  --strategy 'DER'
python train_segEM2d_CL_avalanche.py  --strategy 'GDumb'
python train_segEM2d_CL_avalanche.py  --strategy 'SynapticIntelligence'
```

