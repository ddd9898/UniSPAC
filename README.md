# UniSPAC [A Unified Segmentation framework for Proofreading and Annotation in Connectomics]

![Supplementary Video 1](./data/Supplementary_Video_a.gif)



***
### Note⏰

This is a PYQT-based demonstration. If you wish to use the UniSPAC tool directly, we recommend using  [napari-UniSPAC](https://github.com/ddd9898/napari-UniSPAC). To reproduce the training process, please refer to the  the `training` folder.

### System requirements

It is recommended to deploy the software on a Linux system. Pre-install PyQt5 (Qt) and PyTorch. Devices that support cuda allow for smoother software usage. 

### Quick Start

Set up the software environment:

```shell
conda create -n UniSPAC python=3.9
conda activate UniSPAC
git clone https://github.com/ddd9898/UniSPAC.git
cd UniSPAC
pip install -r requirements.txt
```

Download test data and checkpoints：

```shell
bash ./download.sh
```

The total files after data and model decompression take up **9.3GB** of storage, so please make sure you have enough capacity. See the downloaded model weights in the `checkpoints` folder and the Hemi-Brain-ROI-1 test data in the `data` folder. 

Finally, launch the software:

```shell
python demo.py
```

**Brief tutorial:** Click the <u>left mouse button</u> to add a **positive** point prompt, and the <u>right mouse button</u> to add a **negative** point prompt. Press <kbd>Q</kbd> to undo the previous point prompt, press <kbd>E</kbd> to clear all prompts.

## Napari plugin
![Supplementary Video 2](./data/Supplementary_Video_b.gif)
If you want to apply UniSPAC to your own data, the  [napari plugin for UniSPAC](https://github.com/ddd9898/napari-UniSPAC)  might come in handy. Assuming you are a veteran napari user, installing [napari-UniSPAC](https://pypi.org/project/napari-UniSPAC/)  with the following command is sufficient.

```shell
pip install napari-UniSPAC
```
The installation should take a few minutes, depending on your network conditions.

## Contact


Feel free to contact djt20@mails.tsinghua.edu.cn if you have issues for any questions.