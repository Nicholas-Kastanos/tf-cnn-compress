# tf-cnn-compress
Structural compression of CNNs in TensorFlow
## Setup
### Requirements
 - Anaconda 3
 - CUDA 11.0 (Required for GPU)

### Setup Instructions
Once anaconda is installed, run the following commands:

`conda env create -f environment.yml`

`conda activate tf-cnn-compress`

## Creating new models
The `train_resnet.py` file contains the training procedure for ResNet50V2. This file can be executed from the command line (use command line option `--help` to see input parameter options). Alternatively, the training function can be imported via `from train_resnet import train`.

For examples, see file `train_resnet_notebook.ipynb`.

Models are stored in the `models` directory, while TensorBoard logs are stored in the `logs` directory.
## Compressing and evaluating models
The `compress_model.py` file contains the compression and evaluation procedure for ResNet50V2. This file can be executed from the command line (use command line option `--help` to see input parameter options).

## Source Code
### ResNet50V2
Source code can be found in `src/resnet`.

### YOLOv4 (unstable)
This code is unstable, and not investigated in the report. The source code can be found in `src/yolo` directory. Running `train_coco.*` will train a YOLOv4 model on the MS-COCO (2017) dataset. 

WARNING:
This requires large compute capability and data storage. It will most likely cause issues with the machine you run it on.

## Documentation
A PDF of the associated report can be found in `docs`.