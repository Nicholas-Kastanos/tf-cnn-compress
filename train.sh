#!/bin/bash
EPOCHS=50
FOLDER_NAME="baseline"
BATCH_SIZE=12
INPUT_SIZE=416
DATA_DIR="/home/nk569/tensorflow_datasets/"

python train_coco.py -n $FOLDER_NAME -e $EPOCHS -b $BATCH_SIZE -i $INPUT_SIZE --data_dir $DATA_DIR