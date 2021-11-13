#!/bin/bash

# --- test whether container nvidia-docker worked
# nvidia-smi && exit

# --- avoids error for headless execution
export QT_QPA_PLATFORM=offscreen

SOURCE_PATH=$1
echo "finding images from: '$SOURCE_PATH'" 

# --- sets up tmp directories
DATASET_PATH="/workspace/workdir"

clear_workdir() {
  rm -rf $DATASET_PATH
  mkdir -p $DATASET_PATH
  mkdir -p $DATASET_PATH/sparse
  mkdir -p $DATASET_PATH/dense
}

# --- automatic pipeline
# clear_workdir
# bucket_to_local
# colmap automatic_reconstructor \
#     --workspace_path $DATASET_PATH \
#     --image_path $DATASET_PATH/images
# local_to_bucket