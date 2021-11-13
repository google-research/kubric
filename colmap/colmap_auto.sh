#!/bin/bash

# --- test whether container nvidia-docker worked
# nvidia-smi && exit

# --- avoids error for headless execution
export QT_QPA_PLATFORM=offscreen

# --- brutal parsing
ITEM_INDEX=${3/"--sceneid="/""}
SOURCE_PATH=$1/$ITEM_INDEX
OUTPUT_PATH=$2/$ITEM_INDEX

echo "reading images from: '$SOURCE_PATH'"
echo "saving results to: '$OUTPUT_PATH'"

# --- sets up tmp directories
DATASET_PATH="/workspace/workdir"

# ------------------------------------------------------------------------------
# echo "DEBUG RUN - EXITING EARLY" && exit
# ------------------------------------------------------------------------------

mkdir -p $DATASET_PATH
mkdir -p $DATASET_PATH/sparse
mkdir -p $DATASET_PATH/dense

# --- copy input data in a local directory (trash previous work)
mkdir -p $DATASET_PATH/images
gsutil -m cp -r $SOURCE_PATH/rgba_*.png $DATASET_PATH/images >> $DATASET_PATH/log.txt 2>&1

# --- automatic pipeline
colmap automatic_reconstructor \
  --workspace_path $DATASET_PATH \
  --image_path $DATASET_PATH/images

# --- copy results back to bucket (but not the images)
rm -rf $DATASET_PATH/images
gsutil -m cp -r $DATASET_PATH $OUTPUT_PATH >> $DATASET_PATH/log.txt 2>&1

# --- Inform hypertune the job concluded
cat > /tmp/hypertune_ok.py <<EOF
import hypertune
hpt = hypertune.HyperTune()
hpt.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag="answer",
    metric_value=42)
EOF