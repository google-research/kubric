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

colmap feature_extractor \
  --database_path $DATASET_PATH/database.db \
  --image_path $DATASET_PATH/images \
  --SiftExtraction.use_gpu 1

colmap exhaustive_matcher \
  --database_path $DATASET_PATH/database.db \
  –-SiftMatching.use_gpu 1

colmap mapper \
  --database_path $DATASET_PATH/database.db \
  --image_path $DATASET_PATH/images \
  --output_path $DATASET_PATH/sparse \
  --Mapper.num_threads 16 \
  
colmap image_undistorter \
  --image_path $DATASET_PATH/images \
  --input_path $DATASET_PATH/sparse/0 \
  --output_path $DATASET_PATH/dense \
  --output_type COLMAP \
  --max_image_size 2000

colmap patch_match_stereo \
  --workspace_path $DATASET_PATH/dense \
  --workspace_format COLMAP \
  --PatchMatchStereo.geom_consistency true

colmap stereo_fusion \
  --workspace_path $DATASET_PATH/dense \
  --workspace_format COLMAP \
  --input_type geometric \
  --output_path $DATASET_PATH/dense/fused.ply

# --- copy results back to bucket (but not the images)
rm -rf $DATASET_PATH/images
gsutil -m cp -r $DATASET_PATH $OUTPUT_PATH >> $DATASET_PATH/log.txt 2>&1

# --- Inform hypertune the job concluded
cat > /tmp/hypertune_job_success.py <<EOF
import hypertune
hpt = hypertune.HyperTune()
hpt.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag="answer",
    metric_value=42)
EOF
python3 /tmp/hypertune_job_success.py