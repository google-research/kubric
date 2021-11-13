#!/bin/bash

# --- test whether container nvidia-docker worked
# nvidia-smi && exit

echo "finding images from: '$1'" 
echo "storing colmap output: '$2'"

# --- sets up tmp directories
mkdir -p /tmp/input
mkdir -p /tmp/output
IMAGE_PATH="/tmp/input"
OUTPUT_PATH="/tmp/output"

# --- copy input data in a local directory
gsutil -m cp -r $1/rgba_*.png /tmp/input

# --- avoids error for headless execution
export QT_QPA_PLATFORM=offscreen
colmap feature_extractor \
  --image_path $IMAGE_PATH \
  --database_path $OUTPUT_PATH/database.db \
  --ImageReader.single_camera 1 \
  --SiftExtraction.use_gpu 1

colmap exhaustive_matcher \
  --database_path $OUTPUT_PATH/database.db \
  –-SiftMatching.use_gpu 1

colmap mapper \
  --image_path $IMAGE_PATH \
  --database_path $OUTPUT_PATH/database.db \
  --output_path $OUTPUT_PATH \
  --Mapper.num_threads 16 \
  --Mapper.init_min_tri_angle 4\
  --Mapper.multiple_models 0\
  --Mapper.extract_colors 0

# --- copy results back to bucket
gsutil -m cp -r /tmp/output $2 