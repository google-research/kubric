#!/bin/bash

# --- test whether container nvidia-docker worked
# nvidia-smi && exit

# --- avoids error for headless execution
export QT_QPA_PLATFORM=offscreen

SOURCE_PATH=$0
echo "finding images from: '$SOURCE_PATH'" 

# --- sets up tmp directories
DATASET_PATH="/workspace/workdir"
mkdir -p $DATASET_PATH
mkdir -p $DATASET_PATH/sparse

exit

# --- copy input data in a local directory (trash previous work)
bucket_to_local() {
  mkdir -p $DATASET_PATH/images
  gsutil -m rm -rf $SOURCE_PATH/workdir
  gsutil -m cp -r $SOURCE_PATH/rgba_*.png $DATASET_PATH/images
}

feature_extractor() {
  colmap feature_extractor \
    --image_path $DATASET_PATH/images \
    --database_path $DATASET_PATH/database.db \
    --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu 1
}

feature_extractor_DEFAULT() {
  colmap feature_extractor \
   --database_path $DATASET_PATH/database.db \
   --image_path $DATASET_PATH/images
}


exhaustive_matcher() {
  colmap exhaustive_matcher \
    --database_path $DATASET_PATH/database.db \
    –-SiftMatching.use_gpu 1
}

mapper() {
  colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse \
    --Mapper.num_threads 16 \
    --Mapper.init_min_tri_angle 4 \
    --Mapper.multiple_models 0 \
    --Mapper.extract_colors 0
}

mapper_DEFAULT() {
  colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse
}

image_undistorter() {
  mkdir -p $DATASET_PATH/dense
  colmap image_undistorter \
    --image_path $DATASET_PATH/images \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/dense \
    --output_type COLMAP
}

image_undistorter_DEFAULT() {
  colmap image_undistorter \
      --image_path $DATASET_PATH/images \
      --input_path $DATASET_PATH/sparse/0 \
      --output_path $DATASET_PATH/dense \
      --output_type COLMAP \
      --max_image_size 2000
}

patch_match_stereo() {
  colmap patch_match_stereo \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.window_radius 9 \
    --PatchMatchStereo.geom_consistency true \
    --PatchMatchStereo.filter_min_ncc .07
}

patch_match_stereo_DEFAULT() {
  colmap patch_match_stereo \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true
}

stereo_fusion() {
  colmap stereo_fusion \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --input_type photometric \
    --output_path $DATASET_PATH/dense/fused.ply
}

stereo_fusion_DEFAULT(){
  colmap stereo_fusion \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $DATASET_PATH/dense/fused.ply
}

poisson_mesher() {
  colmap poisson_mesher \
    --input_path $DATASET_PATH/dense/fused.ply \
    --output_path $DATASET_PATH/dense/meshed-poisson.ply
}

local_to_bucket() {
  # --- copy results back to bucket (but not the images)
  rm -rf $DATASET_PATH/images
  gsutil -m cp -r $DATASET_PATH $SOURCE_PATH
}

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# --- automatic pipeline
# bucket_to_local
# colmap automatic_reconstructor \
#     --workspace_path $DATASET_PATH \
#     --image_path $DATASET_PATH/images
# local_to_bucket

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# --- manual pipeline (sasa)
# bucket_to_local
# feature_extractor
# exhaustive_matcher
# mapper
# image_undistorter
# patch_match_stereo
# stereo_fusion
# poisson_mesher
# local_to_bucket

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# --- DEFAULT pipeline
# https://colmap.github.io/cli.html#example
bucket_to_local
feature_extractor_DEFAULT
exhaustive_matcher
mapper_DEFAULT
image_undistorter_DEFAULT
patch_match_stereo_DEFAULT
stereo_fusion_DEFAULT
poisson_mesher
local_to_bucket
