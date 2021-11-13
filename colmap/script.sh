IMAGE_PATH=/workspace/semistatic
OUTPUT_PATH=/workspace/semistatic_output
mkdir $OUTPUT_PATH

# --- copy input data in a local directory
gsutil -m cp -r gs://kubric-public/colmap/input/semistatic1 /tmp/semistatic1

# export QT_QPA_PLATFORM=offscreen
colmap feature_extractor \
  --image_path $IMAGE_PATH \
  --database_path $OUTPUT_PATH/database.db \
  --ImageReader.single_camera 1 \
  --SiftExtraction.use_gpu 0

colmap exhaustive_matcher \
  --database_path $OUTPUT_PATH/database.db \
  –-SiftMatching.use_gpu 0

colmap mapper \
  --image_path $IMAGE_PATH \
  --database_path $OUTPUT_PATH/database.db \
  --output_path $OUTPUT_PATH \
  --Mapper.num_threads 16 \
  --Mapper.init_min_tri_angle 4\
  --Mapper.multiple_models 0\
  --Mapper.extract_colors 0\
