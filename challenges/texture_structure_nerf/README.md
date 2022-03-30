# Texture and Structure in NeRF

This dataset contains randomly placed shapes with procedural textures for the purpose of measuring the correlation of texture frequency and solid structure in NeRF reconstructions.

The dataset is available at `gs://kubric-public/data/texture_structure_nerf`

# Running the worker

    sudo docker run --rm --interactive \
      --user $(id -u):$(id -g)         \
      --volume "$(pwd):/kubric"        \
      kubricdockerhub/kubruntu         \
      /usr/bin/python3                 \
      examples/nerf_texture.py
    
Parameters:

 - `num_objects` How many objects to generate.
 - `num_frequency_bands` How many discrete frequency bands to use.
 - `min_log_frequency` Minimum frequency value in log-scale (base 10).
 - `max_log_frequency` Maximum frequency value in log-scale (base 10).
 - `num_train_frames` How many frames to render in the training split.
 - `num_validation_frames` How many frames to render in the validation split.
 - `num_test_frames` How many frames to render in the testing split.

# Output Format

The script directly generates output that can be used as input by JAXNeRF with the 'blender' configuration.
The resulting folder structure is:
 - `[train|val|test]/*.png` RGB color images.
 - `[train|val|test]/*_segmentation.png` Segmentation maps indicating which frequency band a pixel belongs to.
 - `transforms_[train|val|test].json` Camera information for each data split.
