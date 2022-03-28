# Copyright 2022 The Kubric Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import kubric as kb
from kubric.renderer import Blender

# TODO: go to https://shapenet.org/ create an account and agree to the terms
#       then find the URL for the kubric preprocessed ShapeNet and put it here:
#SHAPENET_PATH = "gs://KUBRIC_SHAPENET_PATH/ShapeNetCore.v2.json"
SHAPENET_PATH = "/usr/local/google/home/klausg/assets/ShapeNetCore.v2.json"



if SHAPENET_PATH == "gs://KUBRIC_SHAPENET_PATH/ShapeNetCore.v2.json":
  raise ValueError("Wrong ShapeNet path. Please visit https://shapenet.org/ "
                   "agree to terms and conditions, and find the correct path.")


# --- CLI arguments
parser = kb.ArgumentParser()
# Configuration for the objects of the scene
parser.set_defaults(frame_end=25, resolution=(512, 512))
FLAGS = parser.parse_args()

# --- Common setups & resources
scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
renderer = Blender(scene, scratch_dir,
                   samples_per_pixel=64,
                   background_transparency=True)
shapenet = kb.AssetSource.from_manifest(SHAPENET_PATH)

logging.info("Adding four (studio) lights to the scene similar to the CLEVR setup...")
scene.add(kb.assets.utils.get_clevr_lights(rng=rng))
scene.ambient_illumination = kb.Color(0.05, 0.05, 0.05)

logging.info("Setting up the Camera...")
scene.camera = kb.PerspectiveCamera(focal_length=35., sensor_width=32)
for frame in range(FLAGS.frame_start, FLAGS.frame_end + 1):
  scene.camera.position = kb.sample_point_in_half_sphere_shell(9., 10.)
  scene.camera.look_at((0, 0, 0))
  scene.camera.keyframe_insert("position", frame)
  scene.camera.keyframe_insert("quaternion", frame)


asset_index = int(kb.as_path(FLAGS.job_dir).name)
logging.info("Adding ShapeNet object nr %d...", asset_index)
asset_ids = sorted(shapenet._assets.keys())
obj = shapenet.create(asset_id=asset_ids[asset_index], scale=6.)
scene.add(obj)

# --- Rendering
logging.info("Rendering the scene ...")
data_stack = renderer.render(return_layers=("rgba",))

# Save to image files
kb.file_io.write_image_dict(data_stack, output_dir)

# --- Metadata
logging.info("Collecting and storing metadata for each object.")
kb.file_io.write_json(filename=output_dir / "metadata.json", data={
    "metadata": kb.get_scene_metadata(scene),
    "camera": kb.get_camera_info(scene.camera),
    "instances": kb.get_instance_info(scene, assets_subset=[obj]),
})

kb.done()
