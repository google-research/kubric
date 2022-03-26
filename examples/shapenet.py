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
import numpy as np

import kubric as kb
from kubric.renderer import Blender

# TODO: go to https://shapenet.org/ create an account and agree to the terms
#       then find the URL for the kubric preprocessed ShapeNet and put it here:
SHAPENET_PATH = "gs://KUBRIC_SHAPENET_PATH/ShapeNetCore.v2.json"

if SHAPENET_PATH == "gs://KUBRIC_SHAPENET_PATH/ShapeNetCore.v2.json":
  raise ValueError("Wrong ShapeNet path. Please visit https://shapenet.org/ "
                   "agree to terms and conditions, and find the correct path.")


# --- CLI arguments
parser = kb.ArgumentParser()
parser.set_defaults(
    frame_end=5,
    resolution=(512, 512),
)
FLAGS = parser.parse_args()

# --- Common setups & resources
scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
renderer = Blender(scene, scratch_dir,
                   samples_per_pixel=64,
                   background_transparency=True)
shapenet = kb.AssetSource.from_manifest(SHAPENET_PATH)

# --- Add Klevr-like lights to the scene
scene += kb.assets.utils.get_clevr_lights(rng=rng)
scene.ambient_illumination = kb.Color(0.05, 0.05, 0.05)

# --- Add shadow-catcher floor
floor = kb.Cube(name="floor", scale=(100, 100, 1), position=(0, 0, -1))
scene += floor
# Make the floor transparent except for catching shadows
# Together with background_transparency=True (above) this results in
# the background being transparent except for the object shadows.
floor.linked_objects[renderer].cycles.is_shadow_catcher = True

# --- Keyframe the camera
scene.camera = kb.PerspectiveCamera()
for frame in range(FLAGS.frame_start, FLAGS.frame_end + 1):
  scene.camera.position = kb.sample_point_in_half_sphere_shell(1.5, 1.7, 0.1)
  scene.camera.look_at((0, 0, 0))
  scene.camera.keyframe_insert("position", frame)
  scene.camera.keyframe_insert("quaternion", frame)

# --- Fetch a random (airplane) asset
airplane_ids = [name for name, spec in shapenet._assets.items()
                if spec["metadata"]["category"] == "airplane"]

asset_id = rng.choice(airplane_ids) #< e.g. 02691156_10155655850468db78d106ce0a280f87
obj = shapenet.create(asset_id=asset_id)
logging.info(f"selected '{asset_id}'")

# --- make object flat on X/Y and not penetrate floor
obj.quaternion = kb.Quaternion(axis=[1, 0, 0], degrees=90)
obj.position = obj.position - (0, 0, obj.aabbox[0][2])  
scene.add(obj)

# --- Rendering
logging.info("Rendering the scene ...")
renderer.save_state(output_dir / "scene.blend")
data_stack = renderer.render()

# --- Postprocessing
kb.compute_visibility(data_stack["segmentation"], scene.assets)
data_stack["segmentation"] = kb.adjust_segmentation_idxs(
    data_stack["segmentation"],
    scene.assets,
    [obj]).astype(np.uint8)

kb.file_io.write_rgba_batch(data_stack["rgba"], output_dir)
kb.file_io.write_depth_batch(data_stack["depth"], output_dir)
kb.file_io.write_segmentation_batch(data_stack["segmentation"], output_dir)

# --- Collect metadata
logging.info("Collecting and storing metadata for each object.")
data = {
  "metadata": kb.get_scene_metadata(scene),
  "camera": kb.get_camera_info(scene.camera),
  "object": kb.get_instance_info(scene, [obj])
}
kb.file_io.write_json(filename=output_dir / "metadata.json", data=data)
kb.done()
