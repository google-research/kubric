# Copyright 2024 The Kubric Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import numpy as np
import bpy

import kubric as kb
from kubric.renderer import Blender

# --- CLI arguments
parser = kb.ArgumentParser()
parser.set_defaults(
    frame_end=1,
    resolution=(512, 512),
)
FLAGS = parser.parse_args()

# --- Common setups & resources
scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
renderer = Blender(scene, scratch_dir,
                   samples_per_pixel=64,
                   background_transparency=True)

# --- Fetch shapenet
source_path = os.getenv("SHAPENET_GCP_BUCKET", "gs://kubric-unlisted/assets/ShapeNetCore.v2.json")
# source_path = "/home/jlidard/Downloads/ShapeNetCore.v2.json"
# source_path = "https://3dwarehouse.sketchup.com/model/371a609f050b4ed3f6497dc58a9a6f8a/SR-71-Blackbird"
shapenet = kb.AssetSource.from_manifest(source_path)

# --- Add Klevr-like lights to the scene
scene += kb.assets.utils.get_clevr_lights(rng=rng)
scene.ambient_illumination = kb.Color(0.05, 0.05, 0.05)

# --- Add shadow-catcher floor
floor = kb.Cube(name="floor", scale=(100, 100, 1), position=(0, 0, -1))
scene += floor
# Make the floor transparent except for catching shadows
# Together with background_transparency=True (above) this results in
# the background being transparent except for the object shadows.
if bpy.app.version > (3, 0, 0):
  floor.linked_objects[renderer].is_shadow_catcher = True
else:
  floor.linked_objects[renderer].cycles.is_shadow_catcher = True

# --- Keyframe the camera
scene.camera = kb.PerspectiveCamera()
for frame in range(1):
  scene.camera.position = kb.sample_point_in_half_sphere_shell(1.5, 1.7, 0.1)
  scene.camera.look_at((0, 0, 0))
  scene.camera.keyframe_insert("position", frame)
  scene.camera.keyframe_insert("quaternion", frame)


  # # --- Fetch a random (airplane) asset
  # airplane_ids = [name for name, spec in shapenet._assets.items()
  #         if spec["metadata"]["category"] == "airplane"]

  # asset_id = rng.choice(airplane_ids) #< e.g. 02691156_10155655850468db78d106ce0a280f87
  # obj = shapenet.create(asset_id=asset_id)
  # logging.info(f"selected '{asset_id}'")

  # # --- make object flat on X/Y and not penetrate floor
  # obj.quaternion = kb.Quaternion(axis=[1, 0, 0], degrees=90)
  # obj.position = obj.position - (0, 0, obj.aabbox[0][2])  
  # scene.add(obj)

  # outdoor objects
  categories = ["bench", "pot", "table",  "tower"]
  selected_objects = []
  positions = np.random.uniform(low=[-2, -2, 0], high=[2, 2, 0], size=(len(categories), 3))
  print(list(set([spec["metadata"]["category"] for name, spec in shapenet._assets.items()])))

  for category, position in zip(categories, positions):
    print(category, position)
    asset_ids = [name for name, spec in shapenet._assets.items()
                 if spec["metadata"]["category"] == category]

    asset_id = rng.choice(asset_ids)
    # obj = shapenet.create(asset_id=asset_id)
    # override
    print(asset_id)
    obj = shapenet.create(asset_id="02843684/a448fb21ce07c5332cba66dc6aeabcd4")
    obj.quaternion = kb.Quaternion(axis=[1, 0, 0], degrees=90)
    obj.position = np.array(position) - np.array([0, 0, obj.aabbox[0][2]])
    scene.add(obj)
    selected_objects.append(obj)
    logging.info(f"selected '{asset_id}' from category '{category}'")


  # Ensure all objects are visible and adjust camera
  min_x, min_y, min_z = np.min([obj.position for obj in selected_objects], axis=0)
  max_x, max_y, max_z = np.max([obj.position for obj in selected_objects], axis=0)
  scene.camera.position = kb.sample_point_in_half_sphere_shell(5.0, 6.0, 0.5)  # Zoom way out
  scene.camera.look_at(((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2 - 0.5))  # Tilt downward
  scene.camera.keyframe_insert("position", frame)
  scene.camera.keyframe_insert("quaternion", frame)


  # --- Rendering
  logging.info("Rendering the scene ...")
  renderer.save_state(output_dir / "scene.blend")
  data_stack = renderer.render()

  # --- Postprocessing
  kb.compute_visibility(data_stack["segmentation"], scene.assets)
  data_stack["segmentation"] = kb.adjust_segmentation_idxs(
    data_stack["segmentation"],
    scene.assets,
    selected_objects).astype(np.uint8)

  kb.file_io.write_rgba_batch(data_stack["rgba"], output_dir)
  kb.file_io.write_depth_batch(data_stack["depth"], output_dir)
  kb.file_io.write_segmentation_batch(data_stack["segmentation"], output_dir)

  # --- Collect metadata
  logging.info("Collecting and storing metadata for each object.")
  data = {
    "metadata": kb.get_scene_metadata(scene),
    "camera": kb.get_camera_info(scene.camera),
    "object": kb.get_instance_info(scene, selected_objects)
  }
  kb.file_io.write_json(filename=output_dir / "metadata.json", data=data)
  kb.done()
