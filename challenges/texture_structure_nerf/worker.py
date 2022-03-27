# Copyright 2021 The Kubric Authors
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
"""Worker script for NeRF texture-structure dataset generation.

The output is a JaxNeRF-compatible scene containing randomly placed,
procedurally textured objects annotated with frequency information.
"""

import logging

import kubric as kb
from kubric.renderer.blender import Blender
import numpy as np
import bpy


BACKGROUND_COLOR = kb.Color(1.0, 1.0, 1.0)
LIGHT_DIRECTION = (-1, -0.5, 3)
MIN_OBJECT_SCALE = 0.1
MAX_OBJECT_SCALE = 0.2

# --- CLI arguments
parser = kb.ArgumentParser()
parser.add_argument("--num_objects", type=int, default=30)
parser.add_argument("--num_frequency_bands", type=int, default=6)
parser.add_argument("--min_log_frequency", type=float, default=-1.0)
parser.add_argument("--max_log_frequency", type=float, default=2.0)

parser.add_argument("--num_train_frames", type=int, default=60)
parser.add_argument("--num_validation_frames", type=int, default=60)
parser.add_argument("--num_test_frames", type=int, default=60)
FLAGS = parser.parse_args()

# --- Common setups & resources
scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
renderer = Blender(scene, scratch_dir, use_denoising=True, adaptive_sampling=False)

# --- Add floor and light
logging.info("Setting up the scene.")
scene.background = BACKGROUND_COLOR
scene += kb.Cube(name="floor", scale=(100, 100, 0.1), position=(0, 0, -0.1))
scene += kb.DirectionalLight(name="sun", position=LIGHT_DIRECTION, look_at=(0, 0, 0), intensity=1.5)

bpy_scene = bpy.context.scene

# --- Add random objects to scene
logging.info("Adding objects to the scene.")
for j in range(FLAGS.num_objects):
  position = (
    (rng.uniform() - 0.5) * 2,
    (rng.uniform() - 0.5) * 2,
    rng.uniform()
  )
  
  # --- Create randomly scaled and rotated cube
  scale = rng.uniform() * (MAX_OBJECT_SCALE - MIN_OBJECT_SCALE) + MIN_OBJECT_SCALE
  instance = kb.Cube(name=f"inst_{j}", scale=scale, position=position, quaternion=kb.random_rotation(rng=rng))
  instance.material = kb.PrincipledBSDFMaterial(name="material")
  instance.material.metallic = 0.0
  instance.material.roughness = 1.0

  # --- Sample log-uniform texture frequency
  fpower = rng.uniform()
  frequency = 10**(fpower * (FLAGS.max_log_frequency - FLAGS.min_log_frequency) + FLAGS.min_log_frequency)

  instance.segmentation_id = 1 + int(fpower * FLAGS.num_frequency_bands)
  scene += instance

  # --- Generate random procedural texture with Blender nodes
  mat = bpy_scene.objects[f"inst_{j}"].active_material
  tree = mat.node_tree

  mat_node = tree.nodes["Principled BSDF"]
  ramp_node = tree.nodes.new(type="ShaderNodeValToRGB")
  tex_node = tree.nodes.new(type="ShaderNodeTexNoise")
  scaling_node = tree.nodes.new(type="ShaderNodeMapping")
  rotation_node = tree.nodes.new(type="ShaderNodeMapping")
  vector_node = tree.nodes.new(type="ShaderNodeNewGeometry")

  tree.links.new(vector_node.outputs["Position"], rotation_node.inputs["Vector"])
  tree.links.new(rotation_node.outputs["Vector"], scaling_node.inputs["Vector"])
  tree.links.new(scaling_node.outputs["Vector"], tex_node.inputs["Vector"])
  tree.links.new(tex_node.outputs["Fac"], ramp_node.inputs["Fac"])
  tree.links.new(ramp_node.outputs["Color"], mat_node.inputs["Base Color"])

  rotation_node.inputs["Rotation"].default_value = (
    rng.uniform() * np.pi,
    rng.uniform() * np.pi,
    rng.uniform() * np.pi,
  )

  scaling_node.inputs["Scale"].default_value = (
    frequency,
    frequency,
    frequency,
  )

  tex_node.inputs["Roughness"].default_value = 0.0
  tex_node.inputs["Detail"].default_value = 0.0

  for x in np.linspace(0.0, 1.0, 5)[1:-1]:
    ramp_node.color_ramp.elements.new(x)

  for element in ramp_node.color_ramp.elements:
    element.color = kb.random_hue_color(rng=rng)

# --- Create the camera
camera = kb.PerspectiveCamera(name="camera", look_at=(0, 0, 1))
scene += camera

def update_camera():
  position = rng.normal(size=(3, ))
  position *= 4 / np.linalg.norm(position)
  position[2] = np.abs(position[2])
  camera.position = position
  camera.look_at((0, 0, 0))
  return camera.matrix_world

def output_split(split_name, n_frames):
  logging.info("Rendering the %s split.", split_name)
  frames = []

  # --- Render a set of frames from random camera poses
  for i in range(n_frames):
    matrix = update_camera()

    frame = renderer.render_still()

    frame["segmentation"] = kb.adjust_segmentation_idxs(frame["segmentation"], scene.assets, [])
    
    kb.write_png(filename=output_dir / split_name / f"{i}.png", data=frame["rgba"])
    kb.write_palette_png(filename=output_dir / split_name / f"{i}_segmentation.png", data=frame["segmentation"])

    frame_data = {
      "transform_matrix": matrix.tolist(),
      "file_path": f"{split_name}/{i}",
    }
    frames.append(frame_data)

  # --- Write the JSON descriptor for this split
  kb.write_json(filename=output_dir / f"transforms_{split_name}.json", data={
    "camera_angle_x": camera.field_of_view,
    "frames": frames,
  })

# --- Write train, validation, and test splits for the nerf data
output_split("train", FLAGS.num_train_frames)
output_split("val", FLAGS.num_validation_frames)
output_split("test", FLAGS.num_test_frames)
