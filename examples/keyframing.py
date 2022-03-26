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

import logging
import kubric as kb
import numpy as np
from kubric.renderer.blender import Blender as KubricRenderer

logging.basicConfig(level="INFO")

# --- create scene and attach a renderer to it
scene = kb.Scene(resolution=(256, 256), frame_start=1, frame_end=20)
renderer = KubricRenderer(scene)

# --- populate the scene with objects, lights, cameras
scene += kb.Sphere(name="floor", scale=1000, position=(0, 0, +1000), background=True)
scene += kb.Cube(name="floor", scale=(.5,.7,1.0), position=(0, 0, 1.1))
scene += kb.PerspectiveCamera(name="camera", position=(3, -1, 4), look_at=(0, 0, 1))

# --- Add Klevr-like lights to the scene
scene += kb.assets.utils.get_clevr_lights()
scene.ambient_illumination = kb.Color(0.05, 0.05, 0.05)

# --- Keyframe a circular camera path around the object (use polar coordinates)
# TODO: this seems to be pretty common logic, move it to a utility file?
original_camera_position = (7.48113, -6.50764, 5.34367)
r = np.sqrt(sum(a * a for a in original_camera_position))
phi = np.arccos(original_camera_position[2] / r)
theta = np.arccos(original_camera_position[0] / (r * np.sin(phi)))
num_phi_values_per_theta = 1  #< only circular motion
theta_change = (2 * np.pi) / ((scene.frame_end - scene.frame_start) / num_phi_values_per_theta)
for frame in range(scene.frame_start, scene.frame_end + 1):
  i = (frame - scene.frame_start)
  theta_new = (i // num_phi_values_per_theta) * theta_change + theta

  # These values of (x, y, z) will lie on the same sphere as the original camera.
  x = r * np.cos(theta_new) * np.sin(phi)
  y = r * np.sin(theta_new) * np.sin(phi)
  z = r * np.cos(phi)
  z_shift_direction = (i % num_phi_values_per_theta) - 1
  z = z + z_shift_direction * 1.2

  scene.camera.position = (x, y, z)
  scene.camera.look_at((0, 0, 0))
  scene.camera.keyframe_insert("position", frame)
  scene.camera.keyframe_insert("quaternion", frame)

# --- save scene for quick inspection
renderer.save_state("output/keyframing.blend")

# --- render the data
data_stack = renderer.render()

# --- save output files
output_dir = kb.as_path("output/")
kb.file_io.write_rgba_batch(data_stack["rgba"], output_dir)