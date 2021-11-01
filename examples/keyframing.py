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
# WARNING: HEMISPHERE WORLD BREAKS DIRECTIONAL LIGHTS!
# scene += kb.DirectionalLight(name="sun", position=(-1, -0.5, 3), look_at=(0, 0, 0), intensity=1.5)


# Render cameras at the same general distance from the origin, but at
# different positions.
#
# We will use spherical coordinates (r, theta, phi) to do this.
#   x = r * cos(theta) * sin(phi)
#   y = r * sin(theta) * sin(phi)
#   z = r * cos(phi)
original_camera_position = (7.48113, -6.50764, 5.34367)
r = np.sqrt(sum(a * a for a in original_camera_position))
phi = np.arccos(original_camera_position[2] / r)
theta = np.arccos(original_camera_position[0] / (r * np.sin(phi)))
num_phi_values_per_theta = 2
theta_change = (2 * np.pi) / ((scene.frame_end - scene.frame_start) / num_phi_values_per_theta)

for frame in range(scene.frame_start, scene.frame_end + 1):
  i = (frame - scene.frame_start)
  theta_new = (i // num_phi_values_per_theta) * theta_change + theta

  # These values of (x, y, z) will lie on the same sphere as the original camera.
  x = r * np.cos(theta_new) * np.sin(phi)
  y = r * np.sin(theta_new) * np.sin(phi)
  z = r * np.cos(phi)

  # To ensure have "roughly LLFF-style" data (multiple z values for the same x and y)
  z_shift_direction = (i % num_phi_values_per_theta) - 1
  z = z + z_shift_direction * 1.2

  scene.camera.position = (x, y, z)
  scene.camera.look_at((0, 0, 0))
  scene.camera.keyframe_insert("position", frame)
  scene.camera.keyframe_insert("quaternion", frame)

# --- save scene for quick inspection
renderer.save_state("output/keyframing.blend")

# --- render the scene
data_stack = renderer.render()
del data_stack["uv"]
del data_stack["forward_flow"]
del data_stack["backward_flow"]
del data_stack["depth"]
del data_stack["normal"]

# --- save data to output folder
job_dir = kb.as_path("output/")
kb.file_io.write_image_dict(data_stack, job_dir)