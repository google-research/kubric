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
from kubric.renderer.blender import Blender as KubricBlender
from kubric.simulator.pybullet import PyBullet as KubricSimulator

logging.basicConfig(level="INFO")  # < CRITICAL, ERROR, WARNING, INFO, DEBUG

# --- create scene and attach a renderer and simulator
scene = kb.Scene(resolution=(256, 256))
scene.frame_end = 48   # < numbers of frames to render
scene.frame_rate = 24  # < rendering framerate
scene.step_rate = 240  # < simulation framerate
renderer = KubricBlender(scene)
simulator = KubricSimulator(scene)

# --- populate the scene with objects, lights, cameras
scene += kb.Cube(name="floor", scale=(3, 3, 0.1), position=(0, 0, -0.1),
                 static=True)
scene += kb.DirectionalLight(name="sun", position=(-1, -0.5, 3),
                             look_at=(0, 0, 0), intensity=1.5)
scene.camera = kb.PerspectiveCamera(name="camera", position=(2, -0.5, 4),
                                    look_at=(0, 0, 0))

# --- generates spheres randomly within a spawn region
spawn_region = [[-1, -1, 0], [1, 1, 1]]
rng = np.random.default_rng()
for i in range(8):
  velocity = rng.uniform([-1, -1, 0], [1, 1, 0])
  material = kb.PrincipledBSDFMaterial(color=kb.random_hue_color(rng=rng))
  sphere = kb.Sphere(scale=0.1, velocity=velocity, material=material)
  scene += sphere
  kb.move_until_no_overlap(sphere, simulator, spawn_region=spawn_region)

# --- executes the simulation (and store keyframes)
simulator.run()

# --- renders the output
renderer.save_state("output/simulator.blend")
frames_dict = renderer.render()
kb.write_image_dict(frames_dict, "output")
