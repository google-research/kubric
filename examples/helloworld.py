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
from kubric.renderer.blender import Blender as KubricRenderer

logging.basicConfig(level="INFO")

# --- create scene and attach a renderer to it
scene = kb.Scene(resolution=(256, 256))
renderer = KubricRenderer(scene)

# --- populate the scene with objects, lights, cameras
scene += kb.Cube(name="floor", scale=(10, 10, 0.1), position=(0, 0, -0.1))
scene += kb.Sphere(name="ball", scale=1, position=(0, 0, 1.))
scene += kb.DirectionalLight(name="sun", position=(-1, -0.5, 3),
                             look_at=(0, 0, 0), intensity=1.5)
scene += kb.PerspectiveCamera(name="camera", position=(3, -1, 4),
                              look_at=(0, 0, 1))

# --- render (and save the blender file)
renderer.save_state("output/helloworld.blend")
frame = renderer.render_still()

# --- save the output as pngs
kb.write_png(frame["rgba"], "output/helloworld.png")
kb.write_palette_png(frame["segmentation"], "output/helloworld_segmentation.png")
scale = kb.write_scaled_png(frame["depth"], "output/helloworld_depth.png")
logging.info("Depth scale: %s", scale)
