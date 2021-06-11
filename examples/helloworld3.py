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

logging.basicConfig(level="WARNING") #< CRITICAL, ERROR, WARNING, INFO, DEBUG

# --- create scene and attach a renderer to it
scene = kb.Scene(resolution=(256, 256))
renderer = kb.renderer.Blender(scene)

# --- populate the scene with objects, lights, cameras
scene += kb.DirectionalLight(position=(-1, -0.5, 3), look_at=(0, 0, 0), intensity=1.5)
scene += kb.Cube(scale=(10, 10, 0.1), position=(0, 0, -0.1))
scene += kb.Sphere(scale=1, position=(0, 0, .5))
scene += kb.PerspectiveCamera(position=(2, -0.5, 4), look_at=(0, 0, 0))

# --- render (and save the blender file) 
renderer.save_state("helloworld.blend")
renderer.render_still("helloworld.png")
