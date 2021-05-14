# Copyright 2021 The Kubric Authors.
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

import numpy as np
import kubric as kb

scene = kb.Scene(resolution=(256, 256))
simulator = kb.simulator.PyBullet(scene)
renderer = kb.renderer.Blender(scene)

floor = kb.Cube(scale=(1, 1, 0.1), position=(0, 0, -0.1), static=True)
north_wall = kb.Cube(scale=(1.2, 0.1, 0.5), position=(0, 1.1, 0.3), static=True)
south_wall = kb.Cube(scale=(1.2, 0.1, 0.5), position=(0, -1.1, 0.3), static=True)
east_wall = kb.Cube(scale=(0.1, 1, 0.5), position=(1.1, 0, 0.3), static=True)
west_wall = kb.Cube(scale=(0.1, 1, 0.5), position=(-1.1, 0, 0.3), static=True)
scene.add([floor, north_wall, south_wall, east_wall, west_wall])

sun = kb.DirectionalLight(position=(-1, -0.5, 3), look_at=(0, 0, 0), intensity=1.5)
scene.add(sun)

scene.camera = kb.PerspectiveCamera(position=(2, -0.5, 4), look_at=(0, 0, 0))

rng = np.random.default_rng()
spawn_region = [[-1, -1, 0], [1, 1, 1]]   # [low, high] bounds of the volume for spawning balls
for i in range(8):
  ball = kb.Sphere(scale=0.1, position=rng.uniform(*spawn_region),
                   velocity=rng.uniform([-1, -1, 0], [1, 1, 0]))
  ball.material = kb.PrincipledBSDFMaterial(color=kb.random_hue_color(rng=rng))
  scene.add(ball)
  kb.move_until_no_overlap(ball, simulator, spawn_region=spawn_region)

renderer.save_state("getting_started.blend")
