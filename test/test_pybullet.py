# Copyright 2023 The Kubric Authors.
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
"""Testing for `kubric.simulator.pybullet` module."""

import kubric as kb
from kubric.simulator.pybullet import PyBullet as KubricSimulator
import numpy as np


def test_basic_simulator():
  scene = kb.Scene(
      gravity=(0, -10, 0),  # A planet slightly larger than Earth.
      frame_end=24,  # One second.
  )
  simulator = KubricSimulator(scene)
  cube = kb.Cube(
      name='box',
      position=[0, 0, 0],
  )
  scene.add(cube)
  simulator.run()
  np.testing.assert_allclose(cube.position[1], -0.5 * 10, atol=0.1)


def test_simulator_in_loop():
  # https://github.com/google-research/kubric/issues/208
  # https://github.com/google-research/kubric/issues/234
  for _ in range(10):
    scene = kb.Scene(
        gravity=(0, -10, 0),  # A planet slightly larger than Earth.
        frame_end=24,  # One second.
    )
    simulator = KubricSimulator(scene)
    cube = kb.Cube(
        name='box',
        position=[0, 0, 0],
    )
    scene.add(cube)
    simulator.run()
    np.testing.assert_allclose(cube.position[1], -0.5 * 10, atol=0.1)
