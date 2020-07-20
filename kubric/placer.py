# Copyright 2020 Google LLC
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
from kubric.simulator import Simulator


class Placer(object):
  # TODO: rename to "Initializer?"
  def __init__(self, template: str = None, simulator: Simulator = None):
    assert template == "sphereworld"
    self.simulator = simulator
    # TODO: where to store planar geometry?
    # self.simulator.load_object("urdf/plane.urdf")

  def place(self, object_id: int):
    # TODO: brutally hardcoded implementation
    if object_id == 0: self.simulator.place_object(object_id, position=(-.2, 0, 1))
    if object_id == 1: self.simulator.place_object(object_id, position=(+.0, 0, 1))
    if object_id == 2: self.simulator.place_object(object_id, position=(+.2, 0, 1))
    # TODO: self.simlator.set_velocity(...)
