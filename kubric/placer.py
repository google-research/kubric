# Copyright 2020 The Kubric Authors
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
from kubric.simulator import Object3D

_NUM_PLACED:int = 0

class Placer(object):
  """TODO: perhaps rename to Initializer?"""
  
  def __init__(self, template: str = None, simulator: Simulator = None):
    assert template == "sphereworld"
    self.simulator = simulator
    # TODO: where to store planar geometry?
    # self.simulator.load_object("urdf/plane.urdf")

  def place(self, obj3d: Object3D):
    global _NUM_PLACED
    # TODO: brutally hardcoded implementation
    if _NUM_PLACED==0: obj3d.position=(-.2, 0, 1)
    if _NUM_PLACED==1: obj3d.position=(+.0, 0, 1)
    if _NUM_PLACED==2: obj3d.position=(+.2, 0, 1)
    _NUM_PLACED=_NUM_PLACED+1

    # TODO: self.simlator.set_velocity(...)

