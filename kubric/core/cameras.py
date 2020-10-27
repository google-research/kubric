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

import numpy as np
import traitlets as tl

from kubric.core import base
from kubric.core import objects

__all__ = ("Camera", "UndefinedCamera", "PerspectiveCamera", "OrthographicCamera")


class Camera(objects.Object3D):
  pass


class UndefinedCamera(Camera, base.Undefined):
  pass


class PerspectiveCamera(Camera):
  focal_length = tl.Float(50)
  sensor_width = tl.Float(36)

  def __init__(self, focal_length=50, sensor_width=36, position=(0., 0., 0.),
               quaternion=None, up="Y", front="-Z", look_at=None, euler=None, **kwargs):
    super().__init__(focal_length=focal_length, sensor_width=sensor_width, position=position,
                     quaternion=quaternion, up=up, front=front, look_at=look_at, euler=euler,
                     **kwargs)

  @property
  def field_of_view(self):
    """ The (horizontal) field of view in radians. """
    return 2 * np.arctan(self.sensor_width / 2 / self.focal_length)


class OrthographicCamera(Camera):
  orthographic_scale = tl.Float(6.0)

  def __init__(self, orthographic_scale=6.0, position=(0., 0., 0.),
               quaternion=None, up="Y", front="-Z", look_at=None, euler=None, **kwargs):
    super().__init__(orthographic_scale=orthographic_scale, position=position,
                     quaternion=quaternion, up=up, front=front, look_at=look_at, euler=euler,
                     **kwargs)
