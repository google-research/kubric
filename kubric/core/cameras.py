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
  """ Base class for all types of cameras. """

  @tl.default("background")
  def _get_background_default(self):
    return True


class UndefinedCamera(Camera, base.Undefined):
  """ Marker object that indicates that a camera instance attribute has not been set. """
  pass


class PerspectiveCamera(Camera):
  """ A :class:`Camera` that uses perspective projection.

  Args:
    focal_length (float)

  Attributes:
    focal_length (float): The focal length of the camera lens in `mm`.
                          `Default = 50`

    sensor_width (float): Horizontal size of the camera sensor in `mm`.
                          `Default = 36`

  """

  focal_length = tl.Float(50)

  sensor_width = tl.Float(36)

  def __init__(self,
               focal_length: float = 50,
               sensor_width: float = 36,
               position=(0., 0., 0.),
               quaternion=None, up="Y", front="-Z", look_at=None, euler=None, **kwargs):
    super().__init__(focal_length=focal_length, sensor_width=sensor_width, position=position,
                     quaternion=quaternion, up=up, front=front, look_at=look_at, euler=euler,
                     **kwargs)

  @property
  def field_of_view(self) -> float:
    """ The (horizontal) field of view in radians.

    .. math:: \\texttt{field_of_view} = 2 * \\arctan{ \\frac{\\texttt{sensor_width}}{2 * \\texttt{focal_length}} }

    Setting the :py:attr:`field_of_view` will internally adjust the :py:obj:`focal_length`, but keep the :py:attr:`sensor_width`.
    """
    return 2 * np.arctan(self.sensor_width / (2 * self.focal_length))

  @field_of_view.setter
  def field_of_view(self, fov: float) -> None:
    self.focal_length = self.sensor_width / (2 * np.tan(fov / 2))


class OrthographicCamera(Camera):
  orthographic_scale = tl.Float(6.0)

  def __init__(self, orthographic_scale=6.0, position=(0., 0., 0.),
               quaternion=None, up="Y", front="-Z", look_at=None, euler=None, **kwargs):
    super().__init__(orthographic_scale=orthographic_scale, position=position,
                     quaternion=quaternion, up=up, front=front, look_at=look_at, euler=euler,
                     **kwargs)
