# Copyright 2022 The Kubric Authors.
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

import numpy as np
import traitlets as tl

from kubric.core.assets import UndefinedAsset
from kubric.core import objects
from kubric.kubric_typing import ArrayLike


class Camera(objects.Object3D):
  """ Base class for all types of cameras. """

  @tl.default("background")
  def _get_background_default(self):
    return True

  @property
  def intrinsics(self):
    raise NotImplementedError

  def project_point(self, point3d, frame=None):
    """ Compute the image space coordinates [0, 1] for a given point in world coordinates."""
    with self.at_frame(frame):
      homo_transform = np.linalg.inv(self.matrix_world)
      homo_intrinsics = np.zeros((3, 4), dtype=np.float32)
      homo_intrinsics[:, :3] = self.intrinsics

      point4d = np.concatenate([point3d, [1.]])
      projected = homo_intrinsics @ homo_transform @ point4d
      image_coords = projected / projected[2]
      image_coords[2] = np.sign(projected[2])
      return image_coords

  def z_to_depth(self, z: ArrayLike) -> np.ndarray:
    raise NotImplementedError


class UndefinedCamera(Camera, UndefinedAsset):
  """ Marker object that indicates that a camera instance attribute has not been set. """


class PerspectiveCamera(Camera):
  """ A :class:`Camera` that uses perspective projection.

  Args:
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
    """ The (horizontal) field of view (fov) in radians.

    .. math:: \\texttt{fov} = 2 * \\arctan{ \\frac{\\texttt{sensor_width}}{2 * \\texttt{fl}} }

    Setting the :py:attr:`field_of_view` will internally adjust the :py:obj:`focal_length` (fl),
    but keep the :py:attr:`sensor_width`.
    """
    return 2 * np.arctan2(self.sensor_width / 2, self.focal_length)

  @field_of_view.setter
  def field_of_view(self, fov: float) -> None:
    self.focal_length = self.sensor_width / (2 * np.tan(fov / 2))

  @property
  def sensor_height(self):
    scene = self.active_scene
    return self.sensor_width / scene.resolution[0] * scene.resolution[1]

  @property
  def intrinsics(self):
    width, height = 1., 1.  # self.active_scene.resolution
    f_x = self.focal_length / self.sensor_width * width
    f_y = self.focal_length / self.sensor_height * height
    p_x = width / 2.
    p_y = height / 2.
    return np.array([
        [f_x, 0, -p_x],
        [0, -f_y, -p_y],
        [0,   0,   -1],
    ])

  def z_to_depth(self, z: ArrayLike) -> np.ndarray:
    z = np.array(z)
    assert z.ndim >= 3
    h, w, _ = z.shape[-3:]

    pixel_centers_x = (np.arange(-w/2, w/2, dtype=np.float32) + 0.5) / w * self.sensor_width
    pixel_centers_y = (np.arange(-h/2, h/2, dtype=np.float32) + 0.5) / h * self.sensor_height
    squared_distance_from_center = np.sum(np.square(np.meshgrid(
        pixel_centers_x,  # X-Axis (columns)
        pixel_centers_y,  # Y-Axis (rows)
        indexing="xy",
    )), axis=0)

    depth_scaling = np.sqrt(1 + squared_distance_from_center / self.focal_length**2)
    depth_scaling = depth_scaling.reshape((1,) * (z.ndim - 3) + depth_scaling.shape + (1,))
    return z * depth_scaling


class OrthographicCamera(Camera):
  """A :class:`Camera` that uses orthographic projection."""
  orthographic_scale = tl.Float(6.0)

  def __init__(self, orthographic_scale=6.0, position=(0., 0., 0.),
               quaternion=None, up="Y", front="-Z", look_at=None, euler=None, **kwargs):
    super().__init__(orthographic_scale=orthographic_scale, position=position,
                     quaternion=quaternion, up=up, front=front, look_at=look_at, euler=euler,
                     **kwargs)

  @property
  def intrinsics(self):
    fx = fy = 2.0 / self.orthographic_scale

    return np.array([
        [fx, 0,  0],
        [0, fy,  0],
        [0,   0,   -1],
    ])

  def z_to_depth(self, z: ArrayLike) -> np.ndarray:
    # not sure if depth is even well defined in orthographic
    # for now just return the z value
    return z
