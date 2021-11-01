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
"""Kubric objects."""

import itertools
import numpy as np
import pyquaternion as pyquat
import traitlets as tl
from typing import Optional, Union, Tuple

from kubric.core import traits as ktl
from kubric.core import assets
from kubric.core import materials
from kubric.kubric_typing import ArrayLike


def ensure_3d_vector(x: ArrayLike) -> np.ndarray:
  x = np.asarray(x, dtype=np.float64)
  if x.shape != (3,):
    raise ValueError(f"Expected shape=(3,), got {x.shape}")
  return x


def normalize(
    x: ArrayLike,
    eps: float = 1.0e-8,
    fallback: Optional[ArrayLike] = None
) -> np.ndarray:
  x = np.asarray(x, dtype=np.float64)
  norm_x = np.linalg.norm(x)
  if norm_x < eps:
    if fallback is None:
      raise ValueError("Expected non-zero vector.")
    else:
      return np.asarray(fallback, dtype=np.float64)
  return x / norm_x


def are_orthogonal(x: ArrayLike, y: ArrayLike, eps: float = 1e-8) -> bool:
  x = np.asarray(x, dtype=np.float64)
  y = np.asarray(y, dtype=np.float64)
  assert x.ndim == 1, x.shape
  assert y.ndim == 1, y.shape
  return x.dot(y) < eps


def convert_str_direction_to_vector(direction: str) -> np.ndarray:
  return {
      "X": np.array([1., 0., 0.], dtype=np.float64),
      "Y": np.array([0., 1., 0.], dtype=np.float64),
      "Z": np.array([0., 0., 1.], dtype=np.float64),
      "-X": np.array([-1., 0., 0.], dtype=np.float64),
      "-Y": np.array([0., -1., 0.], dtype=np.float64),
      "-Z": np.array([0., 0., -1.], dtype=np.float64),
  }[direction.upper()]


def look_at_quat(
    position: ArrayLike,
    target: ArrayLike,
    up: Union[str, ArrayLike] = "Y",
    front: Union[str, ArrayLike] = "-Z",
) -> Tuple[float, float, float, float]:
  # convert directions to vectors if needed
  world_up = convert_str_direction_to_vector("Z")
  world_right = convert_str_direction_to_vector("X")
  if isinstance(up, str):
    up = convert_str_direction_to_vector(up)
  if isinstance(front, str):
    front = convert_str_direction_to_vector(front)

  up = normalize(ensure_3d_vector(up))
  front = normalize(ensure_3d_vector(front))
  right = np.cross(up, front)

  target = ensure_3d_vector(target)
  position = ensure_3d_vector(position)

  # construct the desired coordinate basis front, right, up
  look_at_front = normalize(target - position)
  look_at_right = normalize(np.cross(world_up, look_at_front), fallback=world_right)
  look_at_up = normalize(np.cross(look_at_front, look_at_right))

  rotation_matrix1 = np.stack([look_at_right, look_at_up, look_at_front])
  rotation_matrix2 = np.stack([right, up, front])
  return tuple(pyquat.Quaternion(matrix=(rotation_matrix1.T @ rotation_matrix2)))


def _euler_to_quat(euler_angles):
  """ Convert three (euler) angles around XYZ to a single quaternion."""
  q1 = pyquat.Quaternion(axis=[1., 0., 0.], angle=euler_angles[0])
  q2 = pyquat.Quaternion(axis=[0., 1., 0.], angle=euler_angles[1])
  q3 = pyquat.Quaternion(axis=[0., 0., 1.], angle=euler_angles[2])
  return tuple(q3 * q2 * q1)


class Object3D(assets.Asset):
  """
  Attributes:
    position (vec3d): the (x, y, z) position of the object.
    quaternion (vec4d): a (W, X, Y, Z) quaternion for describing the rotation.
    up (str): which direction to consider as "up" (required for look_at).
    front (str): which direction to consider as "front" (required for look_at).
  """
  position = ktl.Vector3D()
  quaternion = ktl.Quaternion()

  up = tl.CaselessStrEnum(["X", "Y", "Z", "-X", "-Y", "-Z"], default_value="Y")
  front = tl.CaselessStrEnum(["X", "Y", "Z", "-X", "-Y", "-Z"], default_value="-Z")

  def __init__(self, position=(0., 0., 0.), quaternion=None,
               up="Y", front="-Z", look_at=None, euler=None, **kwargs):
    if look_at is not None:
      assert quaternion is None and euler is None
      quaternion = look_at_quat(position, look_at, up, front)
    elif euler is not None:
      assert look_at is None and quaternion is None
      quaternion = _euler_to_quat(euler)
    elif quaternion is None:
      quaternion = (1., 0., 0., 0.)

    super().__init__(position=position, quaternion=quaternion, up=up,
                     front=front, **kwargs)

  def look_at(self, target):
    self.quaternion = look_at_quat(self.position, target, self.up, self.front)

  @property
  def rotation_matrix(self):
    """ Returns the rotation matrix corresponding to self.quaternion."""
    return pyquat.Quaternion(*self.quaternion).rotation_matrix

  @property
  def matrix_world(self):
    """ Returns the homogeneous transformation mapping points from world to object coordinates."""
    transformation = np.eye(4)
    transformation[:3, :3] = self.rotation_matrix
    transformation[:3, 3] = self.position
    return transformation


class PhysicalObject(Object3D):
  """ Base class for all 3D objects with a geometry and that can participate in physics simulation.

  Attributes:
    scale (vec3d): By how much the object is scaled along each of the 3 cardinal directions.

    velocity (vec3d): Vector of velocities along (X, Y, Z).
    angular_velocity (vec3d): Angular velocity of the object around the X, Y, and Z axis.

    static (bool): Whether this object is considered static or movable (default) by
                   the physics simulation.
    mass (float): Mass of the object in kg.
    friction (float): Friction coefficient used for physics simulation of this object
                      (between 0 and 1).
    restitution (float): Restitution (bounciness) coefficient used for physics simulation of this
                         object (between 0 and 1).

    bounds (Tuple[vec3d, vec3d]): An axis aligned bounding box around the object relative to its
                                  center, but ignoring any scaling or rotation.

    material (Material): Material assigned to this object.

    segmentation_id (int): The integer id that will be used for this object in segmentation maps.
                           Can be set to None, in which case the segmentation_id will correspond to
                           the index of the object within the scene.
  """

  scale = ktl.Scale()

  velocity = ktl.Vector3D()
  angular_velocity = ktl.Vector3D()

  static = tl.Bool(False)
  mass = tl.Float(1.0)
  friction = tl.Float(0.0)
  restitution = tl.Float(0.5)

  # TODO: a tuple of two numpy arrays is annoying to work with
  #       either convert to single 2D array or to as tuples
  bounds = tl.Tuple(ktl.Vector3D(), ktl.Vector3D(),
                    default_value=((0., 0., 0.), (0., 0., 0.)))

  material = ktl.AssetInstance(materials.Material,
                               default_value=materials.UndefinedMaterial())

  # If the segmentation_id is None, we use the "default" segmentation label for
  # this object. Otherwise, we use the segmentation_id specified here. This
  # allows us to perform things like "instance segmentation" and
  # "semantic segmentation", from a labelmap in the Kubric worker, even if the
  # objects in the scene are generated in different orders on different runs.
  segmentation_id = tl.Integer(None, allow_none=True)

  @tl.validate("mass")
  def _valid_mass(self, proposal):
    mass = proposal["value"]
    if mass < 0:
      raise tl.TraitError(f"mass cannot be negative ({mass})")
    return mass

  @tl.validate("friction")
  def _valid_friction(self, proposal):
    friction = proposal["value"]
    if friction < 0:
      raise tl.TraitError(f"friction cannot be negative ({friction})")
    if friction > 1.0:
      raise tl.TraitError(f"friction cannot be larger than 1.0 ({friction})")
    return friction

  @tl.validate("restitution")
  def _valid_restitution(self, proposal):
    restitution = proposal["value"]
    if restitution < 0:
      raise tl.TraitError(f"restitution cannot be negative ({restitution})")
    if restitution > 1.0:
      raise tl.TraitError(f"restitution cannot be larger than 1.0 ({restitution})")
    return restitution

  @tl.validate("bounds")
  def _valid_bounds(self, proposal):
    lower, upper = proposal["value"]
    for l, u in zip(lower, upper):
      if l > u:
        raise tl.TraitError(f"lower bound cannot be larger than upper bound ({lower} !<= {upper})")
    return lower, upper

  @property
  def bbox_3d(self):
    """ 3D bounding box as an array of 8 corners (shape = [8, 3])"""
    bounds = np.array(self.bounds, dtype=np.float)
    # scale bounds:
    bounds *= self.scale
    # construct list of bbox corners
    bbox_points = itertools.product(bounds[:, 0], bounds[:, 1], bounds[:, 2])
    # rotate the bbox
    obj_orientation = pyquat.Quaternion(*self.quaternion)
    rotated_bbox_points = [obj_orientation.rotate(x) for x in bbox_points]
    # shift by self.position and convert to single np.array
    return np.array([self.position + x for x in rotated_bbox_points])

  @property
  def aabbox(self):
    """ Axis-aligned bounding box [(min_x, min_y, min_y), (max_x, max_y, max_z)]."""
    bbox3d = self.bbox_3d
    # compute an axis aligned bounding box around the rotated bbox
    axis_aligned_bbox = np.array([bbox3d.min(axis=0), bbox3d.max(axis=0)])
    return axis_aligned_bbox


class Cube(PhysicalObject):
  @tl.default("bounds")
  def _get_bounds_default(self):
    return (-1, -1, -1), (1, 1, 1)


class Sphere(PhysicalObject):
  @tl.default("bounds")
  def _get_bounds_default(self):
    return (-1, -1, -1), (1, 1, 1)


class FileBasedObject(PhysicalObject):
  asset_id = tl.Unicode()

  # TODO: use tfds.core.utils.type_utils.ReadWritePath instead
  simulation_filename = tl.Unicode(allow_none=True)
  render_filename = tl.Unicode(allow_none=True)
  render_import_kwargs = tl.Dict(key_trait=tl.ObjectName())

  # TODO: trigger error when changing filenames or asset-id after the fact
