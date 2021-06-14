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

import itertools
import mathutils
import numpy as np
import traitlets as tl

from kubric.core import traits as ktl
from kubric.core import assets
from kubric.core import materials


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
      direction = mathutils.Vector(look_at) - mathutils.Vector(position)
      quaternion = direction.to_track_quat(self.front.upper(), self.up.upper())
    elif euler is not None:
      assert look_at is None and quaternion is None
      quaternion = mathutils.Euler(euler).to_quaternion()
    elif quaternion is None:
      quaternion = (1., 0., 0., 0.)

    super().__init__(position=position, quaternion=quaternion, up=up,
                     front=front, **kwargs)

  def look_at(self, target):
    direction = mathutils.Vector(target) - mathutils.Vector(self.position)
    self.quaternion = direction.to_track_quat(self.front.upper(), self.up.upper())

  @property
  def euler_xyz(self):
    return np.array(mathutils.Quaternion(self.quaternion).to_euler())

  @property
  def R(self):
    """Rotation matrix that rotates from world to object coordinates"""
    return np.array(mathutils.Quaternion(self.quaternion).to_matrix())

  @property
  def matrix_world(self):
    """ Affine transformation 4x4 matrix to map points from world to object coordinates."""
    RT = np.eye(4)
    RT[:3, :3] = self.R
    RT[:3, 3] = self.position
    return RT


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
    # construct list of bbox edges
    bbox_points = [mathutils.Vector(x)
                   for x in itertools.product(bounds[:, 0], bounds[:, 1], bounds[:, 2])]
    # rotate the
    obj_orientation = mathutils.Quaternion(self.quaternion)
    for x in bbox_points:
      x.rotate(obj_orientation)  # rotates x in place by the object orientation

    # shift by self.position and convert to np.array
    return np.array([self.position + tuple(x) for x in bbox_points])

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
