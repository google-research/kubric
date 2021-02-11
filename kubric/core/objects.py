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

import mathutils
import numpy as np
import traitlets as tl

from kubric.core import traits as ktl
from kubric.core import base
from kubric.core import materials

__all__ = ("Object3D", "PhysicalObject", "Sphere", "Cube", "FileBasedObject")


class Object3D(base.Asset):
  """
  Attributes:
    position (vec3d): the (x, y, z) position of the object.
    quaternion (vec4d): a (W, X, Y, Z) quaternion for describing the rotation.
    up (str): which direction to consider as "up" (required for look_at).
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


class PhysicalObject(Object3D):
  scale = ktl.Scale()

  velocity = ktl.Vector3D()
  angular_velocity = ktl.Vector3D()

  static = tl.Bool(False)
  mass = tl.Float(1.0)
  friction = tl.Float(0.0)
  restitution = tl.Float(0.5)

  bounds = tl.Tuple(ktl.Vector3D(), ktl.Vector3D(),
                    default_value=((0., 0., 0.), (0., 0, 0.)))

  material = ktl.AssetInstance(materials.Material,
                               default_value=materials.UndefinedMaterial())

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


class Cube(PhysicalObject):
  pass


class Sphere(PhysicalObject):
  pass


class FileBasedObject(PhysicalObject):
  asset_id = tl.Unicode()

  simulation_filename = tl.Unicode()   # TODO: use pathlib.Path instead
  render_filename = tl.Unicode()       # TODO: use pathlib.Path instead

  # TODO: trigger error when changing filenames or asset-id after the fact
