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
"""Testing for `kubric.core` module."""

import re

import pytest
import numpy as np
from numpy.testing import assert_allclose
import pyquaternion as pyquat
from traitlets import TraitError
from unittest import mock

from kubric.core import Asset
from kubric.core import UndefinedAsset
from kubric.core import objects
from kubric.core import materials


def test_asset_default_uid():
  # first instance of an object should have the name of object class
  a = Asset()
  assert a.uid
  assert isinstance(a.uid, str)
  assert a.uid == a.__class__.__name__


def test_asset_progressive_uids():
  # first instance of an object should have the name of object class
  a = Asset(name="Foo")
  b = Asset(name="Foo")
  c = Asset(name="Foo")
  assert a.uid == "Foo"
  assert b.uid == "Foo.001"
  assert c.uid == "Foo.002"


def test_asset_name_readonly():
  a = Asset()
  with pytest.raises(TraitError):
    a.name = "Foo"


def test_asset_uid_readonly():
  a = Asset()
  with pytest.raises(TraitError, match=r".*uid.* trait is read-only.*"):
    a.uid = "abc"


def test_undefined_asset_uid():
  a = UndefinedAsset()
  b = UndefinedAsset()
  assert a.uid == "UndefinedAsset"
  assert b.uid == "UndefinedAsset"


def test_asset_raises_unknown_traits():
  with pytest.raises(KeyError, match=r".*'doesnotexist'.*"):
    Asset(doesnotexist=1)


def test_asset_hash_and_eq():
  a = b = Asset()
  c = Asset()

  assert hash(a)
  assert hash(a) == hash(b)
  assert hash(a) != hash(c)
  assert a == a == b
  assert a != c


def test_asset_repr():
  a = Asset()
  assert re.match(r"^<Asset.[0-9][0-9][0-9].*>$", repr(a)) is not None


def test_object3d_constructor_default_args():
  obj = objects.Object3D()
  assert_allclose(obj.position, (0, 0, 0))
  assert_allclose(obj.quaternion, (1, 0, 0, 0))
  assert obj.up == "Y"
  assert obj.front == "-Z"


def test_object3d_constructor():
  obj = objects.Object3D(position=(1, 2, 3), quaternion=(0, 1, 0, 0),
                         up="Z", front="Y")
  assert_allclose(obj.position, (1, 2, 3))
  assert_allclose(obj.quaternion, (0, 1, 0, 0))
  assert obj.up == "Z"
  assert obj.front == "Y"


def test_object3d_constructor_raises_unknown_trait():
  with pytest.raises(KeyError):
    objects.Object3D(location=(1, 2, 3))


def test_look_at_quat_front_points_toward_target():
  position = (0., 0., 0.)
  target = (1., 1., 1.)
  front = "-Z"
  up = "Y"
  q = pyquat.Quaternion(*objects.look_at_quat(position, target, up, front))
  direction = q.rotate(np.array([0, 0, -1.]))
  dir_expected = np.array(target) / np.linalg.norm(target)

  assert np.allclose(direction, dir_expected)


def test_object3d_constructor_look_at():
  obj = objects.Object3D(look_at=(0, 0, 1))
  assert_allclose(obj.quaternion, (0, 0, 1, 0), atol=1e-6)  # TODO: double check


def test_object3d_constructor_euler():
  obj = objects.Object3D(euler=(np.pi, 0, 0))
  assert_allclose(obj.quaternion, (0, 1, 0, 0), atol=1e-6)


def test_object3d_look_at():
  obj = objects.Object3D()
  obj.look_at((0, 0, 1))
  assert_allclose(obj.quaternion, (0, 0, 1, 0), atol=1e-6)  # TODO: double check


def test_physical_object_constructor_default_args():
  obj = objects.PhysicalObject()
  assert_allclose(obj.position, (0, 0, 0))
  assert_allclose(obj.quaternion, (1, 0, 0, 0))
  assert obj.up == "Y"
  assert obj.front == "-Z"
  assert_allclose(obj.scale, (1, 1, 1))
  assert_allclose(obj.velocity, (0, 0, 0))
  assert_allclose(obj.angular_velocity, (0, 0, 0))
  assert obj.mass == 1.0
  assert obj.friction == 0
  assert obj.restitution == 0.5
  assert obj.static is False
  assert_allclose(obj.bounds, ((0, 0, 0), (0, 0, 0)))
  assert isinstance(obj.material, materials.UndefinedMaterial)


def test_physicalobject_constructor_look_at():
  obj = objects.PhysicalObject(look_at=(0, 0, 1))
  assert_allclose(obj.quaternion, (0, 0, 1, 0), atol=1e-6)  # TODO: double check


def test_physicalobject_constructor_euler():
  obj = objects.PhysicalObject(euler=(np.pi, 0, 0))
  assert_allclose(obj.quaternion, (0, 1, 0, 0), atol=1e-6)


def test_physicalobject_constructor():
  mat = materials.FlatMaterial()
  obj = objects.PhysicalObject(position=(1, 2, 4), quaternion=(0, 0, 1, 0), scale=(3, 3, 3),
                               up="Y", front="X", velocity=(2, 3, 4), angular_velocity=(-1, -1, -1),
                               mass=2.4, friction=0.8, restitution=1.0, static=True,
                               bounds=((-1, -2, -3), (1, 2, 3)), material=mat)
  assert_allclose(obj.position, (1, 2, 4))
  assert_allclose(obj.quaternion, (0, 0, 1, 0))
  assert obj.up == "Y"
  assert obj.front == "X"
  assert_allclose(obj.velocity, (2, 3, 4))
  assert_allclose(obj.angular_velocity, (-1, -1, -1))
  assert obj.mass == 2.4
  assert obj.friction == 0.8
  assert obj.restitution == 1.0
  assert obj.static is True
  assert_allclose(obj.bounds, ((-1, -2, -3), (1, 2, 3)))
  assert obj.material is mat


def test_physicalobject_mass_validation():
  with pytest.raises(TraitError):
    obj = objects.PhysicalObject(mass=-1)

  obj = objects.PhysicalObject()
  with pytest.raises(TraitError):
    obj.mass = -2


def test_physicalobject_friction_validation():
  with pytest.raises(TraitError):
    obj = objects.PhysicalObject(friction=-1)

  obj = objects.PhysicalObject()
  with pytest.raises(TraitError):
    obj.friction = -2

  obj = objects.PhysicalObject()
  with pytest.raises(TraitError):
    obj.friction = 2


def test_physicalobject_restitution_validation():
  with pytest.raises(TraitError):
    obj = objects.PhysicalObject(restitution=-1)

  obj = objects.PhysicalObject()
  with pytest.raises(TraitError):
    obj.restitution = -2

  obj = objects.PhysicalObject()
  with pytest.raises(TraitError):
    obj.restitution = 2


def test_physicalobject_bounds_validation():
  with pytest.raises(TraitError):
    obj = objects.PhysicalObject(bounds=((1, 1, 1), (-1, 2, 2)))

  obj = objects.PhysicalObject()
  with pytest.raises(TraitError):
    obj.bounds = ((1, 1, 1), (3, 0, 2))


def test_cube_default_bounds():
  cube = objects.Cube(position=(2, 2, 2))
  assert np.all(cube.bounds[0] == (-1, -1, -1))
  assert np.all(cube.bounds[1] == (1, 1, 1))


def test_sphere_default_bounds():
  sphere = objects.Sphere(position=(1, 2, 3))
  assert np.all(sphere.bounds[0] == (-1, -1, -1))
  assert np.all(sphere.bounds[1] == (1, 1, 1))


def test_object_aabbox_translation_and_scale():
  cube = objects.Cube(position=(1, 2, 3), scale=(1, 2, 0.5))
  lower, upper = cube.aabbox

  assert np.all(lower == (0, 0, 2.5))
  assert np.all(upper == (2, 4, 3.5))


def test_object_aabbox_rotation():
  cube = objects.Cube(look_at=(1, 1, 0))  # 45 degree rotation around z
  lower, upper = cube.aabbox

  sqrt2 = np.sqrt(2)
  np.testing.assert_allclose(lower, (-sqrt2, -sqrt2, -1), atol=1e-5)
  np.testing.assert_allclose(upper, (sqrt2, sqrt2, 1), atol=1e-5)


def test_keyframe_insert_raises_for_unknown_trait():
  obj = objects.Object3D()
  with pytest.raises(KeyError):
    obj.keyframe_insert("doesnotexist", 7)


def test_keyframe_insert():
  obj = objects.Object3D()
  handler = mock.Mock()
  obj.observe(handler, "position", type="keyframe")

  obj.position = (3, 3, 3)
  obj.keyframe_insert("position", 7)

  assert handler.call_count == 1
  change_argument = handler.call_args[0][0]
  assert change_argument.name == "position"
  assert change_argument.owner == obj
  assert change_argument.frame == 7
  assert change_argument.type == "keyframe"

