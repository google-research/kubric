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

import re

import pytest
import numpy as np
from numpy.testing import assert_allclose
from traitlets import TraitError
from unittest import mock

from kubric.core import base
from kubric.core import objects


def test_asset_has_uid():
  a = base.Asset()
  assert a.uid
  assert isinstance(a.uid, str)
  assert re.match(r"^Asset:[0-9][0-9]$", a.uid) is not None


def test_asset_uid_readonly():
  a = base.Asset()
  with pytest.raises(TraitError, match=r".*uid.* trait is read-only.*"):
    a.uid = "abc"


def test_undefined_asset_uid():
  a = base.UndefinedAsset()
  b = base.UndefinedAsset()
  assert a.uid == "UndefinedAsset"
  assert b.uid == "UndefinedAsset"


def test_asset_raises_unknown_traits():
  with pytest.raises(KeyError, match=r".*'doesnotexist'.*") as e:
    base.Asset(doesnotexist=1)


def test_asset_hash_and_eq():
  a = b = base.Asset()
  c = base.Asset()

  assert hash(a)
  assert hash(a) == hash(b)
  assert hash(a) != hash(c)
  assert a == a == b
  assert a != c


def test_asset_repr():
  a = base.Asset()
  assert re.match(r"^<Asset:[0-9][0-9]>$", repr(a)) is not None


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


def test_object3d_constructor_look_at():
  obj = objects.Object3D(look_at=(0, 0, 1))
  assert_allclose(obj.quaternion, (0, 0, -1, 0), atol=1e-6)  # TODO: double check


def test_object3d_constructor_euler():
  obj = objects.Object3D(euler=(np.pi, 0, 0))
  assert_allclose(obj.quaternion, (0, 1, 0, 0), atol=1e-6)


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

