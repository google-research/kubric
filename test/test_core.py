import re

import pytest
import numpy as np
from numpy.testing import assert_allclose
from traitlets import TraitError

from kubric.core import base
from kubric.core import objects


def test_asset_has_uid():
  a = base.Asset()
  assert a.uid
  assert isinstance(a.uid, str)
  assert re.match(r"^[0-9][0-9][0-9]:Asset$", a.uid) is not None


def test_asset_uid_readonly():
  a = base.Asset()
  with pytest.raises(TraitError, match=r".*uid.* trait is read-only.*"):
    a.uid = "abc"


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
  assert re.match(r"^<[0-9][0-9][0-9]:Asset>$", repr(a)) is not None


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
