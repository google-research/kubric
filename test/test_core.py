import re
from unittest import mock

import pytest
import numpy as np
from numpy.testing import assert_allclose
from traitlets import TraitError

from kubric import core


def test_asset_has_uid():
  a = core.Asset()
  assert a.uid
  assert isinstance(a.uid, str)
  assert re.match(r"^[0-9][0-9][0-9]:Asset$", a.uid) is not None


def test_asset_uid_readonly():
  a = core.Asset()
  with pytest.raises(TraitError, match=r".*uid.* trait is read-only.*"):
    a.uid = "abc"


def test_asset_raises_unknown_traits():
  with pytest.raises(KeyError, match=r".*'doesnotexist'.*") as e:
    core.Asset(doesnotexist=1)


def test_asset_hash_and_eq():
  a = b = core.Asset()
  c = core.Asset()

  assert hash(a)
  assert hash(a) == hash(b)
  assert hash(a) != hash(c)
  assert a == a == b
  assert a != c


def test_asset_destruction_callback():
  a = core.Asset()
  callback = mock.Mock()
  a.destruction_callbacks.append(callback)

  del a

  callback.assert_called_once()


def test_asset_repr():
  a = core.Asset()
  assert re.match(r"^<[0-9][0-9][0-9]:Asset>$", repr(a)) is not None


def test_object3d_constructor_default_args():
  obj = core.Object3D()
  assert_allclose(obj.position, (0, 0, 0))
  assert_allclose(obj.quaternion, (1, 0, 0, 0))
  assert obj.up == "Y"
  assert obj.front == "-Z"


def test_object3d_constructor():
  obj = core.Object3D(position=(1, 2, 3), quaternion=(0, 1, 0, 0),
                      up="Z", front="Y")
  assert_allclose(obj.position, (1, 2, 3))
  assert_allclose(obj.quaternion, (0, 1, 0, 0))
  assert obj.up == "Z"
  assert obj.front == "Y"


def test_object3d_constructor_raises_unknown_trait():
  with pytest.raises(KeyError):
    core.Object3D(location=(1, 2, 3))


def test_object3d_constructor_look_at():
  obj = core.Object3D(look_at=(0, 0, 1))
  assert_allclose(obj.quaternion, (0, 0, -1, 0), atol=1e-6)  # TODO: double check


def test_object3d_constructor_euler():
  obj = core.Object3D(euler=(np.pi, 0, 0))
  assert_allclose(obj.quaternion, (0, 1, 0, 0), atol=1e-6)
