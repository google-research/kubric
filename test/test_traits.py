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
import pytest
import traitlets as tl

import kubric.traits as ktl


@pytest.fixture
def obj():
  class TestObject(tl.HasTraits):
    quaternion = ktl.Quaternion()
    position = ktl.Vector3D()
    scale = ktl.Vector3D(default_value=(1., 1., 1.))
    rgb = ktl.RGB()
    rgba = ktl.RGBA()

  return TestObject()


def test_default_values(obj):
  assert obj.quaternion == (1, 0, 0, 0)
  assert obj.position == (0, 0, 0)
  assert obj.scale == (1, 1, 1)
  assert obj.rgb == (0, 0, 0)
  assert obj.rgba == (0, 0, 0, 1)


def test_set_sequence(obj):
  obj.quaternion = (1, 2, 3, 4)
  assert obj.quaternion == (1, 2, 3, 4)

  obj.quaternion = [1, 2, 3, 4]
  assert obj.quaternion == (1, 2, 3, 4)

  obj.position = (3, 2, 1)
  assert obj.position == (3, 2, 1)

  obj.position = np.array([0.1, 0.2, 0.3])
  assert obj.position == (0.1, 0.2, 0.3)

  obj.scale = [2, 2, 2]
  assert obj.scale == (2, 2, 2)

  obj.rgb = (0.5, 0.2, 0.1)
  assert obj.rgb == (0.5, 0.2, 0.1)

  obj.rgba = [1., 0.8, 0.6, 0.4]
  assert obj.rgba == (1., 0.8, 0.6, 0.4)


def test_raises_on_invalid_sequence_length(obj):
  with pytest.raises(tl.TraitError):
    obj.position = (1, 2, 3, 4)

  with pytest.raises(tl.TraitError):
    obj.quaternion = (1, 2, 3)

  with pytest.raises(tl.TraitError):
    obj.rgb = (1, 1, 1, 1)

  with pytest.raises(tl.TraitError):
    obj.rgba = (1, 1)


