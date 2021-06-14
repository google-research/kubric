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
"""Kubric lights module."""

import traitlets as tl

from kubric.core import traits as ktl

from kubric.core import objects
from kubric.core.assets import UndefinedAsset
from kubric.core.color import get_color


class Light(objects.Object3D):
  color = ktl.RGB(default_value=get_color("white").rgb)
  intensity = tl.Float(1.)

  @tl.default("background")
  def get_background_default(self):
    return True


class UndefinedLight(Light, UndefinedAsset):
  pass


class DirectionalLight(Light):
  shadow_softness = tl.Float(0.2)


class RectAreaLight(Light):
  width = tl.Float(1)
  height = tl.Float(1)


class PointLight(Light):
  pass
