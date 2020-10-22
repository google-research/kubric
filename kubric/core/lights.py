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

import traitlets as tl

import kubric.core.traits as ktl
from kubric.core import base
from kubric.core import color
from kubric.core import objects


__all__ = ("Light", "UndefinedLight", "DirectionalLight", "RectAreaLight", "PointLight")


class Light(objects.Object3D):
  color = ktl.RGB(default_value=color.get_color("white").rgb)
  intensity = tl.Float(1.)


class UndefinedLight(Light, base.Undefined):
  pass


class DirectionalLight(Light):
  shadow_softness = tl.Float(0.2)


class RectAreaLight(Light):
  width = tl.Float(1)
  height = tl.Float(1)


class PointLight(Light):
  pass
