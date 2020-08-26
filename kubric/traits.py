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

from kubric.color import Color


class Vector3D(tl.TraitType):
  default_value = (0, 0, 0)
  info_text = "a 3D vector of floats"

  def _validate(self, obj, value):
    value = tuple([float(v) for v in value])
    if len(value) != 3:
      self.error(obj, value)
    else:
      return value


class Quaternion(tl.TraitType):
  default_value = (1, 0, 0, 0)
  info_text = "a 4D vector (quaternion) of floats"

  def _validate(self, obj, value):
    value = tuple([float(v) for v in value])
    if len(value) != 4:
      self.error(obj, value)
    else:
      return value


class RGBA(tl.TraitType):
  default_value = Color(0., 0., 0., 1.0)
  info_text = "an RGBA color"

  def _validate(self, obj, value):
    if isinstance(value, Color):
      color = value
    elif isinstance(value, int):
      color = Color.from_hexint(value)
    elif isinstance(value, str):
      color = Color.from_hexstr(value)
    elif len(value) in [3, 4]:
      color = Color(*value)
    else:
      return self.error(obj, value)

    if not len(color) == 4:
      self.error(obj, value)
    if not all([0 <= x <= 1 for x in color]):
      self.error(obj, value)

    return color


# TODO: it is inconsistent to use Color object for RGBA and a regular tuple for RGB.
#       But we do need both types. So maybe we should have both ColorRGBA and ColorRGB classes?
class RGB(tl.TraitType):
  default_value = (0., 0., 0.)
  info_text = "an RGB color"

  def _validate(self, obj, value):
    if isinstance(value, Color):
      color = value.rgb
    elif isinstance(value, int):
      color = Color.from_hexint(value).rgb
    elif isinstance(value, str):
      color = Color.from_hexstr(value).rgb
    elif len(value) == 3:
      color = value
    else:
      return self.error(obj, value)

    if not len(color) == 3:
      self.error(obj, value)
    if not all([0 <= x <= 1 for x in color]):
      self.error(obj, value)

    return color
