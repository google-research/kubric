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

from kubric import color


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
  info_text = "a 4D vector (WXYZ quaternion) of floats"

  def _validate(self, obj, value):
    value = tuple([float(v) for v in value])
    if len(value) != 4:
      self.error(obj, value)
    else:
      return value


class RGBA(tl.TraitType):
  default_value = color.Color(0., 0., 0., 1.0)
  info_text = "an RGBA color"

  def _validate(self, obj, value):
    if isinstance(value, color.Color):
      rgba = value
    elif isinstance(value, int):
      rgba = color.Color.from_hexint(value)
    elif isinstance(value, str):
      rgba = color.Color.from_hexstr(value)
    elif len(value) in [3, 4]:
      rgba = color.Color(*value)
    else:
      return self.error(obj, value)

    if not all([0 <= x <= 1 for x in rgba]):
      self.error(obj, value)

    return rgba


# TODO: it is inconsistent to use Color object for RGBA and a regular tuple for RGB.
#       But we do need both types. So maybe we should have both ColorRGBA and ColorRGB classes?
class RGB(tl.TraitType):
  default_value = (0., 0., 0.)
  info_text = "an RGB color"

  def _validate(self, obj, value):
    if isinstance(value, color.Color):
      rgb = value.rgb
    elif isinstance(value, int):
      rgb = color.Color.from_hexint(value).rgb
    elif isinstance(value, str):
      rgb = color.Color.from_hexstr(value).rgb
    elif len(value) == 3:
      rgb = color.Color(*value).rgb
    else:
      return self.error(obj, value)

    if not all([0 <= x <= 1 for x in rgb]):
      self.error(obj, value)

    return rgb
