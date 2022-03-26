# Copyright 2022 The Kubric Authors.
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
"""Kubric color classes."""

import colorsys
from typing import NamedTuple, Tuple, Union


class Color(NamedTuple):
  """Represents a color in terms of float values for RGBA between 0.0 and 1.0."""

  r: float
  g: float
  b: float
  a: float = 1.

  @property
  def rgb(self):
    return self.r, self.g, self.b

  @property
  def hsv(self):
    return colorsys.rgb_to_hsv(self.r, self.g, self.b)

  @property
  def hexstr(self):
    r, g, b, a = [int(x * 255) for x in iter(self)]
    return f"#{r:02x}{g:02x}{b:02x}{a:02x}"

  @property
  def hexstr_short(self):
    r, g, b, a = [int(x * 15) for x in iter(self)]
    return f"#{r:01x}{g:01x}{b:01x}{a:01x}"

  @classmethod
  def from_hsv(cls, h: float, s: float, v: float, alpha=1.0):
    if not 0 <= h <= 1:
      raise ValueError(f"Hue has to be between 0.0 and 1.0 (was {h})")
    if not 0 <= s <= 1:
      raise ValueError(f"Saturation has to be between 0.0 and 1.0 (was {s})")
    if not 0 <= v <= 1:
      raise ValueError(f"Value has to be between 0.0 and 1.0 (was {v})")
    return cls(*colorsys.hsv_to_rgb(h, s, v), a=alpha)

  @classmethod
  def from_hexint(cls, hexint: int, alpha: float = 1.0):
    """Create a Color instance from a hex integer like 0xaaff33 and an optional alpha value."""
    if not 0 <= hexint <= 0xffffff:
      raise ValueError(f"hexint not [0x000000 ... 0xffffff] (was 0x{hexint:06x})")
    if not 0. <= alpha <= 1.0:
      raise ValueError(f"alpha has to be between 0.0 and 1.0 (was {alpha})")
    b = hexint & 255
    g = (hexint >> 8) & 255
    r = (hexint >> 16) & 255
    return cls(r / 255.0, g / 255.0, b / 255.0, alpha)

  @classmethod
  def from_hexstr(cls, hexstr: str):
    """Create a Color instance from a hex string like #ffaa22 or #11aa88ff.

    Supports both long and short form (i.e. #ffffff is the same as #fff), and also an optional
    alpha value (e.g. #112233ff or #123f).
    """
    if hexstr[0] == "#":  # get rid of leading #
      hexstr = hexstr[1:]
    if len(hexstr) == 3:
      r = int(hexstr[0], 16) / 15.
      g = int(hexstr[1], 16) / 15.
      b = int(hexstr[2], 16) / 15.
      return cls(r, g, b)
    elif len(hexstr) == 4:
      r = int(hexstr[0], 16) / 15.
      g = int(hexstr[1], 16) / 15.
      b = int(hexstr[2], 16) / 15.
      a = int(hexstr[3], 16) / 15.
      return cls(r, g, b, a)
    elif len(hexstr) == 6:
      r = int(hexstr[0:2], 16) / 255.0
      g = int(hexstr[2:4], 16) / 255.0
      b = int(hexstr[4:6], 16) / 255.0
      return cls(r, g, b)
    elif len(hexstr) == 8:
      r = int(hexstr[0:2], 16) / 255.0
      g = int(hexstr[2:4], 16) / 255.0
      b = int(hexstr[4:6], 16) / 255.0
      a = int(hexstr[6:8], 16) / 255.0
      return cls(r, g, b, a)
    else:
      raise ValueError("invalid color hex string")

  @classmethod
  def from_name(cls, name: str):
    return {
        "aqua":    cls.from_hexstr("#00ffff"),
        "black":   cls.from_hexstr("#000000"),
        "blue":    cls.from_hexstr("#0000ff"),
        "fuchsia": cls.from_hexstr("#ff00ff"),
        "green":   cls.from_hexstr("#008000"),
        "gray":    cls.from_hexstr("#808080"),
        "lime":    cls.from_hexstr("#00ff00"),
        "maroon":  cls.from_hexstr("#800000"),
        "navy":    cls.from_hexstr("#000080"),
        "olive":   cls.from_hexstr("#808000"),
        "purple":  cls.from_hexstr("#800080"),
        "red":     cls.from_hexstr("#ff0000"),
        "silver":  cls.from_hexstr("#c0c0c0"),
        "teal":    cls.from_hexstr("#008080"),
        "white":   cls.from_hexstr("#ffffff"),
        "yellow":  cls.from_hexstr("#ffff00"),
    }[name.lower()]


def get_color(color: Union[int, str, Tuple]) -> Color:
  if isinstance(color, str):
    if color.startswith("#"):
      return Color.from_hexstr(color)
    else:
      return Color.from_name(color)
  elif isinstance(color, int):
    return Color.from_hexint(color)
  else:
    return Color(*color)
