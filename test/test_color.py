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


import pytest

from hypothesis import given
from hypothesis.strategies import floats, tuples, text


from kubric.color import Color


@pytest.mark.parametrize('hexint, expected', [
    (0x000000, Color(0., 0., 0., 1.)),
    (0xff0000, Color(1., 0., 0., 1.)),
    (0x00ff00, Color(0., 1., 0., 1.)),
    (0x0000ff, Color(0., 0., 1., 1.)),
    (0xffffff, Color(1., 1., 1., 1.)),
])
def test_hex_to_rgb(hexint, expected):
  assert Color.from_hexint(hexint) == expected


def test_hex_to_rgba():
  assert Color.from_hexint(0xffffff, alpha=0.5) == (1., 1., 1., 0.5)


def test_hex_to_rgba_invalid():
  with pytest.raises(ValueError):
    Color.from_hexint(-1)

  with pytest.raises(ValueError):
    Color.from_hexint(0xfffffff)

  with pytest.raises(ValueError):
    Color.from_hexint(0x000000, alpha=-0.1)

  with pytest.raises(ValueError):
    Color.from_hexint(0x123456, alpha=1.1)


@given(tuples(floats(0., 1.0), floats(0.01, 1.0), floats(0.01, 1.0)))
def test_hsv_conversion_is_invertable(hsv):
  assert Color.from_hsv(*hsv).hsv == pytest.approx(hsv)


@given(text(alphabet='0123456789abcdef', min_size=6, max_size=6))
def test_hexstr_rgb_conversion_is_invertable(hexstr):
  hexstr = '#' + hexstr
  assert Color.from_hexstr(hexstr).hexstr == hexstr + 'ff'


@given(text(alphabet='0123456789abcdef', min_size=8, max_size=8))
def test_hexstr_rgba_conversion_is_invertable(hexstr):
  hexstr = '#' + hexstr
  assert Color.from_hexstr(hexstr).hexstr == hexstr


@given(text(alphabet='0123456789abcdef', min_size=3, max_size=3))
def test_hexstr_short_rgb_conversion_is_invertable(hexstr_short):
  hexstr_short = '#' + hexstr_short
  assert Color.from_hexstr(hexstr_short).hexstr_short == hexstr_short + 'f'


@given(text(alphabet='0123456789abcdef', min_size=4, max_size=4))
def test_hexstr_short_rgba_conversion_is_invertable(hexstr_short):
  hexstr_short = '#' + hexstr_short
  assert Color.from_hexstr(hexstr_short).hexstr_short == hexstr_short
