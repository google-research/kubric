# Copyright 2023 The Kubric Authors.
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

import pytest

from kubric.core import cameras


def test_orthographic_camera_constructor():
  cam = cameras.OrthographicCamera(orthographic_scale=7)
  assert cam.orthographic_scale == 7


def test_perspective_camera_constructor():
  cam = cameras.PerspectiveCamera(focal_length=22, sensor_width=33)
  assert cam.focal_length == 22
  assert cam.sensor_width == 33


def test_perspective_camera_field_of_view():
  cam = cameras.PerspectiveCamera(focal_length=28, sensor_width=36)
  assert cam.field_of_view == pytest.approx(1.1427, abs=1e-4)  # ca 65.5°


