# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PNG rescaling must invert the full integer range used by the writers."""

import hashlib

import numpy as np
import png
import pytest

from kubric import file_io


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('channels', [1, 3, 4])
@pytest.mark.parametrize('bounds', [(0.0, 1.0), (-7.0, 5.0), (3.0, 10.0), (4.0, 4.0)])
def test_rescaling_uses_both_integer_endpoints(tmp_path, dtype, channels, bounds):
  maximum = np.iinfo(dtype).max
  values = np.array([0, 1, maximum // 2, maximum], dtype=dtype).reshape(2, 2, 1)
  values = np.repeat(values, channels, axis=-1)
  path = tmp_path / 'encoded.png'
  file_io.write_png(values, path)
  digest = hashlib.sha256(path.read_bytes()).digest()

  restored = file_io.read_png(path, rescale_range=bounds)
  expected = np.interp(values.ravel(), [0, maximum], bounds).reshape(values.shape)
  np.testing.assert_allclose(restored, expected, rtol=0, atol=2e-14)
  np.testing.assert_array_equal(restored[0, 0], np.full(channels, bounds[0]))
  np.testing.assert_array_equal(restored[-1, -1], np.full(channels, bounds[1]))
  assert restored.dtype == np.float64
  assert hashlib.sha256(path.read_bytes()).digest() == digest


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('channels', [1, 3, 4])
def test_unscaled_reads_retain_exact_integer_values(tmp_path, dtype, channels):
  values = np.linspace(0, np.iinfo(dtype).max, 12 * channels).astype(dtype).reshape(3, 4, channels)
  path = tmp_path / 'raw.png'
  file_io.write_png(values, path)
  actual = file_io.read_png(path)
  np.testing.assert_array_equal(actual, values)
  assert actual.dtype == dtype


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_scaled_png_round_trip_reaches_recorded_extrema(tmp_path, dtype):
  values = np.linspace(-8, 24, 36, dtype=dtype).reshape(3, 4, 3)
  before = values.copy()
  path = tmp_path / 'scaled.png'
  scaling = file_io.write_scaled_png(values, path)
  bounds = (scaling['min'], scaling['max'])
  restored = file_io.read_png(path, rescale_range=bounds)
  assert restored.min() == bounds[0]
  assert restored.max() == bounds[1]
  np.testing.assert_allclose(restored, values, rtol=0, atol=32 / 65535 + 2e-6)
  np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize(
    'writer,name,template',
    [
        (file_io.write_forward_flow_batch, 'forward_flow', 'forward_flow_{:05d}.png'),
        (file_io.write_backward_flow_batch, 'backward_flow', 'backward_flow_{:05d}.png'),
    ],
)
def test_flow_batch_round_trip_uses_recorded_range(tmp_path, writer, name, template):
  values = np.arange(48, dtype=np.float64).reshape(2, 3, 4, 2) - 24
  before = values.copy()
  writer(values, tmp_path, max_write_threads=2)
  scaling = file_io.read_json(tmp_path / 'data_ranges.json')[name]
  bounds = (scaling['min'], scaling['max'])
  restored = np.stack(
      [
          file_io.read_png(tmp_path / template.format(i), rescale_range=bounds)[..., :2]
          for i in range(2)
      ]
  )
  assert restored.min() == values.min()
  assert restored.max() == values.max()
  np.testing.assert_allclose(restored, values, rtol=0, atol=47 / 65535 + 1e-12)
  np.testing.assert_array_equal(values, before)


def test_missing_file_error_is_preserved(tmp_path):
  with pytest.raises(FileNotFoundError):
    file_io.read_png(tmp_path / 'absent.png', rescale_range=(0.0, 1.0))


def test_unsupported_bit_depth_is_preserved(tmp_path):
  path = tmp_path / 'one_bit.png'
  with path.open('wb') as stream:
    png.Writer(width=2, height=1, greyscale=True, bitdepth=1).write(stream, [[0, 1]])
  with pytest.raises(NotImplementedError, match='Unsupported bitdepth: 1'):
    file_io.read_png(path, rescale_range=(0.0, 1.0))
