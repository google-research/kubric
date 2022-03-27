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

import numpy as np
import pytest

from kubric import file_io


def test_write_read_grayscale_uint8_png(tmpdir):
  filename = tmpdir / "grayscale_uint8.png"
  img_data = np.arange(256, dtype=np.uint8).reshape((16, 16, 1))
  file_io.write_png(img_data, filename)
  img_data_recovered = file_io.read_png(filename)
  np.testing.assert_array_equal(img_data, img_data_recovered)


def test_write_read_rgb_uint8_png(tmpdir):
  filename = tmpdir / "rgb_uint8.png"
  red = np.arange(256, dtype=np.uint8).reshape((16, 16, 1))
  img_data = np.concatenate([red, np.zeros_like(red), np.ones_like(red)], axis=-1)
  file_io.write_png(img_data, filename)
  img_data_recovered = file_io.read_png(filename)
  np.testing.assert_array_equal(img_data, img_data_recovered)


def test_write_read_grayscale_uint16_png(tmpdir):
  filename = tmpdir / "grayscale_uint16.png"
  img_data = np.arange(0, 65536, 64, dtype=np.uint16).reshape((32, 32, 1))
  for dtype in [np.uint16, np.uint32, np.uint64]:
    file_io.write_png(img_data.astype(dtype), filename)
    img_data_recovered = file_io.read_png(filename)
    np.testing.assert_array_equal(img_data, img_data_recovered)


def test_write_read_rgb_uint16_png(tmpdir):
  filename = tmpdir / "rgb_uint16.png"
  red = np.arange(0, 65536, 64, dtype=np.uint16).reshape((32, 32, 1))
  img_data = np.concatenate([red, np.zeros_like(red), np.ones_like(red)], axis=-1)
  file_io.write_png(img_data, filename)
  img_data_recovered = file_io.read_png(filename)
  np.testing.assert_array_equal(img_data, img_data_recovered)


def test_write_read_rgba_uint16_png(tmpdir):
  filename = tmpdir / "rgba_uint16.png"
  red = np.arange(0, 65536, 64, dtype=np.uint16).reshape((32, 32, 1))
  img_data = np.concatenate([red, np.zeros_like(red), np.ones_like(red), np.ones_like(red)],
                            axis=-1)
  file_io.write_png(img_data, filename)
  img_data_recovered = file_io.read_png(filename)
  np.testing.assert_array_equal(img_data, img_data_recovered)


def test_write_read_2_layer_png(tmpdir):
  """When writing 2 channel images (e.g. flow) they are padded to RGB by adding a 0 blue channel."""
  filename = tmpdir / "2_layer_uint16.png"
  img_data = np.arange(0, 65536, 32, dtype=np.uint16).reshape((32, 32, 2))
  file_io.write_png(img_data, filename)
  img_data_recovered = file_io.read_png(filename)
  img_data_padded = np.concatenate([img_data, np.zeros_like(img_data[:, :, :1])], axis=-1)
  np.testing.assert_array_equal(img_data_recovered, img_data_padded)


def test_write_float32_png_fails_for_values_not_between_0_and_1(tmpdir):
  img_data = np.linspace(0, 2., 8*8, dtype=np.float32).reshape((8, 8, 1))
  with pytest.raises(ValueError):
    file_io.write_png(img_data, tmpdir / "ValueError.png")


def test_write_read_rgb_float32_png(tmpdir):
  """Float32 images are automatically converted to uint16."""
  filename = tmpdir / "rgb_float32.png"
  img_data = np.linspace(0, 1., 32*32*3, dtype=np.float32).reshape((32, 32, 3))
  file_io.write_png(img_data, filename)
  img_data_recovered = file_io.read_png(filename)
  img_data_uint16 = (img_data * 65535).astype(np.uint16)
  np.testing.assert_array_equal(img_data_recovered, img_data_uint16)


def test_write_read_float32_tiff(tmpdir):
  filename = tmpdir / "grayscale_float32.tiff"
  img_data = np.linspace(0, 1., 32*32, dtype=np.float32).reshape((32, 32, 1))
  file_io.write_tiff(img_data, filename)
  img_data_recovered = file_io.read_tiff(filename)
  np.testing.assert_array_equal(img_data_recovered, img_data)


def test_write_image_dict(tmpdir):
  img_dict = {
      "rgb": np.arange(4*4*4*3, dtype=np.uint8).reshape((4, 4, 4, 3)),
      "rgba": np.arange(4*4*4*4, dtype=np.uint8).reshape((4, 4, 4, 4)),
      "depth": np.linspace(0, 100000., 4*4*4, dtype=np.float32).reshape((4, 4, 4, 1)),
      "uv": np.arange(4*4*4*3, dtype=np.uint8).reshape((4, 4, 4, 3)),
      "normal": np.arange(4*4*4*3, dtype=np.uint8).reshape((4, 4, 4, 3)),
      "flow": np.linspace(0, 100., 4*4*4*2, dtype=np.float32).reshape((4, 4, 4, 2)),
      "forward_flow": np.linspace(0, 100., 4*4*4*2, dtype=np.float32).reshape((4, 4, 4, 2)),
      "backward_flow": np.linspace(0, 100., 4*4*4*2, dtype=np.float32).reshape((4, 4, 4, 2)),
      "segmentation": np.ones(4*4*4, dtype=np.uint8).reshape((4, 4, 4, 1)),
  }
  file_templates = {
      "rgb": "rgb_{:05d}.png",
      "rgba": "rgba_{:05d}.png",
      "depth": "depth_{:05d}.tiff",
      "uv": "uv_{:05d}.png",
      "normal": "normal_{:05d}.png",
      "flow": "flow_{:05d}.png",
      "forward_flow": "forward_flow_{:05d}.png",
      "backward_flow": "backward_flow_{:05d}.png",
      "segmentation": "segmentation_{:05d}.png",
  }
  file_io.write_image_dict(img_dict, tmpdir)

  data_ranges = file_io.read_json(tmpdir / "data_ranges.json")

  for key, img_batch in img_dict.items():
    if key in data_ranges:
      data_range = data_ranges[key]['min'], data_ranges[key]['max']
    else:
      data_range = None

    for i, img in enumerate(img_batch):
      filename = tmpdir / file_templates[key].format(i)
      assert filename.exists()
      print(filename)
      if str(filename).endswith('.png'):
        img_recovered = file_io.read_png(filename, rescale_range=data_range)
      elif str(filename).endswith('.tiff'):
        img_recovered = file_io.read_tiff(filename)
      else:
        assert False

      if img.shape[-1] == 2:  # pad image with a zero blue channel
        img = np.concatenate([img, np.zeros_like(img[:, :, :1])], axis=-1)

      assert img.shape == img_recovered.shape
      np.testing.assert_allclose(img_recovered, img, rtol=1e-4, atol=1e-4)
