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

import contextlib
import functools
import logging
import json
import multiprocessing
import pickle
import threading
from typing import Any, Dict, Tuple

import imageio
import numpy as np
import png
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

from kubric import plotting
from kubric.custom_types import PathLike


logger = logging.getLogger(__name__)


def as_path(path: PathLike) -> tfds.core.ReadWritePath:
  """Convert str or pathlike object to tfds.core.ReadWritePath.

  Instead of pathlib.Paths, we use the TFDS path because they transparently
  support paths to GCS buckets such as "gs://kubric-public/GSO".
  """
  return tfds.core.as_path(path)


@contextlib.contextmanager
def gopen(filename: PathLike, mode: str = "w"):
  """Simple contextmanager to open a file using tf.io.gfile (and ensure the parent dir exists)."""
  filename = as_path(filename)
  if mode[0] in {"w", "a"}:  # if writing mode ...
    # ensure directory exists
    filename.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Writing to '%s'", filename)
  with tf.io.gfile.GFile(str(filename), mode=mode) as fp:
    yield fp


def write_pkl(data: Any, filename: PathLike) -> None:
  with gopen(filename, "wb") as fp:
    pickle.dump(data, fp)


def write_json(data: Any, filename: PathLike) -> None:
  with gopen(filename, "w") as fp:
    json.dump(data, fp, sort_keys=True, indent=4, cls=_NumpyEncoder)


class _NumpyEncoder(json.JSONEncoder):
  def default(self, o):
    if isinstance(o, np.ndarray):
      return o.tolist()
    return json.JSONEncoder.default(self, o)


def write_png(data: np.array, filename: PathLike) -> None:
  """Writes data as a png file (and convert datatypes if necessary)."""

  if data.dtype in [np.uint32, np.uint64]:
    max_value = np.amax(data)
    if max_value > 65535:
      logger.warning("max_value %d exceeds uint16 bounds for %s.",
                     max_value, filename)
    data = data.astype(np.uint16)
  elif data.dtype in [np.float32, np.float64]:
    min_value = np.amin(data)
    max_value = np.amax(data)
    assert min_value >= 0.0, min_value
    assert max_value <= 1.0, max_value

    data = (data * 65535).astype(np.uint16)
  elif data.dtype in [np.uint8, np.uint16]:
    pass
  else:
    raise NotImplementedError(f"Cannot handle {data.dtype}.")

  bitdepth = 8 if data.dtype == np.uint8 else 16

  assert data.ndim == 3, data.shape
  height, width, channels = data.shape
  greyscale = (channels == 1)
  alpha = (channels == 4)
  w = png.Writer(height, width, greyscale=greyscale, bitdepth=bitdepth, alpha=alpha)

  if channels == 2:
    # Pad two-channel images with a zero channel.
    data = np.concatenate([data, np.zeros_like(data[:, :, :1])], axis=-1)

  # pypng expects 2d arrays
  # see https://pypng.readthedocs.io/en/latest/ex.html#reshaping
  data = data.reshape(height, -1)
  with gopen(filename, "wb") as fp:
    w.write(fp, data)


def write_palette_png(data: np.array, filename: PathLike,
                      palette: np.ndarray = None):
  """Writes grayscale data as pngs to path using a fixed palette (e.g. for segmentations)."""
  assert data.ndim == 3, data.shape
  height, width, channels = data.shape
  assert channels == 1, "Must be grayscale"

  if data.dtype in [np.uint16, np.uint32, np.uint64]:
    max_value = np.amax(data)
    if max_value > 255:
      logger.warning("max_value %d exceeds uint bounds for %s.",
                     max_value, filename)
    data = data.astype(np.uint8)
  elif data.dtype == np.uint8:
    pass
  else:
    raise NotImplementedError(f"Cannot handle {data.dtype}.")

  if palette is None:
    palette = plotting.hls_palette(np.max(data) + 1)

  w = png.Writer(height, width, palette=palette, bitdepth=8)
  with gopen(filename, "wb") as fp:
    w.write(fp, data[:, :, 0])


def write_scaled_png(data: np.array, filename: PathLike) -> Dict[str, float]:
  """Scales data to [0, 1] and then saves as png and returns the scale.

  Args:
    data: the image (H, W, C) to be written (has to be float32 or float64).
    filename: the filename to write to (can be a GCS path).

  Returns:
    {"min": min_value, "max": max_value}
  """
  assert data.dtype in [np.float32, np.float64], data.dtype
  min_value = np.min(data)
  max_value = np.max(data)
  scaling = {"min": min_value.item(), "max": max_value.item()}
  data = (data - min_value) * 65535 / (max_value - min_value)
  data = data.astype(np.uint16)
  write_png(data, filename)
  return scaling


def write_tiff(data: np.ndarray, filename: PathLike):
  """Save data as as tif image (which natively supports float values)."""
  assert data.ndim == 3, data.shape
  assert data.shape[2] in [1, 3, 4], "Must be grayscale, RGB, or RGBA"

  with gopen(filename, "wb") as fp:
    imageio.imwrite(fp, data, format=".tif")


def write_many_threaded(data: np.ndarray, path_template: PathLike, write_fn=write_png,
                        max_write_threads=16, **kwargs):
  # Pre-load image libs to avoid race-condition in multi-thread.
  imageio.plugins.tifffile.load_lib()
  num_threads = min(len(data.shape[0]), max_write_threads)
  with multiprocessing.pool.ThreadPool(num_threads) as pool:
    args = [(img, str(path_template).format(i)) for i, img in enumerate(data)]
    write_single_image_fn = lambda arg: write_fn(arg[0], arg[1], **kwargs)
    pool.map(write_single_image_fn, args)
    pool.close()
    pool.join()

def write_single_record(
    kv: Tuple[str, np.array],
    path_prefix: PathLike,
    scalings: Dict[str, Dict[str, float]],
    lock: threading.Lock,
    segmentation_palette=None,
):
  """Write single record."""
  key = kv[0]
  img = kv[1]
  if key == "depth":
    scaling = write_tiff(img, path_prefix / key)
  elif key == "segmentation":
    write_palette_png(img, path_prefix / key, segmentation_palette)
    scaling = None
  else:
    scaling = write_scaled_png(img, path_prefix / key)
  if scaling is not None:
    with lock:
      scalings[key] = scaling


def write_image_dict(data: Dict[str, np.array], path_prefix: PathLike, max_write_threads=16,
    segmentation_palette=None):
  # Pre-load image libs to avoid race-condition in multi-thread.
  imageio.plugins.tifffile.load_lib()
  scalings = {}
  lock = threading.Lock()
  num_threads = min(len(data.items()), max_write_threads)
  with multiprocessing.pool.ThreadPool(num_threads) as pool:
    args = [(key, img) for key, img in data.items()]  # pylint: disable=unnecessary-comprehension
    write_single_record_fn = functools.partial(
        write_single_record,
        path_prefix=path_prefix,
        scalings=scalings,
        lock=lock,
        segmentation_palette=segmentation_palette)
    pool.map(write_single_record_fn, args)
    pool.close()
    pool.join()

  with tf.io.gfile.GFile(path_prefix / "data_ranges.json", "w") as fp:
    json.dump(scalings, fp)


def read_png(path: PathLike):
  path = tfds.core.as_path(path)
  png_reader = png.Reader(bytes=path.read_bytes())
  width, height, pngdata, info = png_reader.read()
  del png_reader
  bitdepth = info["bitdepth"]
  if bitdepth == 8:
    dtype = np.uint8
  elif bitdepth == 16:
    dtype = np.uint16
  else:
    raise NotImplementedError(f"Unsupported bitdepth: {bitdepth}")
  plane_count = info["planes"]
  pngdata = np.vstack(list(map(dtype, pngdata)))
  return pngdata.reshape((height, width, plane_count))
