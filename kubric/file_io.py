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

import contextlib
import functools
import logging
import json
import multiprocessing
import pickle
from typing import Any, Dict

from etils import epath
import imageio
import numpy as np
import png
import tensorflow as tf

from kubric import plotting
from kubric.kubric_typing import PathLike


logger = logging.getLogger(__name__)


def as_path(path: PathLike) -> epath.Path:
  """Convert str or pathlike object to epath.Path.

  Instead of pathlib.Path, we use `epath` because it transparently
  supports paths to GCS buckets such as "gs://kubric-public/GSO".
  """
  return epath.Path(path)


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


def read_json(filename: PathLike) -> Any:
  with gopen(filename, "r") as fp:
    return json.load(fp, )


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
      raise ValueError(f"max value of {max_value} exceeds uint16 bounds for {filename}")
    data = data.astype(np.uint16)
  elif data.dtype in [np.float32, np.float64]:
    min_value = np.amin(data)
    max_value = np.amax(data)
    if min_value < 0.0 or max_value > 1.0:
      raise ValueError(f"Values need to be in range [0, 1] but got [{min_value}, {max_value}] "
                       f"for {filename}")
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
  w = png.Writer(width=width, height=height, greyscale=greyscale, bitdepth=bitdepth, alpha=alpha)

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

  w = png.Writer(width=width, height=height, palette=palette, bitdepth=8)
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


def read_png(filename: PathLike, rescale_range=None) -> np.ndarray:
  filename = as_path(filename)
  png_reader = png.Reader(bytes=filename.read_bytes())
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
  if rescale_range is not None:
    minv, maxv = rescale_range
    pngdata = pngdata / 2**bitdepth * (maxv - minv) + minv

  return pngdata.reshape((height, width, plane_count))


def write_tiff(data: np.ndarray, filename: PathLike):
  """Save data as as tif image (which natively supports float values)."""
  assert data.ndim == 3, data.shape
  assert data.shape[2] in [1, 3, 4], "Must be grayscale, RGB, or RGBA"

  img_as_bytes = imageio.imwrite("<bytes>", data, format="tiff")
  filename = as_path(filename)
  filename.write_bytes(img_as_bytes)


def read_tiff(filename: PathLike) -> np.ndarray:
  filename = as_path(filename)
  img = imageio.imread(filename.read_bytes(), format="tiff")
  if img.ndim == 2:
    img = img[:, :, None]
  return img


def multi_write_image(data: np.ndarray, path_template: str, write_fn=write_png,
                      max_write_threads=16, **kwargs):
  """Write a batch of images to a series of files using a ThreadPool.
  Args:
    data: Batch of images to write. Shape = (batch_size, height, width, channels)
    path_template: a template for the filenames (e.g. "rgb_frame_{:05d}.png").
      Will be formatted with the index of the image.
    write_fn: the function used for writing the image to disk.
      Must take an image array as its first and a filename as its second argument.
      May take other keyword arguments. (Defaults to the write_png function)
    max_write_threads: number of threads to use for writing images. (default = 16)
    **kwargs: additional kwargs to pass to the write_fn.
  """
  num_threads = min(data.shape[0], max_write_threads)
  with multiprocessing.pool.ThreadPool(num_threads) as pool:
    args = [(img, path_template.format(i)) for i, img in enumerate(data)]

    def write_single_image_fn(arg):
      write_fn(*arg, **kwargs)

    for result in pool.imap_unordered(write_single_image_fn, args):
      if isinstance(result,  Exception):
        logger.warning("Exception while writing image %s", result)

    pool.close()
    pool.join()


def write_rgb_batch(data, directory, file_template="rgb_{:05d}.png", max_write_threads=16):
  assert data.ndim == 4 and data.shape[-1] == 3, data.shape
  path_template = str(as_path(directory) / file_template)
  multi_write_image(data, path_template, write_fn=write_png, max_write_threads=max_write_threads)


def write_rgba_batch(data, directory, file_template="rgba_{:05d}.png", max_write_threads=16):
  assert data.ndim == 4 and data.shape[-1] == 4, data.shape
  path_template = str(as_path(directory) / file_template)
  multi_write_image(data, path_template, write_fn=write_png, max_write_threads=max_write_threads)


def write_uv_batch(data, directory, file_template="uv_{:05d}.png", max_write_threads=16):
  assert data.ndim == 4 and data.shape[-1] == 3, data.shape
  path_template = str(as_path(directory) / file_template)
  multi_write_image(data, path_template, write_fn=write_png, max_write_threads=max_write_threads)


def write_normal_batch(data, directory, file_template="normal_{:05d}.png", max_write_threads=16):
  assert data.ndim == 4 and data.shape[-1] == 3, data.shape
  path_template = str(as_path(directory) / file_template)
  multi_write_image(data, path_template, write_fn=write_png, max_write_threads=max_write_threads)


def write_coordinates_batch(data, directory, file_template="object_coordinates_{:05d}.png",
                            max_write_threads=16):
  assert data.ndim == 4 and data.shape[-1] == 3, data.shape
  path_template = str(as_path(directory) / file_template)
  multi_write_image(data, path_template, write_fn=write_png, max_write_threads=max_write_threads)


def write_depth_batch(data, directory, file_template="depth_{:05d}.tiff", max_write_threads=16):
  assert data.ndim == 4 and data.shape[-1] == 1, data.shape
  path_template = str(as_path(directory) / file_template)
  multi_write_image(data, path_template, write_fn=write_tiff, max_write_threads=max_write_threads)


def write_segmentation_batch(data, directory, file_template="segmentation_{:05d}.png",
                             max_write_threads=16):
  assert data.ndim == 4 and data.shape[-1] == 1, data.shape
  assert data.dtype in [np.uint8, np.uint16, np.uint32, np.uint64], data.dtype
  path_template = str(as_path(directory) / file_template)
  palette = plotting.hls_palette(np.max(data) + 1)
  multi_write_image(data, path_template, write_fn=write_palette_png,
                    max_write_threads=max_write_threads, palette=palette)


def write_flow_batch(data, directory, file_template="flow_{:05d}.png", name="flow",
                     max_write_threads=16, range_file="data_ranges.json"):
  assert data.ndim == 4 and data.shape[-1] == 2, data.shape
  assert data.dtype in [np.float32, np.float64], data.dtype
  directory = as_path(directory)
  path_template = str(directory / file_template)
  range_file_path = directory / range_file
  min_value = np.min(data)
  max_value = np.max(data)
  scaling = {"min": min_value.item(), "max": max_value.item()}
  data = (data - min_value) * 65535 / (max_value - min_value)
  data = data.astype(np.uint16)
  multi_write_image(data, path_template, write_fn=write_png,
                    max_write_threads=max_write_threads)

  if range_file_path.exists():
    ranges = read_json(range_file_path)
  else:
    ranges = {}
  ranges[name] = scaling
  write_json(ranges, range_file_path)


write_forward_flow_batch = functools.partial(write_flow_batch, name="forward_flow",
                                             file_template="forward_flow_{:05d}.png")
write_backward_flow_batch = functools.partial(write_flow_batch, name="backward_flow",
                                              file_template="backward_flow_{:05d}.png")

DEFAULT_WRITERS = {
    "rgb": write_rgb_batch,
    "rgba": write_rgba_batch,
    "depth": write_depth_batch,
    "uv": write_uv_batch,
    "normal": write_normal_batch,
    "flow": write_flow_batch,
    "forward_flow": write_forward_flow_batch,
    "backward_flow": write_backward_flow_batch,
    "segmentation": write_segmentation_batch,
    "object_coordinates": write_coordinates_batch,
}


def write_image_dict(data_dict: Dict[str, np.ndarray], directory: PathLike,
                     file_templates: Dict[str, str] = (), max_write_threads=16):
  for key, data in data_dict.items():
    if key in file_templates:
      DEFAULT_WRITERS[key](data, directory, file_template=file_templates[key],
                           max_write_threads=max_write_threads)
    else:
      DEFAULT_WRITERS[key](data, directory, max_write_threads=max_write_threads)
