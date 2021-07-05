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
import argparse
import collections
import copy
import functools
import io
import json
import logging
import multiprocessing
import multiprocessing.pool
import pathlib
import pickle
import pprint
import shutil
import sys
import tempfile
import threading
from typing import Dict, Tuple

import imageio
import numpy as np
import png
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

from kubric.custom_types import PathLike
from kubric import core

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------------------
# Kubric argparser
# --------------------------------------------------------------------------------------------------

class ArgumentParser(argparse.ArgumentParser):
  """An argumentparser with default options, and compatibility with the Blender REPL."""

  def __init__(self, *args, **kwargs):
    argparse.ArgumentParser.__init__(self, *args, **kwargs)

    # --- default arguments for kubric
    self.add_argument("--frame_rate", type=int, default=24,
                      help="number of rendered frames per second (default: 24)")
    self.add_argument("--step_rate", type=int, default=240,
                      help="number of simulation steps per second. "
                           "Has to be an integer multiple of --frame_rate (default: 240)")
    self.add_argument("--frame_start", type=int, default=1,
                      help="index of the first frame to render. "
                           "Note that simulation always starts at frame 0 (default: 1)")
    self.add_argument("--frame_end", type=int, default=24,
                      help="index of the last frame to render (default: 24)")  # 1 second
    self.add_argument("--logging_level", type=str, default="INFO")
    self.add_argument("--seed", type=int, default=None,
                      help="(int) seed for random sampling in the worker (default: None)")
    self.add_argument("--width", type=int, default=512,
                      help="width of the output image/video in pixels (default: 512)")
    self.add_argument("--height", type=int, default=512,
                      help="height of the output image/video in pixels (default: 512)")
    self.add_argument("--scratch_dir", type=str, default=tempfile.mkdtemp(),
                      help="local directory for storing intermediate files such as "
                           "downloaded assets, raw output of renderer, ... (default: temp dir)")
    self.add_argument("--job-dir", type=str, default="output",
                      help="target directory for storing the worker output (default: ./output)")

  def parse_args(self, args=None, namespace=None):
    # --- parse argument in a way compatible with blender REPL
    if args is not None and "--" in sys.argv:
      args = sys.argv[sys.argv.index("--")+1:]
      flags = super().parse_args(args=args, namespace=namespace)
    else:
      flags = super().parse_args(args=args)
    return flags


# --------------------------------------------------------------------------------------------------
# Helpers for workers
# --------------------------------------------------------------------------------------------------

def setup(flags):
  setup_logging(flags.logging_level)
  log_my_flags(flags)

  seed = flags.seed if flags.seed else np.random.randint(0, 2147483647)
  rng = np.random.RandomState(seed=seed)
  scene = core.scene.Scene.from_flags(flags)

  scratch_dir, output_dir = setup_directories(flags)
  return scene, rng, output_dir, scratch_dir


def setup_logging(logging_level):
  logging.basicConfig(level=logging_level)


def log_my_flags(flags):
  flags_string = pprint.pformat(vars(flags), indent=2, width=100)
  logger.info(flags_string)


def done():
  logging.info("Done!")

  # -- report generated_images to hyperparameter tuner
  import hypertune  # pylint: disable=import-outside-toplevel

  hpt = hypertune.HyperTune()
  hpt.report_hyperparameter_tuning_metric(
      hyperparameter_metric_tag="answer",
      metric_value=42)


# --------------------------------------------------------------------------------------------------
# Collect metadata
# --------------------------------------------------------------------------------------------------

def get_scene_metadata(scene, **kwargs):
  metadata = {
      "width": scene.resolution[0],
      "height": scene.resolution[1],
      "num_frames": scene.frame_end - scene.frame_start + 1,
      "num_instances": len(scene.foreground_assets),
  }
  metadata.update(kwargs)
  return metadata


def get_camera_info(camera, **kwargs):
  camera_info = {
      "focal_length": camera.focal_length,
      "sensor_width": camera.sensor_width,
      "field_of_view": camera.field_of_view,
      "positions": camera.get_values_over_time("position"),
      "quaternions": camera.get_values_over_time("quaternion"),
      "K": camera.intrinsics,
      "R": camera.matrix_world,
  }
  camera_info.update(kwargs)
  return camera_info


def get_instance_info(scene):
  instance_info = []
  # extract the framewise position, quaternion, and velocity for each object
  for instance in scene.foreground_assets:
    info = copy.copy(instance.metadata)
    info["positions"] = instance.get_values_over_time("position")
    info["quaternions"] = instance.get_values_over_time("quaternion")
    info["velocities"] = instance.get_values_over_time("velocity")
    info["angular_velocities"] = instance.get_values_over_time("angular_velocity")
    info["mass"] = instance.mass
    info["friction"] = instance.friction
    info["restitution"] = instance.restitution
    frame_range = range(scene.frame_start, scene.frame_end+1)
    info["image_positions"] = np.array([scene.camera.project_point(point3d=p, frame=f)[:2]
                                        for f, p in zip(frame_range, info["positions"])],
                                       dtype=np.float32)
    instance_info.append(info)
  return instance_info


def process_collisions(collisions, scene):
  def get_obj_index(obj):
    try:
      return scene.foreground_assets.index(obj)
    except ValueError:
      return -1

  return [{
      "instances": (get_obj_index(c["instances"][0]), get_obj_index(c["instances"][1])),
      "contact_normal": c["contact_normal"],
      "frame": c["frame"],
      "force": c["force"],
      "position": c["position"],
      "image_position": scene.camera.project_point(c["position"])[:2],
  } for c in collisions]


# --------------------------------------------------------------------------------------------------
# File IO helpers
# --------------------------------------------------------------------------------------------------

def str2path(path: str) -> PathLike:
  return tfds.core.as_path(path)


def setup_directories(flags):
  assert flags.scratch_dir is not None
  scratch_dir = str2path(flags.scratch_dir)
  if scratch_dir.exists():
    logging.info("Deleting content of old scratch-dir: %s", scratch_dir)
    shutil.rmtree(scratch_dir)
  scratch_dir.mkdir(parents=True)
  logging.info("Using scratch directory: %s", scratch_dir)

  output_dir = tfds.core.as_path(flags.job_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  logging.info("Using output directory: %s", output_dir)
  return scratch_dir, output_dir


def is_local_path(path):
  """ Determine if a given path is local or remote. """
  first_part = pathlib.Path(path).parts[0]
  if first_part.endswith(":") and len(first_part) > 2:
    return False
  else:
    return True


def save_as_pkl(filename, data):
  with tf.io.gfile.GFile(filename, "wb") as fp:
    logging.info("Writing to '%s'", fp.name)
    pickle.dump(data, fp)


class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return json.JSONEncoder.default(self, obj)


def save_as_json(filename, data):
  with tf.io.gfile.GFile(filename, "wb") as fp:
    logging.info("Writing to '%s'", fp.name)
    json.dump(data, fp, sort_keys=True, indent=4, cls=NumpyEncoder)


def write_scaled_png(data: np.array, path_prefix: PathLike) -> Dict[str, float]:
  """Writes data as pngs to path and optionally returns rescaling values.

  data is expected to be of shape [B, H, W, C] and has to have
  between 1 and 4 channels. A two channel image is padded using a zero channel.
  uint8 and uint16 are written as is.
  other dtypes are rescaled to uint16. When rescaling, this method returns the
  min and max values of the original data.

  Args:
    data: the image to be written
    path_prefix: the path prefix to write to

  Returns:
    {"min": value, "max": value} if rescaling was applied, None otherwise.
  """
  assert len(data.shape) == 4, data.shape
  scaling = None
  if data.dtype in [np.uint32, np.uint64]:
    max_value = np.amax(data)
    if max_value > 65535:
      logger.warning("max_value %d exceeds uint16 bounds for %s.",
                     max_value, path_prefix)
    data = data.astype(np.uint16)
  elif data.dtype in [np.float32, np.float64]:
    min_value = np.amin(data)
    max_value = np.amax(data)
    scaling = {"min": min_value.item(), "max": max_value.item()}
    data = (data - min_value) * 65535 / (max_value - min_value)
    data = data.astype(np.uint16)
  elif data.dtype in [np.uint8, np.uint16]:
    pass
  else:
    raise NotImplementedError(f"Cannot handle {data.dtype}.")
  bitdepth = 8 if data.dtype == np.uint8 else 16
  greyscale = (data.shape[-1] == 1)
  alpha = (data.shape[-1] == 4)
  w = png.Writer(data.shape[1], data.shape[2], greyscale=greyscale,
                 bitdepth=bitdepth, alpha=alpha)
  for i in range(data.shape[0]):
    img = data[i].copy()
    if img.shape[-1] == 2:
      # Pad two-channel images with a zero channel.
      img = np.concatenate([img, np.zeros_like(img[..., :1])], axis=-1)
    # pypng expects 2d arrays
    # see https://pypng.readthedocs.io/en/latest/ex.html#reshaping
    img = img.reshape(img.shape[0], -1)
    with tf.io.gfile.GFile(f"{path_prefix}_{i:05d}.png", "wb") as fp:
      w.write(fp, img)
  return scaling


def write_tiff(
    all_imgs: np.ndarray,
    path_prefix: PathLike,
) -> None:
  """Save single-channel float images as tiff.."""
  assert len(all_imgs.shape) == 4
  assert all_imgs.shape[-1] == 1  # Single channel image
  assert all_imgs.dtype in [np.float32, np.float64]
  for i, img in enumerate(all_imgs):
    path = path_prefix.parent / f"{path_prefix.name}_{i:05d}.tiff"

    # Save tiff in an intermediate buffer
    buffer = io.BytesIO()
    imageio.imwrite(buffer, img, format=".tif")
    path.write_bytes(buffer.getvalue())


def write_single_record(
    kv: Tuple[str, np.array],
    path_prefix: PathLike,
    scalings: Dict[str, Dict[str, float]],
    lock: threading.Lock,
):
  """Write single record."""
  key = kv[0]
  img = kv[1]
  if key == "depth":
    write_tiff(img, path_prefix / key)
  else:
    scaling = write_scaled_png(img, path_prefix / key)
    if scaling:
      with lock:
        scalings[key] = scaling


def write_image_dict(data: Dict[str, np.array], path_prefix: PathLike, max_write_threads=16):
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
        lock=lock)
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


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

def next_global_count(name, reset=False):
  """A global counter to create UIDs.
  Return the total number of times (0-indexed) the function has been called with the given name.
  Used to create the increasing UID counts for each class (e.g. "Sphere.007").
  When passing reset=True, then all counts are reset.
  """
  if reset or not hasattr(next_global_count, "counter"):
    next_global_count.counter = collections.defaultdict(int)
    next_global_count.lock = multiprocessing.Lock()

  with next_global_count.lock:
    counter = next_global_count.counter[name]
    next_global_count.counter[name] += 1
    return counter
