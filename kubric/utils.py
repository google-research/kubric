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

from absl.flags import argparse_flags

import collections
import copy
import logging
import multiprocessing
import multiprocessing.pool
import pathlib
import pprint
import shutil
import sys
import tempfile

from etils import epath
import numpy as np

from kubric import core
from kubric import file_io

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------------------
# Kubric argparser
# --------------------------------------------------------------------------------------------------

class ArgumentParser(argparse_flags.ArgumentParser):
  """An argumentparser with default options, and compatibility with the Blender REPL."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

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
                      help="index of the last frame to render (default: 24)")
    self.add_argument("--logging_level", type=str, default="INFO")
    self.add_argument("--seed", type=int, default=None,
                      help="(int) seed for random sampling in the worker (default: None)")
    self.add_argument("--resolution", type=str, default="512x512",
                      help="height and width of rendered image/video in pixels"
                           "Can be given as single number for square images or "
                           "in the form {height}x{width}. (default: 512x512)")
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

  def set_defaults(self, **kwargs):
    """Same as argparse.ArgumentParser.set_defaults() but with safety checks."""
    valid_names = [action.dest for action in self._actions]
    for key in kwargs:
      assert key in valid_names, f"Specifying default for an undefined argument '{key}'"
    super().set_defaults(**kwargs)


# --------------------------------------------------------------------------------------------------
# Helpers for workers
# --------------------------------------------------------------------------------------------------

def setup(flags):
  setup_logging(flags.logging_level)
  log_my_flags(flags)

  seed = flags.seed if flags.seed else np.random.randint(0, 2147483647)
  rng = np.random.RandomState(seed=seed)
  scene = core.scene.Scene.from_flags(flags)
  scene.metadata["seed"] = seed

  scratch_dir, output_dir = setup_directories(flags)
  return scene, rng, output_dir, scratch_dir


def setup_logging(logging_level):
  logging.basicConfig(level=logging_level)


def log_my_flags(flags):
  flags_string = pprint.pformat(vars(flags), indent=2, width=100)
  logger.info(flags_string)


def done():
  logging.info("Done!")

  from kubric import assets  # pylint: disable=import-outside-toplevel
  assets.ClosableResource.close_all()
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
      "resolution": scene.resolution,
      "frame_rate": scene.frame_rate,
      "step_rate": scene.step_rate,
      "gravity": scene.gravity,
      "num_frames": scene.frame_end - scene.frame_start + 1,
  }
  metadata.update(scene.metadata)
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


def get_instance_info(scene, assets_subset=None):
  instance_info = []
  # extract the framewise position, quaternion, and velocity for each object
  assets_subset = scene.foreground_assets if assets_subset is None else assets_subset
  for instance in assets_subset:
    info = copy.copy(instance.metadata)
    if hasattr(instance, "asset_id"):
      info["asset_id"] = instance.asset_id
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
    bboxes3d = []
    for frame in frame_range:
      with instance.at_frame(frame):
        bboxes3d.append(instance.bbox_3d)
    info["bboxes_3d"] = np.stack(bboxes3d)
    instance_info.append(info)
  return instance_info


def process_collisions(collisions, scene, assets_subset=None):
  assets_subset = scene.foreground_assets if assets_subset is None else assets_subset

  def get_obj_index(obj):
    try:
      return assets_subset.index(obj)
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

def setup_directories(flags):
  assert flags.scratch_dir is not None
  scratch_dir = file_io.as_path(flags.scratch_dir)
  if scratch_dir.exists():
    logging.info("Deleting content of old scratch-dir: %s", scratch_dir)
    shutil.rmtree(scratch_dir)
  scratch_dir.mkdir(parents=True)
  logging.info("Using scratch directory: %s", scratch_dir)

  output_dir = epath.Path(flags.job_dir)
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
