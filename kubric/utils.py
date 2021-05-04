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
import copy
import json
import logging
import pathlib
import pickle
import pprint
import shutil
import sys
import tempfile

import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

logger = logging.getLogger(__name__)


class ArgumentParser(argparse.ArgumentParser):
  def __init__(self, *args, **kwargs):
    argparse.ArgumentParser.__init__(self, *args, **kwargs)

    # --- default arguments for kubric
    self.add_argument("--frame_rate", type=int, default=24,
                      help='number of rendered frames per second (default: 24)')
    self.add_argument("--step_rate", type=int, default=240,
                      help='number of simulation steps per second. '
                           'Has to be an integer multiple of --frame_rate (default: 240)')
    self.add_argument("--frame_start", type=int, default=1,
                      help='index of the first frame to render. '
                           'Note that simulation always starts at frame 0 (default: 1)')
    self.add_argument("--frame_end", type=int, default=24,
                      help='index of the last frame to render (default: 24)')  # 1 second
    self.add_argument("--logging_level", type=str, default="INFO")
    self.add_argument("--seed", type=int, default=None,
                      help="(int) seed to be used for random sampling in the worker (default: None)")
    self.add_argument("--width", type=int, default=512,
                      help="width of the output image/video in pixels (default: 512)")
    self.add_argument("--height", type=int, default=512,
                      help="height of the output image/video in pixels (default: 512)")
    self.add_argument("--scratch_dir", type=str, default=None,
                      help="local directory for storing intermediate files such as "
                           "downloaded assets, raw output of renderer, ... (default: temp dir)")
    self.add_argument("--job-dir", type=str, default="output",
                      help="target directory for storing the output of the worker (default: ./output)")

  def parse_args(self, args=None, namespace=None):
    # --- parse argument in a way compatible with blender's REPL
    if args is not None and "--" in sys.argv:
      args = sys.argv[sys.argv.index("--")+1:]
      flags = super(ArgumentParser, self).parse_args(args=args, namespace=namespace)
    else:
      flags = super(ArgumentParser, self).parse_args(args=args)
    return flags


def setup_directories(FLAGS):
  if FLAGS.scratch_dir is None:
    scratch_dir = tfds.core.as_path(tempfile.mkdtemp())
  else:
    scratch_dir = tfds.core.as_path(FLAGS.scratch_dir)
    if scratch_dir.exists():
      logging.info("Deleting content of old scratch-dir: %s", scratch_dir)
      shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True)
  logging.info("Using scratch directory: %s", scratch_dir)

  output_dir = tfds.core.as_path(FLAGS.job_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  logging.info("Using output directory: %s", output_dir)
  return scratch_dir, output_dir


def is_local_path(path):
  """ Determine if a given path is local or remote. """
  first_part = pathlib.Path(path).parts[0]
  if first_part.endswith(':') and len(first_part) > 2:
    return False
  else:
    return True


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

def setup_logging(logging_level):
  logging.basicConfig(level=logging_level)


def log_my_flags(flags):
  flags_string = pprint.pformat(vars(flags), indent=2, width=100)
  logger.info(flags_string)

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
      "K": camera.K,
      "R": camera.R,
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


def save_as_pkl(filename, data):
  with tf.io.gfile.GFile(filename, "wb") as fp:
    logging.info(f"Writing to {fp.name}")
    pickle.dump(data, fp)


class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return json.JSONEncoder.default(self, obj)


def save_as_json(filename, data):
  with tf.io.gfile.GFile(filename, "wb") as fp:
    logging.info(f"Writing to {fp.name}")
    json.dump(data, fp, sort_keys=True, indent=4, cls=NumpyEncoder)


def done():
  logging.info("Done!")

  # -- report generated_images to hyperparameter tuner
  import hypertune

  hpt = hypertune.HyperTune()
  hpt.report_hyperparameter_tuning_metric(
      hyperparameter_metric_tag="answer",
      metric_value=42)
