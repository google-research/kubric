# Copyright 2022 The Kubric Authors
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
# pylint: disable=line-too-long, unexpected-keyword-arg
"""TODO(klausg): description."""
import json

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from kubric import file_io

DEFAULT_LAYERS = ("rgba", "segmentation", "forward_flow", "backward_flow",
                  "depth", "normal", "object_coordinates")

def load_scene_directory(scene_dir, target_size, layers=DEFAULT_LAYERS):
  scene_dir = file_io.as_path(scene_dir)
  example_key = f"{scene_dir.name}"

  with tf.io.gfile.GFile(str(scene_dir / "data_ranges.json"), "r") as fp:
    data_ranges = json.load(fp)

  with tf.io.gfile.GFile(str(scene_dir / "metadata.json"), "r") as fp:
    metadata = json.load(fp)

  with tf.io.gfile.GFile(str(scene_dir / "events.json"), "r") as fp:
    events = json.load(fp)

  num_frames = metadata["metadata"]["num_frames"]

  result = {
      "metadata": {
          "video_name": example_key,
          "width": target_size[1],
          "height": target_size[0],
          "num_frames": num_frames,
          "num_instances": metadata["metadata"]["num_instances"],
      },
      "instances": [format_instance_information(obj)
                    for obj in metadata["instances"]],
      "camera": format_camera_information(metadata),
      "events": format_events_information(events),
  }

  resolution = metadata["metadata"]["resolution"]

  assert resolution[0] / target_size[0] == resolution[1] / target_size[1]
  scale = resolution[0] / target_size[0]
  assert scale == resolution[0] // target_size[0]

  paths = {
      key: [scene_dir / f"{key}_{f:05d}.png" for f in range(num_frames)]
      for key in layers if key != "depth"
  }

  if "depth" in layers:
    depth_paths = [scene_dir / f"depth_{f:05d}.tiff" for f in range(num_frames)]
    depth_frames = np.array([
      subsample_nearest_neighbor(file_io.read_tiff(frame_path), target_size)
      for frame_path in depth_paths])
    depth_min, depth_max = np.min(depth_frames), np.max(depth_frames)
    result["depth"] = convert_float_to_uint16(depth_frames, depth_min, depth_max)
    result["metadata"]["depth_range"] = [depth_min, depth_max]

  if "forward_flow" in layers:
    result["metadata"]["forward_flow_range"] = [
        data_ranges["forward_flow"]["min"] / scale,
        data_ranges["forward_flow"]["max"] / scale]
    result["forward_flow"] = [
        subsample_nearest_neighbor(file_io.read_png(frame_path)[..., :2],
                                   target_size)
        for frame_path in paths["forward_flow"]]

  if "backward_flow" in layers:
    result["metadata"]["backward_flow_range"] = [
        data_ranges["backward_flow"]["min"] / scale,
        data_ranges["backward_flow"]["max"] / scale]
    result["backward_flow"] = [
        subsample_nearest_neighbor(file_io.read_png(frame_path)[..., :2],
                                   target_size)
        for frame_path in paths["backward_flow"]]

  for key in ["normal", "object_coordinates", "uv"]:
    if key in layers:
      result[key] = [
          subsample_nearest_neighbor(file_io.read_png(frame_path),
                                     target_size)
        for frame_path in paths[key]]

  if "segmentation" in layers:
    # somehow we ended up calling this "segmentations" in TFDS and
    # "segmentation" in kubric. So we have to treat it separately.
    result["segmentations"] = [
        subsample_nearest_neighbor(file_io.read_png(frame_path),
                                   target_size)
        for frame_path in paths["segmentation"]]

  if "rgba" in layers:
    result["video"] = [
        subsample_avg(file_io.read_png(frame_path), target_size)[..., :3]
        for frame_path in paths["rgba"]]

  return example_key, result, metadata


def get_camera_features(seq_length):
  return {
      "focal_length": tf.float32,
      "sensor_width": tf.float32,
      "field_of_view": tf.float32,
      "positions": tfds.features.Tensor(shape=(seq_length, 3),
                                        dtype=tf.float32),
      "quaternions": tfds.features.Tensor(shape=(seq_length, 4),
                                          dtype=tf.float32),
   }


def format_camera_information(metadata):
  return {
      "focal_length": metadata["camera"]["focal_length"],
      "sensor_width": metadata["camera"]["sensor_width"],
      "field_of_view": metadata["camera"]["field_of_view"],
      "positions": np.array(metadata["camera"]["positions"], np.float32),
      "quaternions": np.array(metadata["camera"]["quaternions"], np.float32),
  }


def get_events_features():
  return {
      "collisions": tfds.features.Sequence({
         "instances": tfds.features.Tensor(shape=(2,), dtype=tf.uint16),
         "frame": tf.int32,
         "force": tf.float32,
         "position": tfds.features.Tensor(shape=(3,), dtype=tf.float32),
         "image_position": tfds.features.Tensor(shape=(2,), dtype=tf.float32),
         "contact_normal": tfds.features.Tensor(shape=(3,), dtype=tf.float32),
      })
  }


def format_events_information(events):
  return {
      "collisions": [{
          "instances": np.array(c["instances"], dtype=np.uint16),
          "frame": c["frame"],
          "force": c["force"],
          "position": np.array(c["position"], dtype=np.float32),
          "image_position": np.array(c["image_position"], dtype=np.float32),
          "contact_normal": np.array(c["contact_normal"], dtype=np.float32),
      } for c in events["collisions"]],
  }


def get_instance_features(seq_length: int):
  return {
      "mass": tf.float32,
      "friction": tf.float32,
      "restitution": tf.float32,

      "positions": tfds.features.Tensor(shape=(seq_length, 3),
                                        dtype=tf.float32),
      "quaternions": tfds.features.Tensor(shape=(seq_length, 4),
                                          dtype=tf.float32),
      "velocities": tfds.features.Tensor(shape=(seq_length, 3),
                                         dtype=tf.float32),
      "angular_velocities": tfds.features.Tensor(shape=(seq_length, 3),
                                                 dtype=tf.float32),
      "bboxes_3d": tfds.features.Tensor(shape=(seq_length, 8, 3),
                                        dtype=tf.float32),

      "image_positions": tfds.features.Tensor(shape=(seq_length, 2),
                                              dtype=tf.float32),
      "bboxes": tfds.features.Sequence(
          tfds.features.BBoxFeature()),
      "bbox_frames": tfds.features.Sequence(
          tfds.features.Tensor(shape=(), dtype=tf.int32)),
      "visibility": tfds.features.Tensor(shape=(seq_length,), dtype=tf.uint16),
  }


def format_instance_information(obj):
  return {
      "mass": obj["mass"],
      "friction": obj["friction"],
      "restitution": obj["restitution"],
      "positions": np.array(obj["positions"], np.float32),
      "quaternions": np.array(obj["quaternions"], np.float32),
      "velocities": np.array(obj["velocities"], np.float32),
      "angular_velocities": np.array(obj["angular_velocities"], np.float32),
      "bboxes_3d": np.array(obj["bboxes_3d"], np.float32),
      "image_positions": np.array(obj["image_positions"], np.float32),
      "bboxes": [tfds.features.BBox(*bbox) for bbox in obj["bboxes"]],
      "bbox_frames": np.array(obj["bbox_frames"], dtype=np.uint16),
      "visibility": np.array(obj["visibility"], dtype=np.uint16),
  }


def subsample_nearest_neighbor(arr, size):
  src_height, src_width, _ = arr.shape
  dst_height, dst_width = size
  height_step = src_height // dst_height
  width_step = src_width // dst_width
  assert height_step * dst_height == src_height
  assert width_step * dst_width == src_width

  height_offset = int(np.floor((height_step-1)/2))
  width_offset = int(np.floor((width_step-1)/2))
  subsampled = arr[height_offset::height_step, width_offset::width_step, :]
  return subsampled


def convert_float_to_uint16(array, min_val, max_val):
  return np.round((array - min_val) / (max_val - min_val) * 65535
                  ).astype(np.uint16)


def subsample_avg(arr, size):
  src_height, src_width, channels = arr.shape
  dst_height, dst_width = size
  height_bin = src_height // dst_height
  width_bin = src_width // dst_width
  return np.round(arr.reshape((dst_height, height_bin,
                               dst_width, width_bin,
                               channels)).mean(axis=(1, 3))).astype(np.uint8)


def is_complete_dir(video_dir, layers=DEFAULT_LAYERS):
  video_dir = file_io.as_path(video_dir)
  filenames = [d.name for d in video_dir.iterdir()]
  if not ("data_ranges.json" in filenames and
          "metadata.json" in filenames and
          "events.json" in filenames):
    return False
  nr_frames_per_category = {
      key: len([fn for fn in filenames if fn.startswith(key)])
      for key in layers}

  nr_expected_frames = nr_frames_per_category["rgba"]
  if nr_expected_frames == 0:
    return False
  if not all(nr_frames == nr_expected_frames
             for nr_frames in nr_frames_per_category.values()):
    return False

  return True
