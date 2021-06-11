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

# Copyright 2021 The Kubric Authors
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
"""Synthetic video dataset of simple rigid objects interacting physically."""

import logging
import os
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
from typing import List

_DESCRIPTION = """
The KLEVR Uni dataset is a synthetic video dataset of simple rigid objects interacting physically.
It is a slightly more complex version based on the the KLEVR dataset, which uses 11 shapes 
(rather than 3), uniform hsv coloring for the objects (instead of 8 colors), uniform hsv color for 
the floor (instead of gray), and uniform size between (rather than 2 sizes).

Generated with:
./launch.sh hyper examples/klevr_worker.py --object_types=ALL --object_colors=uniform \
                                           --floor_color=uniform --object_sizes=uniform

"""

_CITATION = """
@misc{greff_tagliasacchi_laradji_stone_pot,
  author={Greff, Klaus and
          Tagliasacchi, Andrea and
          Laradji, Issam and
          Stone, Austin and
          Pot, Etienne}
  title={Kubric},
  url={https://github.com/google-research/kubric},
  journal={GitHub},
}
"""


class KlevrUniConfig(tfds.core.BuilderConfig):
  """"Configuration for Klevr video dataset."""

  def __init__(
      self, *, height: int, width: int, num_frames: int,
      validation_ratio: float = 0.1, **kwargs
  ):
    """Defines a particular configuration of tensorflow records.

    Args:
      height (int): The target resolution height.
      width (int): The target resolution width.
      num_frames (int): The target number of frames.
      validation_ratio (float): The proportion of examples to use for validation.
      **kwargs: Keyword arguments to the base class.
    """
    super().__init__(**kwargs)
    self.height = height
    self.width = width
    self.num_frames = num_frames
    self.validation_ratio = validation_ratio


class KlevrUni(tfds.core.BeamBasedBuilder):
  """DatasetBuilder for klevr dataset."""
  VERSION = tfds.core.Version("1.0.0")
  RELEASE_NOTES = {
      "1.0.0": "Initial release.",
  }

  BUILDER_CONFIGS = [
      KlevrUniConfig(
          name="master",
          description="Full resolution of 256x256 and a framerate of 12fps",
          height=256,
          width=256,
          num_frames=24,
          validation_ratio=0.2,
      ),
      KlevrUniConfig(
          name="128x128",
          description="Downscaled to 128x128",
          height=128,
          width=128,
          num_frames=24,
          validation_ratio=0.2,
      ),
      KlevrUniConfig(
          name="64x64",
          description="Downscaled to 64x64",
          height=64,
          width=64,
          num_frames=24,
          validation_ratio=0.2,
      )
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""

    h = self.builder_config.height
    w = self.builder_config.width
    s = self.builder_config.num_frames

    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            "metadata": {
                "video_name": tfds.features.Text(),
                "width": tf.int32,
                "height": tf.int32,
                "num_frames": tf.int32,
                "num_instances": tf.uint16,

                "depth_range": tfds.features.Tensor(shape=(2,), dtype=tf.float32),
                "flow_range": tfds.features.Tensor(shape=(2,), dtype=tf.float32),
                "flow_magnitude_range": tfds.features.Tensor(shape=(2,), dtype=tf.float32),
            },
            "background": {
                "floor_color": tfds.features.Tensor(shape=(3,), dtype=tf.uint8),
            },
            "instances": tfds.features.Sequence(feature={
                "shape": tfds.features.ClassLabel(names=["cube", "cylinder", "sphere", "cone",
                                                         "torus", "gear", "torusknot", "sponge",
                                                         "spot", "teapot", "suzanne"]),
                "size": tf.float32,
                "material": tfds.features.ClassLabel(names=["metal", "rubber"]),
                "color": tfds.features.Tensor(shape=(3,), dtype=tf.uint8),
                "mass": tf.float32,
                "friction": tf.float32,
                "restitution": tf.float32,

                "positions": tfds.features.Tensor(shape=(s, 3), dtype=tf.float32),
                "quaternions": tfds.features.Tensor(shape=(s, 4), dtype=tf.float32),
                "velocities": tfds.features.Tensor(shape=(s, 3), dtype=tf.float32),
                "angular_velocities": tfds.features.Tensor(shape=(s, 3), dtype=tf.float32),

                "image_positions": tfds.features.Tensor(shape=(s, 2), dtype=tf.float32),
                "bboxes": tfds.features.Sequence(
                    tfds.features.BBoxFeature()),
                "bbox_frames": tfds.features.Sequence(
                    tfds.features.Tensor(shape=(), dtype=tf.int32)),
            }),
            "camera": {
                "focal_length": tf.float32,
                "sensor_width": tf.float32,
                "field_of_view": tf.float32,
                "positions": tfds.features.Tensor(shape=(s, 3), dtype=tf.float32),
                "quaternions": tfds.features.Tensor(shape=(s, 4), dtype=tf.float32),
            },
            "events": {
                "collisions": tfds.features.Sequence({
                    "instances": tfds.features.Tensor(shape=(2,), dtype=tf.uint16),
                    "frame": tf.int32,
                    "force": tf.float32,
                    "position": tfds.features.Tensor(shape=(3,), dtype=tf.float32),
                    "image_position": tfds.features.Tensor(shape=(2,), dtype=tf.float32),
                    "contact_normal": tfds.features.Tensor(shape=(3,), dtype=tf.float32),
                }),
            },
            "video":  tfds.features.Video(shape=(s, h, w, 3)),
            "segmentations": tfds.features.Sequence(
                tfds.features.Image(shape=(h, w, 1), dtype=tf.uint16), length=s),
            "flow": tfds.features.Sequence(
                tfds.features.Tensor(shape=(h, w, 4), dtype=tf.uint16), length=s),
            "depth": tfds.features.Sequence(
                tfds.features.Tensor(shape=(h, w, 1), dtype=tf.uint16), length=s),
            "uv": tfds.features.Sequence(
                tfds.features.Tensor(shape=(h, w, 3), dtype=tf.uint16), length=s),
            "normal": tfds.features.Sequence(
                tfds.features.Tensor(shape=(h, w, 3), dtype=tf.uint16), length=s),
        }),
        supervised_keys=None,
        homepage="https://github.com/google-research/kubric",
        citation=_CITATION)

  def _split_generators(self, unused_dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    del unused_dl_manager
    path = tfds.core.as_path("gs://research-brain-kubric-xgcp/jobs/klevr_uni_v0_10k",)
    all_subdirs = sorted(list(path.glob("*")), key=lambda x: int(x.name))
    logging.info("Found %d sub-folders in master path: %s", len(all_subdirs), path)

    str_directories = [str(d) for d in all_subdirs]
    validation_ratio = self.builder_config.validation_ratio
    validation_examples = round(len(str_directories) * validation_ratio)
    training_examples = len(str_directories) - validation_examples
    logging.info("Using %s of examples for validation for a total of %d",
                 "{:.2%}".format(validation_ratio), validation_examples)
    logging.info("Using the other %d examples for training", training_examples)

    return {
        tfds.Split.TRAIN: self._generate_examples(str_directories[:training_examples]),
        tfds.Split.VALIDATION: self._generate_examples(str_directories[training_examples:]),
    }

  def _generate_examples(self, directories: List[str]):
    """Yields examples."""

    target_size = (self.builder_config.height, self.builder_config.width)
    assert 24 % self.builder_config.num_frames == 0
    frame_subsampling = 24 // self.builder_config.num_frames

    def _process_example(video_dir):
      video_dir = tfds.core.as_path(video_dir)
      key = f"{video_dir.name}"
      files = sorted([str(f) for f in video_dir.glob("frame*.pkl")])

      with tf.io.gfile.GFile(str(video_dir / "metadata.pkl"), "rb") as fp:
        metadata = pickle.load(fp)

      def convert_float_to_uint16(array, min_val, max_val):
        return np.round((array - min_val) / (max_val - min_val) * 65535).astype(np.uint16)

      num_frames = metadata["instances"][0]["positions"].shape[0]
      num_instances = metadata["metadata"]["num_instances"]
      assert len(files) == num_frames, f"{len(files)} != {num_frames}"
      metalen = len(metadata["instances"])
      assert len(metadata["instances"]) == num_instances, f"{metalen} != {num_instances}"

      frames = []
      for frame_file in files[::frame_subsampling]:
        with tf.io.gfile.GFile(str(frame_file), "rb") as fp:
          data = pickle.load(fp)
          frames.append({
              "rgb": data["rgba"][..., :3],
              "segmentation": data["segmentation"].astype(np.uint16),
              "flow": data["flow"],
              "depth": data["depth"],
              "uv": data["uv"],
              "normal": data["normal"]
          })

      # compute the range of the depth map
      min_depth = np.min([np.min(f["depth"]) for f in frames])
      max_depth = np.max([np.max(f["depth"]) for f in frames])

      min_flow = np.min([np.min(f["flow"]) for f in frames])
      max_flow = np.max([np.max(f["flow"]) for f in frames])

      # compute the range of magnitudes of the optical flow (both forward and backward)
      def flow_magnitude(vec):
        bwd_flow_magnitude = np.linalg.norm(vec[:, :, :2], axis=-1)
        min_bwd_flow_mag = np.min(bwd_flow_magnitude)
        max_bwd_flow_mag = np.max(bwd_flow_magnitude)
        fwd_flow_magnitude = np.linalg.norm(vec[:, :, 2:], axis=-1)
        min_fwd_flow_mag = np.min(fwd_flow_magnitude)
        max_fwd_flow_mag = np.max(fwd_flow_magnitude)
        return min(min_fwd_flow_mag, min_bwd_flow_mag), max(max_fwd_flow_mag, max_bwd_flow_mag)

      flow_magnitude_ranges = [flow_magnitude(frame["flow"]) for frame in frames]
      min_flow_mag = np.min([fmr[0] for fmr in flow_magnitude_ranges])
      max_flow_mag = np.max([fmr[1] for fmr in flow_magnitude_ranges])

      # compute bboxes
      for i, obj in enumerate(metadata["instances"]):
        obj["bboxes"] = []
        obj["bbox_frames"] = []
        for j, frame in zip(range(0, num_frames, frame_subsampling), frames):
          seg = frame["segmentation"][:, :, 0]
          idxs = np.array(np.where(seg == i+1), dtype=np.float32)
          if idxs.size > 0:
            idxs /= np.array(seg.shape)[:, np.newaxis]
            bbox = tfds.features.BBox(ymin=float(idxs[0].min()), xmin=float(idxs[1].min()),
                                      ymax=float(idxs[0].max()), xmax=float(idxs[1].max()))
            obj["bboxes"].append(bbox)
            obj["bbox_frames"].append(j//frame_subsampling)

      src_height, src_width = frames[0]["rgb"].shape[:2]
      assert src_height % target_size[0] == 0
      assert src_width % target_size[1] == 0

      return key, {
          "metadata": {
              "video_name": os.fspath(key),
              "width": target_size[1],
              "height": target_size[0],
              "num_frames": len(frames),
              "num_instances": num_instances,
              "depth_range": np.array([min_depth, max_depth], dtype=np.float32),
              "flow_range": np.array([min_flow, max_flow], dtype=np.float32),
              "flow_magnitude_range": np.array([min_flow_mag, max_flow_mag], dtype=np.float32)
          },
          "background": {
              "floor_color": np.round(np.array(metadata["background"]["floor_color"])*255
                                      ).astype(np.uint8),
          },
          "instances": [{
              "shape": obj["shape"].lower(),
              "size": obj["size"],
              "material": obj["material"],
              "color": np.round(np.array(obj["color"])*255).astype(np.uint8),
              "mass": obj["mass"],
              "friction": obj["friction"],
              "restitution": obj["restitution"],
              "positions": obj["positions"][::frame_subsampling].astype(np.float32),
              "quaternions": obj["quaternions"][::frame_subsampling].astype(np.float32),
              "velocities": obj["velocities"][::frame_subsampling].astype(np.float32),
              "angular_velocities": obj["angular_velocities"][::frame_subsampling].
                astype(np.float32),
              "image_positions": obj["image_positions"][::frame_subsampling].astype(np.float32),
              "bboxes": obj["bboxes"],
              "bbox_frames": np.array(obj["bbox_frames"], dtype=np.uint16),
          } for obj in metadata["instances"]],
          "camera": {
              "focal_length": metadata["camera"]["focal_length"],
              "sensor_width": metadata["camera"]["sensor_width"],
              "field_of_view": metadata["camera"]["field_of_view"],
              "positions": metadata["camera"]["positions"][::frame_subsampling].astype(np.float32),
              "quaternions": metadata["camera"]["quaternions"][::frame_subsampling].
                astype(np.float32),
          },
          "events": {
              "collisions": [{
                  "instances": np.array(c["instances"], dtype=np.uint16),
                  "contact_normal": np.array(c["contact_normal"], dtype=np.float32),
                  "frame": c["frame"],
                  "force": c["force"],
                  "position": np.array(c["position"], dtype=np.float32),
                  "image_position": np.array(c["image_position"], dtype=np.float32),
              } for c in metadata["events"]["collisions"]],
          },
          "video": [subsample_avg(f["rgb"], target_size) for f in frames],
          "segmentations": [subsample_nearest_neighbor(f["segmentation"], target_size)
                            for f in frames],
          "flow": [convert_float_to_uint16(subsample_nearest_neighbor(f["flow"], target_size),
                                           min_flow, max_flow)
                   for f in frames],
          "depth": [convert_float_to_uint16(subsample_nearest_neighbor(f["depth"], target_size),
                                            min_depth, max_depth)
                    for f in frames],
          "uv": [convert_float_to_uint16(subsample_nearest_neighbor(f["uv"], target_size), 0., 1.)
                 for f in frames],
          "normal": [convert_float_to_uint16(subsample_nearest_neighbor(f["normal"], target_size),
                                             -1., 1.) for f in frames],
      }

    beam = tfds.core.lazy_imports.apache_beam
    return beam.Create(directories) | beam.Map(_process_example)


def _get_files_from_subdir(path: str) -> List[str]:
  path = tfds.core.as_path(path)
  files = [str(f) for f in path.glob("frame*.pkl")]
  logging.info("Found %d files in path: %s", len(files), path)
  return files


def subsample_nearest_neighbor(arr, size):
  src_height, src_width, _ = arr.shape
  dst_height, dst_width = size
  height_step = src_height // dst_height
  width_step = src_width // dst_width
  height_offset = int(np.floor((height_step-1)/2))
  width_offset = int(np.floor((width_step-1)/2))
  return arr[height_offset::height_step, width_offset::width_step, :]


def subsample_avg(arr, size):
  src_height, src_width, channels = arr.shape
  dst_height, dst_width = size
  height_bin = src_height // dst_height
  width_bin = src_width // dst_width
  return np.round(arr.reshape((dst_height, height_bin,
                               dst_width, width_bin,
                               channels)).mean(axis=(1, 3))).astype(np.uint8)

