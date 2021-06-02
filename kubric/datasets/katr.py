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

import logging
import os
import json

import numpy as np
import png
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
from typing import List

_DESCRIPTION = ""

_CITATION = ""


class KatrConfig(tfds.core.BuilderConfig):
  """"Configuration for Katr video dataset."""

  def __init__(
      self, *, height: int, width: int,
      validation_ratio: float = 0.1, **kwargs
  ):
    """Defines a particular configuration of tensorflow records.

    Args:
      height (int): The target resolution height.
      width (int): The target resolution width.
      validation_ratio (float): The proportion of examples to use for validation.
      **kwargs: Keyword arguments to the base class.
    """
    super(KatrConfig, self).__init__(**kwargs)
    self.height = height
    self.width = width
    self.validation_ratio = validation_ratio


class Katr(tfds.core.BeamBasedBuilder):
  """DatasetBuilder for Katr dataset."""
  VERSION = tfds.core.Version('1.1.1')
  RELEASE_NOTES = {
      '1.1.1': "small test",
      '1.1.0': 'split flow into -> forward_flow and backward_flow',
      '1.0.0': 'initial release',
  }

  BUILDER_CONFIGS = [
      KatrConfig(
          name='master',
          description='Full resolution of 512x512 and a framerate of 12fps',
          height=512,
          width=512,
          validation_ratio=0.2,
      ),
      KatrConfig(
          name='256x256',
          description='Downscaled to 256x256',
          height=256,
          width=256,
          validation_ratio=0.2,
      ),
      KatrConfig(
          name='128x128',
          description='Downscaled to 128x128',
          height=128,
          width=128,
          validation_ratio=0.2,
      ),
      KatrConfig(
          name='64x64',
          description='Downscaled to 64x64',
          height=64,
          width=64,
          validation_ratio=0.2,
      )
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""

    h = self.builder_config.height
    w = self.builder_config.width
    s = 24

    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'metadata': {
                'video_name': tfds.features.Text(),
                'width': tf.int32,
                'height': tf.int32,
                'num_frames': tf.int32,
                'num_instances': tf.uint16,

                'depth_range': tfds.features.Tensor(shape=(2,), dtype=tf.float32),
                'forward_flow_range': tfds.features.Tensor(shape=(2,), dtype=tf.float32),
                'backward_flow_range': tfds.features.Tensor(shape=(2,), dtype=tf.float32),
            },
            'background': {
                'hdri': tfds.features.Text(),
            },
            'instances': tfds.features.Sequence(feature={
                'asset_id': tfds.features.Text(),
                'mass': tf.float32,
                'friction': tf.float32,
                'restitution': tf.float32,

                'positions': tfds.features.Tensor(shape=(s, 3), dtype=tf.float32),
                'quaternions': tfds.features.Tensor(shape=(s, 4), dtype=tf.float32),
                'velocities': tfds.features.Tensor(shape=(s, 3), dtype=tf.float32),
                'angular_velocities': tfds.features.Tensor(shape=(s, 3), dtype=tf.float32),

                'image_positions': tfds.features.Tensor(shape=(s, 2), dtype=tf.float32),
                'bboxes': tfds.features.Sequence(
                    tfds.features.BBoxFeature()),
                'bbox_frames': tfds.features.Sequence(
                    tfds.features.Tensor(shape=(), dtype=tf.int32)),
            }),
            'camera': {
                'focal_length': tf.float32,
                'sensor_width': tf.float32,
                'field_of_view': tf.float32,
                'positions': tfds.features.Tensor(shape=(s, 3), dtype=tf.float32),
                'quaternions': tfds.features.Tensor(shape=(s, 4), dtype=tf.float32),
            },
            'events': {
                'collisions': tfds.features.Sequence({
                    'instances': tfds.features.Tensor(shape=(2,), dtype=tf.uint16),
                    'frame': tf.int32,
                    'force': tf.float32,
                    'position': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
                    'image_position': tfds.features.Tensor(shape=(2,), dtype=tf.float32),
                    'contact_normal': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
                }),
            },
            'video':  tfds.features.Video(shape=(s, h, w, 3)),
            'segmentations': tfds.features.Sequence(
                tfds.features.Image(shape=(h, w, 1), dtype=tf.uint16), length=s),
            'forward_flow': tfds.features.Sequence(
                tfds.features.Tensor(shape=(h, w, 2), dtype=tf.uint16), length=s),
            'backward_flow': tfds.features.Sequence(
                tfds.features.Tensor(shape=(h, w, 2), dtype=tf.uint16), length=s),
            'depth': tfds.features.Sequence(
                tfds.features.Tensor(shape=(h, w, 1), dtype=tf.uint16), length=s),
            'uv': tfds.features.Sequence(
                tfds.features.Tensor(shape=(h, w, 3), dtype=tf.uint16), length=s),
            'normal': tfds.features.Sequence(
                tfds.features.Tensor(shape=(h, w, 3), dtype=tf.uint16), length=s),
        }),
        supervised_keys=None,
        homepage='https://github.com/google-research/kubric',
        citation=_CITATION)

  def _split_generators(self, unused_dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    del unused_dl_manager
    path = tfds.core.as_path('gs://research-brain-kubric-xgcp/jobs/kubric_may03_145256',)
    all_subdirs = sorted([d for d in path.glob('*')], key=lambda x: int(x.name))
    logging.info('Found %d sub-folders in master path: %s', len(all_subdirs), path)

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

    def _process_example(video_dir):
      video_dir = tfds.core.as_path(video_dir)
      key = f'{video_dir.name}'

      with tf.io.gfile.GFile(str(video_dir / 'data_ranges.json'), 'rb') as fp:
        data_ranges = json.load(fp)

      with tf.io.gfile.GFile(str(video_dir / 'metadata.json'), 'rb') as fp:
        metadata = json.load(fp)

      with tf.io.gfile.GFile(str(video_dir / 'bboxes.json'), 'rb') as fp:
        bboxes = json.load(fp)

      num_frames = metadata["metadata"]["num_frames"]
      num_instances = metadata["metadata"]["num_instances"]
      assert len(metadata["instances"]) == num_instances, f"{len(metadata['instances'])} != {num_instances}"

      rgba_frame_paths = [video_dir / f"rgba_{f:05d}.png" for f in range(num_frames)]
      segmentation_frame_paths = [video_dir / f"segmentation_{f:05d}.png" for f in range(num_frames)]
      fwd_flow_frame_paths = [video_dir / f"forward_flow_{f:05d}.png" for f in range(num_frames)]
      bwd_flow_frame_paths = [video_dir / f"backward_flow_{f:05d}.png" for f in range(num_frames)]
      depth_frame_paths = [video_dir / f"depth_{f:05d}.png" for f in range(num_frames)]
      uv_frame_paths = [video_dir / f"uv_{f:05d}.png" for f in range(num_frames)]
      normal_frame_paths = [video_dir / f"normal_{f:05d}.png" for f in range(num_frames)]

      return key, {
          'metadata': {
              'video_name': os.fspath(key),
              'width': target_size[1],
              'height': target_size[0],
              'num_frames': num_frames,
              'num_instances': num_instances,
              'depth_range': [data_ranges["depth"]["min"], data_ranges["depth"]["max"]],
              'forward_flow_range': [data_ranges["forward_flow"]["min"],
                                     data_ranges["forward_flow"]["max"]],
              'backward_flow_range': [data_ranges["backward_flow"]["min"],
                                      data_ranges["backward_flow"]["max"]],
          },
          'background': {
              'hdri': metadata['background']['hdri'],
          },
          'instances': [{
              'asset_id': obj['asset_id'],
              'mass': obj['mass'],
              'friction': obj['friction'],
              'restitution': obj['restitution'],
              'positions': np.array(obj['positions'], np.float32),
              'quaternions': np.array(obj['quaternions'], np.float32),
              'velocities': np.array(obj['velocities'], np.float32),
              'angular_velocities': np.array(obj['angular_velocities'], np.float32),
              'image_positions': np.array(obj['image_positions'], np.float32),
              'bboxes': [tfds.features.BBox(*bbox) for bbox in bboxes[i]['bboxes']],
              'bbox_frames': np.array(bboxes[i]['bbox_frames'], dtype=np.uint16),
          } for i, obj in enumerate(metadata["instances"])],
          'camera': {
              "focal_length": metadata["camera"]["focal_length"],
              "sensor_width": metadata["camera"]["sensor_width"],
              "field_of_view": metadata["camera"]["field_of_view"],
              "positions": np.array(metadata["camera"]["positions"], np.float32),
              "quaternions": np.array(metadata["camera"]["quaternions"], np.float32),
          },
          'events': {
              'collisions': [{
                  "instances": np.array(c["instances"], dtype=np.uint16),
                  "frame": c["frame"],
                  "force": c["force"],
                  "position": np.array(c["position"], dtype=np.float32),
                  "image_position": np.array(c["image_position"], dtype=np.float32),
                  "contact_normal": np.array(c["contact_normal"], dtype=np.float32),
              } for c in metadata["events"]["collisions"]],
          },
          'video': [subsample_avg(read_png(frame_path), target_size)[..., :3]
                    for frame_path in rgba_frame_paths],
          'segmentations': [subsample_nearest_neighbor(read_png(frame_path), target_size)
                            for frame_path in segmentation_frame_paths],
          'forward_flow': [subsample_nearest_neighbor(read_png(frame_path), target_size)[..., :2]
                           for frame_path in fwd_flow_frame_paths],
          'backward_flow': [subsample_nearest_neighbor(read_png(frame_path), target_size)[..., :2]
                            for frame_path in bwd_flow_frame_paths],
          'depth': [subsample_nearest_neighbor(read_png(frame_path), target_size)
                    for frame_path in depth_frame_paths],
          'uv': [subsample_nearest_neighbor(read_png(frame_path), target_size)
                 for frame_path in uv_frame_paths],
          'normal': [subsample_nearest_neighbor(read_png(frame_path), target_size)
                     for frame_path in normal_frame_paths],
      }

    beam = tfds.core.lazy_imports.apache_beam
    return beam.Create(directories) | beam.Map(_process_example)


def _get_files_from_subdir(path: str) -> List[str]:
  path = tfds.core.as_path(path)
  files = [str(f) for f in path.glob('frame*.pkl')]
  logging.info('Found %d files in path: %s', len(files), path)
  return files


def subsample_nearest_neighbor(arr, size):
  src_height, src_width, channels = arr.shape
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


def read_png(path: os.PathLike):
  path = tfds.core.as_path(path)
  pngReader = png.Reader(bytes=path.read_bytes())
  width, height, pngdata, info = pngReader.read()
  del pngReader
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
