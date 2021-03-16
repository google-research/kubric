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
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
from typing import List, Tuple

_DESCRIPTION = """
The Klevr dataset is a synthetic video dataset of simple rigid objects interacting physically.
It is based on the the CLEVR dataset:
https://cs.stanford.edu/people/jcjohns/clevr/

It includes RGB, optical flow, metric depth, and object segmentation, as well as 
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


class KlevrConfig(tfds.core.BuilderConfig):
  """"Configuration for Klevr video dataset."""

  def __init__(
      self, *, height: int, width: int, num_frames: int, subset=None, validation_percentage:float = 0.1, **kwargs
  ):
    """Defines a particular configuration of tensorflow records.

    Args:
      path: The path to the data. Path can be a directory containing
        subfolders of frames, or an individual folder containing frames. The
        code assumes this is a path to a google cloud bucket, but the code will
        also work for, e.g., local directories.
      height: The height of the provided data.
      width: The width of the provided data.
      max_objects: maximum number of objects that occur within a single scene
                  (counting the background as one object).
      **kwargs: Keyword arguments to the base class.
    """
    super(KlevrConfig, self).__init__(**kwargs)
    self.height = height
    self.width = width
    self.num_frames = num_frames
    self.subset = subset
    self.validation_percentage = validation_percentage


class Klevrheavy(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for klevr dataset."""
  VERSION = tfds.core.Version('1.3.2')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
      '1.0.1': 'mini version with just 100 examples',
      '1.0.3': 'simplified version without instances, camera, and events',
      '1.0.4': 'minimal version with just video',
      '1.0.5': 'minimal + segmentation',
      '1.0.6': 'minimal + depth + uv + normal + flow',
      '1.0.7': 'everything except depth/uv/normal/flow',
      '1.0.8': 'like 1.0.7 but with all examples',
      '1.0.9': 'full dataset but with depth/uv/normal/flow converted to uint16',
      '1.1.0': 'use 10% of examples as validation split, sparse bboxes, no depth/uv/normal/flow',
      '1.2.0': 'include subsampled variants',
      '1.3.0': 'test full sized dataset',
      '1.3.1': 'without validation',
      '1.3.2': 'non-legacy generator, small ',
  }

  BUILDER_CONFIGS = [
      KlevrConfig(
          name='master',
          description='Full resolution of 512x512 and full framerate of 48fps',
          height=512,
          width=512,
          num_frames=48,
          subset=10,
          validation_percentage=0.0,
      ),
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
            'metadata': {
                'video_name': tfds.features.Text(),
                'width': tf.int32,
                'height': tf.int32,
                'num_frames': tf.int32,
                'num_instances': tf.uint16,

                'depth_range': tfds.features.Tensor(shape=(2,), dtype=tf.float32),
                'flow_range': tfds.features.Tensor(shape=(2,), dtype=tf.float32),
                'flow_magnitude_range': tfds.features.Tensor(shape=(2,), dtype=tf.float32),
            },
            'instances': tfds.features.Sequence(feature={
                'shape': tfds.features.ClassLabel(names=["cube", "cylinder", "sphere"]),
                'size': tfds.features.ClassLabel(names=["small", "large"]),
                'material': tfds.features.ClassLabel(names=["metal", "rubber"]),
                'color': tfds.features.ClassLabel(names=["blue", "brown", "cyan", "gray",
                                                         "green", "purple", "red", "yellow"]),
                'mass': tf.float32,
                'friction': tf.float32,
                'restitution': tf.float32,

                'positions': tfds.features.Tensor(shape=(s, 3), dtype=tf.float32),
                'quaternions': tfds.features.Tensor(shape=(s, 4), dtype=tf.float32),
                'velocities': tfds.features.Tensor(shape=(s, 3), dtype=tf.float32),
                'angular_velocities': tfds.features.Tensor(shape=(s, 3), dtype=tf.float32),
                'image_positions': tfds.features.Tensor(shape=(s, 2), dtype=tf.float32),
                # TODO: Sequence(BBoxFeature(), length=s)  crashes but that might be a tfds bug
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
                'collision': tfds.features.Sequence({
                    'instances': tfds.features.Tensor(shape=(2,), dtype=tf.uint16),
                    'contact_normal': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
                    'frame': tf.int32,
                    'force': tf.float32,
                }),
            },
            'video':  tfds.features.Video(shape=(s, h, w, 3)),
            'segmentations': tfds.features.Sequence(
                tfds.features.Image(shape=(h, w, 1), dtype=tf.uint16), length=s),
            'flow': tfds.features.Sequence(
                tfds.features.Tensor(shape=(h, w, 4), dtype=tf.float32), length=s),
            'depth': tfds.features.Sequence(
                tfds.features.Tensor(shape=(h, w, 1), dtype=tf.float32), length=s),
            'uv': tfds.features.Sequence(
                tfds.features.Tensor(shape=(h, w, 3), dtype=tf.float32), length=s),
            'normal': tfds.features.Sequence(
                tfds.features.Tensor(shape=(h, w, 3), dtype=tf.float32), length=s),
        }),
        supervised_keys=None,
        homepage='https://github.com/google-research/kubric',
        citation=_CITATION)

  def _split_generators(self, unused_dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    del unused_dl_manager
    path = tfds.core.as_path('gs://research-brain-kubric-xgcp/jobs/klevr_v0',)
    all_subdirs = sorted([d for d in path.glob('*')
                          if (d / 'metadata.pkl').exists()], key=lambda x: int(x.name))
    logging.info('Found %d sub-folders in master path: %s', len(all_subdirs), path)

    if self.builder_config.subset:
      all_subdirs = all_subdirs[:self.builder_config.subset]
      logging.info('Using a subset of %d folders.', len(all_subdirs))

    str_directories = [str(d) for d in all_subdirs]
    validation_percentage = self.builder_config.validation_percentage
    validation_examples = round(len(str_directories) * validation_percentage)
    training_examples = len(str_directories) - validation_examples
    logging.info("Using %s of examples for validation for a total of %d",
                 "{:.2%}".format(validation_percentage), validation_examples)
    logging.info("Using the other %d examples for training", training_examples)

    return {
        tfds.Split.TRAIN: self._generate_examples(str_directories),
    }

  def _generate_examples(self, directories: List[str]):
    """Yields examples."""

    target_size = (self.builder_config.height, self.builder_config.width)
    assert 48 % self.builder_config.num_frames == 0
    frame_subsampling = 48 // self.builder_config.num_frames

    def _process_example(video_dir):
      video_dir = tfds.core.as_path(video_dir)
      key = f'{video_dir.name}'
      files = sorted([str(f) for f in video_dir.glob('frame*.pkl')])

      with tf.io.gfile.GFile(str(video_dir / 'metadata.pkl'), 'rb') as fp:
        metadata = pickle.load(fp)

      def convert_float_to_uint16(array, min_val, max_val):
        return np.round((array - min_val) / (max_val - min_val) * 65535).astype(np.uint16)

      frames = []
      for frame_file in files:
        with tf.io.gfile.GFile(str(frame_file), 'rb') as fp:
          data = pickle.load(fp)
          frames.append({
              'rgb': data['rgba'][..., :3],
              'segmentation': data['segmentation'].astype(np.uint16),
              'flow': data['flow'],
              'depth': data['depth'],
              'uv': data['uv'],
              'normal': data['normal']
          })


      num_frames = metadata["objects"][0]["positions"].shape[0]
      assert len(frames) == num_frames, f"{len(frames)} != {num_frames}"
      assert len(metadata["objects"]) == metadata["nr_objects"], f"{len(metadata['objects'])} != {metadata['nr_objects']}"

      # subsample frames
      frames = frames[::frame_subsampling]

      # compute the range of the depth map
      min_depth = np.min([np.min(f['depth']) for f in frames])
      max_depth = np.max([np.max(f['depth']) for f in frames])

      min_flow = np.min([np.min(f['flow']) for f in frames])
      max_flow = np.max([np.max(f['flow']) for f in frames])

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
      for i, obj in enumerate(metadata["objects"]):
        obj['bboxes'] = []
        obj['bbox_frames'] = []
        for j, frame in zip(range(0, num_frames, frame_subsampling), frames):
          seg = frame['segmentation'][:, :, 0]
          idxs = np.array(np.where(seg == i+1), dtype=np.float32)
          if idxs.size > 0:
            idxs /= np.array(seg.shape)[:, np.newaxis]
            bbox = tfds.features.BBox(ymin=float(idxs[0].min()), xmin=float(idxs[1].min()),
                                      ymax=float(idxs[0].max()), xmax=float(idxs[1].max()))
            obj["bboxes"].append(bbox)
            obj["bbox_frames"].append(j)

      src_height, src_width = frames[0]['rgb'].shape[:2]
      assert src_height % target_size[0] == 0
      assert src_width % target_size[1] == 0

      return key, {
          'metadata': {
              'video_name': os.fspath(key),
              'width': target_size[1],
              'height': target_size[0],
              'num_frames': len(frames),
              'num_instances': metadata['nr_objects'],
              'depth_range': [min_depth, max_depth],
              'flow_range': [min_flow, max_flow],
              'flow_magnitude_range': [min_flow_mag, max_flow_mag]
          },
          'instances': [{
              'shape': obj['shape'],
              'size': obj['size'],
              'material': obj['material'],
              'color': obj['color'],
              'mass': obj['mass'],
              'friction': obj['friction'],
              'restitution': obj['restitution'],
              'positions': obj['positions'][::frame_subsampling],
              'quaternions': obj['quaternions'][::frame_subsampling],
              'velocities': obj['velocities'][::frame_subsampling],
              'angular_velocities': obj['angular_velocities'][::frame_subsampling],
              'image_positions': obj['image_positions'][::frame_subsampling],
              'bboxes': obj['bboxes'],
              'bbox_frames': obj['bbox_frames'],
          } for obj in metadata["objects"]],
          'camera': {
              "focal_length": metadata["camera"]["focal_length"],
              "sensor_width": metadata["camera"]["sensor_width"],
              "field_of_view": metadata["camera"]["field_of_view"],
              "positions": metadata["camera"]["positions"][::frame_subsampling],
              "quaternions": metadata["camera"]["quaternions"][::frame_subsampling],
          },
          'events': {"collision": []},
          'video': [subsample_avg(f['rgb'], target_size) for f in frames],
          'segmentations': [subsample_nearest_neighbor(f['segmentation'], target_size)
                            for f in frames],
          'flow': [subsample_nearest_neighbor(f['flow'], target_size) for f in frames],
          'depth': [subsample_nearest_neighbor(f['depth'], target_size) for f in frames],
          'uv': [subsample_nearest_neighbor(f['uv'], target_size) for f in frames],
          'normal': [subsample_nearest_neighbor(f['normal'], target_size) for f in frames],
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

