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

import tensorflow as tf
import tensorflow_datasets.public_api as tfds


_DESCRIPTION = """
The Klevr dataset is a semi-synthetic dataset created from scans of real world
objects.
The Klevr dataset includes RGBA, optical flow, metric depth, object segmentation
 and object UV texture coordinates.
"""


_CITATION = """
@misc{greff_tagliasacchi_liu_laradji_litany_prasso,
  author={Greff, Klaus and
          Tagliasacchi, Andrea and
          Liu, Derek and
          Laradji, Issam and
          Litany, Or and
          Prasso, Luca}
  title={Kubric},
  url={https://github.com/google-research/kubric},
  journal={GitHub},
}
"""


class KlevrConfig(tfds.core.BuilderConfig):
  """"Configuration for Klevr video dataset."""

  def __init__(
      self, *, path: str, is_master: bool, height: int, width: int, **kwargs
  ):
    """Defines a particular configuration of tensorflow records.

    Args:
      path: The path to the data. Path can be a directory containing
        subfolders of frames, or an individual folder containing frames. The
        code assumes this is a path to a google cloud bucket, but the code will
        also work for, e.g., local directories.
      is_master: If True, the folder provided at path contains subfolders of
        frames. If False, the provided path contains no subfolders.
      height: The height of the provided data.
      width: The width of the provided data.
      **kwargs: Keyword arguments to the base class.
    """
    super(KlevrConfig, self).__init__(**kwargs)
    self.path = path
    self.is_master = is_master
    self.height = height
    self.width = width


def _get_files_from_master(path):
  all_files = []
  all_subdirs = list(path.glob('*'))
  for subdir in all_subdirs:
    all_files.extend(_get_files_from_subdir(subdir))
  logging.info('Found %d sub-folders in master path: %s', len(all_subdirs),
               path)
  return all_files


def _get_files_from_subdir(path):
  files = list(path.glob('frame*.pkl'))
  logging.info('Found %d files in path: %s', len(files), path)
  return files


class Klevr(tfds.core.BeamBasedBuilder):
  """DatasetBuilder for klevr dataset."""
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  BUILDER_CONFIGS = [
      KlevrConfig(
          name='master',
          description='All the klevr files.',
          path=tfds.core.as_path('gs://kubric/tfds/klevr'),
          is_master=True,
          height=512,
          width=512,
      ),

  ]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""

    h = self.builder_config.height
    w = self.builder_config.width
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'source':
                tfds.features.Text(),
            'RGBA':
                tfds.features.Image(shape=(h, w, 4), dtype=tf.uint8),
            'segmentation':
                tfds.features.Tensor(shape=(h, w, 1), dtype=tf.uint32),
            'flow':
                tfds.features.Tensor(shape=(h, w, 3), dtype=tf.float32),
            'depth':
                tfds.features.Tensor(shape=(h, w, 1), dtype=tf.float32),
            'UV':
                tfds.features.Tensor(shape=(h, w, 3), dtype=tf.float32)
        }),
        supervised_keys=None,
        homepage='https://github.com/google-research/kubric',
        citation=_CITATION)

  def _split_generators(self, unused_dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    del unused_dl_manager
    if self.builder_config.is_master:
      all_files = _get_files_from_master(self.builder_config.path)
    else:
      all_files = _get_files_from_subdir(self.builder_config.path)

    return {
        tfds.Split.TRAIN: self._generate_examples(all_files),
    }

  def _generate_examples(self, files):
    """Yields examples."""

    beam = tfds.core.lazy_imports.apache_beam

    def _process_example(file):
      with tf.io.gfile.GFile(file, 'rb') as fp:
        data = pickle.load(fp)
        key = f'{file.parent.name}/{file.name}'
        return key, {
            'source': os.fspath(file),
            'RGBA': data['RGBA'],
            'segmentation': data['segmentation'],
            'flow': data['flow'],
            'depth': data['depth'],
            'UV': data['UV']
        }
    return beam.Create(files) | beam.Map(_process_example)
