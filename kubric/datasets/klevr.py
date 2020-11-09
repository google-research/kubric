# Copyright 2020 The Kubric Authors
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
import os
import numpy as np
import tempfile
import logging
import pickle
import argparse
import math

# --- Tensorflow imports
import tensorflow as tf
import tensorflow_datasets as tfds

VERSION = tfds.core.Version('1.0.0')
CITATION = """TODO."""  # TODO: later
DESCRIPTION = """TODO."""  # TODO: later

# TODO: this information should have been available in metadata?
#       perhaps *what* we write to metadata should be modified
W, H = 512, 512

FEATURES = tfds.features.FeaturesDict({
  "source": tf.string,
  "RGBA": tfds.features.Tensor(shape=(H, W, 4), dtype=tf.uint8),
  "segmentation": tfds.features.Tensor(shape=(H, W, 1), dtype=tf.uint32),
  "flow": tfds.features.Tensor(shape=(H, W, 3), dtype=tf.float32),
  "depth": tfds.features.Tensor(shape=(H, W, 1), dtype=tf.float32),
  "UV": tfds.features.Tensor(shape=(H, W, 3), dtype=tf.float32)
})

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class Klevr(tfds.core.GeneratorBasedBuilder):
  VERSION = VERSION

  def __init__(self, path, train_to_test_ratio=.7, *args, **kwargs):
    super().__init__(*args, **kwargs)
    logging.info(f"Klevr(path={path})")
    self.path = path

    # --- open metadata
    with tf.io.gfile.GFile(str(self.path + "/metadata.pkl"), "rb") as fp:
      self.metadata = pickle.load(fp)

    # --- split information
    self.n_frames = len(self.metadata)
    self.n_train = math.floor(self.n_frames * train_to_test_ratio)
    self.n_test = self.n_frames - self.n_train

  def _info(self):
    return tfds.core.DatasetInfo(
      builder=self,
      description=DESCRIPTION,
      features=FEATURES,
      citation=CITATION)

  def _split_generators(self, _):
    # --- split information
    shuffled_ids = np.random.permutation(np.arange(self.n_frames+1)[1:])
    ids_train = shuffled_ids[0:self.n_train]
    ids_test = shuffled_ids[self.n_train:]

    # --- instantiate splits (→_generate_examples)
    return {
      "train": self._generate_examples(ids_train),
      "test": self._generate_examples(ids_test) 
    }

  def _generate_examples(self, ids):
    for frame_id in ids:
      frame_path = self.path + f"/frame_{frame_id:04d}.pkl"

      with tf.io.gfile.GFile(frame_path, "rb") as fp:
        data = pickle.load(fp)

        yield frame_path, {
          "source": frame_path,
          "RGBA": data["RGBA"],
          "segmentation": data["segmentation"],
          "flow": data["flow"],
          "depth": data["depth"],
          "UV": data["UV"]
        }
    
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class MasterKlevr(tfds.core.GeneratorBasedBuilder):
  VERSION = VERSION

  def __init__(self, path, *args, **kwargs):
    super().__init__(*args, **kwargs)
    
    # --- create a builder for each subfolder
    self._builders = list()
    subdirs = tf.io.gfile.listdir(path)
    # removes trailing "/"" caused by listdir, and recovers full path
    subdirs = [path+"/"+subdir[:-1] for subdir in subdirs]

    # --- generate the dataset builders 
    for isubdir, subdir in enumerate(subdirs):
      builder = Klevr(path=subdir)
      builder.download_and_prepare()
      self._builders.append(builder)

  def _info(self):
    return tfds.core.DatasetInfo(
      builder=self,
      description=DESCRIPTION,
      features=FEATURES,
      citation=CITATION)

  def _split_generators(self, _):
    datasets_train = [builder.as_dataset(split="train") for builder in self._builders]
    datasets_test = [builder.as_dataset(split="test") for builder in self._builders]
    return {
      "train": self._generate_examples(datasets_train),
      "test": self._generate_examples(datasets_test) 
    }

  def _generate_examples(self, datasets):
    for dataset in datasets:
      for example in dataset:
        key = example["source"]
        yield key, example

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

if __name__ == "__main__":
  # --- basic setup
  logging.basicConfig(level="INFO")

  # --- example query SINGLE folder dataset
  if False:
    path = "gs://kubric/tfds/klevr/7508d33"
    builder = Klevr(path)
    builder.download_and_prepare()
    ds_train = builder.as_dataset(split="train")
    for example in ds_train.take(2):
      print(f"From {example['source']}:")
      print(f"  example['RGBA'].shape={example['RGBA'].shape}")
      print(f"  example['segmentation'].shape={example['segmentation'].shape}")
      print(f"  example['flow'].shape={example['flow'].shape}")
      print(f"  example['depth'].shape={example['depth'].shape}")
      print(f"  example['UV'].shape={example['UV'].shape}")

  # --- example query HIERARCHICAL folder dataset
  if True:
    path = "gs://kubric/tfds/klevr"
    builder = MasterKlevr(path)
    builder.download_and_prepare()
    ds_train = builder.as_dataset(split="train")
    for example in ds_train.take(2):
      print(f"From {example['source']}:")
      print(f"  example['RGBA'].shape={example['RGBA'].shape}")
      print(f"  example['segmentation'].shape={example['segmentation'].shape}")
      print(f"  example['flow'].shape={example['flow'].shape}")
      print(f"  example['depth'].shape={example['depth'].shape}")
      print(f"  example['UV'].shape={example['UV'].shape}")