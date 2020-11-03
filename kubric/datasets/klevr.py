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
import pathlib
import logging
import pickle
import argparse
import math

# --- Tensorflow imports
import tensorflow as tf
import tensorflow_datasets as tfds

class Klevr(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version('1.0.0')
  CITATION = """Something TODO to write."""
  DESCRIPTION = """Something TODO to write."""
  URL = "gs://kubric/tfds/klevr/7508d33"  #< remote output dir

  @staticmethod
  def load(*args, **kwargs):
    return tfds.load("klevr", *args, **kwargs)

  def _info(self):
    # --- open metadata
    with tf.io.gfile.GFile(self.URL + "/metadata.pkl", "rb") as fp:
      self.metadata = pickle.load(fp)

    # TODO: this information should have been available in metadata?
    #       perhaps *what* we write to metadata should be modified
    W, H = 512, 512
    
    # TODO: Is there a standard way to provide parameters?
    train_to_test_ratio = .7
    self.n_frames = len(self.metadata)
    self.n_train = math.floor(self.n_frames * train_to_test_ratio)
    self.n_test = self.n_frames - self.n_train

    features = tfds.features.FeaturesDict({
      "source": tf.string,
      "RGBA": tfds.features.Tensor(shape=(H, W, 4), dtype=tf.uint8),
      "segmentation": tfds.features.Tensor(shape=(H, W, 1), dtype=tf.uint32),
      "flow": tfds.features.Tensor(shape=(H, W, 3), dtype=tf.float32),
      "depth": tfds.features.Tensor(shape=(H, W, 1), dtype=tf.float32),
      "UV": tfds.features.Tensor(shape=(H, W, 3), dtype=tf.float32)
    })

    return tfds.core.DatasetInfo(
      builder=self,
      description=self.DESCRIPTION,
      features=features,
      citation=self.CITATION)

  def _split_generators(self, _):
    # TODO: implement the logic for when folder contains multiple kubric runs (i.e. subfolders)
    # subdirs = tf.io.gfile.listdir("gs://kubric/tfds/klevr/")

    # --- split information
    shuffled_ids = np.random.permutation(np.arange(self.n_frames+1)[1:])
    ids_train = shuffled_ids[0:self.n_train]
    ids_test = shuffled_ids[self.n_train:]

    # --- instantiate splits (→_generate_examples)
    train_split = tfds.core.SplitGenerator(name=tfds.Split.TRAIN, gen_kwargs=dict(ids=ids_train))
    test_split = tfds.core.SplitGenerator(name=tfds.Split.TEST, gen_kwargs=dict(ids=ids_test))
    return [train_split, test_split]

  def _generate_examples(self, ids):
    for frame_id in ids:
      frame_path = self.URL + f"/frame_{frame_id:04d}.pkl"

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

if __name__ == "__main__":
  # --- basic setup
  logging.basicConfig(level="INFO")

  # --- where TFDS saves records
  data_dir = "~/tensorflow_datasets"
  data_dir = tempfile.mkdtemp()  #TODO: for development purposes only!

  # --- example query the dataet
  ds_train, info = Klevr.load(split="train", data_dir=data_dir, with_info=True)
  for example in ds_train.take(2):
    print(f"From {example['source']}:")
    print(f"  example['RGBA'].shape={example['RGBA'].shape}")
    print(f"  example['segmentation'].shape={example['segmentation'].shape}")
    print(f"  example['flow'].shape={example['flow'].shape}")
    print(f"  example['depth'].shape={example['depth'].shape}")
    print(f"  example['UV'].shape={example['UV'].shape}")
