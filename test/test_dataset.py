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

import pathlib
import tensorflow as tf
from kubric.datasets import klevr
import tensorflow_datasets.public_api as tfds


class KlevrTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for klevr dataset."""
  BUILDER_CONFIG_NAMES_TO_TEST = ['test']
  DATASET_CLASS = klevr.Klevr
  SPLITS = {
      tfds.Split.TRAIN: 4,  # Number of fake train example
  }

  @classmethod
  def setUpClass(cls):
    klevr.Klevr.BUILDER_CONFIGS = [klevr.KlevrConfig(
        name='test',
        description='Dummy test.',
        path=pathlib.Path(__file__).parent / "dummy_data",
        is_master=True,
        height=28,
        width=28)]
    super().setUpClass()

  def _download_and_prepare_as_dataset(self, builder):
    super()._download_and_prepare_as_dataset(builder)

    if not tf.executing_eagerly():  # Only test the following in eager mode.
      return

    with self.subTest('source_check'):
      splits = builder.as_dataset()
      train_ex = list(splits[tfds.Split.TRAIN])[0]
      # Check that the shapes of the dataset examples is correct.
      self.assertIn('dummy_data', str(train_ex['source'].numpy()))


if __name__ == '__main__':
  tfds.testing.test_main()
