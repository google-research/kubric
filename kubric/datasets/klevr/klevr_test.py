"""Tests for the klevr dataset."""

import tensorflow as tf
from kubric.datasets.klevr import klevr
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
    klevr.Klevr.BUILDER_CONFIGS = [klevr.KlevrConfig(name='test',
                                                     description='Dummy test.',
                                                     path=cls.dummy_data,
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