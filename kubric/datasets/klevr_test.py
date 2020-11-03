# TODO: write test case once API is consolidated
"""
import tempfile
import tensorflow_datasets as tfds
from kubric.datasets import Klevr

class KlevrTest(tfds.testing.DatasetBuilderTestCase):
  DATASET_CLASS = Klevr
  SPLITS = {
      "train": 24,  # Number of fake train example
      "test": 16,  # Number of fake test example
  }

  DL_EXTRACT_RESULT = ""
  SKIP_CHECKSUMS = True

  @classmethod
  def setUpClass(cls):
    cls.EXAMPLE_DIR = tempfile.mkdtemp()

  if __name__ == "__main__":
    tfds.testing.test_main()
"""
