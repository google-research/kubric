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
# pylint: disable=line-too-long
import itertools
import json
import logging

import numpy as np
import png
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
from typing import List

_DESCRIPTION = """
A dataset of each ShapeNet object rendered from 25 random perspectives on transparent background.
Images are rendered at 512x512 and then cropped to fit the object, so they vary in size.

The dataset contains the following information:
- "image_id": str
- "asset_id": str
  The id of the ShapeNet object. E.g. "02691156/1021a0914a7207aff927ed529ad90a11".
  
- "label": tfds.features.ClassLabel
  One of the 55 Shapenet classes:
     ["airplane", "ashcan", "bag", "basket", "bathtub", "bed", "bench", "birdhouse",
      "bookshelf", "bottle", "bowl", "bus", "cabinet", "camera", "can", "cap", "car",
      "cellular telephone", "chair", "clock", "computer keyboard", "dishwasher",
      "display", "earphone", "faucet", "file", "guitar", "helmet", "jar", "knife",
      "lamp", "laptop", "loudspeaker", "mailbox", "microphone", "microwave",
      "motorcycle", "mug", "piano", "pillow", "pistol", "pot", "printer",
      "remote control", "rifle", "rocket", "skateboard", "sofa", "stove",
      "table", "telephone", "tower", "train", "vessel", "washer"]
- "camera_position": (3,) [float32]
     position of the camera in a half-sphere shell with inner radius 9 and outer radius 10. 
     The object sits at the origin.
- "image":  (None, None, 4) [uint8]
  The rendered image in RGBA format, cropped to fit the object. 
"""

_CITATION = """\
@inproceedings{greff2022kubric,
title = {Kubric: a scalable dataset generator}, 
    author = {Klaus Greff and Francois Belletti and Lucas Beyer and Carl Doersch and
              Yilun Du and Daniel Duckworth and David J Fleet and Dan Gnanapragasam and
              Florian Golemo and Charles Herrmann and Thomas Kipf and Abhijit Kundu and
              Dmitry Lagun and Issam Laradji and Hsueh-Ti (Derek) Liu and Henning Meyer and
              Yishu Miao and Derek Nowrouzezahrai and Cengiz Oztireli and Etienne Pot and
              Noha Radwan and Daniel Rebain and Sara Sabour and Mehdi S. M. Sajjadi and Matan Sela and
              Vincent Sitzmann and Austin Stone and Deqing Sun and Suhani Vora and Ziyu Wang and
              Tianhao Wu and Kwang Moo Yi and Fangcheng Zhong and Andrea Tagliasacchi},
    booktitle = {{IEEE} Conference on Computer Vision and Pattern Recognition, {CVPR}},
    year = {2022},
    publisher = {Computer Vision Foundation / {IEEE}},
}"""


class ShapenetPretraining(tfds.core.BeamBasedBuilder):
  """TFDS definition for ShapenetPretraining dataset"""

  VERSION = tfds.core.Version("1.0.0")
  RELEASE_NOTES = {
      "1.0.0": "initial release",
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""

    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            "image_id": tfds.features.Text(),
            "asset_id": tfds.features.Text(),
            "label": tfds.features.ClassLabel(names=[
                "airplane", "ashcan", "bag", "basket", "bathtub", "bed", "bench", "birdhouse",
                "bookshelf", "bottle", "bowl", "bus", "cabinet", "camera", "can", "cap", "car",
                "cellular telephone", "chair", "clock", "computer keyboard", "dishwasher",
                "display", "earphone", "faucet", "file", "guitar", "helmet", "jar", "knife",
                "lamp", "laptop", "loudspeaker", "mailbox", "microphone", "microwave",
                "motorcycle", "mug", "piano", "pillow", "pistol", "pot", "printer",
                "remote control", "rifle", "rocket", "skateboard", "sofa", "stove",
                "table", "telephone", "tower", "train", "vessel", "washer"]),
            "camera_position": tfds.features.Tensor(shape=(3,), dtype=tf.float32),
            "image":  tfds.features.Image(shape=(None, None, 4)),
        }),
        supervised_keys=("image", "label"),
        homepage="https://github.com/google-research/kubric",
        citation=_CITATION,
    )

  def _split_generators(self, unused_dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    del unused_dl_manager

    # find all available directories
    path = tfds.core.as_path("gs://research-brain-kubric-xgcp/jobs/shapenet_demo_7/")
    all_subdirs = list(path.glob("*"))  # if (d / "metadata.json").exists()]
    # figure out how many images per directory exist
    nr_images_per_dir = len(list(all_subdirs[0].glob("rgba_*.png")))
    logging.info("Found %d sub-folders with %d images each in master path: %s",
                 len(all_subdirs), nr_images_per_dir, path)

    # we pick one view for each object for validation and the others for train
    # views are random so we can just pick the first one for validation
    val_all_image_paths = [str(d / "rgba_00000.png") for d in all_subdirs]
    train_all_image_paths = [str(d / "rgba_{:05d}.png".format(i))
                             for d, i in itertools.product(all_subdirs,
                                                           range(1, nr_images_per_dir))]
    # directories are sorted by categories, so we shuffle
    np.random.shuffle(train_all_image_paths)

    logging.info("Using 1 image per object for validation for a total of %d images",
                 len(val_all_image_paths))
    logging.info("Using the other %d images for train", len(train_all_image_paths))

    return {
        tfds.Split.TRAIN: self._generate_examples(train_all_image_paths),
        tfds.Split.VALIDATION: self._generate_examples(val_all_image_paths),
    }

  def _generate_examples(self, directories: List[str]):
    """Yields examples."""

    def _process_example(image_path):
      image_path = tfds.core.as_path(image_path)
      image_dir = image_path.parent
      image_index = int(image_path.name[-9:-4])  # int("rgba_00008.png"[-9:-4]) -> 8
      key = f"{image_dir.name}_{image_index:05d}"

      with tf.io.gfile.GFile(str(image_dir / "metadata.json"), "r") as fp:
        metadata = json.load(fp)

      bbox = metadata["instances"][0]["bboxes"][image_index]
      y_min, x_min, y_max, x_max = [int(v*512) for v in bbox]
      img = read_png(image_path)

      return key, {
          "image_id": key,
          "asset_id": metadata["instances"][0]["asset_id"].replace("_", "/"),
          "label": metadata["instances"][0]["category"],
          "camera_position": np.array(metadata["camera"]["positions"][image_index], np.float32),
          "image":  img[y_min:y_max+1, x_min:x_max+1],
      }

    beam = tfds.core.lazy_imports.apache_beam
    return beam.Create(directories) | beam.Map(_process_example)


def read_png(filename) -> np.ndarray:
  filename = tfds.core.as_path(filename)
  png_reader = png.Reader(bytes=filename.read_bytes())
  width, height, pngdata, info = png_reader.read()
  del png_reader

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
