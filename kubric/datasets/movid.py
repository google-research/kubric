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
# pylint: disable=line-too-long, unexpected-keyword-arg
"""TODO(klausg): description."""

import dataclasses
import logging
import os
import json
import imageio

import numpy as np
import png
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
from typing import List, Dict

_DESCRIPTION = "TODO(klausg)."

_CITATION = "TODO(klausg)."


@dataclasses.dataclass
class MovidConfig(tfds.core.BuilderConfig):
  """"Configuration for Multi-Object Video (MOVid) dataset."""
  height: int = 256
  width: int = 256
  validation_ratio: float = 0.1
  train_val_path: str = None
  test_split_paths: Dict[str, str] = dataclasses.field(default_factory=dict)
  shape_info: str = "None"  # also export shape, material, color, size (for A and B)


class Movid(tfds.core.BeamBasedBuilder):
  """DatasetBuilder for Katr dataset."""
  VERSION = tfds.core.Version("1.2.1")
  RELEASE_NOTES = {
    "1.0.0": "initial release",
    "1.1.0": "fixed segmentation, and various other minor issues",
    "1.2.0": "higher resolution, added object coordinates",
    "1.2.1": "more examples, remove UV",
  }

  BUILDER_CONFIGS = [
      MovidConfig(
          name="E_512x512",
          description="Static objects, moving camera, full resolution of 512x512",
          height=512,
          width=512,
          validation_ratio=0.02,
          train_val_path="gs://research-brain-kubric-xgcp/jobs/movid_e_v121",
          test_split_paths={
              # "test_held_out_objects": "gs://research-brain-kubric-xgcp/jobs/movid_e_v11_test_obj",
              # "test_held_out_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_e_v11_test_bg1",
              # "test_held_out_objects_and_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_e_v11_test_objbg",
              # "test_all_same": "gs://research-brain-kubric-xgcp/jobs/movid_e_v11_test_same2",
          }
      ),
      MovidConfig(
          name="E_256x256",
          description="Static objects, moving camera, full resolution of 256x256",
          height=256,
          width=256,
          validation_ratio=0.02,
          train_val_path="gs://research-brain-kubric-xgcp/jobs/movid_e_v121",
          test_split_paths={
              # "test_held_out_objects": "gs://research-brain-kubric-xgcp/jobs/movid_e_v11_test_obj",
              # "test_held_out_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_e_v11_test_bg1",
              # "test_held_out_objects_and_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_e_v11_test_objbg",
              # "test_all_same": "gs://research-brain-kubric-xgcp/jobs/movid_e_v11_test_same2",
          }
      ),
      MovidConfig(
          name="E_128x128",
          description="Static objects, moving camera, downscaled to  128x128",
          height=128,
          width=128,
          validation_ratio=0.02,
          train_val_path="gs://research-brain-kubric-xgcp/jobs/movid_e_v121",
          test_split_paths={
              # "test_held_out_objects": "gs://research-brain-kubric-xgcp/jobs/movid_e_v11_test_obj",
              # "test_held_out_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_e_v11_test_bg1",
              # "test_held_out_objects_and_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_e_v11_test_objbg",
              # "test_all_same": "gs://research-brain-kubric-xgcp/jobs/movid_e_v11_test_same2",
          }
      ),
      # MovidConfig(
      #     name="E_64x64",
      #     description="Static objects, moving camera, downscaled to 64x64",
      #     height=64,
      #     width=64,
      #     validation_ratio=0.1,
      #     train_val_path="gs://research-brain-kubric-xgcp/jobs/movid_e_v11",
      #     test_split_paths={
      #         "test_held_out_objects": "gs://research-brain-kubric-xgcp/jobs/movid_e_v11_test_obj",
      #         "test_held_out_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_e_v11_test_bg1",
      #         "test_held_out_objects_and_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_e_v11_test_objbg",
      #         "test_all_same": "gs://research-brain-kubric-xgcp/jobs/movid_e_v11_test_same2",
      #     }
      # ),
      # MovidConfig(
      #     name="D_256x256",
      #     description="Static objects, random camera, full resolution of 256x256",
      #     height=256,
      #     width=256,
      #     validation_ratio=0.1,
      #     train_val_path="gs://research-brain-kubric-xgcp/jobs/movid_d_v11",
      #     test_split_paths={
      #         "test_held_out_objects": "gs://research-brain-kubric-xgcp/jobs/movid_d_v11_testobj",
      #         "test_held_out_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_d_v11_testbg",
      #         "test_held_out_objects_and_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_d_v11_testobjbg",
      #         "test_all_same": "gs://research-brain-kubric-xgcp/jobs/movid_d_v11_testsame",
      #         "test_many": "gs://research-brain-kubric-xgcp/jobs/movid_d_v11_testmany",
      #     }
      # ),
      # MovidConfig(
      #     name="D_128x128",
      #     description="Static objects, random camera, downscaled to  128x128",
      #     height=128,
      #     width=128,
      #     validation_ratio=0.1,
      #     train_val_path="gs://research-brain-kubric-xgcp/jobs/movid_d_v11",
      #     test_split_paths={
      #         "test_held_out_objects": "gs://research-brain-kubric-xgcp/jobs/movid_d_v11_testobj",
      #         "test_held_out_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_d_v11_testbg",
      #         "test_held_out_objects_and_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_d_v11_testobjbg",
      #         "test_all_same": "gs://research-brain-kubric-xgcp/jobs/movid_d_v11_testsame",
      #         "test_many": "gs://research-brain-kubric-xgcp/jobs/movid_d_v11_testmany",
      #     }
      # ),
      # MovidConfig(
      #     name="D_64x64",
      #     description="Static objects, random camera, downscaled to 64x64",
      #     height=64,
      #     width=64,
      #     validation_ratio=0.1,
      #     train_val_path="gs://research-brain-kubric-xgcp/jobs/movid_d_v11",
      #     test_split_paths={
      #         "test_held_out_objects": "gs://research-brain-kubric-xgcp/jobs/movid_d_v11_testobj",
      #         "test_held_out_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_d_v11_testbg",
      #         "test_held_out_objects_and_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_d_v11_testobjbg",
      #         "test_all_same": "gs://research-brain-kubric-xgcp/jobs/movid_d_v11_testsame",
      #         "test_many": "gs://research-brain-kubric-xgcp/jobs/movid_d_v11_testmany",
      #     }
      # ),
      # MovidConfig(
      #     name="C_256x256",
      #     description="Dynamic objects, random camera, full resolution of 256x256",
      #     height=256,
      #     width=256,
      #     validation_ratio=0.1,
      #     train_val_path="gs://research-brain-kubric-xgcp/jobs/movid_c_v11",
      #     test_split_paths={
      #         "test_held_out_objects": "gs://research-brain-kubric-xgcp/jobs/movid_c_v11_testobj",
      #         "test_held_out_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_c_v11_testbg",
      #         "test_held_out_objects_and_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_c_v11_testobjbg",
      #         "test_all_same": "gs://research-brain-kubric-xgcp/jobs/movid_c_v11_testsame",
      #         "test_many": "gs://research-brain-kubric-xgcp/jobs/movid_c_v11_testmany",
      #     }
      # ),
      # MovidConfig(
      #     name="C_128x128",
      #     description="Dynamic objects, random camera, downscaled to  128x128",
      #     height=128,
      #     width=128,
      #     validation_ratio=0.1,
      #     train_val_path="gs://research-brain-kubric-xgcp/jobs/movid_c_v11",
      #     test_split_paths={
      #         "test_held_out_objects": "gs://research-brain-kubric-xgcp/jobs/movid_c_v11_testobj",
      #         "test_held_out_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_c_v11_testbg",
      #         "test_held_out_objects_and_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_c_v11_testobjbg",
      #         "test_all_same": "gs://research-brain-kubric-xgcp/jobs/movid_c_v11_testsame",
      #         "test_many": "gs://research-brain-kubric-xgcp/jobs/movid_c_v11_testmany",
      #     }
      # ),
      # MovidConfig(
      #     name="C_64x64",
      #     description="Static objects, random camera, downscaled to 64x64",
      #     height=64,
      #     width=64,
      #     validation_ratio=0.1,
      #     train_val_path="gs://research-brain-kubric-xgcp/jobs/movid_c_v11",
      #     test_split_paths={
      #         "test_held_out_objects": "gs://research-brain-kubric-xgcp/jobs/movid_c_v11_testobj",
      #         "test_held_out_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_c_v11_testbg",
      #         "test_held_out_objects_and_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_c_v11_testobjbg",
      #         "test_all_same": "gs://research-brain-kubric-xgcp/jobs/movid_c_v11_testsame",
      #         "test_many": "gs://research-brain-kubric-xgcp/jobs/movid_c_v11_testmany",
      #     }
      # ),
      # MovidConfig(
      #     name="CC_256x256",
      #     description="Dynamic objects, moving camera, full resolution of 256x256",
      #     height=256,
      #     width=256,
      #     validation_ratio=0.1,
      #     train_val_path="gs://research-brain-kubric-xgcp/jobs/movid_cc_v11",
      #     test_split_paths={
      #         # "test_held_out_objects": "gs://research-brain-kubric-xgcp/jobs/movid_cc_v11_testobj",
      #         # "test_held_out_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_c_v11_testbg",
      #         # "test_held_out_objects_and_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_c_v11_testobjbg",
      #         # "test_all_same": "gs://research-brain-kubric-xgcp/jobs/movid_c_v11_testsame",
      #         # "test_many": "gs://research-brain-kubric-xgcp/jobs/movid_c_v11_testmany",
      #     }
      # ),
      # MovidConfig(
      #     name="CC_128x128",
      #     description="Dynamic objects, moving camera, downscaled to  128x128",
      #     height=128,
      #     width=128,
      #     validation_ratio=0.1,
      #     train_val_path="gs://research-brain-kubric-xgcp/jobs/movid_cc_v11",
      #     test_split_paths={
      #         # "test_held_out_objects": "gs://research-brain-kubric-xgcp/jobs/movid_c_v11_testobj",
      #         # "test_held_out_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_c_v11_testbg",
      #         # "test_held_out_objects_and_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_c_v11_testobjbg",
      #         # "test_all_same": "gs://research-brain-kubric-xgcp/jobs/movid_c_v11_testsame",
      #         # "test_many": "gs://research-brain-kubric-xgcp/jobs/movid_c_v11_testmany",
      #     }
      # ),
      # MovidConfig(
      #     name="CC_64x64",
      #     description="Static objects, moving camera, downscaled to 64x64",
      #     height=64,
      #     width=64,
      #     validation_ratio=0.1,
      #     train_val_path="gs://research-brain-kubric-xgcp/jobs/movid_cc_v11",
      #     test_split_paths={
      #         # "test_held_out_objects": "gs://research-brain-kubric-xgcp/jobs/movid_c_v11_testobj",
      #         # "test_held_out_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_c_v11_testbg",
      #         # "test_held_out_objects_and_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_c_v11_testobjbg",
      #         # "test_all_same": "gs://research-brain-kubric-xgcp/jobs/movid_c_v11_testsame",
      #         # "test_many": "gs://research-brain-kubric-xgcp/jobs/movid_c_v11_testmany",
      #     }
      # ),
      #
      # MovidConfig(
      #     name="B_256x256",
      #     description="Random color background, Kubasic objects, random camera, full resolution of 256x256",
      #     height=256,
      #     width=256,
      #     validation_ratio=0.1,
      #     train_val_path="gs://research-brain-kubric-xgcp/jobs/movid_b_v11",
      #     shape_info="KuBasic",
      #     test_split_paths={
      #         "test_held_out_objects": "gs://research-brain-kubric-xgcp/jobs/movid_b_v11_testobj",
      #         "test_held_out_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_b_v11_testbg",
      #         "test_held_out_objects_and_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_b_v11_testobjbg",
      #         "test_all_same": "gs://research-brain-kubric-xgcp/jobs/movid_b_v11_testsame",
      #         "test_many": "gs://research-brain-kubric-xgcp/jobs/movid_b_v11_testmany",
      #     }
      # ),
      # MovidConfig(
      #     name="B_128x128",
      #     description="Random color background, Kubasic objects, random camera, downscaled to  128x128",
      #     height=128,
      #     width=128,
      #     validation_ratio=0.1,
      #     train_val_path="gs://research-brain-kubric-xgcp/jobs/movid_b_v11",
      #     shape_info="KuBasic",
      #     test_split_paths={
      #         "test_held_out_objects": "gs://research-brain-kubric-xgcp/jobs/movid_b_v11_testobj",
      #         "test_held_out_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_b_v11_testbg",
      #         "test_held_out_objects_and_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_b_v11_testobjbg",
      #         "test_all_same": "gs://research-brain-kubric-xgcp/jobs/movid_b_v11_testsame",
      #         "test_many": "gs://research-brain-kubric-xgcp/jobs/movid_b_v11_testmany",
      #     }
      # ),
      # MovidConfig(
      #     name="B_64x64",
      #     description="Random color background, Kubasic objects, random camera, downscaled to 64x64",
      #     height=64,
      #     width=64,
      #     validation_ratio=0.1,
      #     train_val_path="gs://research-brain-kubric-xgcp/jobs/movid_b_v11",
      #     shape_info="KuBasic",
      #     test_split_paths={
      #         "test_held_out_objects": "gs://research-brain-kubric-xgcp/jobs/movid_b_v11_testobj",
      #         "test_held_out_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_b_v11_testbg",
      #         "test_held_out_objects_and_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_b_v11_testobjbg",
      #         "test_all_same": "gs://research-brain-kubric-xgcp/jobs/movid_b_v11_testsame",
      #         "test_many": "gs://research-brain-kubric-xgcp/jobs/movid_b_v11_testmany",
      #     }
      # ),
      # MovidConfig(
      #     name="A_256x256",
      #     description="CLEVR setup, full resolution of 256x256",
      #     height=256,
      #     width=256,
      #     validation_ratio=0.1,
      #     train_val_path="gs://research-brain-kubric-xgcp/jobs/movid_a_v11",
      #     shape_info="CLEVR",
      #     test_split_paths={
      #         "test_held_out_objects": "gs://research-brain-kubric-xgcp/jobs/movid_a_v11_test_obj",
      #         "test_held_out_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_a_v11_test_bg",
      #         "test_held_out_objects_and_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_a_v11_test_objbg",
      #         "test_all_same": "gs://research-brain-kubric-xgcp/jobs/movid_a_v11_test_same",
      #         "test_many": "gs://research-brain-kubric-xgcp/jobs/movid_a_v11_test_many",
      #     }
      # ),
      # MovidConfig(
      #     name="A_128x128",
      #     description="CLEVR setup, downscaled to  128x128",
      #     height=128,
      #     width=128,
      #     validation_ratio=0.1,
      #     train_val_path="gs://research-brain-kubric-xgcp/jobs/movid_a_v11",
      #     shape_info="CLEVR",
      #     test_split_paths={
      #         "test_held_out_objects": "gs://research-brain-kubric-xgcp/jobs/movid_a_v11_test_obj",
      #         "test_held_out_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_a_v11_test_bg",
      #         "test_held_out_objects_and_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_a_v11_test_objbg",
      #         "test_all_same": "gs://research-brain-kubric-xgcp/jobs/movid_a_v11_test_same",
      #         "test_many": "gs://research-brain-kubric-xgcp/jobs/movid_a_v11_test_many",
      #     }
      # ),
      # MovidConfig(
      #     name="A_64x64",
      #     description="CLEVR setup, downscaled to 64x64",
      #     height=64,
      #     width=64,
      #     validation_ratio=0.1,
      #     train_val_path="gs://research-brain-kubric-xgcp/jobs/movid_a_v11",
      #     shape_info="CLEVR",
      #     test_split_paths={
      #         "test_held_out_objects": "gs://research-brain-kubric-xgcp/jobs/movid_a_v11_test_obj",
      #         "test_held_out_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_a_v11_test_bg",
      #         "test_held_out_objects_and_backgrounds": "gs://research-brain-kubric-xgcp/jobs/movid_a_v11_test_objbg",
      #         "test_all_same": "gs://research-brain-kubric-xgcp/jobs/movid_a_v11_test_same",
      #         "test_many": "gs://research-brain-kubric-xgcp/jobs/movid_a_v11_test_many",
      #     }
      # ),

  ]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""

    h = self.builder_config.height
    w = self.builder_config.width
    s = 24

    instance_features = {
        "asset_id": tfds.features.Text(),
        "is_dynamic": tfds.features.Tensor(shape=(), dtype=tf.bool),
        "mass": tf.float32,
        "friction": tf.float32,
        "restitution": tf.float32,

        "positions": tfds.features.Tensor(shape=(s, 3), dtype=tf.float32),
        "quaternions": tfds.features.Tensor(shape=(s, 4), dtype=tf.float32),
        "velocities": tfds.features.Tensor(shape=(s, 3), dtype=tf.float32),
        "angular_velocities": tfds.features.Tensor(shape=(s, 3), dtype=tf.float32),
        "bboxes_3d": tfds.features.Tensor(shape=(s, 8, 3), dtype=tf.float32),

        "image_positions": tfds.features.Tensor(shape=(s, 2), dtype=tf.float32),
        "bboxes": tfds.features.Sequence(
            tfds.features.BBoxFeature()),
        "bbox_frames": tfds.features.Sequence(
            tfds.features.Tensor(shape=(), dtype=tf.int32)),
        "visibility": tfds.features.Tensor(shape=(s,), dtype=tf.uint16),
    }

    if self.builder_config.shape_info == "CLEVR":
      instance_features["shape_label"] = tfds.features.ClassLabel(names=["Cube", "Cylinder", "Sphere"])
      instance_features["size_label"] = tfds.features.ClassLabel(names=["small", "large"])
      instance_features["size"] = tfds.features.Tensor(shape=(3,), dtype=tf.float32)
      instance_features["material_label"] = tfds.features.ClassLabel(names=["Metal", "Rubber"])
      instance_features["color_label"] = tfds.features.ClassLabel(names=[
          "blue", "brown", "cyan", "gray", "green", "purple", "red", "yellow"])
      instance_features["color"] = tfds.features.Tensor(shape=(3,), dtype=tf.float32)

    elif self.builder_config.shape_info == "KuBasic":
      instance_features["shape_label"] = tfds.features.ClassLabel(names=[
          "Cube", "Cylinder", "Sphere", "Cone", "Torus", "Gear", "TorusKnot", "Sponge", "Spot",
          "Teapot", "Suzanne"])
      instance_features["size"] = tf.float32
      instance_features["material_label"] = tfds.features.ClassLabel(names=["Metal", "Rubber"])
      instance_features["color"] = tfds.features.Tensor(shape=(3,), dtype=tf.float32)

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
                "forward_flow_range": tfds.features.Tensor(shape=(2,), dtype=tf.float32),
                "backward_flow_range": tfds.features.Tensor(shape=(2,), dtype=tf.float32),
            },
            "background": tfds.features.Text(),
            "instances": tfds.features.Sequence(feature=instance_features),
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
            "forward_flow": tfds.features.Sequence(
                tfds.features.Tensor(shape=(h, w, 2), dtype=tf.uint16), length=s),
            "backward_flow": tfds.features.Sequence(
                tfds.features.Tensor(shape=(h, w, 2), dtype=tf.uint16), length=s),
            "depth": tfds.features.Sequence(
                tfds.features.Image(shape=(h, w, 1), dtype=tf.uint16), length=s),
            "normal": tfds.features.Sequence(
                tfds.features.Image(shape=(h, w, 3), dtype=tf.uint16), length=s),
            "object_coordinates": tfds.features.Sequence(
                tfds.features.Image(shape=(h, w, 3), dtype=tf.uint16), length=s),
        }),
        supervised_keys=None,
        homepage="https://github.com/google-research/kubric",
        citation=_CITATION)

  def _split_generators(self, unused_dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    del unused_dl_manager
    path = tfds.core.as_path(self.builder_config.train_val_path)
    all_subdirs = [d for d in path.iterdir()]  # if (d / "events.json").exists()]
    all_subdirs = sorted(all_subdirs, key=lambda x: int(x.name))
    all_subdirs = [str(d) for d in all_subdirs]
    logging.info("Found {subfolder} sub-folders in master path: {path}",
        subfolders=len(all_subdirs), path=path)

    validation_ratio = self.builder_config.validation_ratio
    validation_examples = round(len(all_subdirs) * validation_ratio)
    training_examples = len(all_subdirs) - validation_examples
    logging.info("Using {ratio} of examples for validation for a total of {examples}}",
        ratio=validation_ratio, examples=validation_examples)
    logging.info("Using the other {examples} examples for training",
        examples=training_examples)

    splits = {
        tfds.Split.TRAIN: self._generate_examples(all_subdirs[:training_examples]),
        tfds.Split.VALIDATION: self._generate_examples(all_subdirs[training_examples:]),
    }

    for key, path in self.builder_config.test_split_paths.items():
      path = tfds.core.as_path(path)
      split_dirs = [d for d in path.iterdir()]  # if (d / "events.json").exists()]
      # sort the directories by their integer number
      split_dirs = sorted(split_dirs, key=lambda x: int(x.name))
      logging.info("Found %d sub-folders in '%s' path: %s", len(split_dirs), key, path)
      splits[key] = self._generate_examples([str(d) for d in split_dirs])

    return splits

  def _generate_examples(self, directories: List[str]):
    """Yields examples."""

    target_size = (self.builder_config.height, self.builder_config.width)

    def _is_complete_dir(video_dir):
      video_dir = tfds.core.as_path(video_dir)
      filenames = [d.name for d in video_dir.iterdir()]
      if not ("data_ranges.json" in filenames and
              "metadata.json" in filenames and
              "events.json" in filenames):
        return False
      nr_frames_per_category = {
          key: len([fn for fn in filenames if fn.startswith(key)])
          for key in ["rgba", "depth", "segmentation", "forward_flow",
                      "backward_flow", "normal", "object_coordinates"]}

      nr_expected_frames = nr_frames_per_category["rgba"]
      if nr_expected_frames == 0:
        return False
      if not all(nr_frames == nr_expected_frames for nr_frames in nr_frames_per_category.values()):
        return False

      return True

    def _process_example(video_dir):
      video_dir = tfds.core.as_path(video_dir)
      key = f"{video_dir.name}"

      with tf.io.gfile.GFile(str(video_dir / "data_ranges.json"), "r") as fp:
        data_ranges = json.load(fp)

      with tf.io.gfile.GFile(str(video_dir / "metadata.json"), "r") as fp:
        metadata = json.load(fp)

      with tf.io.gfile.GFile(str(video_dir / "events.json"), "r") as fp:
        events = json.load(fp)

      num_frames = metadata["metadata"]["num_frames"]
      num_instances = metadata["metadata"]["num_instances"]

      assert len(metadata["instances"]) == num_instances, f"{len(metadata['instances'])} != {num_instances}"

      # assert "depth" in data_ranges, f"ERROR {key}\t{video_dir}\t{data_ranges}"
      assert "forward_flow" in data_ranges, f"ERROR {key}\t{video_dir}\t{data_ranges}"
      assert "backward_flow" in data_ranges, f"ERROR {key}\t{video_dir}\t{data_ranges}"
      # depth_min, depth_max = data_ranges["depth"]["min"], data_ranges["depth"]["max"]

      rgba_frame_paths = [video_dir / f"rgba_{f:05d}.png" for f in range(num_frames)]
      segmentation_frame_paths = [video_dir / f"segmentation_{f:05d}.png" for f in range(num_frames)]
      fwd_flow_frame_paths = [video_dir / f"forward_flow_{f:05d}.png" for f in range(num_frames)]
      bwd_flow_frame_paths = [video_dir / f"backward_flow_{f:05d}.png" for f in range(num_frames)]
      depth_frame_paths = [video_dir / f"depth_{f:05d}.tiff" for f in range(num_frames)]
      normal_frame_paths = [video_dir / f"normal_{f:05d}.png" for f in range(num_frames)]
      object_coordinates_frame_paths = [video_dir / f"object_coordinates_{f:05d}.png" for f in range(num_frames)]

      scale = 512 / target_size[0]

      depth_frames = np.array([subsample_nearest_neighbor(read_tiff(frame_path), target_size)
                               for frame_path in depth_frame_paths])
      depth_min, depth_max = np.min(depth_frames), np.max(depth_frames)
      depth_frames_uint16 = convert_float_to_uint16(depth_frames, depth_min, depth_max)

      def get_instance_info(obj):
        instance_info = {
            "asset_id": obj["asset_id"],
            "is_dynamic": bool(obj["is_dynamic"]),
            "mass": obj["mass"],
            "friction": obj["friction"],
            "restitution": obj["restitution"],
            "positions": np.array(obj["positions"], np.float32),
            "quaternions": np.array(obj["quaternions"], np.float32),
            "velocities": np.array(obj["velocities"], np.float32),
            "angular_velocities": np.array(obj["angular_velocities"], np.float32),
            "bboxes_3d": np.array(obj["bboxes_3d"], np.float32),
            "image_positions": np.array(obj["image_positions"], np.float32),
            "bboxes": [tfds.features.BBox(*bbox) for bbox in obj["bboxes"]],
            "bbox_frames": np.array(obj["bbox_frames"], dtype=np.uint16),
            "visibility": np.array(obj["visibility"], dtype=np.uint16),
        }
        if self.builder_config.shape_info == "CLEVR":
          instance_info["shape_label"] = obj["shape_label"]
          instance_info["size_label"] = obj["size_label"]
          instance_info["size"] = obj["size"]
          instance_info["material_label"] = obj["material"]
          instance_info["color_label"] = obj["color_label"]
          instance_info["color"] = obj["color"]

        elif self.builder_config.shape_info == "KuBasic":
          instance_info["shape_label"] = obj["shape_label"]
          instance_info["size"] = obj["size"]
          instance_info["material_label"] = obj["material"]
          instance_info["color"] = obj["color"]
        return instance_info

      return key, {
          "metadata": {
              "video_name": os.fspath(key),
              "width": target_size[1],
              "height": target_size[0],
              "num_frames": num_frames,
              "num_instances": num_instances,
              "depth_range": [depth_min, depth_max],
              "forward_flow_range": [data_ranges["forward_flow"]["min"] / scale * 512,
                                     data_ranges["forward_flow"]["max"] / scale * 512],
              "backward_flow_range": [data_ranges["backward_flow"]["min"] / scale * 512,
                                      data_ranges["backward_flow"]["max"] / scale * 512],
          },
          "background": metadata["metadata"]["background"],
          "instances": [get_instance_info(obj) for obj in metadata["instances"]],
          "camera": {
              "focal_length": metadata["camera"]["focal_length"],
              "sensor_width": metadata["camera"]["sensor_width"],
              "field_of_view": metadata["camera"]["field_of_view"],
              "positions": np.array(metadata["camera"]["positions"], np.float32),
              "quaternions": np.array(metadata["camera"]["quaternions"], np.float32),
          },
          "events": {
              "collisions": [{
                  "instances": np.array(c["instances"], dtype=np.uint16),
                  "frame": c["frame"],
                  "force": c["force"],
                  "position": np.array(c["position"], dtype=np.float32),
                  "image_position": np.array(c["image_position"], dtype=np.float32),
                  "contact_normal": np.array(c["contact_normal"], dtype=np.float32),
              } for c in events["collisions"]],
          },
          "video": [subsample_avg(read_png(frame_path), target_size)[..., :3]
                    for frame_path in rgba_frame_paths],
          "segmentations": [subsample_nearest_neighbor(read_png(frame_path).astype(np.uint16),
                                                       target_size)
                            for frame_path in segmentation_frame_paths],
          "forward_flow": [subsample_nearest_neighbor(read_png(frame_path)[..., :2], target_size)
                           for frame_path in fwd_flow_frame_paths],
          "backward_flow": [subsample_nearest_neighbor(read_png(frame_path)[..., :2], target_size)
                            for frame_path in bwd_flow_frame_paths],
          "depth": depth_frames_uint16,
          "normal": [subsample_nearest_neighbor(read_png(frame_path), target_size)
                     for frame_path in normal_frame_paths],
          "object_coordinates":  [subsample_nearest_neighbor(read_png(frame_path), target_size)
                                  for frame_path in object_coordinates_frame_paths],
      }

    beam = tfds.core.lazy_imports.apache_beam
    return beam.Create(directories) | beam.Filter(_is_complete_dir) | beam.Map(_process_example)


def subsample_nearest_neighbor(arr, size):
  src_height, src_width, _ = arr.shape
  dst_height, dst_width = size
  height_step = src_height // dst_height
  width_step = src_width // dst_width
  height_offset = int(np.floor((height_step-1)/2))
  width_offset = int(np.floor((width_step-1)/2))
  subsampled = arr[height_offset::height_step, width_offset::width_step, :]
  return subsampled


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
  png_reader = png.Reader(bytes=path.read_bytes())
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


def read_tiff(path: os.PathLike):
  img_bytes = tfds.core.as_path(path).read_bytes()
  return imageio.imread(img_bytes, format="tif")[:, :, None]


def convert_float_to_uint16(array, min_val, max_val):
  return np.round((array - min_val) / (max_val - min_val) * 65535).astype(np.uint16)
