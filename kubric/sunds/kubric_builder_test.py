# Copyright 2022 The Kubric Authors.
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
"""Tests for tfds_worker."""

from __future__ import annotations

import tempfile

import kubric as kb
import kubric.renderer.blender  # pylint: disable=g-import-not-at-top,unused-import
import kubric.sunds  # pylint: disable=g-import-not-at-top,unused-import
import tensorflow_datasets as tfds


class HelloBeamWorker(kb.sunds.KubricBuilder):
  VERSION = tfds.core.Version("1.0.0")
  RELEASE_NOTES = {
      "1.0.0": "Initial version",
  }
  SCENE_CONFIG = kb.sunds.SceneConfig()

  def _info(self) -> tfds.core.DatasetInfo:
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({
            "rgb":
              tfds.features.Image(shape=(*self.SCENE_CONFIG.resolution, 3)),
        }),
    )

  def split_to_scene_configs(self) -> dict[str, list[kb.sunds.SceneConfig]]:
    return {
        "train": [self.SCENE_CONFIG.replace(seed=i) for i in range(10)],
    }

  def generate_scene(self, config: kb.sunds.SceneConfig):
    scene = config.as_scene()
    renderer = kb.renderer.blender.Blender(scene, config.scratch_dir)
    scene += kb.Cube(name="floor", scale=(10, 10, 0.1), position=(0, 0, -0.1))
    scene += kb.Sphere(name="ball", scale=1, position=(0, 0, 1.))
    scene += kb.DirectionalLight(
        name="sun", position=(-1, -0.5, 3), look_at=(0, 0, 0), intensity=1.5)
    scene += kb.PerspectiveCamera(
        name="camera", position=(3, -1, 4), look_at=(0, 0, 1))

    render_layers = renderer.render_still()
    return {"rgb": render_layers["rgba"][:, :, :3]}


class KubricBuilderTest(tfds.testing.DatasetBuilderTestCase):
  DATASET_CLASS = HelloBeamWorker
  EXAMPLE_DIR = tempfile.mkdtemp()  # No dummy data, yet still required
  SPLITS = {
      # Number of fake train example (10 scene * 1 example per scene)
      "train": 10,
  }

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    # Mock Kubric inside the test.
    cm = kb.sunds.mock_render(num_frames=1)
    cm.__enter__()
    # IMPORTANT: If the contextmanager is not bound. The Python never execute
    # it. So we have to set `cls._kubric_render_cm = cm` even if it is never
    # used.
    cls._kubric_render_cm = cm


if __name__ == "__main__":
  tfds.testing.test_main()