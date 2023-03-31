# Copyright 2023 The Kubric Authors.
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
"""Kubric worker."""

from __future__ import annotations

import abc
import collections
import dataclasses
import itertools
import tempfile
from typing import Any, Dict, Iterator, List, Optional, TypeVar, Union

import apache_beam as beam
from etils import edc
from etils import epath
import kubric as kb
import tensorflow_datasets as tfds

_T = TypeVar('_T')

# Mapping <split-name> -> _T
SplitDict = Dict[str, _T]

ExDict = Dict[str, Any]
Exs = Iterator[ExDict]

# Example can be a single example (e.g. video) or an iterator of exs
SceneExs = Union[Exs, ExDict]
# User can return examples directly, or a mapping split-name -> examples
SceneOutput = Union[SplitDict[SceneExs], SceneExs]

# Normalized scene outputs
SplitToSceneExs = SplitDict[List[ExDict]]


@edc.dataclass(kw_only=True)
@dataclasses.dataclass(frozen=True)
class SceneConfig:
  """Parameters of a single scene."""
  seed: int = 42
  resolution: tuple[int, int] = (256, 256)
  frame_start: int = 1
  frame_end: int = 24
  frame_rate: int = 24
  step_rate: int = 240

  scratch_dir: Optional[epath.Path] = edc.field(
      validate=lambda p: p if p is None else epath.Path(p),
      default=None,  # pytype: disable=annotation-type-mismatch
  )

  def replace(self, **kwargs: Any) -> SceneConfig:
    return dataclasses.replace(self, **kwargs)

  def as_scene(self, **kwargs: Any) -> kb.Scene:
    """Create the `kb.Scene` associated with the config."""
    # Would likely be best to have a `scene = kb.Scene.from_config(config)`
    # instead

    return kb.Scene(
        frame_start=self.frame_start,
        frame_end=self.frame_end,
        frame_rate=self.frame_rate,
        step_rate=self.step_rate,
        resolution=self.resolution,
        **kwargs,
    )


class KubricBuilder(tfds.core.GeneratorBasedBuilder):
  """Base TFDS builder.

  The functions to overwrite are:

  * `_info`: Defines dataset features and metadata (see TFDS docs:
    https://www.tensorflow.org/datasets/add_dataset#_info_dataset_metadata)
  * `split_to_scene_configs`: Define the splits and scenes
  * `generate_scene`: Generate a single scene and returns the examples.

  Usage: See `HelloBeamWorker` for an example.

  """

  def _split_generators(
      self,
      dl_manager: tfds.download.DownloadManager,
      pipeline: beam.Pipeline,
  ) -> SplitDict[beam.PCollection[tuple[tfds.typing.Key, ExDict]]]:
    """Returns the split PCollection.

    The pipeline is as follow:

    1. Collect all scenes to generate (`split_to_scene_configs`)
    2. Generate each scene in parallel (`_generate_single_scene`)
    3. For each scene, flatten the examples to be distributed across splits (
       `_flatten_split_examples`)
    4. Group examples per split

    Args:
      dl_manager: Unused
      pipeline: Root beam pipeline

    Returns:
      The dict mapping split names to PCollection of `(key, examples)`
    """
    del dl_manager

    # Step 1.
    split_to_scenes = SplitScenesMapping(self.split_to_scene_configs())

    data_per_split = (
        pipeline
        | 'Creating scenes' >> beam.Create(
        split_to_scenes.scene_id_to_scene_config.items())
        # tuple[<scene_id>, SceneConfig]
        # Step 2.
        | 'Generate scenes' >> beam.Map(
        self._generate_single_scene,
        split_to_scenes=split_to_scenes,
    )
        # tuple[<key>, SplitToSceneExs]
        # Step 3.
        | 'Select split' >> beam.FlatMap(self._flatten_split_examples)
        # tuple[<split_name>, <key>, ExDict]
        # Step 4.
        | 'Partition by split' >> beam.Partition(
        lambda ex, _: split_to_scenes.split_name_to_split_index[ex[0]],  # pylint: disable=too-many-function-args
        len(split_to_scenes.split_name_to_split_index),
    )
        # Each split is a separate PCollection
        # tuple[<split_name>, <key>, ExDict]
    )

    split_to_exs = dict(zip(split_to_scenes.split_names, data_per_split))
    # Remove the split name:
    # tuple[<split_name>, <key>, ExDict] -> tuple[<key>, ExDict]
    split_to_exs = {
        split_name:
          examples | f'Remove {split_name}' >> beam.Map(lambda ex: ex[1:])
        for split_name, examples in split_to_exs.items()
    }

    # Returns the mapping <split_name> -> <PCollection>
    return split_to_exs

  def _generate_single_scene(
      self,
      id_and_config: tuple[int, SceneConfig],
      *,
      split_to_scenes: SplitScenesMapping,
  ) -> tuple[int, SplitToSceneExs]:
    """Generate a single scene."""
    scene_id, scene_config = id_and_config

    with tempfile.TemporaryDirectory() as tmp_dir:
      scene_config = scene_config.replace(scratch_dir=tmp_dir)

      scene_output = self.generate_scene(scene_config)
      scene_output = self._normalize_scene_output(
          scene_output=scene_output,
          scene_id=scene_id,
          split_to_scenes=split_to_scenes,
      )
      return scene_id, scene_output

  def _normalize_scene_output(
      self,
      *,
      scene_id: int,
      scene_output: SceneOutput,
      split_to_scenes: SplitScenesMapping,
  ) -> SplitToSceneExs:
    """Validate and normalize the scene outputs."""
    expected_splits = split_to_scenes.scene_id_to_split_names[scene_id]

    # 2 cases:
    # * User returned `SceneExs` directly
    # * User returned `SplitDict[SceneExs]` mapping split to examples

    if (isinstance(scene_output, dict) and
        set(scene_output) == set(expected_splits)):
      # If the user did specified the wrong splits, the error will be raised
      # in `_assert_single_split` bellow.
      # If the user specify a single wrong split, the error will be raised
      # during encoding.
      return {
          split_name: self._normalize_scene_examples(exs)
          for split_name, exs in scene_output.items()
      }
    else:
      _assert_single_split(expected_splits)
      split, = expected_splits
      return {split: self._normalize_scene_examples(scene_output)}

  def _normalize_scene_examples(self, exs: SceneExs) -> list[ExDict]:
    """Validate and normalize the examples."""
    # 2 cases:
    if isinstance(exs, dict):  # Example is singleton (e.g. video)
      return [exs]
    else:  # Example is an iterable of examples
      return list(exs)  # pytype: disable=bad-return-type

  def _flatten_split_examples(
      self,
      id_and_split_to_exs: tuple[int, SplitToSceneExs],
  ) -> Iterator[tuple[str, tfds.typing.Key, ExDict]]:
    """Flatten each scene examples."""
    scene_id, split_to_exs = id_and_split_to_exs

    for split_name, scene_examples in split_to_exs.items():
      for i, ex in enumerate(scene_examples):
        yield split_name, f'{scene_id}_{i}', ex

  def _generate_examples(self):
    raise AssertionError('Should not be called')

  @abc.abstractmethod
  def split_to_scene_configs(self) -> dict[str, list[SceneConfig]]:
    """Returns the scene to generate per split.

    Example:

    ```python
    def split_to_scene_configs(self):

      # Train split has 2 scenes, test split had 1 scene
      scene0 = kb.sunds.SceneConfig()
      scene1 = kb.sunds.SceneConfig()
      scene2 = kb.sunds.SceneConfig()

      # The config passed here will be generated in `generate_scene`
      return {
         'train': [scene0, scene1],
         'test': [scene2],
      }
    ```

    In the above example, the train and test scenes are completely separate.
    If you want to use the same scene in both train and test, you can
    pass the same instance in both train/test.

    ```
    def split_to_scene_configs(self):
      # Use a single scene, shared between train and test
      scene0 = kb.sunds.SceneConfig()
      return {
         'train': [scene0],
         'test': [scene0],
      }
    ```

    Returns:
      Dict mapping splits to scenes
    """
    raise NotImplementedError

  @abc.abstractmethod
  def generate_scene(
      self,
      scene_config: SceneConfig,
  ) -> SceneOutput:
    """Generate a single scene.

    This is the function which creates and render the scene.

    ```
    def generate_scene(self, scene_config: kb.sunds.SceneConfig):
      scene = kb.Scene.from_config(scene_config)
      ...
      render_outputs = renderer.render()

      # Returns the list[ex0, ex1, ...]
      return etree.unzip(render_outputs)
    ```

    Example(s) can be:

    * Singleton: 1 scene == 1 example (e.g., video, conditional nerf,...)
    * Iterable of examples. (e.g. each example is a single frame)

    Returns value can be:

    * The scene example(s): If scene is only in one split
    * A `dict` of `{'train': scene example(s)}`

    All the following are valid, assuming `ex` is a valid example dict (.e.g
    `{'rgb': ..., 'segmentation': ...}`):

    ```python
    return ex
    return [ex0, ex1, ...]
    return {'train': ex0, 'test': ex1}
    return {'train': [ex0, ex1,...], 'test': [...]}
    ```

    Args:
      scene_config: Config of the scene to generate.

    Returns:
      scene_data: All examples from the scene.
    """
    raise NotImplementedError


class SplitScenesMapping:
  """Utils to query split/scenes info."""

  def __init__(self, split_to_scene_configs: SplitDict[list[SceneConfig]]):
    self.split_to_scene_configs = split_to_scene_configs

    # Ideally, we could use `@functools.cached_property` directly, however
    # because of lazy-computation, `id()` executed on the workers processes
    # might not match `id()` computed in master thread.
    self.split_names = self._split_names()
    self.split_name_to_split_index = self._split_name_to_split_index()
    self.all_scene_configs = self._all_scene_configs()
    self.scene_id_to_scene_config = self._scene_id_to_scene_config()
    self.split_name_to_scene_ids = self._split_name_to_scene_ids()
    self.scene_id_to_split_names = self._scene_id_to_split_names()

  def _split_names(self) -> list[str]:
    """Returns the list of split names `['train', 'test', ...]`."""
    return list(self.split_to_scene_configs.keys())

  def _split_name_to_split_index(self) -> dict[str, int]:
    """Returns the mapping split->index `{'train': 0, 'test': 1}`."""
    return {name: index for index, name in enumerate(self.split_names)}

  def _all_scene_configs(self) -> list[SceneConfig]:
    """Returns the flatten list of all scene configs."""
    return list(itertools.chain(*self.split_to_scene_configs.values()))

  def _scene_id_to_scene_config(self) -> dict[int, SceneConfig]:
    """Returns the mapping id(scene_config) -> scene_config."""
    # Use `id` because the same scene can be defined in 2 splits.
    return {
        id(scene_config): scene_config
        for scene_config in self.all_scene_configs  # pylint: disable=not-an-iterable
    }

  def _split_name_to_scene_ids(self) -> dict[str, list[int]]:
    """Returns the mapping split_name->list[id(scene_config)]`."""
    return {
        split_name: [id(scene_config) for scene_config in scenes_configs]
        for split_name, scenes_configs in self.split_to_scene_configs.items()
    }

  def _scene_id_to_split_names(self) -> dict[int, list[str]]:
    """Returns the mapping id(scene_config)->list[split_name]`."""
    out = collections.defaultdict(list)
    for split_name, scene_ids in self.split_name_to_scene_ids.items():
      for scene_id in scene_ids:
        out[scene_id].append(split_name)
    return out


def _assert_single_split(splits):
  if len(splits) != 1:
    raise ValueError(
        'Cannot infer how to distribute the examples among splits: '
        f'{splits}. `generate_scene` should return `dict[split_names: ex]`')
