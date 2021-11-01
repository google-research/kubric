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

import json
import logging
import pathlib
import shutil
import tarfile
import tempfile

import pandas as pd
import tensorflow as tf
import tensorflow_datasets.public_api as tfds


from typing import Optional
import weakref

from kubric.kubric_typing import PathLike
from kubric.core import objects
from kubric.core import materials


class ClosableResource:
  _set_of_open_resources = weakref.WeakSet()

  def __init__(self):
    super().__init__()
    self.is_closed = False
    self._set_of_open_resources.add(self)

  def close(self):
    try:
      self._set_of_open_resources.remove(self)
    except (ValueError, KeyError):
      pass  # not listed anymore. Ignore.

  @classmethod
  def close_all(cls):
    while True:
      try:
        r = cls._set_of_open_resources.pop()
      except KeyError:
        break
      r.close()


class AssetSource(ClosableResource):
  """TODO(klausg): documentation."""

  def __init__(self, path: PathLike, scratch_dir: Optional[PathLike] = None):
    super().__init__()
    self.remote_dir = tfds.core.as_path(path)
    name = self.remote_dir.name
    logging.info("Adding AssetSource '%s' with URI='%s'", name, self.remote_dir)

    self.local_dir = pathlib.Path(tempfile.mkdtemp(prefix="assets", dir=scratch_dir))

    manifest_path = self.remote_dir / "manifest.json"
    if manifest_path.exists():
      self.db = pd.read_json(tf.io.gfile.GFile(manifest_path, "r"))
      logging.info("Found manifest file. Loaded information about %d assets", self.db.shape[0])
    else:
      assets_list = [p.name[:-7] for p in self.remote_dir.iterdir() if p.name.endswith(".tar.gz")]
      self.db = pd.DataFrame(assets_list, columns=["id"])
      logging.info("No manifest file. Found %d assets.", self.db.shape[0])

  def close(self):
    if self.is_closed:
      return
    try:
      shutil.rmtree(self.local_dir)
    finally:
      super().close()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()

  def create(self, asset_id: str, **kwargs) -> objects.FileBasedObject:
    assert asset_id in self.db["id"].values, kwargs
    sim_filename, vis_filename, properties = self.fetch(asset_id)

    for pname in ["mass", "friction", "restitution", "bounds", "render_import_kwargs"]:
      if pname in properties and pname not in kwargs:
        kwargs[pname] = properties[pname]

    return objects.FileBasedObject(asset_id=asset_id,
                                   simulation_filename=str(sim_filename),
                                   render_filename=str(vis_filename),
                                   **kwargs)

  def fetch(self, object_id):
    remote_path = self.remote_dir / (object_id + ".tar.gz")
    local_path = self.local_dir / (object_id + ".tar.gz")
    if not local_path.exists():
      logging.debug("Copying %s to %s", str(remote_path), str(local_path))
      tf.io.gfile.copy(remote_path, local_path)

      with tarfile.open(local_path, "r:gz") as tar:
        list_of_files = tar.getnames()

        if object_id in list_of_files and tar.getmember(object_id).isdir():
          # tarfile contains directory with name object_id, so we can just extract
          assert f"{object_id}/data.json" in list_of_files, list_of_files
          tar.extractall(self.local_dir)
        else:
          # tarfile contains files only, so extract into a new directory
          assert "data.json" in list_of_files, list_of_files
          tar.extractall(self.local_dir / object_id)
        logging.debug("Extracted %s", repr([m.name for m in tar.getmembers()]))

    json_path = self.local_dir / object_id / "data.json"
    with open(json_path, "r", encoding="utf-8") as f:
      properties = json.load(f)
      logging.debug("Loaded properties %s", repr(properties))

    # paths
    vis_path = properties["paths"]["visual_geometry"]
    if isinstance(vis_path, list):
      vis_path = vis_path[0]
    vis_path = self.local_dir / object_id / vis_path
    urdf_path = properties["paths"]["urdf"]
    if isinstance(urdf_path, list):
      urdf_path = urdf_path[0]
    urdf_path = self.local_dir / object_id / urdf_path

    return urdf_path, vis_path, properties

  def get_test_split(self, fraction=0.1):
    """
    Generates a train/test split for the asset source.

    Args:
      fraction: the fraction of the asset source to use for the heldout set.

    Returns:
      train_objects: list of asset ID strings
      held_out_objects: list of asset ID strings
    """
    held_out_objects = list(self.db.sample(frac=fraction, replace=False, random_state=42)["id"])
    train_objects = [i for i in self.db["id"] if i not in held_out_objects]
    return train_objects, held_out_objects


class TextureSource(ClosableResource):
  """TODO(klausg): documentation."""

  def __init__(self, path: PathLike, scratch_dir: Optional[PathLike] = None):
    super().__init__()
    self.remote_dir = tfds.core.as_path(path)
    name = self.remote_dir.name
    logging.info("Adding TextureSource '%s' with URI='%s'", name, self.remote_dir)
    self.local_dir = tfds.core.as_path(tempfile.mkdtemp(prefix="textures", dir=scratch_dir))

    manifest_path = self.remote_dir / "manifest.json"
    if manifest_path.exists():
      self.db = pd.read_json(tf.io.gfile.GFile(manifest_path, "r"))
      logging.info("Found manifest file. Loaded information about %d assets", self.db.shape[0])
    else:
      assets_list = [p.name for p in self.remote_dir.iterdir()]
      self.db = pd.DataFrame(assets_list, columns=["id"])
      logging.info("No manifest file. Found %d assets.", self.db.shape[0])

  def close(self):
    if self.is_closed:
      return
    try:
      shutil.rmtree(self.local_dir)
    finally:
      super().close()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()

  def create(self, texture_name: str, **kwargs) -> materials.Texture:
    texture_path = self.fetch(texture_name)
    return materials.Texture(filename=str(texture_path), **kwargs)

  def fetch(self, texture_name):
    remote_path = self.remote_dir / texture_name
    local_path = self.local_dir / texture_name
    if not local_path.exists():
      logging.debug("Copying %s to %s", str(remote_path), str(local_path))
      tf.io.gfile.copy(remote_path, local_path)
    return local_path

  def get_test_split(self, fraction=0.1):
    held_out_textures = list(self.db.sample(frac=fraction, replace=False, random_state=42)["id"])
    train_textures = [i for i in self.db["id"] if i not in held_out_textures]
    return train_textures, held_out_textures
