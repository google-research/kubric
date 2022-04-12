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

import logging
import pathlib
import shutil
import tarfile
import tempfile

import numpy as np
import tensorflow as tf
import thefuzz.process

from typing import Optional, Dict, Any, Type
import weakref

from kubric import core
from kubric import file_io
from kubric.kubric_typing import PathLike


class ClosableResource:
  """TODO(klausg): documentation."""
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

  @classmethod
  def from_manifest(
      cls,
      manifest_path: PathLike,
      scratch_dir: Optional[PathLike] = None
  ) -> "AssetSource":
    if manifest_path == "gs://kubric-public/assets/ShapeNetCore.v2.json":
      raise ValueError(f"The path `{manifest_path}` is a placeholder for the real path. "
                       "Please visit https://shapenet.org, agree to terms and conditions."
                       "After logging in, you will find the manifest URL here:"
                       "https://shapenet.org/download/kubric")

    manifest_path = file_io.as_path(manifest_path)
    manifest = file_io.read_json(manifest_path)
    name = manifest.get("name", manifest_path.stem)  # default to filename
    data_dir = manifest.get("data_dir", manifest_path.parent)  # default to manifest dir
    assets = manifest["assets"]
    return cls(name=name, data_dir=data_dir, assets=assets, scratch_dir=scratch_dir)

  def __init__(
      self,
      name: str,
      data_dir: PathLike,
      assets: Dict[str, Any],
      scratch_dir: Optional[PathLike] = None
  ):
    super().__init__()
    self.name = name
    self.data_dir = file_io.as_path(data_dir)
    logging.info("Created AssetSource '%s' with '%d' assets at URI='%s'",
                 name, len(assets), self.data_dir)
    self.local_dir = pathlib.Path(tempfile.mkdtemp(prefix=name, dir=scratch_dir))
    self._assets = assets

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

  @staticmethod
  def _resolve_asset_type(asset_type: str) -> Type:
    types = {
        "FileBasedObject": core.FileBasedObject,
        "Texture": core.Texture,
    }
    if asset_type not in types:
      raise KeyError(f"Unknown asset_type {asset_type!r}. "
                     f"Available types: {types!r}")
    return types[asset_type]

  def _resolve_asset_path(self, path: Optional[str], asset_id: str) -> Optional[PathLike]:
    if path is None:
      return None
    elif path == "":
      path = f"{asset_id}.tar.gz"

    return self.data_dir / path

  @staticmethod
  def _adjust_paths(asset_kwargs: Dict[str, Any], asset_dir: PathLike) -> Dict[str, Any]:
    """If present, replace '{asset_dir}' prefix with actual asset_dir in each kwarg value."""
    def _adjust_path(p):
      if isinstance(p, str) and p.startswith("{asset_dir}/"):
        return str(asset_dir / p[12:])
      elif isinstance(p, dict):
        return {key: _adjust_path(value) for key, value in p.items()}
      else:
        return p

    return {k: _adjust_path(v) for k, v in asset_kwargs.items()}

  def create(self, asset_id: str, add_metadata: bool = True, **kwargs) -> Type[core.Asset]:
    """
    Create an instance of an asset by a given id.

    Performs the following steps
    1. check if asset_id is found in manifest and retrieve entry
    2. determine Asset class and full path (can be remote or local cache or missing)
    3. if path is not none, then fetch and unpack the zipped asset to scratch_dir
    4. construct kwargs from asset_entry->kwargs, override with **kwargs and then
    adjust paths (ones that start with “{{asset_dir}}”
    5. create asset by calling constructor with kwargs
    6. set metadata (if add_metadata is True)
    7. return asset

    Args:
        asset_id (str): the id of the asset to be created
                        (corresponds to its key in the manifest file and
                        typically also to the filename)
        add_metadata (bool): whether to add the metadata from the asset to the instance
        **kwargs: additional kwargs to be passed to the asset constructor

    Returns:
      An instance of the specified asset (subtype of kubric.core.Asset)
    """
    # find corresponding asset entry
    asset_entry = self._assets.get(asset_id)
    if not asset_entry:
      near_match, _ = thefuzz.process.extractOne(asset_id, choices=self._assets.keys())
      raise KeyError(f"Unknown asset with id='{asset_id}'. Did you mean '{near_match}'?")

    # determine type and path
    asset_type = self._resolve_asset_type(asset_entry["asset_type"])
    asset_path = self._resolve_asset_path(asset_entry.get("path", ""), asset_id)

    # fetch and unpack tar.gz file if necessary
    asset_dir = None if asset_path is None else self.fetch(asset_path, asset_id)

    # construct kwargs
    asset_kwargs = asset_entry.get("kwargs", {})
    asset_kwargs.update(kwargs)
    asset_kwargs = self._adjust_paths(asset_kwargs, asset_dir)
    if asset_type == core.FileBasedObject:
      asset_kwargs["asset_id"] = asset_id
    # create the asset
    asset = asset_type(**asset_kwargs)
    # set the metadata
    if add_metadata:
      asset.metadata.update(asset_entry.get("metadata", {}))

    return asset

  def fetch(self, asset_path, asset_id):
    local_path = self.local_dir / (asset_id + ".tar.gz")
    if not local_path.exists():
      logging.debug("Copying %s to %s", str(asset_path), str(local_path))
      local_path.parent.mkdir(parents=True, exist_ok=True)
      tf.io.gfile.copy(asset_path, local_path)

      with tarfile.open(local_path, "r:gz") as tar:
        # We support two kinds of archives:
        #  1. flat archives that do not contain any directories
        #  2. archives where the content is in a directory with the name of the asset
        list_of_files = tar.getnames()
        if asset_id in list_of_files and tar.getmember(asset_id).isdir():
          # tarfile contains directory with name object_id, so we can just extract
          assert f"{asset_id}/data.json" in list_of_files, list_of_files
          tar.extractall(self.local_dir)
        else:
          # tarfile contains files only, so extract into a new directory
          assert "data.json" in list_of_files, list_of_files
          tar.extractall(self.local_dir / asset_id)
        logging.debug("Extracted %s", repr([m.name for m in tar.getmembers()]))

    return self.local_dir / asset_id

  def get_test_split(self, fraction=0.1):
    """
    Generates a train/test split for the asset source.

    Args:
      fraction: the fraction of the asset source to use for the heldout set.

    Returns:
      train_ids: list of asset ID strings
      test_ids: list of asset ID strings
    """
    rng = np.random.default_rng(42)
    test_size = int(round(len(self._assets) * fraction))
    test_ids = rng.choice(list(self._assets.keys()), size=test_size, replace=False)
    train_ids = [i for i in self._assets if i not in test_ids]
    return train_ids, test_ids
