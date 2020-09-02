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

import json
import logging
import pathlib
import shutil
import tarfile
import tempfile

import pandas as pd
from google.cloud import storage
import urllib.parse

from kubric import core


logger = logging.getLogger(__name__)


class AssetSource(object):
  # see: https://googleapis.dev/python/storage/latest

  def __init__(self, uri: str):
    sections = urllib.parse.urlparse(uri)
    self.local_temp_folder = tempfile.TemporaryDirectory()
    self.local_path = pathlib.Path(self.local_temp_folder.name)

    if sections.scheme == "gs":   # cloud
      self.protocol = "gs"
      self.bucket_name = sections.netloc
      self.path = sections.path
      self.client = storage.Client()
      self.bucket = self.client.get_bucket(self.bucket_name)

    elif sections.scheme == "":  # local
      self.protocol = "local"
      self.path = pathlib.Path(uri)
    else:
      raise ValueError("Unknown protocol for {}".format(uri))

    # TODO handle missing details list file
    manifest = self._download_file("details_list.json")
    self.db = pd.read_json(manifest)

  def __del__(self):
    logger.info("removing tmp dir: \"%s\"", self.local_temp_folder)
    self.local_temp_folder.cleanup()

  def create(self, asset_id: str, **kwargs) -> core.FileBasedObject:
    assert asset_id in self.db["id"].values, kwargs
    sim_filename, vis_filename, properties = self.fetch(asset_id)

    for pname in ["mass", "friction", "restitution", "bounds"]:
      if pname in properties and pname not in kwargs:
        kwargs[pname] = properties[pname]

    return core.FileBasedObject(asset_id=asset_id,
                                simulation_filename=str(sim_filename),
                                render_filename=str(vis_filename),
                                **kwargs)

  def fetch(self, object_id):
    object_path = self._download_file(object_id + ".tar.gz")
    with tarfile.open(object_path, "r:gz") as tar:
      tar.extractall(self.local_path)

    urdf = self.local_path / object_id / "object.urdf"
    vis = self.local_path / object_id / "visual_geometry.obj"
    json_path = self.local_path / object_id / "data.json"

    with open(json_path, "r") as f:
      properties = json.load(f)
    return urdf, vis, properties

  def _download_file(self, filename):
    target_path = self.local_path / filename
    if self.protocol == "gs":
      remote_path = f"gs://{self.bucket_name}{self.path}/{filename}"
      logger.info("Downloading %s to %s", remote_path, str(target_path))
      blob = storage.blob.Blob.from_string(remote_path)
      blob.download_to_filename(str(target_path), client=self.client)
    elif self.protocol == "local":
      remote_path = f"{self.path}/{filename}"
      logger.info("Copying %s to %s", remote_path, str(target_path))
      shutil.copyfile(remote_path, target_path)

    return pathlib.Path(target_path)

