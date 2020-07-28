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
import os
import re
import logging
import pathlib
import random
import tempfile

from google.cloud import storage
from google.cloud.storage.blob import Blob


class AssetSource(object):
  # see: https://googleapis.dev/python/storage/latest

  def __init__(self, path: str):
    self.path = path

    if self.path.startswith("gs://"):
      bucket_name, prefix = re.findall("gs://(.*)/(.*)", self.path)[0]
      self.bucket_name, self.prefix = bucket_name, prefix
      self.local_temp_folder = tempfile.mkdtemp()
      self.client = storage.Client()
      self.bucket = self.client.get_bucket(self.bucket_name)
      self.manifest = self._download_manifest(
        "gs://" + bucket_name + "/" + prefix + "/manifest.txt")
    else:
      localpath = pathlib.Path(path).expanduser()

      # TODO: if manifest does not exist
      # self.manifest = glob.glob(path+"/**/*.urdf")
      with open(localpath / "manifest.txt") as f:
        self.manifest = f.read().splitlines()

  def take(self, num_objects: int):
    random_folders = random.sample(self.manifest, num_objects)

    # --- download to local temp folder
    if self.path.startswith("gs://"):
      [self._copy_folder(folder) for folder in random_folders]
      prefix = self.local_temp_folder
    else:
      prefix = self.path

    # --- pick a few local models
    local_prefix = pathlib.Path(prefix).expanduser()
    local_folders = [local_prefix / folder for folder in random_folders]

    # --- fetch URDF files in the folders
    # --- TODO: unchecked assumption one URDF per folder!!!
    urdf_paths = [folder.glob("*.urdf") for folder in local_folders]
    urdf_paths = [next(urdf_path) for urdf_path in urdf_paths]
    return urdf_paths

  def _download_manifest(self, remote_path):
    assert remote_path.startswith("gs://")
    logging.info("Downloading manifest: {}".format(remote_path))
    blob = Blob.from_string(remote_path)
    string = blob.download_as_string(client=self.client)
    lines = [line.decode('utf-8') for line in string.splitlines()]
    return lines

  def _copy_folder(self, subfolder: str):
    remote_subfolder = ["gs://" + self.bucket_name + "/" + self.prefix + "/" + subfolder]
    local_folder = os.path.join(self.local_temp_folder, subfolder)
    logging.info("Copying '{}' to '{}'".format(remote_subfolder, local_folder))
    remote_blobs = self.bucket.list_blobs(prefix=self.prefix + "/" + subfolder)
    for remote_blob in remote_blobs:
      local_blob_name = remote_blob.name.replace(self.prefix + "/", "")
      local_blob_path = os.path.join(self.local_temp_folder,
                                     local_blob_name)  # where to download
      pathlib.Path(local_blob_path).parent.mkdir(parents=True,
                                                 exist_ok=True)  # parents must exist
      remote_blob.download_to_filename(local_blob_path)
