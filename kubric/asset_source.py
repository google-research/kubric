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
import glob
import logging
import pathlib
import random
import tempfile

from google.cloud import storage
from google.cloud.storage.blob import Blob


class AssetSource(object):
  # see: https://googleapis.dev/python/storage/latest

  def __init__(self, path):
    self.path = path

    if self.path.startswith("gs://"):
      bucket_name, prefix = re.findall("gs://(.*)/(.*)", self.path)[0]
      self.bucket_name, self.prefix = bucket_name, prefix
      self.local_temp_folder = tempfile.mkdtemp()
      self.client = storage.Client()
      self.bucket = self.client.get_bucket(self.bucket_name)
      self.manifest = self.download_manifest("gs://"+bucket_name+"/"+prefix+"/manifest.txt")
      self.manifest = ["gs://"+bucket_name+"/"+prefix+"/"+line for line in self.manifest]
    else:
      # TODO: if manifest does not exist
      # self.manifest = glob.glob(path+"/**/*.urdf")
      with open(os.path.join(path, "manifest.txt")) as f:
        self.manifest = f.read().splitlines()

  def take(self, num_objects: int):
    if self.path.startswith("gs://"):
      # --- pick a few, and mirror locally
      remote_folders = random.sample(self.manifest, num_objects)
      local_folders = [self.copy_folder(folder) for folder in remote_folders]
    else:
      # --- pick a few local models
      local_folders = random.sample(self.manifest, num_objects)
      local_folders = [os.path.join(self.path, folder) for folder in local_folders]

    # --- fetch URDF files in the folders
    urdf_paths = [glob.glob(folder+"/*.urdf") for folder in local_folders]
    # --- TODO: unchecked assumption one URDF per folder!!!
    urdf_paths = [urdf_path[0] for urdf_path in urdf_paths]
    return urdf_paths

  def download_manifest(self, remote_path):
    assert remote_path.startswith("gs://")
    logging.info("Downloading manifest: {}".format(remote_path))
    blob = Blob.from_string(remote_path)
    string = blob.download_as_string(client=self.client)
    lines = [line.decode('utf-8') for line in string.splitlines()]
    return lines

  def copy_folder(self, remote_path: str):
    assert remote_path.startswith("gs://")
    remote_subfolder_name = remote_path.replace(self.path+"/", "")
    local_folder = os.path.join(self.local_temp_folder, remote_subfolder_name)
    logging.info("Copying '{}' to '{}'".format(remote_path, local_folder))
    remote_blobs = self.bucket.list_blobs(prefix=self.prefix+"/"+remote_subfolder_name)
    for remote_blob in remote_blobs:
      local_blob_name = remote_blob.name.replace(self.prefix+"/", "")
      local_blob_path = os.path.join(self.local_temp_folder, local_blob_name)  # where to download
      pathlib.Path(local_blob_path).parent.mkdir(parents=True, exist_ok=True)  # parents must exist
      remote_blob.download_to_filename(local_blob_path)
    return local_folder
