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
"""Integration tests."""
import pytest


@pytest.mark.skip
def test_upload():
  # TODO: figure out how to properly test cloud storage access
  # example: writes frame.png → gs://kubric/subfolder/target.png
  from google.cloud import storage
  bucket = storage.Client().bucket("kubric") #< gs://kubric
  blob = bucket.blob("subfolder/target.png")  #< position on bucket
  blob.upload_from_filename("frame.png") #< position on local system