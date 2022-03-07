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

import argparse
import copy
import functools
import logging
import multiprocessing
import pathlib
import shutil
import tarfile
import urllib.error
import urllib.parse
import urllib.request

import requests
import tqdm

from kubric import file_io
from kubric.kubric_typing import PathLike


def collect_list_of_available_assets(
    catalogue_path="hdri_haven_catalogue.json"):
  catalogue_path = file_io.as_path(catalogue_path)
  if catalogue_path.exists():
    return file_io.read_json(catalogue_path)

  # Get a list of available assets
  response = requests.get("https://api.polyhaven.com/assets?t=hdris")
  catalogue = [
      {"id": k,
       "asset_type": "Texture",
       "license": "CC0 1.0",
       "url": "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/4k/{name}_4k.hdr".format(
         name=urllib.request.quote(k)),
       "kwargs": {
       },
       "metadata": {
           "authors": list(v["authors"].keys()),
           "resolution": "4k",
           "coords": v.get("coords"),
           "date_taken": v["date_taken"],
           "tags": v["tags"],
           "categories": v["categories"],
       },
       } for k, v in response.json().items()]
  file_io.write_json(catalogue, catalogue_path)
  return catalogue


def download_asset(a, download_dir):
  filename = pathlib.Path(urllib.parse.urlparse(a["url"]).path).name
  download_dir = pathlib.Path(download_dir)
  target_path = download_dir / filename
  if not target_path.exists():
    try:
      opener = urllib.request.URLopener()
      opener.addheader('User-Agent', 'Mozilla/5.0')
      opener.retrieve(a["url"], target_path)
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
      logging.warning(f"FAILED! skipping '{a['name']}'", e)


def download_all(assets_list, num_processes=16, download_dir='GSO_raw'):
  download_dir = pathlib.Path(download_dir)
  download_dir.mkdir(parents=True, exist_ok=True)
  download_func = functools.partial(download_asset, download_dir=download_dir)
  with tqdm.tqdm(total=len(assets_list)) as pbar:
    with multiprocessing.Pool(num_processes, maxtasksperchild=1) as pool:
      promise = pool.imap_unordered(download_func, assets_list)
      for _ in promise:
        pbar.update(1)


def kubricify(asset, source_dir, target_dir):
  name = asset["id"]
  source_dir = file_io.as_path(source_dir)
  target_dir = file_io.as_path(target_dir)

  hdri_source_path = source_dir / f"{name}_4k.hdr"
  tmp_dir = target_dir / name
  if tmp_dir.exists():
    shutil.rmtree(tmp_dir)
  tmp_dir.mkdir(parents=True, exist_ok=False)

  json_path = tmp_dir / "data.json"
  tar_path = target_dir / f"{name}.tar.gz"

  if tar_path.exists():
    with tarfile.open(tar_path, "r:gz") as tar:
      tar.extract("data.json", tmp_dir)
    asset_entry = file_io.read_json(json_path)
    if asset_entry and "id" in asset_entry:
      shutil.rmtree(tmp_dir)
      return asset_entry["id"], asset_entry

  asset_entry = copy.deepcopy(asset)
  del asset_entry["url"]
  asset_entry["kwargs"]["filename"] = "environment_4k.hdr"

  file_io.write_json(asset_entry, json_path)

  with tarfile.open(tar_path, "w:gz") as tar:
    tar.add(hdri_source_path, asset_entry["kwargs"]["filename"])
    tar.add(json_path, "data.json")

  shutil.rmtree(tmp_dir)

  return name, asset_entry

def main(
    download_dir: PathLike = "GSO_raw",
    target_dir: PathLike = "GSO",
    keep_raw_assets=False
):
  download_dir = file_io.as_path(download_dir)
  target_dir = file_io.as_path(target_dir)
  catalogue = collect_list_of_available_assets()
  download_all(catalogue, download_dir=download_dir)
  assets = {}
  with tqdm.tqdm(total=len(catalogue)) as pbar:
    with multiprocessing.Pool(32, maxtasksperchild=1) as pool:
      promise = pool.imap_unordered(functools.partial(kubricify,
                                                      source_dir=download_dir,
                                                      target_dir=target_dir),
                                    catalogue)
      for name, entry in promise:
        assets[name] = entry
        pbar.update(1)

  manifest_path = "HDRI_haven.json"
  manifest = {
      "name": "HDRI_haven",
      "data_dir": str(target_dir),
      "version": "1.0",
      "assets": assets
  }
  file_io.write_json(manifest, manifest_path)
  if not keep_raw_assets:
    logging.info("Deleting the raw (unconverted) assets...")
    shutil.rmtree(download_dir)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--download_dir", type=str, default="HDRI_haven_raw")
  parser.add_argument("--target_dir", type=str, default="HDRI_haven")
  parser.add_argument("--keep_raw_assets", type=bool, default=False)
  FLAGS, unused = parser.parse_known_args()
  main(download_dir=FLAGS.download_dir, target_dir=FLAGS.target_dir,
       keep_raw_assets=FLAGS.keep_raw_assets)
