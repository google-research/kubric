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

# Copyright 2022 The Kubric Authors
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
import argparse
import functools
import itertools
import json
import logging
import multiprocessing
import pathlib
import re
import shutil
import ssl
import sys
import tarfile
import urllib
import urllib.request
import urllib.error
import zipfile

import pybullet as pb
import tqdm
import trimesh as tm
import trimesh.exchange.obj as tri_obj

from kubric import file_io
from kubric import redirect_io
from kubric.kubric_typing import PathLike


URDF_TEMPLATE = """
<robot name="{id}">
    <link name="base">
        <inertial>
            <origin xyz="{center_mass[0]} {center_mass[1]} {center_mass[2]}" />
            <mass value="{mass}" />
            <inertia ixx="{inertia[0][0]}" ixy="{inertia[0][1]}" 
                     ixz="{inertia[0][2]}" iyy="{inertia[1][1]}" 
                     iyz="{inertia[1][2]}" izz="{inertia[2][2]}" />
        </inertial>
        <visual>
            <origin xyz="0 0 0" />
            <geometry>
                <mesh filename="visual_geometry.obj" />
            </geometry>
        </visual>
        <collision>
            <origin xyz="0 0 0" />
            <geometry>
                <mesh filename="collision_geometry.obj" />
            </geometry>
        </collision>
    </link>
</robot>
"""


def get_object_properties(tmesh, density=1.0):
  properties = {
      "nr_vertices": len(tmesh.vertices),
      "nr_faces": len(tmesh.faces),
      "bounds": tmesh.bounds.tolist(),
      "surface_area": tmesh.area,
      "volume": tmesh.volume,
      "center_mass": tmesh.center_mass.tolist(),
      "inertia": tmesh.moment_inertia.tolist(),
      "mass": tmesh.volume * density,
  }
  return properties


def bypass_ssl_certification():
  if getattr(ssl, "_create_unverified_context", None):
    ssl._create_default_https_context = ssl._create_unverified_context


def get_metadata(a):
  name = a["name"]
  return {
      "name": name,
      "category": a.get("categories", ["None"])[0],
      "description": a["description"],
      "url": "https://fuel.ignitionrobotics.org/1.0/GoogleResearch/models/{name}/1/{name}.zip".format(name=urllib.request.quote(name)),
  }


def collect_list_of_available_assets(catalogue_path="GSO_catalogue.json"):
  catalogue_path = file_io.as_path(catalogue_path)
  if catalogue_path.exists():
    return file_io.read_json(catalogue_path)

  bypass_ssl_certification()
  # Get a list of available assets
  list_of_assets = []
  for page in itertools.count(start=1):
    url = f"https://fuel.ignitionrobotics.org/1.0/GoogleResearch/models?page={page}"
    try:
      with urllib.request.urlopen(url) as f:
        list_of_assets.extend(json.load(f))
    except urllib.error.HTTPError:
      assets = [get_metadata(a) for a in list_of_assets]
      file_io.write_json(assets, catalogue_path)
      return assets


def download_asset(a, download_dir):
  filename = download_dir / f"{a['name']}.zip"
  if not filename.exists():
    try:
      urllib.request.urlretrieve(a["url"], filename)
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
      logging.warning(f"FAILED! skipping '{a['name']}'", e)


def download_all(
    assets_list,
    num_processes: int = 16,
    download_dir: PathLike = 'GSO_raw'
):
  download_dir = pathlib.Path(download_dir)
  download_dir.mkdir(parents=True, exist_ok=True)
  with tqdm.tqdm(total=len(assets_list)) as pbar:
    with multiprocessing.Pool(num_processes, maxtasksperchild=1) as pool:
      promise = pool.imap_unordered(functools.partial(download_asset, download_dir=download_dir), assets_list)
      for _ in promise:
        pbar.update(1)


def kubricify(asset, source_dir, target_dir):
  source_dir = pathlib.Path(source_dir)
  target_dir = pathlib.Path(target_dir)
  target_dir.mkdir(parents=True, exist_ok=True)
  name = asset["name"]

  # --- source paths
  asset_raw_path = source_dir / (name + ".zip")
  asset_tmp_dir = source_dir / name

  texture_source_path = asset_tmp_dir / "materials" / "textures" / "texture.png"
  obj_source_path = asset_tmp_dir / "meshes" / "model.obj"
  mat_source_path = asset_tmp_dir / "meshes" / "model.mtl"

  # --- target paths
  target_asset_dir = target_dir / name
  vis_path = target_asset_dir / "visual_geometry.obj"
  mat_path = target_asset_dir / "visual_geometry.mtl"
  coll_path = target_asset_dir / "collision_geometry.obj"
  urdf_path = target_asset_dir / "object.urdf"
  json_path = target_asset_dir / "data.json"
  tex_path = target_asset_dir / "texture.png"
  tar_path = target_dir / (name + ".tar.gz")

  if tar_path.exists():
    with tarfile.open(tar_path, "r:gz") as tar:
      tar.extract("data.json", target_asset_dir)
    asset_entry = file_io.read_json(json_path)
    if asset_entry and "id" in asset_entry:
      shutil.rmtree(target_asset_dir)
      return asset_entry["id"], asset_entry

  # extract source to asset_tmp_dir
  with zipfile.ZipFile(asset_raw_path, "r") as zip_ref:
    zip_ref.extractall(asset_tmp_dir)

  # clear target path
  if target_asset_dir.exists():
    shutil.rmtree(target_asset_dir)
  target_asset_dir.mkdir(parents=True, exist_ok=True)

  # --- convert
  # copy the texture to a place that trimesh will find
  shutil.copy(texture_source_path, asset_tmp_dir / "meshes" / "texture.png")

  # import mesh into trimesh
  tmesh = tm.load_mesh(obj_source_path, file_type="obj")
  # center it around center of mass
  tmesh.apply_translation(-tmesh.center_mass)
  # export to obj again
  obj_content = tri_obj.export_obj(tmesh)
  obj_content = re.sub("mtllib material_0.mtl\nusemtl material_0\n",
                       f"mtllib {mat_path.name}\nusemtl material_0\n", obj_content)
  with open(vis_path, "w") as f:
    f.write(obj_content)

  shutil.move(mat_source_path, mat_path)
  shutil.move(texture_source_path, tex_path)

  with redirect_io.RedirectStream(stream=sys.stdout):
    pb.vhacd(str(vis_path), str(coll_path), str(asset_tmp_dir / "pybullet.log"))

  properties = get_object_properties(tmesh)

  with open(urdf_path, "w") as f:
    f.write(URDF_TEMPLATE.format(id=asset["name"], **properties))

  asset_entry = {
      "id": asset["name"],
      "asset_type": "FileBasedObject",
      "kwargs": {
          "bounds": properties["bounds"],
          "mass": properties["mass"],
          "render_filename": "{asset_dir}/" + vis_path.name,
          "simulation_filename": "{asset_dir}/" + urdf_path.name,
      },
      "license": "CC BY-SA 4.0",
      "metadata": {
          "nr_faces": properties["nr_faces"],
          "nr_vertices": properties["nr_vertices"],
          "surface_area": properties["surface_area"],
          "volume": properties["volume"],
          "category": asset["category"],
          "description": asset["description"],
      },
  }

  with open(json_path, "w") as f:
    json.dump(asset_entry, f, indent=4, sort_keys=True)

  with tarfile.open(tar_path, "w:gz") as tar:
    for p in target_asset_dir.glob("*"):
      tar.add(p, arcname=p.name)

  shutil.rmtree(asset_tmp_dir)
  shutil.rmtree(target_asset_dir)

  return name, asset_entry


def main(
    download_dir: PathLike = "GSO_raw",
    target_dir: PathLike = "GSO",
    keep_raw_assets=False
):
  download_dir = file_io.as_path(download_dir)
  target_dir = file_io.as_path(target_dir)
  assets = collect_list_of_available_assets()
  download_all(assets, download_dir=download_dir)
  assets_list = []
  with tqdm.tqdm(total=len(assets)) as pbar:
    with multiprocessing.Pool(32, maxtasksperchild=1) as pool:
      promise = pool.imap_unordered(functools.partial(kubricify,
                                                      source_dir=download_dir,
                                                      target_dir=target_dir),
                                    assets)
      for result in promise:
        assets_list.append(result)
        pbar.update(1)

  manifest_path = "GSO.json"
  manifest = {
      "name": "GSO",
      "data_dir": str(target_dir),
      "version": "1.0",
      "assets": {k: v for k, v in assets_list}
  }
  file_io.write_json(manifest, manifest_path)
  if not keep_raw_assets:
    logging.info("Deleting the raw (unconverted) assets...")
    shutil.rmtree(download_dir)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--download_dir", type=str, default="GSO_raw")
  parser.add_argument("--target_dir", type=str, default="GSO")
  parser.add_argument("--keep_raw_assets", type=bool, default=False)
  FLAGS, unused = parser.parse_known_args()
  main(download_dir=FLAGS.download_dir, target_dir=FLAGS.target_dir,
       keep_raw_assets=FLAGS.keep_raw_assets)
