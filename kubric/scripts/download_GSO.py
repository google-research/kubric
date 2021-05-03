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

# Copyright 2021 The Kubric Authors
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
import numpy as np
import os
import pybullet as pb
import re
import shutil
import ssl
import tarfile
import trimesh as tm
import trimesh.exchange.obj as tri_obj
import urllib.request
import zipfile
import argparse
import logging
import itertools


from pathlib import Path

URDF_TEMPLATE = """
<robot name="{id}">
    <link name="base">
        <contact>
            <lateral_friction value="{friction}" />  
        </contact>
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


def bypass_ssl_certification():
  if getattr(ssl, "_create_unverified_context", None):
    ssl._create_default_https_context = ssl._create_unverified_context


def collect_list_of_available_assets():
  bypass_ssl_certification()
  # Get a list of available assets
  list_of_assets = []
  for page in itertools.count(start=1):
    url = f"https://fuel.ignitionrobotics.org/1.0/GoogleResearch/models?page={page}"
    try:
      with urllib.request.urlopen(url) as f:
        list_of_assets.extend(json.load(f))
    except urllib.error.HTTPError:
      return list_of_assets


def download_gso_assets(download_dir: Path):
  download_dir.mkdir(parents=True, exist_ok=True)
  list_of_assets = collect_list_of_available_assets()
  # download
  for i, asset in enumerate(list_of_assets):
    name = asset["name"]
    url = "https://fuel.ignitionrobotics.org/1.0/GoogleResearch/models/{name}/1/{name}.zip".format(name=urllib.request.quote(name))
    filename = download_dir / f"{name}.zip"
    if not filename.exists():
      try:
        logging.info(f"{i+1:4d}/{len(list_of_assets)}: Fetching '{name}' from {url} into {filename}")
        urllib.request.urlretrieve(url, filename)
      except (urllib.error.HTTPError, urllib.error.URLError) as e:
        logging.warning(f"FAILED! skipping '{name}'", e)


def get_object_properties(tmesh, name, density=None):
  if density is None:
    tmesh.density = 1000.0

  rounda = lambda x: np.round(x, decimals=6).tolist()
  roundf = lambda x: float(np.round(x, decimals=6))

  return {
      "id": name,
      "density": roundf(tmesh.density),
      "friction": 0.5,
      "nr_vertices": len(tmesh.vertices),
      "nr_faces": len(tmesh.faces),
      "bounds": rounda(tmesh.bounds),
      "area": roundf(tmesh.area),
      "volume": roundf(tmesh.volume),
      "mass": roundf(tmesh.mass),
      "center_mass": rounda(tmesh.center_mass),
      "inertia": rounda(tmesh.moment_inertia),
      "is_convex": tmesh.is_convex,
      "euler_number": tmesh.euler_number,  # used for topological analysis (see: http://max-limper.de/publications/Euler/index.html),
  }


def convert_gso_asset(path, target_dir):
  name = path.stem
  asset_dir = path.parent / name
  with zipfile.ZipFile(path, "r") as zip_ref:
    zip_ref.extractall(asset_dir)
  texture_path = asset_dir / "materials" / "textures" / "texture.png"
  obj_path = asset_dir / "meshes" / "model.obj"
  mat_path = asset_dir / "meshes" / "model.mtl"

  target_asset_dir = target_dir / name
  if target_asset_dir.exists():
    shutil.rmtree(target_asset_dir)
  target_asset_dir.mkdir(parents=True, exist_ok=True)

  vis_path = target_asset_dir / "visual_geometry.obj"
  coll_path = target_asset_dir / "collision_geometry.obj"
  urdf_path = target_asset_dir / "object.urdf"
  json_path = target_asset_dir / "data.json"
  tex_path = target_asset_dir / "texture.png"
  tar_path = target_dir / (name + ".tar.gz")

  shutil.copy(texture_path, asset_dir / "meshes" / "texture.png")
  # import mesh into trimesh
  tmesh = tm.load_mesh(obj_path, file_type="obj")
  # center it around center of mass
  tmesh.apply_translation(-tmesh.center_mass)
  # export to obj again
  obj_content = tri_obj.export_obj(tmesh)
  obj_content = re.sub("mtllib material0.mtl\nusemtl material0\n",
                       "mtllib visual_geometry.mtl\nusemtl material_0\n", obj_content)
  with open(vis_path, "w") as f:
    f.write(obj_content)

  # compute a collision mesh using pybullets VHACD
  pb.vhacd(str(vis_path),
           str(coll_path),
           str(target_asset_dir / "pybullet_logs.txt"))

  # move material and texture
  shutil.move(mat_path, target_asset_dir / "visual_geometry.mtl")
  shutil.move(texture_path, tex_path)

  properties = get_object_properties(tmesh, name)

  with open(urdf_path, "w") as f:
    f.write(URDF_TEMPLATE.format(**properties))

  properties["paths"] = {
      "visual_geometry": ["visual_geometry.obj"],
      "collision_geometry": ["collision_geometry.obj"],
      "urdf": ["object.urdf"],
      "texture": ["texture.png"],
  }

  with open(json_path, "w") as f:
    json.dump(properties, f, indent=4, sort_keys=True)

  logging.info("  saving as", tar_path)
  with tarfile.open(tar_path, "w:gz") as tar:
    tar.add(target_asset_dir, arcname=name)

  shutil.rmtree(asset_dir)
  shutil.rmtree(target_asset_dir)
  return properties


def convert_all_gso_assets_from_dir(download_dir: Path, target_dir: Path):
  details_list = []
  list_of_asset_files = sorted(download_dir.glob("*.zip"))
  for i, path in enumerate(list_of_asset_files):
    logging.info(f"{i:4d}/{len(list_of_asset_files)}: Converting {path}...")
    properties = convert_gso_asset(path, target_dir)
    details_list.append(properties)

  with open(target_dir / "manifest.json", "w") as f:
    json.dump(details_list, f, indent=4, sort_keys=True)


def main(download_dir: Path = Path("../Assets/GSO_raw"), target_dir: Path = Path("../Assets/GSO"),
         keep_raw_assets=False):
  download_gso_assets(download_dir)
  target_dir.mkdir(parents=True, exist_ok=True)
  convert_all_gso_assets_from_dir(download_dir, target_dir)
  logging.info("Done!")
  if not keep_raw_assets:
    logging.info("Deleting the raw (unconverted) assets...")
    shutil.rmtree(download_dir)


if __name__ == '__main__':
  parser = argparse.Parser()
  parser.add_argument("--download_dir", type=str, default="Assets/GSO_raw")
  parser.add_argument("--target_dir", type=str, default="Assets/GSO")
  parser.add_argument("--keep_raw_assets", type=bool, default=False)
  FLAGS, unused = parser.parse_known_args()
  main(download_dir=FLAGS.download_dir, target_dir=FLAGS.target_dir,
       keep_raw_assets=FLAGS.keep_raw_assets)
