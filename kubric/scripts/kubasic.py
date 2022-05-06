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
""" Convert KuBasic.blend (or other suitable blend files) into asset source format."""

import argparse
import logging
import os
import re
import sys
import tarfile
import tempfile
from typing import Optional

import bpy
import numpy as np
import pybullet as pb

from kubric import file_io
from kubric.renderer import blender_utils
from kubric.kubric_typing import PathLike
from kubric import redirect_io

logger = logging.getLogger(__name__)

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


def get_object_properties(obj, density=1.0):
  tmesh = blender_utils.bpy_mesh_object_to_trimesh(obj)

  rounda = lambda x: np.round(x, decimals=4).tolist()
  roundf = lambda x: float(np.round(x, decimals=4))

  properties = {
      "nr_vertices": len(tmesh.vertices),
      "nr_faces": len(tmesh.faces),
      "bounds": rounda(tmesh.bounds),
      "surface_area": roundf(tmesh.area),
      "volume": roundf(tmesh.volume),
      "center_mass": rounda(tmesh.center_mass),
      "inertia": rounda(tmesh.moment_inertia),
      "mass": roundf(tmesh.volume * density),
  }
  return properties


def kubricify(obj: bpy.types.Object, output_path: PathLike, asset_license):
  logger.info("Kubricifying %s", obj.name)
  with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = file_io.as_path(tmpdir)
    logger.info("applying transformations ...")
    blender_utils.apply_transformations(obj)
    logger.info("triangulating ...")
    blender_utils.triangulate(obj)
    logger.info("centering mesh around center of mass")
    blender_utils.center_mesh_around_center_of_mass(obj)

    logger.info("computing and exporting properties of object to data.json")
    properties = get_object_properties(obj)
    properties["id"] = obj.name
    properties["license"] = asset_license
    properties["asset_type"] = "FileBasedObject"
    properties["paths"] = {
        "visual_geometry": "visual_geometry.obj",
        "collision_geometry": "collision_geometry.obj",
        "urdf": "object.urdf"
    }

    asset_entry = {
        "id": properties["id"],
        "asset_type": properties["asset_type"],
        "license": asset_license,
        "kwargs": {
            "bounds": properties["bounds"],
            "mass": properties["mass"],
            "simulation_filename": "{asset_dir}/" + properties["paths"]["urdf"],
            "render_filename": "{asset_dir}/" + properties["paths"]["visual_geometry"],
        },
        "metadata": {
            "nr_faces": properties["nr_faces"],
            "nr_vertices": properties["nr_vertices"],
            "volume": properties["volume"],
            "surface_area": properties["surface_area"],
        },
    }

    file_io.write_json(asset_entry, tmpdir / "data.json")

    logger.info("exporting visual geometry")
    vis_path = tmpdir / properties["paths"]["visual_geometry"]
    with blender_utils.selected(obj), blender_utils.centered(obj):
      bpy.ops.export_scene.obj(filepath=str(vis_path),
                               axis_forward="Y", axis_up="Z", use_selection=True,
                               use_materials=True)
    logger.info("generating and exporting collision geometry")
    coll_path = tmpdir / properties["paths"]["collision_geometry"]
    with redirect_io.RedirectStream(stream=sys.stdout):
      pb.vhacd(str(vis_path), str(coll_path), str(tmpdir / "pybullet.log"))
    os.remove(tmpdir / "pybullet.log")

    logger.info("exporting URDF file")
    urdf_path = tmpdir / properties["paths"]["urdf"]
    with open(urdf_path, "w", encoding="utf-8") as f:
      f.write(URDF_TEMPLATE.format(**properties))

    out_path = output_path / (properties["id"] + ".tar.gz")
    logger.info("packing all files into single %s", out_path)
    with tarfile.open(out_path, "w:gz") as tar:
      for p in tmpdir.glob("*"):
        tar.add(p, arcname=p.name)

    return properties["id"], asset_entry


def main(
    output_dir: str,
    scene_path: str,
    name: Optional[str] = None,
    version: Optional[str] = None,
):
  scene_path = file_io.as_path(scene_path).expanduser()
  if name is None:
    name = scene_path.stem
    logging.info("Inferred name from scene path as %s", name)

  if version is None:
    n, sep, v = name.rpartition("_v")
    if sep and v and re.match("[0-9.]+", v):
      version = v
      name = n
      logging.info("Inferred version from name as %s and shortened name to %s", version, name)

  logging.info("name = %s and version = %s", name, version)
  output_dir = file_io.as_path(output_dir).expanduser()
  base_dir = output_dir / name
  logging.info(f"Converting {scene_path} into asset source format in directory {base_dir}")
  base_dir.mkdir(parents=True, exist_ok=True)

  logging.info(f"Resetting Blender and loading scene %s", scene_path)
  bpy.ops.wm.read_factory_settings(use_empty=True)
  bpy.ops.wm.open_mainfile(filepath=str(scene_path))

  manifest = {
      "name": name,
      "data_dir": str(base_dir),
      "version": version,
      "assets": {}
  }
  for obj in bpy.data.objects:
   asset_id, asset_entry = kubricify(obj, base_dir, asset_license="CC BY-SA 4.0")
   manifest["assets"][asset_id] = asset_entry

  manifest_path = output_dir / (name + ".json")
  logging.info("writing manifest to '%s'", manifest_path)
  file_io.write_json(manifest, manifest_path)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--output_dir", type=str, default="Assets")
  parser.add_argument("--scene_path", type=str, default="KuBasic_v1.0.blend")
  parser.add_argument("--name", type=str, default=None)
  parser.add_argument("--version", type=str, default=None)
  parser.add_argument("--log_level", type=str, default="INFO")
  FLAGS = parser.parse_args()

  logging.basicConfig(level=FLAGS.log_level)
  main(FLAGS.output_dir, FLAGS.scene_path, FLAGS.name, FLAGS.version)
