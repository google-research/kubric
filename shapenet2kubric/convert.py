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

# pylint: disable=logging-fstring-interpolation
# see: https://docs.python.org/3/library/subprocess.html

import argparse
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from typing import Tuple

import trimesh

from shapenet_synsets import CATEGORY_NAMES
from trimesh_utils import get_object_properties
import trimesh_utils
from urdf_template import URDF_TEMPLATE

_DEFAULT_LOGGER = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def stage0(object_folder: Path, logger=_DEFAULT_LOGGER):
  source_path = object_folder / 'models' / 'model_normalized.obj'
  target_path = object_folder / 'kubric' / 'visual_geometry_pre.glb'

  if target_path.is_file():
    logger.debug(f'skipping stage0 on "{object_folder}"')
    return  # stage already completed; skipping
  
  # --- pre-condition
  if not source_path.is_file():
    logger.error(f'stage0 pre-condition failed, file does not exist "{source_path}"')

  # --- body
  logger.debug(f'stage0 running on "{object_folder}"')
  cmd = f'obj2gltf -i {source_path} -o {target_path}'
  retobj = subprocess.run(cmd, capture_output=True, shell=True, text=True)
  if 'ENOENT' in retobj.stdout:
    logger.error(f'{retobj.stdout}')
  if retobj.returncode != 0:
    logger.error(f'obj2gltf failed on "f{object_folder}"')
    if retobj.stdout != '': logger.error(f'{retobj.stdout}')
    if retobj.stderr != '': logger.error(f'{retobj.stderr}')

  # --- post-condition
  if not target_path.is_file():
    logger.error(f'Post-condition failed, file does not exist "{target_path}"')


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def stage1(object_folder: Path, logger=_DEFAULT_LOGGER):
  source_path = object_folder / 'models' / 'model_normalized.obj'
  target_path = object_folder / 'kubric' / 'model_watertight.obj'

  if target_path.is_file():
    logger.debug(f'skipping stage1 on "{object_folder}"')
    return  # stage already completed; skipping

  # --- pre-condition
  if not source_path.is_file():
    logger.error(f'stage1 pre-condition failed, file does not exist "{source_path}"')

  # --- body
  logger.debug(f'stage1 running on "{object_folder}"')
  cmd = f'manifold --input {source_path} --output {target_path}'
  retobj = subprocess.run(cmd, capture_output=True, shell=True, text=True)
  if retobj.returncode != 0:
    logger.error(f'manifold failed on "f{object_folder}"')
    if retobj.stdout != '': logger.error(f'{retobj.stdout}')
    if retobj.stderr != '': logger.error(f'{retobj.stderr}')

  # --- post-condition
  if not target_path.is_file():
    logger.error(f'stage1 post-condition failed, file does not exist "{target_path}"')


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
    
def stage2(object_folder: Path, logger=_DEFAULT_LOGGER):
  source_path = object_folder / 'kubric' / 'model_watertight.obj'
  target_path = object_folder / 'kubric' / 'collision_geometry.obj'
  log_path = object_folder / 'kubric' / 'stage2_logs.txt'
  stdout_path = str(object_folder / 'kubric' / 'stage2_stdout.txt')

  if target_path.is_file():
    logger.debug(f'skipping stage2 on "{object_folder}"')
    return  # stage already completed; skipping

  # --- pre-condition
  if not source_path.is_file():
    logger.error(f'stage2 pre-condition failed, file does not exist "{source_path}"')

  # --- body
  logger.debug(f'stage2 running on "{object_folder}"')
  # TODO: how to monitor errors? should we move to "raw" VHCD?
  command_string = f"python pybullet_vhacd.py " \
                   f"--source_path={source_path} --target_path={target_path} " \
                   f"--stdout_path={stdout_path} > {log_path}"

  retobj = subprocess.run(command_string, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  if retobj.returncode != 0:
    logger.error(f'stage2 failed with return code {retobj.returncode}')

  # --- post-condition
  if not target_path.is_file():
    logger.error(f'stage2 post-condition failed, file does not exist "{target_path}"')


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def stage3(object_folder: Path, logger=_DEFAULT_LOGGER):
  source_path = object_folder / 'kubric' / 'visual_geometry_pre.glb'
  log_path = object_folder / 'kubric' / 'stage3_logs.txt'
  target_path = object_folder / 'kubric' / 'visual_geometry.glb'

  if target_path.is_file():
    logger.debug(f'skipping stage3 on "{object_folder}"')
    return  # stage already completed; skipping
  logger.debug(f'stage3 running on "{object_folder}"')

  asset_id = str(object_folder.relative_to(object_folder.parent.parent))

  command_string = f"python bpy_clean_mesh.py " \
                   f"--source_path={source_path} --target_path={target_path} " \
                   f"--asset_id={asset_id} > {log_path}"

  retobj = subprocess.run(command_string, shell=True, check=True)
  if retobj.returncode != 0:
    logger.error(f'stage3 failed with return code {retobj.returncode}')

  if not target_path.is_file():
    logger.error(f'stage3 post-condition failed, file does not exist "{target_path}"')


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def get_asset_id_and_category(object_folder: Path) -> Tuple[str, str, str]:
  category_id = str(object_folder.parent.relative_to(object_folder.parent.parent))
  asset_id = str(object_folder.relative_to(object_folder.parent))
  category_name = CATEGORY_NAMES[category_id]
  return asset_id, category_id, category_name

def get_object_volume(obj_path:Path, logger=_DEFAULT_LOGGER, density=1.0):
  # --- override the trimesh logger
  trimesh.util.log = logger

  tmesh = trimesh_utils.get_tmesh(str(obj_path))

  properties = {
      "volume": tmesh.volume,
      "surface_area": tmesh.area,
      "mass": tmesh.volume * density,
  }
  return properties

def get_visual_properties(obj_path:Path, logger=_DEFAULT_LOGGER):
  # --- override the trimesh logger
  trimesh.util.log = logger

  tmesh = trimesh_utils.get_tmesh(str(obj_path))

  properties = {
      "nr_vertices": len(tmesh.vertices),
      "nr_faces": len(tmesh.faces),
  }
  return properties



def stage4(object_folder: Path, logger=_DEFAULT_LOGGER):
  # TODO: we should probably use a mixture of model_normalized and model_wateright here?
  source_path = object_folder / 'kubric' / 'collision_geometry.obj'
  watertight_mesh_path = object_folder / 'kubric' / 'model_watertight.obj'
  vis_mesh_path = object_folder / 'kubric' / 'visual_geometry.glb'
  target_urdf_path = object_folder / 'kubric' / 'object.urdf'
  target_json_path = object_folder / 'kubric' / 'data.json'

  if target_urdf_path.is_file() and target_json_path.is_file():
    return

  # --- pre-condition
  if not source_path.is_file():
    logger.error(f'stage4 pre-condition failed, file does not exist "{source_path}"')
  if not watertight_mesh_path.is_file():
    logger.error(f'stage4 pre-condition failed, file does not exist "{watertight_mesh_path}"')
  if not vis_mesh_path.is_file():
    logger.error(f'stage4 pre-condition failed, file does not exist "{vis_mesh_path}"')

  logger.debug(f'stage4 running on "{object_folder}"')

  # --- body1: object.urdf file
  properties = get_object_properties(source_path, logger)
  properties.update(get_object_volume(watertight_mesh_path))
  properties.update(get_visual_properties(vis_mesh_path))


  asset_id, category_id, category_name = get_asset_id_and_category(object_folder)
  properties["id"] = asset_id
  urdf_str = URDF_TEMPLATE.format(**properties)
  with open(target_urdf_path, 'w') as fd:
    fd.write(urdf_str)

  # --- body2: data.json file
  asset_entry = {
      "id": asset_id,
      "asset_type": "FileBasedObject",
      "kwargs": {
          "bounds": properties["bounds"],
          "mass": properties["mass"],
          "render_filename": "visual_geometry.glb",
          "simulation_filename": "object.urdf",
      },
      "license": "https://shapenet.org/terms",
      "metadata": {
          "category": category_name,
          "category_id": category_id,
          "watertight_mesh_filename": "model_watertight.obj",
          "nr_faces": properties["nr_faces"],
          "nr_vertices": properties["nr_vertices"],
          "surface_area": properties["surface_area"],
          "volume": properties["volume"],
      }
  }

  with open(target_json_path, "w") as fd:
    json.dump(asset_entry, fd, indent=4, sort_keys=True)
  
  # --- post-condition
  if not target_urdf_path.is_file():
    logger.error(f'stage4 post-condition failed, file does not exist "{target_urdf_path}"')
  if not target_json_path.is_file():
    logger.error(f'stage4 post-condition failed, file does not exist "{target_json_path}"')

  return properties


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def stage5(object_folder: Path, logger=_DEFAULT_LOGGER):
  target_path = object_folder / 'kubric.tar.gz'

  if target_path.is_file():
    logger.debug(f'skipping stage5 on "{object_folder}"')
    return  # stage already completed; skipping
  logger.debug(f'stage5 running on "{object_folder}"')

  # --- dumps file into tar (pre-conditions auto-verified by exceptions)
  with tarfile.open(target_path, 'w:gz') as tar:
    tar.add(object_folder / 'kubric' / 'visual_geometry.glb',
            arcname='visual_geometry.glb')
    tar.add(object_folder / 'kubric' / 'collision_geometry.obj',
            arcname='collision_geometry.obj')
    tar.add(object_folder / 'kubric' / 'model_watertight.obj',
            arcname='model_watertight.obj')
    tar.add(object_folder / 'kubric' / 'object.urdf',
            arcname='object.urdf')
    tar.add(object_folder / 'kubric' / 'data.json',
            arcname='data.json')

  if not target_path.is_file():
    logger.error(f'stage5 post-condition failed, file does not exist "{target_path}"')


def stage6(object_folder: Path, logger=_DEFAULT_LOGGER):
  asset_id, category_id, category_name = get_asset_id_and_category(object_folder)
  source_path = object_folder / 'kubric.tar.gz'
  target_path = object_folder.parent.parent / 'kubric' / f'{category_id}_{asset_id}.tar.gz'

  if target_path.is_file():
    logger.debug(f'skipping stage6 on "{object_folder}"')
    return

  if not source_path.is_file():
    logger.error(f'stage6 pre-condition failed, file does not exist "{source_path}"')
    return

  logger.debug(f'stage6 running on "{object_folder}"')

  target_path.parent.mkdir(exist_ok=True)
  shutil.move(str(source_path), str(target_path))
  #shutil.rmtree(str(object_folder / 'kubric'))

  if not target_path.is_file():
    logger.error(f'stage6 post-condition failed, file does not exist "{target_path}"')

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--datadir', default='/ShapeNetCore.v2')
  parser.add_argument('--model', default='02933112/718f8fe82bc186a086d53ab0fe94e911')
  parser.add_argument('--stages', nargs='+')
  args = parser.parse_args()

  # --- setup logger (→stdout)
  logger = logging.getLogger(__name__)
  logger.setLevel(logging.DEBUG)
  handler = logging.StreamHandler(sys.stdout)
  logger.addHandler(handler)

  # --- execute functors
  object_folder = Path(args.datadir)/args.model
  stages = [int(stage) for stage in args.stages]
  if 0 in stages: stage0(object_folder, logger)
  if 1 in stages: stage1(object_folder, logger)
  if 2 in stages: stage2(object_folder, logger)
  if 3 in stages: stage3(object_folder, logger)
  if 4 in stages: properties = stage4(object_folder, logger)
  if 5 in stages: stage5(object_folder, logger)
  if 6 in stages: stage6(object_folder, logger)
