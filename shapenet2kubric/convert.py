# pylint: disable=logging-fstring-interpolation
# see: https://docs.python.org/3/library/subprocess.html

import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys
import tarfile

import bpy
import pybullet as pb

from redirect_io import RedirectStream  # < duplicate?
from trimesh_utils import get_object_properties
from urdf_template import URDF_TEMPLATE

_DEFAULT_LOGGER = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def stage0(object_folder: Path, logger=_DEFAULT_LOGGER):
  logger.debug(f'stage0 running on "{object_folder}"')
  source_path = object_folder / 'models' / 'model_normalized.obj'
  target_path = object_folder / 'kubric' / 'visual_geometry.glb'
  
  # --- pre-condition
  if not source_path.is_file():
    logger.error(f'stage0 pre-condition failed, file does not exist "{source_path}"')

  # --- body
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
  logger.debug(f'stage1 running on "{object_folder}"')
  source_path = object_folder / 'models' / 'model_normalized.obj'
  target_path = object_folder / 'kubric' / 'model_watertight.obj'

  # --- pre-condition
  if not source_path.is_file():
    logger.error(f'stage1 pre-condition failed, file does not exist "{source_path}"')

  # --- body
  cmd = f'manifold --input {source_path} --output {target_path}'
  retobj = subprocess.run(cmd, capture_output=True, shell=True, text=True)
  if retobj.returncode != 0:
    logger.error(f'obj2gltf failed on "f{object_folder}"')
    if retobj.stdout != '': logger.error(f'{retobj.stdout}')
    if retobj.stderr != '': logger.error(f'{retobj.stderr}')

  # --- post-condition
  if not target_path.is_file():
    logger.error(f'stage1 post-condition failed, file does not exist "{target_path}"')


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
    
def stage2(object_folder: Path, logger=_DEFAULT_LOGGER):
  logger.debug(f'stage2 running on "{object_folder}"')
  source_path = object_folder / 'kubric' / 'model_watertight.obj'
  target_path = object_folder / 'kubric' / 'collision_geometry.obj'
  log_path = object_folder / 'kubric' / 'stage2_logs.txt'
  redirect_log_path = str(object_folder / 'kubric' / 'stage2_stdout.txt')

  # --- pre-condition
  if not source_path.is_file():
    logger.error(f'stage2 pre-condition failed, file does not exist "{source_path}"')

  # --- body
  # TODO: how to monitor errors? should we move to "raw" VHCD?
  with RedirectStream(stream=sys.stdout, filename=str(log_path)):
    pb.vhacd(str(source_path), str(target_path), str(redirect_log_path))

  # --- post-condition
  if not target_path.is_file():
    logger.error(f'stage2 post-condition failed, file does not exist "{target_path}"')


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def stage3(object_folder: Path, logger=_DEFAULT_LOGGER):
  # TODO: we should probably use a mixture of model_normalized and model_wateright here?

  logger.debug(f'stage3 running on "{object_folder}"')
  source_path = object_folder / 'kubric' / 'collision_geometry.obj'
  target_urdf_path = object_folder / 'kubric' / 'object.urdf'
  target_json_path = object_folder / 'kubric' / 'data.json'

  # --- pre-condition
  if not source_path.is_file():
    logger.error(f'stage3 pre-condition failed, file does not exist "{source_path}"')

  # --- body1: object.urdf file
  properties = get_object_properties(source_path)
  properties['id'] = str(object_folder.relative_to(object_folder.parent.parent))
  properties['density'] = 1
  properties['friction'] = .5
  urdf_str = URDF_TEMPLATE.format(**properties)
  with open(target_urdf_path, 'w') as fd:
    fd.write(urdf_str)

  # --- body2: data.json file
  properties['paths'] = {
    'visual_geometry': 'visual_geometry.glb',
    'collision_geometry': 'collision_geometry.obj',
    'urdf': 'object.urdf',
  }
  with open(target_json_path, "w") as fd:
    json.dump(properties, fd, indent=4, sort_keys=True)
  
  # --- post-condition
  if not target_urdf_path.is_file():
    logger.error(f'stage3 post-condition failed, file does not exist "{target_urdf_path}"')
  if not target_json_path.is_file():
    logger.error(f'stage3 post-condition failed, file does not exist "{target_json_path}"')

  return properties


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def stage35(object_folder: Path, logger=_DEFAULT_LOGGER):
  logger.debug(f'stage3.5 running on "{object_folder}"')

  source_path = object_folder / 'kubric' / 'visual_geometry.glb'
  source_backup_path = object_folder / 'kubric' / 'visual_geometry_bak.glb'
  log_path = object_folder / 'kubric' / 'stage3.5_logs.txt'
  target_path = object_folder / 'kubric' / 'visual_geometry.glb'

  asset_id = str(object_folder.relative_to(object_folder.parent.parent))

  bpy.ops.wm.read_factory_settings(use_empty=True)
  bpy.context.scene.world = bpy.data.worlds.new("World")

  with RedirectStream(stream=sys.stdout, filename=str(log_path)):
    bpy.ops.import_scene.gltf(filepath=str(source_path), loglevel=50)

    bpy.ops.object.select_all(action='DESELECT')

    for obj in bpy.data.objects:
      # remove duplicate vertices
      bpy.context.view_layer.objects.active = obj
      bpy.ops.object.mode_set(mode='EDIT')
      bpy.ops.mesh.remove_doubles(threshold=1e-06)
      bpy.ops.object.mode_set(mode='OBJECT')
      # disable auto-smoothing
      obj.data.use_auto_smooth = False
      # split edges with an angle above 70 degrees (1.22 radians)
      m = obj.modifiers.new("EdgeSplit", "EDGE_SPLIT")
      m.split_angle = 1.22173
      bpy.ops.object.modifier_apply(modifier="EdgeSplit")
      # move every face an epsilon in the direction of its normal, to reduce clipping artifacts
      m = obj.modifiers.new("Displace", "DISPLACE")
      m.strength = 0.00001
      bpy.ops.object.modifier_apply(modifier="Displace")

    # join all objects together
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.join()

    # set the name of the asset
    bpy.context.active_object.name = asset_id

    # rename the source file
    source_path.rename(source_backup_path)

    # store new visual geometry
    bpy.ops.export_scene.gltf(filepath=str(target_path), check_existing=True)

  if not target_path.is_file():
    logger.error(f'stage3 post-condition failed, file does not exist "{target_path}"')
  if not source_backup_path.is_file():
    logger.error(f'stage3 post-condition failed, file does not exist "{source_backup_path}"')


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def stage4(object_folder: Path, logger=_DEFAULT_LOGGER):
  logger.debug(f'stage4 running on "{object_folder}"')
  target_path = object_folder / 'kubric.tar.gz'

  # --- dumps file into tar (pre-conditions auto-verified by exceptions)
  with tarfile.open(target_path, 'w:gz') as tar:
    tar.add(object_folder / 'kubric' / 'visual_geometry.glb')
    tar.add(object_folder / 'kubric' / 'collision_geometry.obj')
    tar.add(object_folder / 'kubric' / 'object.urdf')
    tar.add(object_folder / 'kubric' / 'data.json') 






# TODO: cleanup
# def stage5():
#   import shutil
#   shutil.rmtree(str(object_folder / 'kubric'))

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
  if 3 in stages: properties = stage3(object_folder, logger)
  if 35 in stages: stage35(object_folder, logger)
  if 4 in stages: stage4(object_folder, logger)
