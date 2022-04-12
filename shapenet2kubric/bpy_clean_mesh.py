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
from kubric.safeimport.bpy import bpy


def cleanup_mesh(asset_id: str, source_path: str, target_path: str):
  # start from a clean slate
  bpy.ops.wm.read_factory_settings(use_empty=True)
  bpy.context.scene.world = bpy.data.worlds.new("World")

  # import source mesh
  bpy.ops.import_scene.gltf(filepath=source_path, loglevel=50)

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

  # export cleaned up mesh
  bpy.ops.export_scene.gltf(filepath=str(target_path), check_existing=True)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--source_path', type=str)
  parser.add_argument('--target_path', type=str)
  parser.add_argument('--asset_id', type=str)
  args = parser.parse_args()
  cleanup_mesh(asset_id=args.asset_id,
               source_path=args.source_path,
               target_path=args.target_path)
