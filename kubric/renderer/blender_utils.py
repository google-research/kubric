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

import functools
from typing import Any, Callable
from kubric import core

import bpy

AddAssetFunction = Callable[[core.View, core.Asset], Any]


def prepare_blender_object(func: AddAssetFunction) -> AddAssetFunction:
  """ Decorator for add_asset methods that takes care of a few Blender specific settings.

  For each asset it:
    - sets the Blender name to the UID of the asset
    - sets the rotation mode to "QUATERNION" (if possible)
    - links it to the current scene collection
  """
  @functools.wraps(func)
  def _func(self, asset: core.Asset):
    blender_obj = func(self, asset)  # create the new blender object
    blender_obj.name = asset.uid  # link the name of the object to the UID
    # if it has a rotation mode, then make sure it is set to quaternions
    if hasattr(blender_obj, "rotation_mode"):
      blender_obj.rotation_mode = "QUATERNION"
    # if object is an actual Object (eg. not a Scene, or a Material)
    # then ensure that it is linked into (used by) the current scene collection
    if isinstance(blender_obj, bpy.types.Object):
      collection = bpy.context.scene.collection.objects
      if blender_obj not in collection.values():
        collection.link(blender_obj)

    return blender_obj

  return _func

