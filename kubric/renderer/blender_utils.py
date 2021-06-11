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
import sys

import bpy

from kubric import core
from kubric.custom_types import AddAssetFunction
from kubric.redirect_io import RedirectStream


def clear_and_reset_blender_scene(verbose=False):
  """ Resets Blender to an entirely empty scene."""
  with RedirectStream(stream=sys.stdout, disabled=verbose):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.world = bpy.data.worlds.new("World")


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


def set_up_exr_output_node(layers=("Image", "Depth", "UV", "Normal", "CryptoObject00")):
  """ Set up the blender compositor nodes required for exporting EXR files.

  The filename can then be set with:
  out_node.base_path = "my/custom/path/prefix_"

  """
  bpy.context.scene.use_nodes = True
  tree = bpy.context.scene.node_tree
  links = tree.links

  # clear existing nodes
  for node in tree.nodes:
    tree.nodes.remove(node)

  # the render node has outputs for all the rendered layers
  render_node = tree.nodes.new(type="CompositorNodeRLayers")

  # create a new FileOutput node
  out_node = tree.nodes.new(type="CompositorNodeOutputFile")
  # set the format to EXR (multilayer)
  out_node.format.file_format = "OPEN_EXR_MULTILAYER"

  out_node.file_slots.clear()
  for layer_name in layers:
    out_node.file_slots.new(layer_name)
    links.new(render_node.outputs.get(layer_name), out_node.inputs.get(layer_name))

  # manually convert to RGBA. See:
  # https://blender.stackexchange.com/questions/175621/incorrect-vector-pass-output-no-alpha-zero-values/175646#175646
  split_rgba = tree.nodes.new(type="CompositorNodeSepRGBA")
  combine_rgba = tree.nodes.new(type="CompositorNodeCombRGBA")
  for channel in "RGBA":
    links.new(split_rgba.outputs.get(channel), combine_rgba.inputs.get(channel))
  out_node.file_slots.new("Vector")
  links.new(render_node.outputs.get("Vector"), split_rgba.inputs.get("Image"))
  links.new(combine_rgba.outputs.get("Image"), out_node.inputs.get("Vector"))

  return out_node


def activate_render_passes(normal: bool = True, optical_flow: bool = True,
                           segmentation: bool = True, uv: bool = True):
    view_layer = bpy.context.scene.view_layers[0]
    view_layer.use_pass_vector = optical_flow
    view_layer.use_pass_uv = uv
    view_layer.use_pass_normal = normal  # surface normals
    view_layer.use_pass_cryptomatte_object = segmentation
    if segmentation:
      view_layer.pass_cryptomatte_depth = 2
