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
import logging
import sys
from typing import Dict, Sequence

import bpy
import numpy as np
import OpenEXR
import Imath
import sklearn.utils

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


def activate_render_passes(normal: bool = True,
                           optical_flow: bool = True,
                           segmentation: bool = True,
                           uv: bool = True):
  view_layer = bpy.context.scene.view_layers[0]
  view_layer.use_pass_vector = optical_flow
  view_layer.use_pass_uv = uv
  view_layer.use_pass_normal = normal  # surface normals
  if bpy.app.version >= (2, 93, 0):
    view_layer.use_pass_cryptomatte_object = segmentation
    if segmentation:
      view_layer.pass_cryptomatte_depth = 2
  else:
    view_layer.cycles.use_pass_crypto_object = segmentation
    if segmentation:
      view_layer.cycles.pass_crypto_depth = 2


def read_channels_from_exr(exr: OpenEXR.InputFile, channel_names: Sequence[str]) -> np.ndarray:
  """Reads a single channel from an EXR file and returns it as a numpy array."""
  channels_header = exr.header()["channels"]
  window = exr.header()["dataWindow"]
  width = window.max.x - window.min.x + 1
  height = window.max.y - window.min.y + 1
  outputs = []
  for channel_name in channel_names:
    channel_type = channels_header[channel_name].type.v
    numpy_type = {
        Imath.PixelType.HALF: np.float16,
        Imath.PixelType.FLOAT: np.float32,
        Imath.PixelType.UINT: np.uint32,
    }[channel_type]
    array = np.frombuffer(exr.channel(channel_name), numpy_type)
    array = array.reshape([height, width])
    outputs.append(array)
  # TODO: verify that the types are all the same?
  return np.stack(outputs, axis=-1)


def get_render_layers_from_exr(filename,
                               background_objects=(),
                               objects=()) -> Dict[str, np.ndarray]:
  exr = OpenEXR.InputFile(str(filename))
  layer_names = set()
  for n, _ in exr.header()["channels"].items():
    layer_name, _, _ = n.partition(".")
    layer_names.add(layer_name)

  output = {}
  if "Image" in layer_names:
    # Image is in RGBA format with range [0, inf]
    # TODO: image is in HDR, so we need some tone-mapping
    output["rgba"] = read_channels_from_exr(exr, ["Image.R", "Image.G", "Image.B", "Image.A"])
  if "Depth" in layer_names:
    # range [0, 10000000000.0]  # the value 1e10 is used for background / infinity
    # TODO: clip to a reasonable value. Is measured in meters so usual range is ~ [0, 10]
    output["depth"] = read_channels_from_exr(exr, ["Depth.V"])
  if "Vector" in layer_names:
    flow = read_channels_from_exr(exr, ["Vector.R", "Vector.G", "Vector.B", "Vector.A"])
    output["backward_flow"] = flow[..., :2]
    output["forward_flow"] = flow[..., 2:]
  if "Normal" in layer_names:
    # range: [-1, 1]
    data = read_channels_from_exr(exr, ["Normal.X", "Normal.Y", "Normal.Z"])
    output["normal"] = ((data + 1) * 65535 / 2).astype(np.uint16)
  if "UV" in layer_names:
    # range [0, 1]
    data = read_channels_from_exr(exr, ["UV.X", "UV.Y", "UV.Z"])
    output["uv"] = (data * 65535).astype(np.uint16)
  if "CryptoObject00" in layer_names:
    # CryptoMatte stores the segmentation of Objects using two kinds of channels:
    #  - index channels (uint32) specify the object index for a pixel
    #  - alpha channels (float32) specify the corresponding mask value
    # there may be many cryptomatte layers, which allows encoding a pixel as belonging to multiple
    # objects at once (up to a maximum of # of layers many objects per pixel)
    # In the EXR this is stored with 2 layers per RGBA image  (CryptoObject00, CryptoObject01, ...)
    # with RG being the first layer and BA being the second
    # So the R and B channels are uint32 and the G and A channels are float32.
    crypto_layers = [n for n in layer_names if n.startswith("CryptoObject")]
    index_channels = [n + "." + c for n in crypto_layers for c in "RB"]
    idxs = read_channels_from_exr(exr, index_channels)
    idxs.dtype = np.uint32
    output["segmentation_indices"] = idxs
    alpha_channels = [n + "." + c for n in crypto_layers for c in "GA"]
    alphas = read_channels_from_exr(exr, alpha_channels)
    output["segmentation_alphas"] = alphas
    # replace crypto-ids with object index for foreground objects and 0 for background objects.
    labelmap = {}
    # Foreground objects: Set the label id to either asset.segmentation_id
    # if it is present, or index + 1 otherwise.
    for idx, asset in enumerate(objects):
      if asset.segmentation_id is not None:
        labelmap[mm3hash(asset.uid)] = asset.segmentation_id
      else:
        labelmap[mm3hash(asset.uid)] = idx + 1
    # All background images are assigned to 0.
    for asset in background_objects:
      labelmap[mm3hash(asset.uid)] = 0
    logging.info("The labelmap is '%s'", labelmap)  # TODO(klausg): check %s appropriate here?

    bg_ids = [mm3hash(obj.uid) for obj in background_objects]
    object_ids = [mm3hash(obj.uid) for obj in objects]
    for bg_id in bg_ids:
      idxs[idxs == bg_id] = labelmap[bg_id]  # assign 0 to all background objects
    for _, object_id in enumerate(object_ids):
      idxs[idxs == object_id] = labelmap[object_id]

  return output


def mm3hash(name):
  """ Compute the uint32 hash that Blenders Cryptomatte uses.
  https://github.com/Psyop/Cryptomatte/blob/master/specification/cryptomatte_specification.pdf
  """
  hash_32 = sklearn.utils.murmurhash3_32(name, positive=True)
  exp = hash_32 >> 23 & 255
  if exp in (0, 255):
    hash_32 ^= 1 << 23
  return hash_32
