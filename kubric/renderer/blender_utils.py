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
from typing import Dict, Sequence

import numpy as np
import OpenEXR
import Imath
import sklearn.utils

from kubric import core
from kubric.kubric_typing import AddAssetFunction, ArrayLike
from kubric.redirect_io import RedirectStream
from kubric.safeimport.bpy import bpy


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


def set_up_exr_output_node(default_layers=("Image", "Depth"),
                           aux_layers=("UV", "Normal", "CryptoObject00", "ObjectCoordinates")):
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
  render_node_aux = tree.nodes.new(type="CompositorNodeRLayers")
  render_node_aux.name = "Render Layers Aux"
  render_node_aux.layer = "AuxOutputs"

  # create a new FileOutput node
  out_node = tree.nodes.new(type="CompositorNodeOutputFile")
  # set the format to EXR (multilayer)
  out_node.format.file_format = "OPEN_EXR_MULTILAYER"

  out_node.file_slots.clear()
  for layer_name in default_layers:
    out_node.file_slots.new(layer_name)
    links.new(render_node.outputs.get(layer_name), out_node.inputs.get(layer_name))

  for layer_name in aux_layers:
    out_node.file_slots.new(layer_name)
    links.new(render_node_aux.outputs.get(layer_name), out_node.inputs.get(layer_name))

  # manually convert to RGBA. See:
  # https://blender.stackexchange.com/questions/175621/incorrect-vector-pass-output-no-alpha-zero-values/175646#175646
  split_rgba = tree.nodes.new(type="CompositorNodeSepRGBA")
  combine_rgba = tree.nodes.new(type="CompositorNodeCombRGBA")
  for channel in "RGBA":
    links.new(split_rgba.outputs.get(channel), combine_rgba.inputs.get(channel))
  out_node.file_slots.new("Vector")
  links.new(render_node_aux.outputs.get("Vector"), split_rgba.inputs.get("Image"))
  links.new(combine_rgba.outputs.get("Image"), out_node.inputs.get("Vector"))

  return out_node


def add_coordinate_material():
  """Create a special material for generating object-coordinates as a separate output pass."""
  mat = bpy.data.materials.new("KubricObjectCoordinatesOverride")

  mat.use_nodes = True
  bsdf_node = mat.node_tree.nodes["Principled BSDF"]
  mat.node_tree.nodes.remove(bsdf_node)
  out_node = mat.node_tree.nodes["Material Output"]
  mat.node_tree.nodes.remove(out_node)

  tex_coordinates = mat.node_tree.nodes.new(type="ShaderNodeTexCoord")
  aov_out_node = mat.node_tree.nodes.new(type="ShaderNodeOutputAOV")
  aov_out_node.name = "ObjectCoordinates"
  unused_mat_out_node = mat.node_tree.nodes.new(type="ShaderNodeOutputMaterial")

  mat.node_tree.links.new(tex_coordinates.outputs.get("Generated"),
                          aov_out_node.inputs.get("Color"))
  mat.node_tree.links.new(tex_coordinates.outputs.get("Generated"),
                          unused_mat_out_node.inputs.get("Surface"))

  return mat


def activate_render_passes(normal: bool = True,
                           optical_flow: bool = True,
                           segmentation: bool = True,
                           uv: bool = True):
  # We use two separate view layers
  # 1) the default view layer renders the image and uses many samples per pixel
  # 2) the aux view layer uses only 1 sample per pixel to avoid anti-aliasing
  default_view_layer = bpy.context.scene.view_layers[0]
  aux_view_layer = bpy.context.scene.view_layers.new("AuxOutputs")
  aux_view_layer.samples = 1  # only use 1 ray per pixel to disable anti-aliasing
  aux_view_layer.use_pass_z = False  # no need for a separate z-pass
  aux_view_layer.material_override = add_coordinate_material()
  object_coords_aov = aux_view_layer.aovs.add()
  object_coords_aov.name = "ObjectCoordinates"
  aux_view_layer.cycles.use_denoising = False

  # For optical flow, uv, and normals we use the aux view layer
  aux_view_layer.use_pass_vector = optical_flow
  aux_view_layer.use_pass_uv = uv
  aux_view_layer.use_pass_normal = normal  # surface normals
  # We use the default view layer for segmentation, so that we can get
  # anti-aliased crypto-matte
  if bpy.app.version >= (2, 93, 0):
    aux_view_layer.use_pass_cryptomatte_object = segmentation
    if segmentation:
      aux_view_layer.pass_cryptomatte_depth = 2
  else:
    aux_view_layer.cycles.use_pass_crypto_object = segmentation
    if segmentation:
      aux_view_layer.cycles.pass_crypto_depth = 2


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
  return np.stack(outputs, axis=-1)


def get_render_layers_from_exr(filename) -> Dict[str, np.ndarray]:
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
    # Blender exports forward and backward flow in a single image,
    # and uses (-delta_col, delta_row) format, but we prefer (delta_row, delta_col)
    # also use normalized coords that range from (0, 0) at the top left to (1, 1) at bottom right
    output["backward_flow"] = np.zeros_like(flow[..., :2])
    output["backward_flow"][..., 0] = flow[..., 1] / flow.shape[-3]
    output["backward_flow"][..., 1] = -flow[..., 0] / flow.shape[-2]

    output["forward_flow"] = np.zeros_like(flow[..., 2:])
    output["forward_flow"][..., 0] = flow[..., 3] / flow.shape[-3]
    output["forward_flow"][..., 1] = -flow[..., 2] / flow.shape[-2]

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
  if "ObjectCoordinates" in layer_names:
    output["object_coordinates"] = read_channels_from_exr(exr, ["ObjectCoordinates.R", "ObjectCoordinates.G", "ObjectCoordinates.B"])
  return output


def replace_cryptomatte_hashes_by_asset_index(
    segmentation_ids: ArrayLike,
    assets: Sequence[core.assets.Asset]):
  """Replace (inplace) the cryptomatte hash (from Blender) by the index of each asset + 1.
  (the +1 is to ensure that the 0 for background does not interfere with asset index 0)

  Args:
    segmentation_ids: Segmentation array of cryptomatte hashes as returned by Blender.
    assets: List of assets to use for replacement.
  """
  # replace crypto-ids with asset index
  new_segmentation_ids = np.zeros_like(segmentation_ids)
  for idx, asset in enumerate(assets, start=1):
    asset_hash = mm3hash(asset.uid)
    new_segmentation_ids[segmentation_ids == asset_hash] = idx
  return new_segmentation_ids


def mm3hash(name):
  """ Compute the uint32 hash that Blenders Cryptomatte uses.
  https://github.com/Psyop/Cryptomatte/blob/master/specification/cryptomatte_specification.pdf
  """
  hash_32 = sklearn.utils.murmurhash3_32(name, positive=True)
  exp = hash_32 >> 23 & 255
  if exp in (0, 255):
    hash_32 ^= 1 << 23
  return hash_32
