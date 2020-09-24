# Copyright 2020 The Kubric Authors
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

import numpy as np
import OpenEXR
import Imath
from typing import Dict, Sequence


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


def get_render_layers_from_exr(filename) -> Dict[str, np.ndarray]:
  exr = OpenEXR.InputFile(str(filename))
  layer_names = set()
  for n, v in exr.header()["channels"].items():
    layer_name, _,  channel_name = n.partition(".")
    layer_names.add(layer_name)

  output = {}
  if "Image" in layer_names:
    # Image is in RGBA format, but we"ll ignore the alpha value
    # range [0, inf]  # TODO: image is in HDR, so we need some tone-mapping
    output["Image"] = read_channels_from_exr(exr, ["Image.R", "Image.G", "Image.B"])
  if "Depth" in layer_names:
    # range [0, 10000000000.0]  # the value 1e10 is used for background / infinity
    # TODO: clip to a reasonable value. Is measured in meters so usual range is ~ [0, 10]
    output["Depth"] = read_channels_from_exr(exr, ["Depth.V"])
  if "Vector" in layer_names:
    # TODO: The vector output of Blender is supposed to have 4 components, but has only 3 (XYZ)
    #       The first two should be movement wrt. previous frame, the other two wrt. next frame.
    #       We"ll focus on the first two for now, no idea what the 3rd is about.
    output["Vector"] = read_channels_from_exr(exr, ["Vector.X", "Vector.Y"])
  if "Normal" in layer_names:
    # range: [-1, 1]
    output["Normal"] = read_channels_from_exr(exr, ["Normal.X", "Normal.Y", "Normal.Z"])
  if "UV" in layer_names:
    # range [0, 1]
    output["UV"] = read_channels_from_exr(exr, ["UV.X", "UV.Y", "UV.Z"])
  if "CryptoObject00" in layer_names:
    # CryptoMatte stores the segmentation of Objects using two kinds of channels:
    #  - index channels (uint32) specify the object index for a pixel
    #  - alpha channels (float32) specify the corresponding mask value
    # there may be many cryptomatte layers, which allows encoding a pixel as belonging to multiple objects at once
    # (up to a maximum of # of layers many objects per pixel)
    # In the EXR this is stored with 2 layers per RGBA image  (CryptoObject00, CryptoObject01, ...)
    # with RG being the first layer and BA being the second
    # So the R and B channels are uint32 and the G and A channels are float32.
    crypto_layers = [n for n in layer_names if n.startswith("CryptoObject")]
    index_channels = [n + "." + c for n in crypto_layers for c in "RB"]
    idxs = read_channels_from_exr(exr, index_channels)
    idxs.dtype = np.uint32
    output["SegmentationIndex"] = idxs
    alpha_channels = [n + "." + c for n in crypto_layers for c in "GA"]
    alphas = read_channels_from_exr(exr, alpha_channels)
    output["SegmentationAlpha"] = alphas
  return output

