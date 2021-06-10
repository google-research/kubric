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

import logging
import numpy as np
import OpenEXR
import Imath
from typing import Dict, Sequence
import kubric.assets


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


def get_render_layers_from_exr(filename, background_objects=(), objects=()) -> Dict[str, np.ndarray]:
  exr = OpenEXR.InputFile(str(filename))
  layer_names = set()
  for n, v in exr.header()["channels"].items():
    layer_name, _,  channel_name = n.partition(".")
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
        labelmap[kubric.assets.mm3hash(asset.uid)] = asset.segmentation_id
      else:
        labelmap[kubric.assets.mm3hash(asset.uid)] = idx + 1
    # All background images are assigned to 0.
    for asset in background_objects:
      labelmap[kubric.assets.mm3hash(asset.uid)] = 0
    logging.info("The labelmap is ", labelmap)

    bg_ids = [kubric.assets.mm3hash(obj.uid) for obj in background_objects]
    object_ids = [kubric.assets.mm3hash(obj.uid) for obj in objects]
    for bg_id in bg_ids:
      idxs[idxs == bg_id] = labelmap[bg_id]  # assign 0 to all background objects
    for i, object_id in enumerate(object_ids):
      idxs[idxs == object_id] = labelmap[object_id]

  return output

def compute_bboxes(segmentation):
  instances = []
  for k in range(1, np.max(segmentation)+1):
    obj = {
        "bboxes": [],
        "bbox_frames": [],
    }
    for t in range(segmentation.shape[0]):
      seg = segmentation[t, ..., 0]
      idxs = np.array(np.where(seg == k), dtype=np.float32)
      if idxs.size > 0:
        idxs /= np.array(seg.shape)[:, np.newaxis]
        obj["bboxes"].append((float(idxs[0].min()), float(idxs[1].min()),
                              float(idxs[0].max()), float(idxs[1].max())))
        obj["bbox_frames"].append(t)

    instances.append(obj)
  return instances
