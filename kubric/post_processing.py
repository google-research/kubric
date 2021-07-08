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

import numpy as np
from typing import Sequence
from kubric.core import assets
from kubric.core import objects
from kubric.custom_types import ArrayLike


def compute_visibility(segmentation: np.ndarray, assets: Sequence[assets.Asset]):
  """Compute how many pixels are visible for each instance at each frame.

  Args:
    segmentation: An integer array that contains segmentation indices.
    assets: The list of assets in the scene (whose ordering corresponds to the segmentation indices)

  """
  for i, asset in enumerate(assets):
    asset.metadata["visibility"] = [int(np.sum(segmentation[t] == i))
                                    for t in range(segmentation.shape[0])]


def adjust_segmentation_idxs(
    segmentation: ArrayLike,
    old_assets_list: Sequence[assets.Asset],
    new_assets_list: Sequence[assets.Asset],
    ignored_label: int = 0):
  """Replaces segmentation ids with either asset.segmentation_id or the index in new_assets_list.

  Note that this starts with index=1 for the first asset in new_assets_list, to leave id=0 for
  background assets.
  """
  for i, asset in enumerate(old_assets_list):
    if isinstance(asset, objects.PhysicalObject) and asset.segmentation_id is not None:
      segmentation[segmentation == i] = asset.segmentation_id
    elif asset in new_assets_list:
      segmentation[segmentation == i] = new_assets_list.index(asset) + 1
    else:
      segmentation[segmentation == i] = ignored_label


def compute_bboxes(segmentation: ArrayLike, asset_list: Sequence[assets.Asset]):
  for k, asset in enumerate(asset_list, start=1):
    for t in range(segmentation.shape[0]):
      seg = segmentation[t, ..., 0]
      idxs = np.array(np.where(seg == k), dtype=np.float32)
      if idxs.size > 0:
        idxs /= np.array(seg.shape)[:, np.newaxis]
        asset.metadata["bboxes"].append((float(idxs[0].min()), float(idxs[1].min()),
                              float(idxs[0].max()), float(idxs[1].max())))
        asset.metadata["bbox_frames"].append(t)

