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

import numpy as np
from typing import Sequence
from kubric import core
from kubric.kubric_typing import ArrayLike


def compute_visibility(segmentation: np.ndarray, assets: Sequence[core.Asset]):
  """Compute how many pixels are visible for each instance at each frame.

  Args:
    segmentation: An integer array that contains segmentation indices.
    assets: The list of assets in the scene (whose ordering corresponds to the segmentation indices)

  """
  for i, asset in enumerate(assets, start=1):
    asset.metadata["visibility"] = [int(np.sum(segmentation[t] == i))
                                    for t in range(segmentation.shape[0])]


def adjust_segmentation_idxs(
    segmentation: ArrayLike,
    old_assets_list: Sequence[core.Asset],
    new_assets_list: Sequence[core.Asset],
    ignored_label: int = 0):
  """Replaces segmentation ids with either asset.segmentation_id or the index in new_assets_list.

  Note that this starts with index=1 for the first asset in new_assets_list, to leave id=0 for
  background assets.
  """
  new_segmentation = np.zeros_like(segmentation)
  for i, asset in enumerate(old_assets_list, start=1):
    if isinstance(asset, core.PhysicalObject) and asset.segmentation_id is not None:
      new_segmentation[segmentation == i] = asset.segmentation_id
    elif asset in new_assets_list:
      new_segmentation[segmentation == i] = new_assets_list.index(asset) + 1
    else:
      new_segmentation[segmentation == i] = ignored_label
  return new_segmentation


def compute_bboxes(segmentation: ArrayLike, asset_list: Sequence[core.Asset]):
  for k, asset in enumerate(asset_list, start=1):
    asset.metadata["bboxes"] = []
    asset.metadata["bbox_frames"] = []
    for t in range(segmentation.shape[0]):
      seg = segmentation[t, ..., 0]
      idxs = np.array(np.where(seg == k), dtype=np.float32)
      if idxs.size > 0:
        y_min = float(idxs[0].min() / seg.shape[0])
        x_min = float(idxs[1].min() / seg.shape[1])
        y_max = float((idxs[0].max() + 1) / seg.shape[0])
        x_max = float((idxs[1].max() + 1) / seg.shape[1])
        asset.metadata["bboxes"].append((y_min, x_min, y_max, x_max))
        asset.metadata["bbox_frames"].append(t)

