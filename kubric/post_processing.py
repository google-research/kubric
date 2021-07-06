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
from typing import List


def compute_visibility(segmentation: np.ndarray) -> List[List[int]]:
  """Compute how many pixels are visible for each instance at each frame."""
  instances = []
  for k in range(1, np.max(segmentation)+1):
      visibility = [int(np.sum(segmentation[t, ..., 0] == k))
                    for t in range(segmentation.shape[0])]
      instances.append(visibility)
  return instances


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
