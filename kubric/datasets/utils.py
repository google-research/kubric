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
import tensorflow_datasets.public_api as tfds
from typing import List, Optional


def get_flow_magnitude_range(flow):
  flow_magnitude = np.linalg.norm(flow, axis=-1)
  return np.min(flow_magnitude), np.max(flow_magnitude)


def get_total_flow_magnitude_range(vec):
  fwd_flow_min, fwd_flow_max = get_flow_magnitude_range(vec[:, :, :2])
  bwd_flow_min, bwd_flow_max = get_flow_magnitude_range(vec[:, :, 2:])
  return min(fwd_flow_min, bwd_flow_min), max(fwd_flow_max, bwd_flow_max)


def get_bboxes(seg_frames: List[np.ndarray], num_obj: Optional[int] = None):
  for t, seg in enumerate(seg_frames):
    if seg.ndim == 3:
      seg = seg[:, :, 0]
    assert seg.ndim == 2
    bboxes = []
    idxs = np.array(np.where(seg == obj_idx), dtype=np.float32)
    if idxs.size > 0:
      mins = idxs.min(axis=1) / seg.shape
      maxs = idxs.min(axis=1) / seg.shape
      bbox = tfds.features.BBox(ymin=float(mins[0]), xmin=float(mins[1]),
                                ymax=float(maxs[0]), xmax=float(maxs[1]))
    else:
      bbox = tfds.features.BBox(ymin=0., xmin=0., ymax=0., xmax=0.)
    bboxes.append(bbox)
  return bboxes
