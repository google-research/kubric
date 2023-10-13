# Copyright 2023 The Kubric Authors.
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

"""Render mock utils."""

from __future__ import annotations

import contextlib
import functools
from typing import Iterator, Optional, Sequence
from unittest import mock

import numpy as np


def _make_array(
    *,
    batch_shape: tuple[int, ...],
    last_dim: int,
    dtype=np.float32,
) -> np.ndarray:
  """Creates a dummy array."""
  return np.zeros(shape=batch_shape + (last_dim,), dtype=dtype)


_RENDERING_LAYERS = {
    'rgba': functools.partial(_make_array, last_dim=4, dtype=np.uint8),
    'segmentation': functools.partial(_make_array, last_dim=1, dtype=np.int32),
    'backward_flow': functools.partial(_make_array, last_dim=2),
    'forward_flow': functools.partial(_make_array, last_dim=2),
    'depth': functools.partial(_make_array, last_dim=1),
    'uv': functools.partial(_make_array, last_dim=3),
    'normal': functools.partial(_make_array, last_dim=3),
}


@contextlib.contextmanager
def mock_render(num_frames: int = 1) -> Iterator[None]:
  """Mocked Blender rendering.

  Args:
    num_frames: The number of frames per scene to render.

  Yields:
    None
  """

  # Use closure instead of `functools.partial` because `partial` does not
  # implement the bound method descriptor.

  def render_fn(*args, **kwargs):
    return _render(*args, **kwargs, num_frames=num_frames)

  with mock.patch('kubric.renderer.blender.Blender.render', render_fn):
    yield


def _render(
    self,
    frames: Optional[Sequence[int]] = None,
    *,
    num_frames: int,
    **kwargs,
) -> dict[str, np.ndarray]:
  """Mocked render."""
  del frames
  batch_shape = (num_frames, *self.scene.resolution)
  return {
      k: make_array_fn(batch_shape=batch_shape)
      for k, make_array_fn in _RENDERING_LAYERS.items()
  }
