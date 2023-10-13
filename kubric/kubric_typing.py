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

"""Kubric type annotations."""

from typing import Any, Callable, Union, Sequence
from etils import epath
import numpy as np
import pyquaternion as pyquat

from kubric import core  # pylint: disable=unused-import

AddAssetFunction = Callable[["core.View", "core.Asset"], Any]

PathLike = Union[str, epath.Path]

ArrayLike = Union[Sequence[float], np.ndarray]

Quaternion = pyquat.Quaternion
