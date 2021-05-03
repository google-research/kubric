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

"""Root of the kubric module."""

from kubric.core import *

from kubric.core.color import Color, get_color
from kubric.renderer import Blender
from kubric.simulator import PyBullet
from kubric.post_processing import get_render_layers_from_exr

from kubric import assets
from kubric.assets import AssetSource, TextureSource
from kubric.assets.utils import mm3hash

from kubric.randomness import random_hue_color
from kubric.randomness import random_rotation
from kubric.randomness import rotation_sampler
from kubric.randomness import position_sampler
from kubric.randomness import resample_while
from kubric.randomness import move_until_no_overlap

from kubric.utils import ArgumentParser
from kubric.utils import setup_logging
from kubric.utils import log_my_flags
from kubric.utils import setup_directories
from kubric.utils import get_scene_metadata
from kubric.utils import get_instance_info
from kubric.utils import get_camera_info
from kubric.utils import process_collisions
from kubric.utils import save_as_pkl
from kubric.utils import save_as_json
from kubric.utils import done

from tensorflow_datasets.core.utils.generic_path import as_path


from kubric.version import __version__
