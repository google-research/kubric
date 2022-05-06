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

"""Root of the kubric module."""

# --- auto-computed by setup.py, source version is always at HEAD
__version__ = "HEAD"

# --- basic kubric types
from pyquaternion import Quaternion

from kubric.core.scene import Scene

from kubric.core.assets import Asset
from kubric.core.assets import UndefinedAsset

from kubric.core.cameras import Camera
from kubric.core.cameras import UndefinedCamera
from kubric.core.cameras import PerspectiveCamera
from kubric.core.cameras import OrthographicCamera

from kubric.core.color import Color
from kubric.core.color import get_color

from kubric.core.lights import Light
from kubric.core.lights import UndefinedLight
from kubric.core.lights import DirectionalLight
from kubric.core.lights import PointLight
from kubric.core.lights import RectAreaLight
from kubric.core.lights import SpotLight

from kubric.core.materials import Material
from kubric.core.materials import UndefinedMaterial
from kubric.core.materials import PrincipledBSDFMaterial
from kubric.core.materials import FlatMaterial
from kubric.core.materials import Texture

from kubric.core.objects import Object3D
from kubric.core.objects import PhysicalObject
from kubric.core.objects import Sphere
from kubric.core.objects import Cube
from kubric.core.objects import FileBasedObject

from kubric.kubric_typing import AddAssetFunction
from kubric.kubric_typing import PathLike

from kubric import assets
from kubric.assets import AssetSource

from kubric.randomness import random_hue_color
from kubric.randomness import random_rotation
from kubric.randomness import rotation_sampler
from kubric.randomness import position_sampler
from kubric.randomness import resample_while
from kubric.randomness import move_until_no_overlap
from kubric.randomness import sample_point_in_half_sphere_shell

from kubric.post_processing import compute_visibility
from kubric.post_processing import compute_bboxes
from kubric.post_processing import adjust_segmentation_idxs

from kubric.file_io import as_path
from kubric.file_io import write_pkl
from kubric.file_io import write_json
from kubric.file_io import write_png
from kubric.file_io import write_palette_png
from kubric.file_io import write_scaled_png
from kubric.file_io import write_tiff
from kubric.file_io import write_image_dict
from kubric.file_io import read_png
from kubric.file_io import read_tiff

from kubric.utils import ArgumentParser
from kubric.utils import done
from kubric.utils import get_camera_info
from kubric.utils import get_instance_info
from kubric.utils import get_scene_metadata
from kubric.utils import log_my_flags
from kubric.utils import process_collisions
from kubric.utils import setup
from kubric.utils import setup_directories
from kubric.utils import setup_logging
