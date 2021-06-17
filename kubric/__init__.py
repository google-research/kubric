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

from kubric.core.materials import Material
from kubric.core.materials import UndefinedMaterial
from kubric.core.materials import PrincipledBSDFMaterial
from kubric.core.materials import FlatMaterial

from kubric.core.objects import Object3D
from kubric.core.objects import PhysicalObject
from kubric.core.objects import Sphere
from kubric.core.objects import Cube
from kubric.core.objects import FileBasedObject

from kubric.core.traits import Vector3D
from kubric.core.traits import Scale
from kubric.core.traits import Quaternion
from kubric.core.traits import RGB
from kubric.core.traits import RGBA
from kubric.core.traits import AssetInstance

from kubric.custom_types import AddAssetFunction
from kubric.custom_types import PathLike

from kubric import assets
from kubric.assets import AssetSource
from kubric.assets import TextureSource

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
from kubric.utils import str2path
from kubric.version import __version__


# TODO: remove and add a test that checks pathlib should NOT be imported?
from tensorflow_datasets.core.utils.generic_path import as_path
