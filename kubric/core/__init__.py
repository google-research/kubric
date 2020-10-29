# Copyright 2020 The Kubric Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" This package defines the basic object hierarchy that forms the center of Kubrics interface.

The root classes are Scene and Asset, which further specializes into:
 * Material
   - PrincipledBSDFMaterial
   - FlatMaterial
 * Object3D
   - PhysicalObject
     > FileBasedObject
     > Cube
     > Sphere
 * Light
   - DirectionalLight
   - RectAreaLight
   - PointLight
 * Camera
   - PerspectiveCamera
   - OrthographicCamera
"""

from kubric.core.color import *
from kubric.core.base import *
# from kubric.core.traits import *
from kubric.core.objects import *
from kubric.core.materials import *
from kubric.core.lights import *
from kubric.core.cameras import *
from kubric.core.scene import *

