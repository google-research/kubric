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

import numpy as np

import kubric as kb
from kubric.assets import asset_source
from kubric import core
from kubric import randomness
from kubric.core import color

KLEVR_ASSETS_IDs = [
  "LargeMetalCube",
  "LargeMetalCylinder",
  "LargeMetalSphere",
  "LargeRubberCube",
  "LargeRubberCylinder",
  "LargeRubberSphere",
  "MetalSpot",
  "RubberSpot",
  "SmallMetalCube",
  "SmallMetalCylinder",
  "SmallMetalSphere",
  "SmallRubberCube",
  "SmallRubberCylinder",
  "SmallRubberSphere",
]

class KLEVR(asset_source.AssetSource):
  def __init__(self, uri: str):
    super().__init__(uri)
    #TODO: this seems obsolete?
    # self.spawn_region = ((-3.5, -3.5, 0), (3.5, 3.5, 2))
    #TODO: this seems obsolete?
    # self.ambient_light = (0.05, 0.05, 0.05)

  # NOTE: moved from worker
  def create_random_object(self, rng=randomness.default_rng()):
    asset_id = rng.choice(KLEVR_ASSETS_IDs)

    if "Metal" in asset_id:
      material = kb.PrincipledBSDFMaterial(
            color=kb.random_hue_color(rng=rng),
            roughness=0.2,
            metallic=1.0,
            ior=2.5)
    else:  # if "Rubber" in asset_id:
      material = kb.PrincipledBSDFMaterial(
          color=kb.random_hue_color(rng=rng),
          roughness=0.7,
          specular=0.33,
          metallic=0.,
          ior=1.25)

    obj = self.create(asset_id=asset_id, material=material)   
    return obj

  def get_floor(self):
    asset = self.create(asset_id="Floor", static=True, position=(0, 0, -0.2), scale=(2, 2, 2),
                        background=True)
    return asset

  def get_lights(self):
    # --- Light settings from CLEVR
    sun = core.DirectionalLight(color=color.Color.from_name("white"), shadow_softness=0.2,
                                intensity=0.45, position=(11.6608, -6.62799, 25.8232))
    sun.look_at((0, 0, 0))
    lamp_back = core.RectAreaLight(color=color.Color.from_name("white"), intensity=50.,
                                   position=(-1.1685, 2.64602, 5.81574))
    lamp_back.look_at((0, 0, 0))
    lamp_key = core.RectAreaLight(color=color.Color.from_hexint(0xffedd0), intensity=100,
                                  width=0.5, height=0.5, position=(6.44671, -2.90517, 4.2584))
    lamp_key.look_at((0, 0, 0))
    lamp_fill = core.RectAreaLight(color=color.Color.from_hexint(0xc2d0ff), intensity=30,
                                   width=0.5, height=0.5, position=(-4.67112, -4.0136, 3.01122))
    lamp_fill.look_at((0, 0, 0))
    return [sun, lamp_back, lamp_key, lamp_fill]

  def get_camera(self):
    camera = core.PerspectiveCamera(focal_length=35., sensor_width=32,
                                    position=(7.48113, -6.50764, 5.34367))
    camera.look_at((0, 0, 0))
    return camera
