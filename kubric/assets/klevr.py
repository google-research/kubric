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

from kubric.assets import asset_source
from kubric import core
from kubric import color


class KLEVR(asset_source.AssetSource):
  objects_list = [
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

  def __init__(self, uri: str):
    super().__init__(uri)
    self.spawn_region = ((-3.5, -3.5, 0), (3.5, 3.5, 2))
    self.ambient_light = (0.05, 0.05, 0.05)

  def get_scene_geometry(self):
    return self.create(asset_id="Floor", static=True, position=(0, 0, -0.2), scale=(2, 2, 2))

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

  def get_scene(self):
    geometry = self.get_scene_geometry()
    lights = self.get_lights()
    camera = self.get_camera()
    return geometry, lights, camera

  def create_material(self, name, color_rgb):
    if name == "metal":
      return core.PrincipledBSDFMaterial(
          color=color_rgb,
          roughness=0.2,
          metallic=1.0,
          ior=2.5)
    elif name == "rubber":
      return core.PrincipledBSDFMaterial(
          color=color_rgb,
          roughness=0.7,
          specular=0.33,
          metallic=0.,
          ior=1.25)

  def get_random_rotation(self, rnd=None):
    """Samples a random rotation around z axis (uniformly distributed)."""
    if rnd is None:
      rnd = np.random.RandomState()
    theta = rnd.uniform(0, 2*np.pi)
    return np.cos(theta), 0, 0, np.sin(theta)

  def get_random_object_pose(self, asset_id, rnd=None):
    if rnd is None:
      rnd = np.random.RandomState()
    properties = self.db[self.db["id"] == asset_id].iloc[0]   # there has to be a better way!
    bounds = np.array(properties.get("bounds", [[0, 0, 0], [0, 0, 0]]))
    rotation = self.get_random_rotation(rnd)
    spawn_region = np.array(self.spawn_region) - bounds
    position = rnd.uniform(*spawn_region)
    return position, rotation

  def get_random_object(self, rnd=None):
    if rnd is None:
      rnd = np.random.RandomState()
    random_id = rnd.choice(self.objects_list)
    position, rotation = self.get_random_object_pose(random_id, rnd)
    velocity = rnd.uniform((-4, -4, 0), (4, 4, 0)) - [position[0], position[1], 0]  # bias towards center
    rgba = color.Color.from_hsv(rnd.random_sample(), .95, 1.0)
    material_type = "metal" if "Metal" in random_id else "rubber"
    material = self.create_material(material_type, rgba)
    return self.create(asset_id=random_id,
                       position=position,
                       quaternion=rotation,
                       velocity=velocity,
                       material=material)
