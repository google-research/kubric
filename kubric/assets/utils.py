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

import kubric as kb
from kubric import core
from kubric import randomness
from kubric.core import color


CLEVR_OBJECTS = ("Cube", "Cylinder", "Sphere")
KUBASIC_OBJECTS = ("Cube", "Cylinder", "Sphere", "Cone", "Torus", "Gear", "TorusKnot",
                   "Sponge", "Spot", "Teapot", "Suzanne")


def get_clevr_lights(
    light_jitter: float = 1.0,
    rng: np.random.RandomState = randomness.default_rng()):
  """ Create lights that match the setup from the CLEVR dataset."""
  sun = core.DirectionalLight(name="sun",
                              color=color.Color.from_name("white"), shadow_softness=0.2,
                              intensity=0.45, position=(11.6608, -6.62799, 25.8232))
  lamp_back = core.RectAreaLight(name="lamp_back",
                                 color=color.Color.from_name("white"), intensity=50.,
                                 position=(-1.1685, 2.64602, 5.81574))
  lamp_key = core.RectAreaLight(name="lamp_key",
                                color=color.Color.from_hexint(0xffedd0), intensity=100,
                                width=0.5, height=0.5, position=(6.44671, -2.90517, 4.2584))
  lamp_fill = core.RectAreaLight(name="lamp_fill",
                                 color=color.Color.from_hexint(0xc2d0ff), intensity=30,
                                 width=0.5, height=0.5, position=(-4.67112, -4.0136, 3.01122))
  lights = [sun, lamp_back, lamp_key, lamp_fill]

  # jitter lights
  for light in lights:
    light.position = light.position + rng.rand(3) * light_jitter
    light.look_at((0, 0, 0))

  return lights


def get_random_kubasic_object(
    asset_source,
    objects_set="kubasic",
    color_strategy="uniform_hue",
    size_strategy="uniform",
    rng=randomness.default_rng()):
  if objects_set == "clevr":
    shape_name = rng.choice(CLEVR_OBJECTS)
  elif objects_set == "kubasic":
    shape_name = rng.choice(KUBASIC_OBJECTS)
  else:
    raise ValueError(f"Unknown object set {objects_set}")

  size_label, size = randomness.sample_sizes(size_strategy)
  color_label, random_color = randomness.sample_color(color_strategy)
  material_name = rng.choice(["Metal", "Rubber"])
  obj = asset_source.create(name=f"{size_label} {color_label} {material_name} {shape_name}",
                            asset_id=shape_name, scale=size)

  if material_name == "Metal":
    obj.material = kb.PrincipledBSDFMaterial(color=random_color, metallic=1.0, roughness=0.2,
                                             ior=2.5)
    obj.friction = 0.4
    obj.restitution = 0.3
    obj.mass *= 2.7 * size**3
  else:  # material_name == "Rubber"
    obj.material = kb.PrincipledBSDFMaterial(color=random_color, metallic=0., ior=1.25,
                                             roughness=0.7, specular=0.33)
    obj.friction = 0.8
    obj.restitution = 0.7
    obj.mass *= 1.1 * size**3

  obj.metadata = {
      "shape": shape_name.lower(),
      "size": size,
      "size_label": size_label,
      "material": material_name.lower(),
      "color": random_color.rgb,
      "color_label": color_label,
  }
  return obj


def add_hdri_dome(hdri_source, scene, background_hdri=None):
  dome_path = hdri_source.fetch("dome.blend")
  dome = kb.FileBasedObject(
      name="BackgroundDome",
      position=(0, 0, 0.01),  # slight offset in z direction to stay above floor plane
      static=True, background=True,
      simulation_filename=None,
      render_filename=str(dome_path),
      render_import_kwargs={
          "filepath": str(dome_path / "Object" / "Dome"),
          "directory": str(dome_path / "Object"),
          "filename": "Dome",
      })
  scene.add(dome)
  # pylint: disable=import-outside-toplevel
  from kubric.renderer import Blender
  import bpy
  blender_renderer = [v for v in scene.views if isinstance(v, Blender)]
  if blender_renderer:
    dome_blender = dome.linked_objects[blender_renderer[0]]
    dome_blender.cycles_visibility.shadow = False
    if background_hdri is not None:
      dome_mat = dome_blender.data.materials[0]
      texture_node = dome_mat.node_tree.nodes["Image Texture"]
      texture_node.image = bpy.data.images.load(background_hdri.filename)
  return dome
