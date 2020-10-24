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
import datetime
import logging

import sys; sys.path.append(".")

import kubric as kb
from kubric.assets import KLEVR

# class KLEVRWorker(kb.Worker):
  # def get_argparser(self):
  #   parser = super().get_argparser()
  #   # add additional commandline arguments
  #   parser.add_argument("--min_nr_objects", type=int, default=4)
  #   parser.add_argument("--max_nr_objects", type=int, default=10)
  #   # overwrite default values
  #   # TODO(klausg): should we move KLEVR.zip to ./Assets too?
  #   parser.set_defaults(asset_source=['./Assets/KLEVR'])
  #   return parser

  # def add_floor(self, asset_source):
  #   floor = asset_source.create(asset_id="Floor", static=True, position=(0, 0, -0.2), scale=(2, 2, 2))
  #   self.add(floor)
  #   return floor

  # def add_lights(self):
  #   # --- Light settings from CLEVR
  #   sun = kb.DirectionalLight(color=kb.get_color("white"), intensity=0.45,
  #                             shadow_softness=0.2,
  #                             position=(11.6608, -6.62799, 25.8232))
  #   sun.look_at((0, 0, 0))
  #   lamp_back = kb.RectAreaLight(color=kb.get_color("white"), intensity=50.,
  #                                position=(-1.1685, 2.64602, 5.81574))
  #   lamp_back.look_at((0, 0, 0))
  #   lamp_key = kb.RectAreaLight(color=kb.get_color(0xffedd0), intensity=100,
  #                               width=0.5, height=0.5,
  #                               position=(6.44671, -2.90517, 4.2584))
  #   lamp_key.look_at((0, 0, 0))
  #   lamp_fill = kb.RectAreaLight(color=kb.get_color(0xc2d0ff), intensity=30,
  #                                width=0.5, height=0.5,
  #                                position=(-4.67112, -4.0136, 3.01122))
  #   lamp_fill.look_at((0, 0, 0))
  #   self.add(sun, lamp_back, lamp_key, lamp_fill)

  # def add_camera(self):
  #   camera = kb.PerspectiveCamera(focal_length=35., sensor_width=32, position=(7.48113, -6.50764, 5.34367))
  #   camera.look_at((0, 0, 0))
  #   self.add(camera)
  #   self.scene.camera = camera


  # def get_random_object(self, rnd, asset_source):
  #   random_id = rnd.choice([
  #       "LargeMetalCube",
  #       "LargeMetalCylinder",
  #       "LargeMetalSphere",
  #       "LargeRubberCube",
  #       "LargeRubberCylinder",
  #       "LargeRubberSphere",
  #       "MetalSpot",
  #       "RubberSpot",
  #       "SmallMetalCube",
  #       "SmallMetalCylinder",
  #       "SmallMetalSphere",
  #       "SmallRubberCube",
  #       "SmallRubberCylinder",
  #       "SmallRubberSphere",
  #   ])
  #   if "Metal" in random_id:
  #     material = kb.PrincipledBSDFMaterial(
  #           color=kb.random_hue_color(rnd=rnd),
  #           roughness=0.2,
  #           metallic=1.0,
  #           ior=2.5)
  #   else:  # if "Rubber" in random_id:
  #     material = kb.PrincipledBSDFMaterial(
  #         color=kb.random_hue_color(rnd=rnd),
  #         roughness=0.7,
  #         specular=0.33,
  #         metallic=0.,
  #         ior=1.25)
  #   return asset_source.create(asset_id=random_id, material=material)

# --- parser
PARSER = kb.ArgumentParser()
PARSER.add_argument("--min_nr_objects", type=int, default=4)
PARSER.add_argument("--max_nr_objects", type=int, default=10)
PARSER.add_argument("--max_placement_trials", type=int, default=100)
PARSER.add_argument("--asset_source", type=str, default="./Assets/KLEVR")
FLAGS = PARSER.parse_args()

# --- common setups
kb.setup_logging(FLAGS)
kb.log_my_flags(FLAGS)
RND = kb.setup_random_state(FLAGS)
ASSET_SOURCE = kb.AssetSource(FLAGS.asset_source) # TODO should this take flags too?

# --- Synchonizer between renderer and physics
class Environment:
  def __init__(self, flags):
    self.scene = kb.Scene.factory(flags) # TODO why couldn't use constructor?
    self.simulator = kb.simulator.PyBullet(self.scene)
    self.renderer = kb.renderer.Blender(self.scene)
    # self.background_objects = []
    # self.objects = []  #TODO: foreground objects?

  def add(self, *assets: kb.Asset, is_background=False):
    for asset in assets:
      logging.info("Added asset %s", asset)
      asset.background = is_background
      self.simulator.add(asset)
      self.renderer.add(asset)

  # WARNING renamed from place_ to add_, as it's an add with a modifier!
  def add_without_overlap(self, obj, pose_samplers, max_trials, rnd):
    self.add(obj)
    max_trials = max_trials if max_trials is not None else self.config.max_placement_trials

    collision = True
    trial = 0
    while collision and trial < max_trials:
      for sampler in pose_samplers:
        sampler(obj, rnd)
      collision = self.simulator.check_overlap(obj)
      trial += 1
    if collision:
      raise RuntimeError("Failed to place", obj)

  def run_simulation(self):
    animation = self.simulator.run()

    # --- Bake the simulation into keyframes
    for obj in animation.keys():
      for frame_id in range(self.scene.frame_end + 1):
        obj.position = animation[obj]["position"][frame_id]
        obj.quaternion = animation[obj]["quaternion"][frame_id]
        obj.keyframe_insert("position", frame_id)
        obj.keyframe_insert("quaternion", frame_id)
    return animation

ENV = Environment(FLAGS)

# --- Setup camera
camera = kb.PerspectiveCamera(focal_length=35., sensor_width=32, position=(7.48113, -6.50764, 5.34367))
camera.look_at((0, 0, 0))
ENV.add(camera)
ENV.scene.camera = camera #TODO: we shouldn't use a setter, but something more transparent here?

# --- Light settings from CLEVR
sun = kb.DirectionalLight(color=kb.get_color("white"), intensity=0.45,
                          shadow_softness=0.2,
                          position=(11.6608, -6.62799, 25.8232))
sun.look_at((0, 0, 0))
lamp_back = kb.RectAreaLight(color=kb.get_color("white"), intensity=50.,
                              position=(-1.1685, 2.64602, 5.81574))
lamp_back.look_at((0, 0, 0))
lamp_key = kb.RectAreaLight(color=kb.get_color(0xffedd0), intensity=100,
                            width=0.5, height=0.5,
                            position=(6.44671, -2.90517, 4.2584))
lamp_key.look_at((0, 0, 0))
lamp_fill = kb.RectAreaLight(color=kb.get_color(0xc2d0ff), intensity=30,
                              width=0.5, height=0.5,
                              position=(-4.67112, -4.0136, 3.01122))
lamp_fill.look_at((0, 0, 0))
ENV.add(sun, lamp_back, lamp_key, lamp_fill)

# --- floor
floor = ASSET_SOURCE.create(asset_id="Floor", static=True, position=(0, 0, -0.2), scale=(2, 2, 2))
ENV.add(floor)

# --- camera
camera = kb.PerspectiveCamera(focal_length=35., sensor_width=32, position=(7.48113, -6.50764, 5.34367))
camera.look_at((0, 0, 0))
ENV.add(camera)
ENV.scene.camera = camera

# --- TODO: to be removed
# worker = KLEVRWorker()
# worker.config.update(vars(FLAGS)) # TODO: why this needed?
# worker.setup(ENV.scene, ENV.simulator, ENV.renderer)
# worker.add_camera()
# worker.add_lights()
# worker.add_floor(ASSET_SOURCE)

spawn_region = [(-4, -4, 0), (4, 4, 3)]
velocity_range = [(-4, -4, 0), (4, 4, 0)]
objects = []

def get_random_object(rnd):
  asset_id = rnd.choice([
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
  ])
  if "Metal" in asset_id:
    material = kb.PrincipledBSDFMaterial(
          color=kb.random_hue_color(rnd=rnd),
          roughness=0.2,
          metallic=1.0,
          ior=2.5)
  else:  # if "Rubber" in asset_id:
    material = kb.PrincipledBSDFMaterial(
        color=kb.random_hue_color(rnd=rnd),
        roughness=0.7,
        specular=0.33,
        metallic=0.,
        ior=1.25)
  return asset_id, material

# --- Placer
nr_objects = RND.randint(FLAGS.min_nr_objects, FLAGS.max_nr_objects)
for i in range(nr_objects):
  asset_id, material = get_random_object(RND)
  obj = ASSET_SOURCE.create(asset_id=asset_id, material=material)
  ENV.add_without_overlap(obj, [kb.rotation_sampler(), kb.position_sampler(spawn_region)], FLAGS.max_placement_trials, rnd=RND)
  obj.velocity = (RND.uniform(*velocity_range) - [obj.position[0], obj.position[1], 0])  # bias velocity towards center
  objects.append(obj)
ENV.simulator.save_state("klevr.bullet")

ENV.run_simulation()
ENV.renderer.save_state("klevr.blend")

# TODO: re-enable later on
# self.render()
# output = self.post_process()
# # --- collect ground-truth factors
# output["factors"] = []
# for i, obj in enumerate(objects):
#   output["factors"].append({
#       "asset_id": obj.asset_id,
#       "material": "Metal" if "Metal" in obj.asset_id else "Rubber",
#       "mass": obj.mass,
#       "color": obj.material.color.rgb,
#       "animation": obj.keyframes,
#   })
# out_path = self.save_output(output)
# name = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
# self.export(self.output_dir, name, files_list=[sim_path, render_path, out_path])