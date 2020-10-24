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

# --- klevr imports
import kubric as kb
from kubric.assets import KLEVR

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

# --- common resources
scene = kb.Scene.factory(FLAGS) # TODO why couldn't use constructor?
simulator = kb.simulator.PyBullet(scene)
renderer = kb.renderer.Blender(scene)

# --- adds assets to all resources
def add_assets(*assets: kb.Asset, is_background=False):
  for asset in assets:
    logging.info("Added asset %s", asset)
    asset.background = is_background
    simulator.add(asset)
    renderer.add(asset)

# --- transfers information between animation and scene
def bake_keyframes(animation, scene):
  # --- Bake the simulation into keyframes
  for obj in animation.keys():
    for frame_id in range(scene.frame_end + 1):
      obj.position = animation[obj]["position"][frame_id]
      obj.quaternion = animation[obj]["quaternion"][frame_id]
      obj.keyframe_insert("position", frame_id)
      obj.keyframe_insert("quaternion", frame_id)

# --- Synchonizer between renderer and physics
def move_till_no_overlap(simulator, obj, max_trials = FLAGS.max_placement_trials, samplers = []):
  if len(samplers) == 0:
    spawn_region = [(-4, -4, 0), (4, 4, 3)]
    samplers.append(kb.rotation_sampler())
    samplers.append(kb.position_sampler(spawn_region))

  collision = True
  trial = 0
  while collision and trial < max_trials:
    for sampler in samplers:
      sampler(obj, RND)
    collision = simulator.check_overlap(obj)
    trial += 1
  if collision:
    raise RuntimeError("Failed to place", obj)

# --- Assemble the basic scene
klevr = KLEVR(FLAGS.asset_source)
camera = klevr.get_camera()
lights = klevr.get_lights()
floor = klevr.get_floor()
add_assets(camera, floor, *lights)
scene.camera = camera #TODO: we shouldn't use a setter, but something more explicit

# --- TODO: to be removed
# worker = KLEVRWorker()
# worker.config.update(vars(FLAGS)) # TODO: why this needed?
# worker.setup(ENV.scene, ENV.simulator, ENV.renderer)
# worker.add_camera()
# worker.add_lights()
# worker.add_floor(ASSET_SOURCE)

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
objects = []
velocity_range = [(-4, -4, 0), (4, 4, 0)]
nr_objects = RND.randint(FLAGS.min_nr_objects, FLAGS.max_nr_objects)
for i in range(nr_objects):
  asset_id, material = get_random_object(RND)
  obj = ASSET_SOURCE.create(asset_id=asset_id, material=material)
  add_assets(obj)
  move_till_no_overlap(simulator, obj)
  obj.velocity = (RND.uniform(*velocity_range) - [obj.position[0], obj.position[1], 0])  # bias velocity towards center
  objects.append(obj)

simulator.save_state("klevr.bullet")
animation = simulator.run()
bake_keyframes(animation, scene)  #< shouldn't 2nd argument locally be "renderer"?
renderer.save_state("klevr.blend")

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