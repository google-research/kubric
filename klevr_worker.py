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
import tempfile
import numpy as np

import sys; sys.path.append(".")

# --- klevr imports
import kubric as kb
from kubric.assets import KLEVR

# --- parser
parser = kb.ArgumentParser()
parser.add_argument("--min_nr_objects", type=int, default=4)
parser.add_argument("--max_nr_objects", type=int, default=10)
parser.add_argument("--max_placement_trials", type=int, default=100)
parser.add_argument("--render_dir", type=str, default=tempfile.mkdtemp())
parser.add_argument("--output_dir", type=str, default="output/")
parser.add_argument("--asset_source", type=str, default="./Assets/KLEVR")
FLAGS = parser.parse_args()

# --- common setups & resources
kb.setup_logging(FLAGS.logging_level)
kb.log_my_flags(FLAGS)
rnd = np.random.RandomState(seed=FLAGS.random_seed)
scene = kb.Scene.factory(FLAGS) # TODO: why couldn't use constructor? →traitlets...
simulator = kb.simulator.PyBullet(scene)
renderer = kb.renderer.Blender(scene)

# --- adds assets to all resources
# TODO(klausg): apply refactor
def add_assets(*assets: kb.Asset, is_background=False):
  for asset in assets:
    logging.info("Added asset %s", asset)
    asset.background = is_background  #TODO: what is background used for?
    simulator.add(asset)
    renderer.add(asset)

# --- Synchonizer between renderer and physics
# TODO: this should be moved somewhere.. perhaps util?
def move_till_no_overlap(simulator, obj, max_trials = FLAGS.max_placement_trials, samplers = []):
  if len(samplers) == 0:
    spawn_region = [(-4, -4, 0), (4, 4, 3)]
    samplers.append(kb.rotation_sampler())
    samplers.append(kb.position_sampler(spawn_region))

  collision = True
  trial = 0
  while collision and trial < max_trials:
    for sampler in samplers:
      sampler(obj, rnd)
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

# --- Placer
objects = [] # TODO: shouldn't the list of objects be a property of scene?
velocity_range = [(-4, -4, 0), (4, 4, 0)]
nr_objects = rnd.randint(FLAGS.min_nr_objects, FLAGS.max_nr_objects)
for i in range(nr_objects):
  obj = klevr.create_random_object(rnd)
  add_assets(obj)
  move_till_no_overlap(simulator, obj)
  # bias velocity towards center
  obj.velocity = (rnd.uniform(*velocity_range) - [obj.position[0], obj.position[1], 0])
  objects.append(obj)

# --- Simulation
if FLAGS.output_dir:
  simulator.save_state(FLAGS.output_dir)
animation = simulator.run()

# --- Transfer simulation to keyframes
for obj in animation.keys():
  for frame_id in range(scene.frame_end + 1): 
    obj.position = animation[obj]["position"][frame_id]
    obj.quaternion = animation[obj]["quaternion"][frame_id]
    obj.keyframe_insert("position", frame_id)
    obj.keyframe_insert("quaternion", frame_id)

# --- Save a copy of the keyframed scene
if FLAGS.output_dir:
  renderer.save_state(path=FLAGS.output_dir)

# --- Rendering
if FLAGS.render_dir:
  renderer.render(path=FLAGS.render_dir)