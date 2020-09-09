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

import argparse
import logging
import pathlib

import numpy as np

import sys; sys.path.append(".")

from kubric import simulator
from kubric import core
from kubric.assets import klevr
from kubric.viewer import blender

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--assets", type=str, default="./KLEVR",
                    help="e.g. '~/datasets/katamari' or 'gs://kubric/katamari'")
parser.add_argument("--frame_rate", type=int, default=24)
parser.add_argument("--step_rate", type=int, default=240)
parser.add_argument("--frame_start", type=int, default=0)
parser.add_argument("--frame_end", type=int, default=96)  # 4 seconds 
parser.add_argument("--logging_level", type=str, default="INFO")
parser.add_argument("--max_placement_trials", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--width", type=int, default=512)
parser.add_argument("--height", type=int, default=512)
# TODO: support cloud storage
parser.add_argument("--output", type=str, default="./output/", help="output directory")

# --- parse argument in a way compatible with blender's REPL
if "--" in sys.argv:
  FLAGS = parser.parse_args(args=sys.argv[sys.argv.index("--")+1:])
else:
  FLAGS = parser.parse_args(args=[])

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# --- Setup logger
logging.basicConfig(level=FLAGS.logging_level)

# --- Configures random generator
if FLAGS.seed:
  rnd = np.random.RandomState(FLAGS.seed)
else:
  rnd = np.random.RandomState()

scene = core.Scene(frame_start=FLAGS.frame_start,
                   frame_end=FLAGS.frame_end,
                   frame_rate=FLAGS.frame_rate,
                   step_rate=FLAGS.step_rate,
                   resolution=(FLAGS.width, FLAGS.height))


# --- Download a few models locally
asset_source = klevr.KLEVR(uri=FLAGS.assets)
simulator = simulator.Simulator(scene)

# --- Scene static geometry
floor, lights, camera = asset_source.get_scene()
simulator.add(floor)

# --- Scene configuration (number of objects randomly scattered in a region)
nr_objects = rnd.randint(4, 10)
objects = [asset_source.get_random_object(rnd) for _ in range(nr_objects)]

for obj in objects:
  collision = simulator.add(obj)
  trial = 0
  while collision and trial < FLAGS.max_placement_trials:
    obj.position, obj.rotation = asset_source.get_random_object_pose(obj.name, rnd)
    collision = simulator.check_overlap(obj)
    trial += 1
  if collision:
    raise RuntimeError("Failed to place", obj)

# --- run the physics simulation
animation = simulator.run()

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
renderer = blender.Blender(scene)
renderer.set_ambient_light()
renderer.add(floor)
for light in lights:
  renderer.add(light)
renderer.add(camera)
scene.camera = camera   # TODO: currently camera has to be added to renderer before assignment. fix!

for obj in objects:
  renderer.add(obj)
  # --- Bake the simulation into keyframes
  for frame_id in range(scene.frame_start, scene.frame_end):
    obj.position = animation[obj]["position"][frame_id]
    obj.quaternion = animation[obj]["quaternion"][frame_id]
    obj.keyframe_insert("position", frame_id)
    obj.keyframe_insert("quaternion", frame_id)

# --- Render or create the .blend file
output_path = pathlib.Path(FLAGS.output)
output_path.mkdir(parents=True, exist_ok=True)
renderer.render(path=output_path)
