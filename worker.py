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

import sys; sys.path.append(".")
from kubric.asset_source import AssetSource
from kubric.placer import Placer
from kubric.simulator import Simulator
from kubric.simulator import Object3D

class Renderer(object):
  def __init__(self, framerate: int):
    pass

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--template", type=str, default="sphereworld")
parser.add_argument("--assets", type=str, default="gs://kubric/katamari",
                    help="e.g. '~/datasets/katamari' or 'gs://kubric/katamari'")
parser.add_argument("--num_objects", type=int, default=3)
parser.add_argument("--frame_rate", type=int, default=24)
parser.add_argument("--step_rate", type=int, default=240)
parser.add_argument("--logging_level", type=str, default="INFO")
FLAGS = parser.parse_args()

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

logging.basicConfig(level=FLAGS.logging_level)

# --- Download a few models locally
asset_source = AssetSource(path=FLAGS.assets)
urdf_paths = asset_source.take(FLAGS.num_objects)

# --- load models & place them in the simulator
simulator = Simulator(frame_rate=FLAGS.frame_rate, step_rate=FLAGS.step_rate)
placer = Placer(template=FLAGS.template, simulator=simulator)

for urdf_path in urdf_paths:
  obj3d = Object3D(sim_filename=urdf_path)
  placer.place(obj3d)
  simulator.place_object(obj3d)

# TODO: Issue #4
# blender → bpy.ops.import_scene.obj(filepath=path, axis_forward='Y', axis_up='Z')
# pubullet → getVisualShapeData

# --- run the simulation
animation = simulator.run(1)
print(animation)
# renderer = Renderer(FLAGS.frame_rate)
