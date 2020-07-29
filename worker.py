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
import logging
import argparse

import sys; sys.path.append(".")
import kubric.viewer.blender as THREE

from kubric.asset_source import AssetSource
from kubric.placer import Placer
from kubric.simulator import Simulator
from kubric.simulator import Object3D

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--template", type=str, default="sphereworld")
parser.add_argument("--assets", type=str, default="~/datasets/katamari",
                    help="e.g. '~/datasets/katamari' or 'gs://kubric/katamari'")
parser.add_argument("--num_objects", type=int, default=3)
parser.add_argument("--frame_rate", type=int, default=24)
parser.add_argument("--step_rate", type=int, default=240)
parser.add_argument("--logging_level", type=str, default="INFO")

# --- parse argument in a way compatible with blender's REPL
if "--" in sys.argv:
  FLAGS = parser.parse_args(args=sys.argv[sys.argv.index("--")+1:])
else:
  FLAGS = parser.parse_args(args=[])

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
  simulator.add(obj3d)

# TODO: Issue #4
# blender → bpy.ops.import_scene.obj(filepath=path, axis_forward='Y', axis_up='Z')
# pubullet → getVisualShapeData

# --- setup the renderer
renderer = THREE.Renderer()
renderer.set_size(200,200)
scene = THREE.Scene()
scene.frame_start = 0
scene.frame_end = 10

camera = THREE.OrthographicCamera(left=-.5, right=+.5, top=.5, bottom=-.5)
camera.position = (2, 2, 2)
camera.look_at(0, 0, .75)

# --- run the simulation
animation = simulator.run(duration=1.0)

# --- dump the simulation data in the renderer
for obj_id in animation:
  import pybullet as pb
  # TODO: why is the [0] needed? [4] is the position of the file path
  mesh_filename = pb.getVisualShapeData(obj_id)[0][4].decode("utf-8")
  obj3d = scene.add_from_file(mesh_filename)
  for frame_id in range(scene.frame_start, scene.frame_end):
    obj3d.position = animation[obj_id]["position"][frame_id]
    obj3d.keyframe_insert("position", frame_id)

renderer.render(scene, camera, path="output.blend")

