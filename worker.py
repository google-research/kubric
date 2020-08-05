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
import numpy as np
import pathlib
import pickle

import sys; sys.path.append(".")

import kubric.viewer.blender as THREE
from kubric.assets.asset_source import AssetSource
from kubric.assets.utils import mm3hash
from kubric.simulator import Simulator
from kubric.post_processing import get_render_layers_from_exr


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--assets", type=str, default="./KLEVR",
                    help="e.g. '~/datasets/katamari' or 'gs://kubric/katamari'")
parser.add_argument("--frame_rate", type=int, default=24)
parser.add_argument("--step_rate", type=int, default=240)
parser.add_argument("--frame_start", type=int, default=0)
parser.add_argument("--frame_end", type=int, default=24)
parser.add_argument("--logging_level", type=str, default="INFO")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--resolution", type=int, default=512)
parser.add_argument("--outpath", type=str, default='./output/')

# --- parse argument in a way compatible with blender's REPL
if "--" in sys.argv:
  FLAGS = parser.parse_args(args=sys.argv[sys.argv.index("--")+1:])
else:
  FLAGS = parser.parse_args(args=[])

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

logging.basicConfig(level=FLAGS.logging_level)
if FLAGS.seed:
  rnd = np.random.RandomState(FLAGS.seed)
else:
  rnd = np.random.RandomState()

# --- Download a few models locally
asset_source = AssetSource(uri=FLAGS.assets)

# scene geometry
floor = asset_source.create({'id': 'Floor', 'static': True})

# random number of objects scattered in a given region
spawn_region = ((-4, -4, 0.4), (4, 4, 0.5))
velocity_range = ((-4, -4, 0), (4, 4, 0))
nr_objects = rnd.randint(4, 10)
objects = []
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

for i in range(nr_objects):
  object_id = rnd.choice(objects_list)
  position = rnd.uniform(*spawn_region)
  velocity = rnd.uniform(*velocity_range)

  objects.append(asset_source.create({'id': object_id,
                                      'position': position,
                                      'linear_velocity': velocity}))


# --- load models & place them in the simulator
simulator = Simulator(frame_rate=FLAGS.frame_rate, step_rate=FLAGS.step_rate)
simulator.add(floor)
for obj in objects:
  collision = simulator.add(obj)
  # TODO do something about detected collisions

# --- run the physics simulation
animation = simulator.run(FLAGS.frame_end)

# --- set up the rendering
renderer = THREE.Renderer()
renderer.set_size(FLAGS.resolution, FLAGS.resolution)
scene = THREE.Scene()
scene.frame_start = FLAGS.frame_start
scene.frame_end = FLAGS.frame_end
renderer.set_up_background(bg_color=(0., 0., 0., 0.))
renderer.set_up_exr_output(path=FLAGS.outpath)

# Camera settings from CLEVR
camera = THREE.PerspectiveCamera(focal_length=35.)
camera.position = (7.48113, -6.50764, 5.34367)
camera.quaternion = (0.7816, 0.481707, 0.212922, 0.334251)
camera.camera.sensor_width = 32

# Light settings from CLEVR
sun = THREE.DirectionalLight(color=0xffffff, intensity=0.45, shadow_softness=0.2)
sun.position = (11.6608, -6.62799, 25.8232)
sun.quaternion = (0.971588, 0.105085, 0.210842, 0.022804)
scene.add(sun)
lamp_back = THREE.RectAreaLight(color=0xffffff, intensity=50., width=1, height=1)
lamp_back.position = (-1.1685, 2.64602, 5.81574)
lamp_back.look_at(0, 0, 0)
scene.add(lamp_back)
lamp_key = THREE.RectAreaLight(color=0xffedd0, intensity=100, width=0.5, height=0.5)
lamp_key.position = (6.44671, -2.90517, 4.2584)
lamp_key.look_at(0, 0, 0)
scene.add(lamp_key)
lamp_fill = THREE.RectAreaLight(color=0xc2d0ff, intensity=30, width=0.5, height=0.5)
lamp_fill.position = (-4.67112, -4.0136, 3.01122)
lamp_fill.look_at(0, 0, 0)
scene.add(lamp_fill)


# --- dump the simulation data in the renderer
room = scene.add_from_file(str(floor.vis_filename))

for obj in objects:
  o = scene.add_from_file(str(obj.vis_filename), name=obj.uid)
  o.position = obj.position
  o.quaternion = obj.rotation

  for frame_id in range(scene.frame_start, scene.frame_end):
    o.position = animation[obj]["position"][frame_id]
    o.quaternion = animation[obj]["orient_quat"][frame_id]
    o.keyframe_insert("position", frame_id)
    o.keyframe_insert("quaternion", frame_id)

outpath = pathlib.Path(FLAGS.outpath)

renderer.render(scene, camera, path=str(outpath / 'out'))

# --- Postprocessing

layers = []
for frame_id in range(scene.frame_start, scene.frame_end):
  layers.append(get_render_layers_from_exr(f'{outpath}/out{frame_id:04d}.exr'))

gt_factors = []
for obj in objects:
  gt_factors.append({
    'mass': obj.mass,
    'asset_id': obj.asset_id,
    'crypto_id': mm3hash(obj.uid),   # TODO: adjust segmentation maps instead
    'animation': animation[obj],
  })


# TODO: convert to TFrecords
with open(outpath + '/layers.pkl', 'wb') as f:
  pickle.dump({'layers': layers, 'factors': gt_factors}, f)

