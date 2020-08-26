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
from kubric.core import Scene
from kubric.viewer.blender import Blender
from kubric import core
from kubric.color import Color

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--assets", type=str, default="gs://kubric/KLEVR",
                    help="e.g. '~/datasets/katamari' or 'gs://kubric/katamari'")
parser.add_argument("--frame_rate", type=int, default=24)
parser.add_argument("--step_rate", type=int, default=240)
parser.add_argument("--frame_start", type=int, default=0)
parser.add_argument("--frame_end", type=int, default=96)  # 4 seconds 
parser.add_argument("--logging_level", type=str, default="INFO")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--resolution", type=int, default=512)
parser.add_argument("--randomize_material", type=bool, default=False)
parser.add_argument("--outpath", type=str, default='./output/')
parser.add_argument("--output", type=str, default='gs://kubric/output')  # TODO: actually copy results there

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
logger = logging.getLogger(__name__)

# --- Configures random generator
if FLAGS.seed:
  rnd = np.random.RandomState(FLAGS.seed)
else:
  rnd = np.random.RandomState()

scene = Scene(frame_start=FLAGS.frame_start,
              frame_end=FLAGS.frame_end,
              frame_rate=FLAGS.frame_rate,
              step_rate=FLAGS.step_rate,
              resolution=(FLAGS.resolution, FLAGS.resolution))


# --- Download a few models locally
asset_source = AssetSource(uri=FLAGS.assets)
simulator = Simulator(scene)
renderer = Blender(scene)

# --- Scene static geometry
floor = asset_source.create('Floor', {'static': True, 'position': (0, 0, -0.2)})
simulator.add(floor)
renderer.add(floor)


# --- Camera settings from CLEVR
camera = core.PerspectiveCamera(focal_length=35., sensor_width=32,
                                position=(7.48113, -6.50764, 5.34367))
camera.look_at((0, 0, 0))
renderer.add(camera)
scene.camera = camera

# --- Light settings from CLEVR
sun = core.DirectionalLight(color=Color.from_name('white'), intensity=0.45, shadow_softness=0.2,
                            position=(11.6608, -6.62799, 25.8232))
sun.look_at((0, 0, 0))
renderer.add(sun)

lamp_back = core.RectAreaLight(color=Color.from_name('white'), intensity=50.,
                               position=(-1.1685, 2.64602, 5.81574))
lamp_back.look_at((0, 0, 0))
renderer.add(lamp_back)

lamp_key = core.RectAreaLight(color=Color.from_hexint(0xffedd0), intensity=100,
                              width=0.5, height=0.5, position=(6.44671, -2.90517, 4.2584))
lamp_key.look_at((0, 0, 0))
renderer.add(lamp_key)

lamp_fill = core.RectAreaLight(color=Color.from_hexint(0xc2d0ff), intensity=30,
                               width=0.5, height=0.5, position=(-4.67112, -4.0136, 3.01122))
lamp_fill.look_at((0, 0, 0))
renderer.add(lamp_fill)

renderer.set_ambient_illumination(color=Color(0.05, 0.05, 0.05))
renderer.set_background(color=Color(0., 0., 0.))


# --- Scene configuration (number of objects randomly scattered in a region)
spawn_region = ((-3.5, -3.5, 0), (3.5, 3.5, 2))
velocity_range = ((-4, -4, 0), (4, 4, 0))
nr_objects = rnd.randint(6, 10)

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


def get_random_rotation(rnd):
  """Samples a random rotation around z axis (uniformly distributed)."""
  theta = rnd.uniform(0, 2*np.pi)
  return np.cos(theta), 0, 0, np.sin(theta)


objects = []
for i in range(nr_objects):
  objects.append(asset_source.create(rnd.choice(objects_list),
                                     {'position': tuple(rnd.uniform(*spawn_region)),
                                      'quaternion': get_random_rotation(rnd),
                                      'linear_velocity': tuple(rnd.uniform(*velocity_range))}))

for obj in objects:
  simulator.add(obj)
  trial = 0
  collision = simulator.check_overlap(obj)
  while collision and trial < 100:
    obj.position = tuple(rnd.uniform(*spawn_region))
    obj.quaternion = get_random_rotation(rnd)
    collision = simulator.check_overlap(obj)
    trial += 1
  if collision:
    raise RuntimeError('Failed to place', obj)

# --- run the physics simulation
animation = simulator.run()

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


for obj in objects:
  # --- Bake the simulation into keyframes
  for frame_id in range(scene.frame_start, scene.frame_end):
    obj.position = animation[obj]["position"][frame_id]
    obj.quaternion = animation[obj]["quaternion"][frame_id]
    obj.keyframe_insert('position', frame_id)
    obj.keyframe_insert('quaternion', frame_id)

# --- Render or create the .blend file
renderer.render(path=FLAGS.output)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# TODO: add option to trigger gather of data from cloud bucket (or local file system)
if False:
  # --- Postprocessing
  layers = []
  for frame_id in range(scene.frame_start, scene.frame_end):
    layers.append(get_render_layers_from_exr(f'{FLAGS.outpath}/out{frame_id:04d}.exr'))

  gt_factors = []
  for obj in objects:
    gt_factors.append({
      'mass': obj.mass,
      'asset_id': obj.asset_id,
      'crypto_id': mm3hash(obj.uid),   # TODO: adjust segmentation maps instead
      'animation': animation[obj],
    })

  # TODO: convert to TFrecords
  with open(FLAGS.outpath + '/layers.pkl', 'wb') as f:
    pickle.dump({'layers': layers, 'factors': gt_factors}, f)
