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

import logging
import tempfile
import pickle
import pathlib

import numpy as np
import kubric as kb

# --- parser
parser = kb.ArgumentParser()
parser.add_argument("--min_nr_balls", type=int, default=4)
parser.add_argument("--max_nr_balls", type=int, default=4)
parser.add_argument("--ball_radius", type=float, default=0.2)
parser.add_argument("--restitution", type=float, default=1.)
parser.add_argument("--friction", type=float, default=.0)
parser.add_argument("--color", type=str, default="cat4")
parser.add_argument("--shape", type=str, default="sphere")
parser.add_argument("--render_dir", type=str, default=tempfile.mkdtemp())
parser.add_argument("--output_dir", type=str, default="output/")
FLAGS = parser.parse_args()

# --- common setups & resources
kb.setup_logging(FLAGS.logging_level)
kb.log_my_flags(FLAGS)
rng = np.random.default_rng(seed=FLAGS.random_seed)


# --- setup scene, simulator, renderer and asset source
scene = kb.Scene.from_flags(FLAGS)
simulator = kb.simulator.PyBullet(scene)
renderer = kb.renderer.Blender(scene)


# --- construct the room
floor_material = kb.FlatMaterial(color=kb.get_color('black'),
                                 indirect_visibility=False)
wall_material = kb.FlatMaterial(color=kb.get_color('white'),
                                indirect_visibility=False)
room_dynamics = {
    "restitution": FLAGS.restitution,
    "friction": FLAGS.friction,
}

floor = kb.Cube(scale=(1, 1, 0.9), position=(0, 0, -0.9),
                material=floor_material, static=True, restitution=0.,
                friction=FLAGS.friction)
north_wall = kb.Cube(scale=(1.2, 0.9, 1), position=(0, 1.9, 0.9),
                     material=wall_material, static=True, **room_dynamics)
south_wall = kb.Cube(scale=(1.2, 0.9, 1), position=(0, -1.9, 0.9),
                     material=wall_material, static=True, **room_dynamics)
east_wall = kb.Cube(scale=(0.9, 1, 1), position=(1.9, 0, 0.9),
                    material=wall_material, static=True, **room_dynamics)
west_wall = kb.Cube(scale=(0.9, 1, 1), position=(-1.9, 0, 0.9),
                    material=wall_material, static=True, **room_dynamics)
scene.add_all(floor, north_wall, south_wall, east_wall, west_wall)

# --- Camera
scene.camera = kb.OrthographicCamera(position=(0, 0, 3), orthographic_scale=2.2)

# --- Balls
nr_objects = rng.integers(FLAGS.min_nr_balls, FLAGS.max_nr_balls+1)

if FLAGS.color == "uniform_hsv":
  colors = [kb.random_hue_color(rng=rng) for _ in range(nr_objects)]
if FLAGS.color == "fixed":
  hues = np.linspace(0, 1., nr_objects, endpoint=False)
  colors = [kb.Color.from_hsv(hue, 1., 1.) for hue in hues]
if FLAGS.color.startswith("cat"):
  num_colors = int(FLAGS.color[3:])
  all_hues = np.linspace(0, 1., num_colors, endpoint=False)
  hues = rng.choice(all_hues, size=nr_objects)
  colors = [kb.Color.from_hsv(hue, 1., 1.) for hue in hues]
if FLAGS.color.startswith("noreplace"):
  num_colors = int(FLAGS.color[9:])
  all_hues = np.linspace(0, 1., num_colors, endpoint=False)
  hues = rng.choice(all_hues, size=nr_objects, replace=False)
  colors = [kb.Color.from_hsv(hue, 1., 1.) for hue in hues]
else:
  raise ValueError(f"Unknown color directive --color={FLAGS.color}")


def get_random_ball(color):
  velocity_range = (-1, -1, 0), (1, 1, 0)
  ball_material = kb.FlatMaterial(color=color,
                                  indirect_visibility=False)
  shape = FLAGS.shape
  if shape == "mixed":
    shape = rng.choice(["cube", "sphere"])
  if shape == "cube":
    ball = kb.Cube(scale=[FLAGS.ball_radius]*3,
                   material=ball_material,
                   friction=FLAGS.friction,
                   restitution=FLAGS.restitution,
                   quaternion=kb.random_rotation([0, 0, 1], rng),
                   velocity=rng.uniform(*velocity_range))
  elif shape == "sphere":
    ball = kb.Sphere(scale=[FLAGS.ball_radius]*3,
                     material=ball_material,
                     friction=FLAGS.friction,
                     restitution=FLAGS.restitution,
                     velocity=rng.uniform(*velocity_range))
  else:
    raise ValueError(f"Unknown shape type '{shape}'")
  return ball


samplers = [
    kb.position_sampler(region=[(-1, -1, 0), (1, 1, 2.1 * FLAGS.ball_radius)]),
    kb.rotation_sampler(axis=(0, 0, 1))]
for color in colors:
  ball = get_random_ball(color)
  scene.add(ball)
  kb.resample_while(ball, samplers, simulator.check_overlap, rng=rng)

# --- simulation
simulator.save_state(FLAGS.output_dir)
simulator.run()

# --- rendering
renderer.save_state(path=FLAGS.output_dir)
renderer.render(path=FLAGS.render_dir)
# Parse renderer-specific output into per-frame numpy pickles
renderer.postprocess(from_dir=FLAGS.render_dir, to_dir=FLAGS.output_dir)

# --- store object-centric metadata
metadata = [{
    "color": obj.material.color.rgb,
    "mass": obj.mass,
    "animation": obj.keyframes,
} for obj in scene.foreground_assets]

with open(pathlib.Path(FLAGS.output_dir) / "metadata.pkl", "wb") as fp:
  logging.info(f"Writing {fp.name}")
  pickle.dump(metadata, fp)
