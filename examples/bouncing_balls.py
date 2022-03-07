# Copyright 2022 The Kubric Authors
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
"""
Very basic scenes that mimics 2D bouncing balls by using an orthographic camera,
looking straight down.
"""

import logging

import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
import numpy as np


# --- CLI arguments
parser = kb.ArgumentParser()
# Configuration for the objects of the scene
parser.add_argument("--min_nr_balls", type=int, default=4)
parser.add_argument("--max_nr_balls", type=int, default=4)
parser.add_argument("--ball_radius", type=float, default=0.15)
parser.add_argument("--restitution", type=float, default=1.)
parser.add_argument("--friction", type=float, default=.0)
parser.add_argument("--shape", type=str, default="sphere")
parser.add_argument("--color", type=str, default="uniform_hsv")

parser.set_defaults(frame_end=24, frame_rate=12, width=128, height=128)
FLAGS = parser.parse_args()

# --- Common setups & resources
scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
simulator = PyBullet(scene, scratch_dir)
renderer = Blender(scene, scratch_dir, use_denoising=True, adaptive_sampling=False)

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
                friction=FLAGS.friction, background=True)
north_wall = kb.Cube(scale=(1.2, 0.9, 1), position=(0, 1.9, 0.9),
                     material=wall_material, static=True, background=True,
                     **room_dynamics)
south_wall = kb.Cube(scale=(1.2, 0.9, 1), position=(0, -1.9, 0.9),
                     material=wall_material, static=True, background=True,
                     **room_dynamics)
east_wall = kb.Cube(scale=(0.9, 1, 1), position=(1.9, 0, 0.9),
                    material=wall_material, static=True, background=True,
                    **room_dynamics)
west_wall = kb.Cube(scale=(0.9, 1, 1), position=(-1.9, 0, 0.9),
                    material=wall_material, static=True, background=True,
                    **room_dynamics)
scene.add([floor, north_wall, south_wall, east_wall, west_wall])

# --- Camera
scene.camera = kb.OrthographicCamera(position=(0, 0, 3), orthographic_scale=2.2)

# --- Balls
nr_objects = rng.randint(FLAGS.min_nr_balls, FLAGS.max_nr_balls+1)

if FLAGS.color == "uniform_hsv":
  colors = [kb.random_hue_color(rng=rng) for _ in range(nr_objects)]
elif FLAGS.color == "fixed":
  hues = np.linspace(0, 1., nr_objects, endpoint=False)
  colors = [kb.Color.from_hsv(hue, 1., 1.) for hue in hues]
elif FLAGS.color.startswith("cat"):
  num_colors = int(FLAGS.color[3:])
  all_hues = np.linspace(0, 1., num_colors, endpoint=False)
  hues = rng.choice(all_hues, size=nr_objects)
  colors = [kb.Color.from_hsv(hue, 1., 1.) for hue in hues]
elif FLAGS.color.startswith("noreplace"):
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

  ball.metadata["color"] = color
  return ball


samplers = [
    kb.position_sampler(region=[(-1, -1, 0), (1, 1, 2.1 * FLAGS.ball_radius)]),
    kb.rotation_sampler(axis=(0, 0, 1))]
for color in colors:
  ball = get_random_ball(color)
  scene.add(ball)
  kb.resample_while(ball, samplers, simulator.check_overlap, rng=rng)

# --- simulation
logging.info("Saving the simulator state to '%s' before starting the simulation.",
             output_dir / "scene.bullet")
simulator.save_state(output_dir / "scene.bullet")

# Run dynamic objects simulation
logging.info("Running the simulation ...")
animation, collisions = simulator.run(frame_start=0, frame_end=scene.frame_end+1)


# --- Rendering
logging.info("Saving the renderer state to '%s' before starting the rendering.",
             output_dir / "scene.blend")
renderer.save_state(output_dir / "scene.blend")

logging.info("Rendering the scene ...")
data_stack = renderer.render()


# --- Postprocessing
kb.compute_visibility(data_stack["segmentation"], scene.assets)
visible_foreground_assets = [asset for asset in scene.foreground_assets
                             if np.max(asset.metadata["visibility"]) > 0]
visible_foreground_assets = sorted(visible_foreground_assets,
                                   key=lambda asset: np.sum(asset.metadata["visibility"]),
                                   reverse=True)
data_stack["segmentation"] = kb.adjust_segmentation_idxs(
    data_stack["segmentation"],
    scene.assets,
    visible_foreground_assets)
scene_metadata = {"num_instances": len(visible_foreground_assets)}

# Save to image files
kb.write_image_dict(data_stack, output_dir)

kb.post_processing.compute_bboxes(data_stack["segmentation"], visible_foreground_assets)

# --- Metadata
logging.info("Collecting and storing metadata for each object.")
kb.write_json(filename=output_dir / "metadata.json", data={
    "metadata": kb.get_scene_metadata(scene, **scene_metadata),
    "instances": kb.get_instance_info(scene, assets_subset=visible_foreground_assets),
})
kb.write_json(filename=output_dir / "events.json", data={
    "collisions":  kb.process_collisions(collisions, scene, assets_subset=visible_foreground_assets),
})

kb.done()
