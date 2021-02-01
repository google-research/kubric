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
import pickle
import pathlib

import numpy as np
import kubric as kb

# --- CLI arguments
parser = kb.ArgumentParser()
parser.add_argument("--min_nr_objects", type=int, default=4)
parser.add_argument("--max_nr_objects", type=int, default=10)
parser.add_argument("--max_placement_trials", type=int, default=100)
parser.add_argument("--output_dir", type=str, default="output/")
parser.add_argument("--assets_dir", type=str, default="./assets/KLEVR")
FLAGS = parser.parse_args()

# --- Common setups & resources
kb.setup_logging(FLAGS.logging_level)
kb.log_my_flags(FLAGS)
rng = np.random.default_rng(seed=FLAGS.random_seed)
output_dir = pathlib.Path(FLAGS.output_dir)
scene = kb.Scene.from_flags(FLAGS)
simulator = kb.simulator.PyBullet(scene)
renderer = kb.renderer.Blender(scene)

logging.info("Loading assets from %s", FLAGS.assets_dir)
klevr = kb.AssetSource(FLAGS.assets_dir)


# --- Populate the scene
logging.info("Creating a large gray cube as the floor...")
floor_material = kb.PrincipledBSDFMaterial(color=kb.get_color('gray'), roughness=1., specular=0.)
floor = kb.Cube(scale=(10, 10, 1), position=(0, 0, -1), material=floor_material, friction=0.3,
                static=True, background=True)
scene.add(floor)

logging.info("Adding several lights to the scene...")
sun = kb.DirectionalLight(color=kb.Color(1, 1, 1), shadow_softness=0.2, intensity=0.45,
                          position=(11.6608, -6.62799, 25.8232), look_at=(0, 0, 0))
lamp_back = kb.RectAreaLight(color=kb.get_color("white"), intensity=50.,
                             position=(-1.1685, 2.64602, 5.81574), look_at=(0, 0, 0))
lamp_key = kb.RectAreaLight(color=kb.get_color(0xffedd0), intensity=100, width=0.5, height=0.5,
                            position=(6.44671, -2.90517, 4.2584), look_at=(0, 0, 0))
lamp_fill = kb.RectAreaLight(color=kb.get_color("#c2d0ff"), intensity=30, width=0.5, height=0.5,
                             position=(-4.67112, -4.0136, 3.01122), look_at=(0, 0, 0))
scene.add([sun, lamp_back, lamp_key, lamp_fill])
scene.ambient_illumination = kb.Color(0.05, 0.05, 0.05)

logging.info("Setting up the Camera...")
scene.camera = kb.PerspectiveCamera(focal_length=35., sensor_width=32,
                                    position=(7.48113, -6.50764, 5.34367), look_at=(0, 0, 0))

# --- Place random objects
nr_objects = rng.integers(FLAGS.min_nr_objects, FLAGS.max_nr_objects)
logging.info("Randomly placing %d objects:", nr_objects)
spawn_region = [(-4, -4, 0), (4, 4, 3)]
velocity_range = [(-4, -4, 0), (4, 4, 0)]

for i in range(nr_objects):
  shape = rng.choice(["Cube", "Cylinder", "Sphere"])
  # "Cone", "Torus", "Spot", "Sponge", "TorusKnot", "Gear", "Teapot", "Suzanne"])
  size = rng.choice([1.4, 0.7])
  color = kb.random_hue_color(rng=rng)
  material = rng.choice(["Metal", "Rubber"])
  obj = klevr.create(asset_id=shape, scale=size)
  if material == "Metal":
    obj.material = kb.PrincipledBSDFMaterial(color=color, metallic=1.0, roughness=0.2, ior=2.5)
    obj.friction = 0.4
    obj.restitution = 0.3
    obj.mass *= 2.7
  else:  # material == "Rubber"
    obj.material = kb.PrincipledBSDFMaterial(color=color, metallic=0., ior=1.25, roughness=0.7,
                                             specular=0.33)
    obj.friction = 0.8
    obj.restitution = 0.7
    obj.mass *= 1.1

  scene.add(obj)
  kb.move_until_no_overlap(obj, simulator, spawn_region=spawn_region,
                           max_trials=FLAGS.max_placement_trials)
  # bias velocity towards center
  obj.velocity = (rng.uniform(*velocity_range) - [obj.position[0], obj.position[1], 0])
  logging.info("    Added %s", obj)


# --- Simulation
logging.info("Saving the simulator state to '%s' before starting the simulation.", output_dir)
simulator.save_state(output_dir / "scene.bullet")
logging.info("Running the Simulation...")
animation = simulator.run()


# --- Rendering
logging.info("Saving the renderer state to '%s' before starting the rendering.", FLAGS.output_dir)
renderer.save_state(output_dir / "scene.blend")
render_dir = output_dir / "render"
logging.info("Rendering the scene and saving results in '%s'", render_dir)
renderer.render(render_dir)


# --- Postprocessing
logging.info("Parse and post-process renderer-specific output into per-frame numpy pickles.")
renderer.postprocess(from_dir=render_dir, to_dir=output_dir)


# --- Metadata
logging.info("Collecting and storing metadata for each object.")
metadata = [{
    "asset_id": obj.asset_id,
    "material": "Metal" if obj.material.metallic > 0.5 else "Rubber",
    "mass": obj.mass,
    "color": obj.material.color.rgb,
    "size": "small" if obj.scale[0] < 1. else "large",
    "animation": obj.keyframes,
  } for obj in scene.foreground_assets]

with open(pathlib.Path(FLAGS.output_dir) / "metadata.pkl", "wb") as fp:
  logging.info(f"Writing to {fp.name}")
  pickle.dump(metadata, fp)

logging.info("Done!")
