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
import tempfile
import numpy as np
import pickle
import pathlib

import sys; sys.path.append(".")

import kubric as kb

# --- parser
parser = kb.ArgumentParser()
parser.add_argument("--min_nr_objects", type=int, default=4)
parser.add_argument("--max_nr_objects", type=int, default=10)
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
klevr = kb.assets.KLEVR("./assets/KLEVR")
gso = kb.AssetSource("./assets/GSO")


scene.camera = kb.PerspectiveCamera(position=(7.5, -6.5, 5.3),
                                    look_at=(0, 0, 0))

floor_material = kb.PrincipledBSDFMaterial(color=kb.get_color('gray'),
                                           roughness=1., specular=0.)
floor = kb.Cube(scale=(10, 10, 1), position=(0, 0, -1),
                material=floor_material, friction=0.3,
                static=True, background=True)
scene.add(floor)

scene.add_all(*klevr.get_lights())

# --- place random objects
spawn_region = [(-4, -4, 0), (4, 4, 3)]
velocity_range = [(-4, -4, 0), (4, 4, 0)]
nr_objects = rng.integers(FLAGS.min_nr_objects, FLAGS.max_nr_objects)
for i in range(nr_objects):
  obj = gso.create(asset_id=gso.db.sample(random_state=rng).iloc[0]['id'],
                   scale=8)
  scene.add(obj)
  kb.move_until_no_overlap(obj, simulator, spawn_region=spawn_region, rng=rng)
  obj.velocity = rng.uniform(*velocity_range) - [obj.position[0], obj.position[1], 0]

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
    "asset_id": obj.asset_id,
    "mass": obj.mass,
    "animation": obj.keyframes,
  } for obj in scene.foreground_assets]

with open(pathlib.Path(FLAGS.output_dir) / "metadata.pkl", "wb") as fp:
  logging.info(f"Writing {fp.name}")
  pickle.dump(metadata, fp)
