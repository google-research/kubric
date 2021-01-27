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
import pickle
import pathlib

import numpy as np
import kubric as kb

# --- parser
parser = kb.ArgumentParser()
parser.add_argument("--min_nr_objects", type=int, default=4)
parser.add_argument("--max_nr_objects", type=int, default=10)
parser.add_argument("--max_placement_trials", type=int, default=100)
parser.add_argument("--render_dir", type=str, default=tempfile.mkdtemp())
parser.add_argument("--output_dir", type=str, default="output/")
parser.add_argument("--asset_source", type=str, default="./assets/KLEVR")
FLAGS = parser.parse_args()

# --- common setups & resources
kb.setup_logging(FLAGS.logging_level)
kb.log_my_flags(FLAGS)
rng = np.random.default_rng(seed=FLAGS.random_seed)
scene = kb.Scene.from_flags(FLAGS)
simulator = kb.simulator.PyBullet(scene)
renderer = kb.renderer.Blender(scene)

# --- Assemble the basic scene
klevr = kb.assets.KLEVR(FLAGS.asset_source)

scene.camera = klevr.get_camera()
scene.add_all(*klevr.get_lights())
scene.add(klevr.get_floor())

# --- Placer
velocity_range = [(-4, -4, 0), (4, 4, 0)]
nr_objects = rng.integers(FLAGS.min_nr_objects, FLAGS.max_nr_objects)
for i in range(nr_objects):
  obj = klevr.create_random_object(rng)
  scene.add(obj)
  kb.move_until_no_overlap(obj, simulator, spawn_region=[(-4, -4, 0), (4, 4, 3)])
  # bias velocity towards center
  obj.velocity = (rng.uniform(*velocity_range) - [obj.position[0], obj.position[1], 0])

# --- Simulation
if FLAGS.output_dir:
  simulator.save_state(FLAGS.output_dir)
animation = simulator.run()

# --- Transfer simulation to renderer keyframes
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

# --- Post-process renderer output
if FLAGS.output_dir and FLAGS.render_dir: 
  # Parse renderer-specific output into per-frame numpy pickles
  renderer.postprocess(from_dir=FLAGS.render_dir, to_dir=FLAGS.output_dir)

  # Extracting metadata.pkl from the simulation
  metadata = [{
      "asset_id": obj.asset_id,
      "material": "Metal" if "Metal" in obj.asset_id else "Rubber",
      "mass": obj.mass,
      "color": obj.material.color.rgb,
      "animation": obj.keyframes,
  } for obj in scene.foreground_assets]
  with open(pathlib.Path(FLAGS.output_dir) / "metadata.pkl", "wb") as fp:
    logging.info(f"Writing {fp.name}")
    pickle.dump(metadata, fp)
