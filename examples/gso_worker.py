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
import copy
import logging
import pickle

import numpy as np
import tensorflow as tf
import kubric as kb


# --- Some configuration values
# the region in which to place objects [(min), (max)]
SPAWN_REGION = [(-4, -4, 0), (4, 4, 3)]
# the range of velocities from which to sample [(min), (max)]
VELOCITY_RANGE = [(-4, -4, 0), (4, 4, 0)]

# --- CLI arguments
parser = kb.ArgumentParser()
parser.add_argument("--min_nr_objects", type=int, default=3)
parser.add_argument("--max_nr_objects", type=int, default=10)
parser.add_argument("--assets_dir", type=str, default="gs://kubric-public/GSO")
parser.set_defaults(frame_end=48)
parser.set_defaults(width=512)
parser.set_defaults(height=512)
FLAGS = parser.parse_args()

# --- Common setups & resources
kb.setup_logging(FLAGS.logging_level)
kb.log_my_flags(FLAGS)
scratch_dir, output_dir = kb.setup_directories(FLAGS)
seed = FLAGS.seed if FLAGS.seed else np.random.randint(0, 2147483647)
rng = np.random.RandomState(seed=seed)
scene = kb.Scene.from_flags(FLAGS)
simulator = kb.simulator.PyBullet(scene, scratch_dir)
renderer = kb.renderer.Blender(scene, scratch_dir)

logging.info("Loading assets from %s", FLAGS.assets_dir)
gso = kb.AssetSource(FLAGS.assets_dir)


# --- Populate the scene
logging.info("Creating a large gray cube as the floor...")
floor_material = kb.PrincipledBSDFMaterial(color=kb.get_color('gray'), roughness=1., specular=0.)
floor = kb.Cube(scale=(100, 100, 1), position=(0, 0, -1), material=floor_material, friction=0.3,
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
nr_objects = rng.randint(FLAGS.min_nr_objects, FLAGS.max_nr_objects)
logging.info("Randomly placing %d objects:", nr_objects)

object_info = []
for i in range(nr_objects):
  asset_id = gso.db.sample(random_state=rng).iloc[0]['id']
  obj = gso.create(asset_id=asset_id, scale=8)
  obj.friction = 0.5
  obj.restitution = 0.5
  scene.add(obj)
  kb.move_until_no_overlap(obj, simulator, spawn_region=SPAWN_REGION)
  # bias velocity towards center
  obj.velocity = (rng.uniform(*VELOCITY_RANGE) - [obj.position[0], obj.position[1], 0])
  logging.info("    Added %s", obj)


# --- Simulation
logging.info("Saving the simulator state to '%s' before starting the simulation.",
             output_dir / "scene.bullet")
simulator.save_state(output_dir / "scene.bullet")
logging.info("Running the Simulation ...")
animation = simulator.run()

# --- Rendering
logging.info("Saving the renderer state to '%s' before starting the rendering.",
             output_dir / "scene.blend")
renderer.save_state(output_dir / "scene.blend")
logging.info("Rendering the scene ...")
renderer.render()


# --- Postprocessing
logging.info("Parse and post-process renderer-specific output into per-frame numpy pickles.")
renderer.postprocess(from_dir=scratch_dir, to_dir=output_dir)

# --- Metadata
logging.info("Collecting and storing metadata for each object.")

object_info = []
frames = list(range(scene.frame_start, scene.frame_end+1))
# extract the framewise position, quaternion, and velocity for each object
for obj in scene.foreground_assets:
  info = copy.copy(obj.metadata)
  info['positions'] = np.array([obj.keyframes['position'][f] for f in frames], dtype=np.float32)
  info['quaternions'] = np.array([obj.keyframes['quaternion'][f] for f in frames], dtype=np.float32)
  info['velocities'] = np.array([obj.keyframes['velocity'][f] for f in frames], dtype=np.float32)
  info['angular_velocities'] = np.array([obj.keyframes['angular_velocity'][f] for f in frames], dtype=np.float32)

  info['mass'] = obj.mass
  info['friction'] = obj.friction
  info['restitution'] = obj.restitution
  info['image_positions'] = np.zeros((len(frames), 2), dtype=np.float32)  # TODO: actually compute
  object_info.append(info)

cam = scene.camera
metadata = {
    "seed": seed,
    "nr_objects": nr_objects,
    "objects": object_info,
    "camera": {
        "focal_length": cam.focal_length,
        "sensor_width": cam.sensor_width,
        "field_of_view": cam.field_of_view,
        "positions": np.array([cam.position for _ in frames], dtype=np.float32),
        "quaternions": np.array([cam.quaternion for _ in frames], dtype=np.float32)
    }
}

with tf.io.gfile.GFile(output_dir / "metadata.pkl", "wb") as fp:
  logging.info(f"Writing to {fp.name}")
  pickle.dump(metadata, fp)

logging.info("Done!")

# -- report generated_images to hyperparameter tuner
import hypertune

hpt = hypertune.HyperTune()
hpt.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag='generated_images',
    metric_value=len(frames))
