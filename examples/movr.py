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

import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
import numpy as np


# --- Some configuration values
# the region in which to place objects [(min), (max)]
SPAWN_REGION = [(-4, -4, 0), (4, 4, 3)]
# the range of velocities from which to sample [(min), (max)]

# the names of the KuBasic assets to use


# --- CLI arguments
parser = kb.ArgumentParser()
parser.add_argument("--min_num_objects", type=int, default=3)
parser.add_argument("--max_num_objects", type=int, default=10)
parser.add_argument("--objects_set", choices=["clevr", "kubasic", "gso"], default="clevr")
parser.add_argument("--background", choices=["clevr", "hdri"], default="clevr")
parser.add_argument("--camera", choices=["clevr", "moving"], default="clevr")
parser.add_argument("--max_camera_movement", type=float, default=1.0)
parser.add_argument("--floor_friction", type=float, default=0.3)
parser.add_argument("--floor_restitution", type=float, default=0.5)
parser.add_argument("--object_friction", type=float, default=None)
parser.add_argument("--object_restitution", type=float, default=None)
parser.add_argument("--object_initial_velocity", type=float, default=4.0)
parser.add_argument("--kubasic_assets_dir", type=str, default="gs://kubric-public/KuBasic")
parser.add_argument("--gso_assets_dir", type=str, default="gs://kubric-public/GSO")
parser.add_argument("--hdri_dir", type=str, default="gs://kubric-public/hdri_haven/4k")
parser.add_argument("--objects_split", choices=["train", "test"], default="train")
parser.add_argument("--backgrounds_split", choices=["train", "test"], default="train")

parser.set_defaults(frame_end=24, frame_rate=12, width=256, height=256)
FLAGS = parser.parse_args()

v = FLAGS.object_initial_velocity
VELOCITY_RANGE = [(-v, -v, 0), (v, v, 0)]
del v

# --- Common setups & resources
scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
simulator = PyBullet(scene, scratch_dir)
renderer = Blender(scene, scratch_dir, use_denoising=False, adaptive_sampling=False)


# --- Populate the scene
logging.info("Creating a large gray floor...")
floor_material = kb.PrincipledBSDFMaterial(color=kb.Color.from_name("gray"),
                                           roughness=1., specular=0.)
scene.add(kb.Cube(name="floor", scale=(100, 100, 1), position=(0, 0, -1),
                  material=floor_material, friction=FLAGS.floor_friction,
                  restitution=FLAGS.floor_restitution,
                  static=True, background=True))

if FLAGS.background == "clevr":
  logging.info("Adding four (studio) lights to the scene similar to the CLEVR setup...")
  scene.add(kb.assets.utils.get_clevr_lights(rng=rng))
  scene.ambient_illumination = kb.Color(0.05, 0.05, 0.05)

elif FLAGS.background == "hdri":
  logging.info("Loading background HDRIs from %s", FLAGS.hdri_dir)
  hdri_source = kb.TextureSource(FLAGS.hdri_dir)
  train_backgrounds, held_out_backgrounds = hdri_source.get_test_split(fraction=0.1)
  if FLAGS.backgrounds_split == "train":
    logging.info("Choosing one of the %d training backgrounds...", len(train_backgrounds))
    background_hdri = hdri_source.create(texture_name=rng.choice(train_backgrounds))
  else:
    logging.info("Choosing one of the %d held-out backgrounds...", len(held_out_backgrounds))
    background_hdri = hdri_source.create(texture_name=rng.choice(held_out_backgrounds))

  dome = kb.assets.utils.add_hdri_dome(hdri_source, background_hdri, scene)
  renderer._set_background_hdri(background_hdri.filename)
  renderer._set_ambient_light_hdri(background_hdri.filename)


def scale_to_norm_range(vec, min_length=0, max_length=np.inf):
  length = np.clip(np.linalg.norm(vec), 1e-6, np.inf)
  if length < min_length:
    return vec / length * min_length
  elif length > max_length:
    return vec / length * max_length
  return vec


logging.info("Setting up the Camera...")
scene.camera = kb.PerspectiveCamera(focal_length=35., sensor_width=32)
if FLAGS.camera == "clevr":
  # Specific position + jitter + look at origin
  scene.camera.position = [7.48113, -6.50764, 5.34367] + rng.rand(3)
  scene.camera.look_at((0, 0, 0))
if FLAGS.camera == "moving":
  # sample two points in the cube ([-10, -10, 1], [10, 10, 3])
  # and move the camera from the first towards the second
  # but no further than FLAGS.max_camera_movement
  camera_range = [[-10, -10, 3], [10, 10, 7]]
  camera_start = scale_to_norm_range(rng.uniform(*camera_range), 8, 12)
  camera_end = scale_to_norm_range(rng.uniform(*camera_range), 8, 12)
  camera_movement = scale_to_norm_range(camera_end - camera_start,
                                        max_length=FLAGS.max_camera_movement)
  camera_end = camera_start + camera_movement
  # linearly interpolate the camera position between these two points
  # while keeping it focused on the center of the scene
  # we start one frame early and end one frame late to ensure that
  # forward and backward flow are still consistent for the last and first frames
  for frame in range(FLAGS.frame_start - 1, FLAGS.frame_end + 2):
    interp = (frame - FLAGS.frame_start + 1) / (FLAGS.frame_end - FLAGS.frame_start + 3)
    scene.camera.position = interp * camera_start + (1 - interp) * camera_end
    scene.camera.look_at((0, 0, 0))
    scene.camera.keyframe_insert("position", frame)
    scene.camera.keyframe_insert("quaternion", frame)


# --- Set up asset sources and held-out splits for random objects
assert FLAGS.objects_set in {"clevr", "kubasic", "gso"}, FLAGS.objects_set

if FLAGS.objects_set in {"clevr", "kubasic"}:
  logging.info("Loading assets from %s", FLAGS.kubasic_assets_dir)
  asset_source = kb.AssetSource(FLAGS.kubasic_assets_dir)
  active_split = None
else:  # FLAGS.objects_set == "gso":
  logging.info("Loading assets from %s", FLAGS.gso_assets_dir)
  asset_source = kb.AssetSource(FLAGS.gso_assets_dir)
  train_split, test_split = asset_source.get_test_split(fraction=0.1)
  active_split = train_split if FLAGS.objects_split == "train" else test_split

# --- Place random objects
num_objects = rng.randint(FLAGS.min_num_objects, FLAGS.max_num_objects)
logging.info("Randomly placing %d objects:", num_objects)
for i in range(num_objects):
  if FLAGS.objects_set == "clevr":
    obj = kb.assets.utils.get_random_kubasic_object(
        asset_source, objects_set="clevr", color_strategy="clevr", size_strategy="clevr", rng=rng)
  elif FLAGS.objects_set == "kubasic":
    obj = kb.assets.utils.get_random_kubasic_object(
        asset_source, objects_set="kubasic", color_strategy="uniform_hue",
        size_strategy="uniform", rng=rng)
  else:  # FLAGS.objec_types == "gso":
    obj = asset_source.create(asset_id=rng.choice(active_split), scale=8.0)
    obj.metadata = {
        "asset_id": obj.asset_id,
        "jft_category": asset_source.db[
          asset_source.db["id"] == obj.asset_id].iloc[0]["jft_category"],
    }

  if FLAGS.object_friction is not None:
    obj.friction = FLAGS.object_friction
  if FLAGS.object_restitution is not None:
    obj.restitution = FLAGS.object_restitution
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
animation, collisions = simulator.run()


# --- Rendering
logging.info("Saving the renderer state to '%s' before starting the rendering.",
             output_dir / "scene.blend")
renderer.save_state(output_dir / "scene.blend")
logging.info("Rendering the scene ...")
renderer.render()

# --- Postprocessing
logging.info("Parse and post-process renderer-specific output into per-frame numpy pickles.")
renderer.postprocess(from_dir=renderer.scratch_dir, to_dir=output_dir)

# --- Metadata
logging.info("Collecting and storing metadata for each object.")
kb.save_as_pkl(output_dir / "metadata.pkl", {
    "metadata": kb.get_scene_metadata(scene),
    "camera": kb.get_camera_info(scene.camera),
    "instances": kb.get_instance_info(scene),
    "events": {"collisions":  kb.process_collisions(collisions, scene)},
})

kb.done()
