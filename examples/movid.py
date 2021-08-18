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
"""
Worker file for the Multi-Object Video (MOVid) dataset.

It creates a scene with a number of static objects lying on the ground,
and a few objects being tossed onto them.
Many aspects of this scene are configurable.

Objects
  * The number of static objects is randomly chosen between
    --min_num_static_objects and --max_num_static_objects
  * The number of dynamic objects is randomly chosen between
    --min_num_dynamic_objects and --max_num_dynamic_objects
  * The objects are randomly chosen from one of three sets (--objects_set):
    1. "clevr" refers to the objects from the CLEVR dataset i.e. plastic or metallic
       cubes, cylinders and spheres in one of eight different colors and two sizes.
    2. "kubasic" is a superset of "clevr" that contains eight additional shapes:
       Cone, Torus, Gear, TorusKnot, Sponge, Spot, Teapot, and Suzanne
       The "kubasic" objects also use uniformly sampled hues as color and vary
       continuously in size
    3. "gso" refers to the set of Google Scanned Objects and consists of roughly
       1000 scanned household items (shoes, toys, appliances, and other products)
       They come with a fixed high-quality texture and are not varied in scale.
  * --object_friction and --object_restitution control the friction and bounciness of
    dynamic objects during the physics simulation. They default to None in which case
    "clevr" and "kubasic" objects have friction and restitution according to their material,
    and "gso" objects have a friction and restitution of 0.5 each.

Background

  * background
    1. "clevr"
    2. "hdri"
  * backgrounds_split
  * --floor_friction and --floor_restitution control the friction and bounciness of
    the floor during the physics simulation.

Camera
  1. clevr
  2. random
  3. linear_movement
    - max_camera_movement


MOVid-A
  --camera=clevr --background=clevr --objects_set=clevr
  --min_num_dynamic_objects=3 --max_num_dynamic_objects=10
  --min_num_static_objects=0 --max_num_static_objects=0

MOVid-B
  --camera=random --background=colored --objects_set=kubasic
  --min_num_dynamic_objects=3 --max_num_dynamic_objects=10
  --min_num_static_objects=0 --max_num_static_objects=0

MOVid-C
  --camera=random --background=hdri --objects_set=gso
  --min_num_dynamic_objects=3 --max_num_dynamic_objects=10
  --min_num_static_objects=0 --max_num_static_objects=0
  --save_state=False

MOVid-D
  --camera=random --background=hdri --objects_set=gso
  --min_num_dynamic_objects=1 --max_num_dynamic_objects=3
  --min_num_static_objects=10 --max_num_static_objects=20
  --save_state=False

MOVid-E
  --camera=linear_movement --background=hdri --objects_set=gso
  --min_num_dynamic_objects=1 --max_num_dynamic_objects=3
  --min_num_static_objects=10 --max_num_static_objects=20
  --save_state=False
"""

import logging

import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
import numpy as np


# --- Some configuration values
# the region in which to place objects [(min), (max)]
STATIC_SPAWN_REGION = [(-7, -7, 0), (7, 7, 4)]
DYNAMIC_SPAWN_REGION = [(-5, -5, 1), (5, 5, 5)]
CAMERA_RANGE = [[-10, -10, 1], [10, 10, 3]]

# --- CLI arguments
parser = kb.ArgumentParser()
# Configuration for the objects of the scene
parser.add_argument("--objects_set", choices=["clevr", "kubasic", "gso"], default="clevr")
parser.add_argument("--objects_split", choices=["train", "test"], default="train")
parser.add_argument("--min_num_static_objects", type=int, default=10,
                    help="minimum number of static (distractor) objects")
parser.add_argument("--max_num_static_objects", type=int, default=20,
                    help="maximum number of static (distractor) objects")
parser.add_argument("--min_num_dynamic_objects", type=int, default=1,
                    help="minimum number of dynamic (tossed) objects")
parser.add_argument("--max_num_dynamic_objects", type=int, default=3,
                    help="maximum number of static (distractor) objects")
parser.add_argument("--object_friction", type=float, default=None)
parser.add_argument("--object_restitution", type=float, default=None)
parser.add_argument("--objects_all_same", action="store_true", default=False)
# Configuration for the floor and background
parser.add_argument("--floor_friction", type=float, default=0.3)
parser.add_argument("--floor_restitution", type=float, default=0.5)
parser.add_argument("--background", choices=["clevr", "colored", "hdri"], default="clevr")
parser.add_argument("--backgrounds_split", choices=["train", "test"], default="train")

# Configuration for the camera
parser.add_argument("--camera", choices=["clevr", "random", "linear_movement"], default="clevr")
parser.add_argument("--max_camera_movement", type=float, default=4.0)

# Configuration for the source of the assets
parser.add_argument("--kubasic_assets_dir", type=str, default="gs://kubric-public/KuBasic")
parser.add_argument("--gso_assets_dir", type=str, default="gs://kubric-public/GSO")
parser.add_argument("--hdri_dir", type=str, default="gs://kubric-public/hdri_haven/4k")

parser.add_argument("--no_save_state", dest="save_state", action="store_false")
parser.add_argument("--save_state", dest="save_state", action="store_true")
parser.set_defaults(save_state=True, frame_end=24, frame_rate=12, width=256, height=256)
FLAGS = parser.parse_args()

# --- Common setups & resources
scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
simulator = PyBullet(scene, scratch_dir)
renderer = Blender(scene, scratch_dir, use_denoising=True, adaptive_sampling=False)


# --- Populate the scene
logging.info("Creating a large gray floor...")
floor_material = kb.PrincipledBSDFMaterial(color=kb.Color.from_name("gray"),
                                           roughness=1., specular=0.)
floor = kb.Cube(name="floor", scale=(100, 100, 1), position=(0, 0, -1),
                material=floor_material, friction=1.0,
                restitution=0,  # friction and restitution are set later
                static=True, background=True)
scene.add(floor)

scene_metadata = {}
if FLAGS.background == "clevr":
  logging.info("Adding four (studio) lights to the scene similar to the CLEVR setup...")
  scene.add(kb.assets.utils.get_clevr_lights(rng=rng))
  scene.ambient_illumination = kb.Color(0.05, 0.05, 0.05)
  scene_metadata["background"] = "clevr"
if FLAGS.background == "colored":
  logging.info("Adding four (studio) lights to the scene similar to the CLEVR setup...")
  scene.add(kb.assets.utils.get_clevr_lights(rng=rng))
  scene.ambient_illumination = kb.Color(0.05, 0.05, 0.05)
  hdri_source = kb.TextureSource(FLAGS.hdri_dir)
  dome = kb.assets.utils.add_hdri_dome(hdri_source, scene, None)
  bg_color = kb.random_hue_color()
  dome.material = kb.PrincipledBSDFMaterial(color=bg_color, roughness=1., specular=0.)
  scene_metadata["background"] = bg_color.hexstr
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

  dome = kb.assets.utils.add_hdri_dome(hdri_source, scene, background_hdri)
  renderer._set_ambient_light_hdri(background_hdri.filename)
  scene_metadata["background"] = kb.as_path(background_hdri.filename).stem


def sample_point_in_half_sphere_shell(inner_radius, outer_radius):
  while True:
    v = rng.uniform((-outer_radius, -outer_radius, 0),
                    (outer_radius, outer_radius, outer_radius))
    len_v = np.linalg.norm(v)
    if inner_radius <= len_v <= outer_radius:
      return v


def get_linear_camera_motion_start_end(inner_radius=8., outer_radius=12.):
  while True:
    camera_start = sample_point_in_half_sphere_shell(inner_radius, outer_radius)
    movement_speed = rng.uniform(low=0., high=FLAGS.max_camera_movement)
    direction = rng.rand(3) - 0.5
    movement = direction / np.linalg.norm(direction) * movement_speed
    camera_end = camera_start + movement
    camera_end_length = np.linalg.norm(camera_end)
    if (inner_radius <= camera_end_length <= outer_radius) and camera_end[2] > 0.:
      return camera_start, camera_end


logging.info("Setting up the Camera...")
scene.camera = kb.PerspectiveCamera(focal_length=35., sensor_width=32)
if FLAGS.camera == "clevr":
  # Specific position + jitter + look at origin
  scene.camera.position = [7.48113, -6.50764, 5.34367] + rng.rand(3)
  scene.camera.look_at((0, 0, 0))
if FLAGS.camera == "random":
  scene.camera.position = sample_point_in_half_sphere_shell(8., 12.)
  scene.camera.look_at((0, 0, 0))
if FLAGS.camera == "linear_movement":
  camera_start, camera_end = get_linear_camera_motion_start_end()
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
  if FLAGS.objects_all_same:
    active_split = [rng.choice(active_split)]


# --- Place random objects
def add_random_object(spawn_region, rng, use_init_velocity=True):
  velocity_range = [(-4., -4., 0.), (4., 4., 0.)]
  if FLAGS.objects_set == "clevr":
    obj = kb.assets.utils.get_random_kubasic_object(
        asset_source, objects_set="clevr", color_strategy="clevr", size_strategy="clevr", rng=rng)
  elif FLAGS.objects_set == "kubasic":
    obj = kb.assets.utils.get_random_kubasic_object(
        asset_source, objects_set="kubasic", color_strategy="uniform_hue",
        size_strategy="uniform", rng=rng)
  else:  # FLAGS.objects_set == "gso":
    obj = asset_source.create(asset_id=rng.choice(active_split), scale=8.0,
                              friction=0.5, restitution=0.5)
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
  kb.move_until_no_overlap(obj, simulator, spawn_region=spawn_region, rng=rng)
  # bias velocity towards center
  if use_init_velocity:
    obj.velocity = rng.uniform(*velocity_range) - [obj.position[0], obj.position[1], 0]
  else:
    obj.velocity = (0., 0., 0.)
  logging.info("    Added %s at %s", obj.asset_id, obj.position)
  return obj


num_static_objects = rng.randint(FLAGS.min_num_static_objects,
                                 FLAGS.max_num_static_objects+1)
logging.info("Randomly placing %d static objects:", num_static_objects)
for i in range(num_static_objects):
  obj = add_random_object(spawn_region=STATIC_SPAWN_REGION, rng=rng, use_init_velocity=False)
  obj.friction = 1.0
  obj.metadata["is_dynamic"] = False


# --- Simulation
logging.info("Running 100 frames of simulation to let static objects settle ...")
_, _ = simulator.run(frame_start=-100, frame_end=0)
for obj in scene.foreground_assets:
  # stop any objects that are still moving/rolling
  if hasattr(obj, "velocity"):
    obj.velocity = (0., 0., 0.)

floor.friction = FLAGS.floor_friction
floor.restitution = FLAGS.floor_restitution

# Add dynamic objects
num_dynamic_objects = rng.randint(FLAGS.min_num_dynamic_objects,
                                  FLAGS.max_num_dynamic_objects+1)
logging.info("Randomly placing %d dynamic objects:", num_dynamic_objects)
for i in range(num_dynamic_objects):
  obj = add_random_object(spawn_region=DYNAMIC_SPAWN_REGION, rng=rng, use_init_velocity=True)
  obj.metadata["is_dynamic"] = True


if FLAGS.save_state:
  logging.info("Saving the simulator state to '%s' before starting the simulation.",
               output_dir / "scene.bullet")
  simulator.save_state(output_dir / "scene.bullet")

# Run dynamic objects simulation
logging.info("Running the simulation ...")
animation, collisions = simulator.run(frame_start=0, frame_end=scene.frame_end+1)


# --- Rendering
if FLAGS.save_state:
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
scene_metadata["num_instances"] = len(visible_foreground_assets)
# Save to image files
kb.utils.write_image_dict(data_stack, output_dir)

kb.post_processing.compute_bboxes(data_stack["segmentation"], visible_foreground_assets)


# --- Metadata
logging.info("Collecting and storing metadata for each object.")
kb.write_json(filename=output_dir / "metadata.json", data={
    "metadata": kb.get_scene_metadata(scene, **scene_metadata),
    "camera": kb.get_camera_info(scene.camera),
    "instances": kb.get_instance_info(scene, assets_subset=visible_foreground_assets),
})
kb.write_json(filename=output_dir / "events.json", data={
    "collisions":  kb.process_collisions(collisions, scene, assets_subset=visible_foreground_assets),
})

kb.done()
