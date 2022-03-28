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
Worker file for the Multi-Object Video (MOVi) C (and CC) datasets.
  * The number of objects is randomly chosen between
    --min_num_objects (3) and --max_num_objects (10)
  * The objects are randomly chosen from the Google Scanned Objects dataset

  * Background is an random HDRI from the HDRI Haven dataset,
    projected onto a Dome (half-sphere).
    The HDRI is also used for lighting the scene.
"""

import logging

import bpy
import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
import numpy as np


# --- Some configuration values
# the region in which to place objects [(min), (max)]
SPAWN_REGION = [(-5, -5, 1), (5, 5, 5)]
VELOCITY_RANGE = [(-4., -4., 0.), (4., 4., 0.)]

# --- CLI arguments
parser = kb.ArgumentParser()
parser.add_argument("--objects_split", choices=["train", "test"],
                    default="train")
# Configuration for the objects of the scene
parser.add_argument("--min_num_objects", type=int, default=3,
                    help="minimum number of objects")
parser.add_argument("--max_num_objects", type=int, default=10,
                    help="maximum number of objects")
# Configuration for the floor and background
parser.add_argument("--floor_friction", type=float, default=0.3)
parser.add_argument("--floor_restitution", type=float, default=0.5)
parser.add_argument("--backgrounds_split", choices=["train", "test"],
                    default="train")

parser.add_argument("--camera", choices=["fixed_random", "linear_movement"],
                    default="fixed_random")
parser.add_argument("--max_camera_movement", type=float, default=4.0)


# Configuration for the source of the assets
parser.add_argument("--kubasic_assets", type=str,
                    default="gs://kubric-public/assets/KuBasic/KuBasic.json")
parser.add_argument("--hdri_assets", type=str,
                    default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json")
parser.add_argument("--gso_assets", type=str,
                    default="gs://kubric-public/assets/GSO/GSO.json")
parser.add_argument("--save_state", dest="save_state", action="store_true")
parser.set_defaults(save_state=False, frame_end=24, frame_rate=12,
                    resolution=256)
FLAGS = parser.parse_args()

# --- Common setups & resources
scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
simulator = PyBullet(scene, scratch_dir)
renderer = Blender(scene, scratch_dir, samples_per_pixel=64)
kubasic = kb.AssetSource.from_manifest(FLAGS.kubasic_assets)
gso = kb.AssetSource.from_manifest(FLAGS.gso_assets)
hdri_source = kb.AssetSource.from_manifest(FLAGS.hdri_assets)


# --- Populate the scene
# background HDRI
train_backgrounds, test_backgrounds = hdri_source.get_test_split(fraction=0.1)
if FLAGS.backgrounds_split == "train":
  logging.info("Choosing one of the %d training backgrounds...", len(train_backgrounds))
  hdri_id = rng.choice(train_backgrounds)
else:
  logging.info("Choosing one of the %d held-out backgrounds...", len(test_backgrounds))
  hdri_id = rng.choice(test_backgrounds)
background_hdri = hdri_source.create(asset_id=hdri_id)
#assert isinstance(background_hdri, kb.Texture)
logging.info("Using background %s", hdri_id)
scene.metadata["background"] = hdri_id
renderer._set_ambient_light_hdri(background_hdri.filename)

# Dome
dome = kubasic.create(asset_id="dome", name="dome",
                      friction=FLAGS.floor_friction,
                      restitution=FLAGS.floor_restitution,
                      static=True, background=True)
assert isinstance(dome, kb.FileBasedObject)
scene += dome
dome_blender = dome.linked_objects[renderer]
texture_node = dome_blender.data.materials[0].node_tree.nodes["Image Texture"]
texture_node.image = bpy.data.images.load(background_hdri.filename)



def get_linear_camera_motion_start_end(
    movement_speed: float,
    inner_radius: float = 8.,
    outer_radius: float = 12.,
    z_offset: float = 0.1,
):
  """Sample a linear path which starts and ends within a half-sphere shell."""
  while True:
    camera_start = np.array(kb.sample_point_in_half_sphere_shell(inner_radius,
                                                                 outer_radius,
                                                                 z_offset))
    direction = rng.rand(3) - 0.5
    movement = direction / np.linalg.norm(direction) * movement_speed
    camera_end = camera_start + movement
    if (inner_radius <= np.linalg.norm(camera_end) <= outer_radius and
        camera_end[2] > z_offset):
      return camera_start, camera_end


# Camera
logging.info("Setting up the Camera...")
scene.camera = kb.PerspectiveCamera(focal_length=35., sensor_width=32)
if FLAGS.camera == "fixed_random":
  scene.camera.position = kb.sample_point_in_half_sphere_shell(
      inner_radius=7., outer_radius=9., offset=0.1)
  scene.camera.look_at((0, 0, 0))
elif FLAGS.camera == "linear_movement":
  camera_start, camera_end = get_linear_camera_motion_start_end(
      movement_speed=rng.uniform(low=0., high=FLAGS.max_camera_movement)
  )
  # linearly interpolate the camera position between these two points
  # while keeping it focused on the center of the scene
  # we start one frame early and end one frame late to ensure that
  # forward and backward flow are still consistent for the last and first frames
  for frame in range(FLAGS.frame_start - 1, FLAGS.frame_end + 2):
    interp = ((frame - FLAGS.frame_start + 1) /
              (FLAGS.frame_end - FLAGS.frame_start + 3))
    scene.camera.position = (interp * np.array(camera_start) +
                             (1 - interp) * np.array(camera_end))
    scene.camera.look_at((0, 0, 0))
    scene.camera.keyframe_insert("position", frame)
    scene.camera.keyframe_insert("quaternion", frame)


# Add random objects
train_split, test_split = gso.get_test_split(fraction=0.1)
if FLAGS.objects_split == "train":
  logging.info("Choosing one of the %d training objects...", len(train_split))
  active_split = train_split
else:
  logging.info("Choosing one of the %d held-out objects...", len(test_split))
  active_split = test_split


num_objects = rng.randint(FLAGS.min_num_objects,
                          FLAGS.max_num_objects+1)
logging.info("Randomly placing %d objects:", num_objects)
for i in range(num_objects):
  obj = gso.create(asset_id=rng.choice(active_split))
  assert isinstance(obj, kb.FileBasedObject)
  scale = rng.uniform(0.75, 3.0)
  obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
  obj.metadata["scale"] = scale
  scene += obj
  kb.move_until_no_overlap(obj, simulator, spawn_region=SPAWN_REGION, rng=rng)
  # initialize velocity randomly but biased towards center
  obj.velocity = (rng.uniform(*VELOCITY_RANGE) -
                  [obj.position[0], obj.position[1], 0])
  logging.info("    Added %s at %s", obj.asset_id, obj.position)


if FLAGS.save_state:
  logging.info("Saving the simulator state to '%s' prior to the simulation.",
               output_dir / "scene.bullet")
  simulator.save_state(output_dir / "scene.bullet")

# Run dynamic objects simulation
logging.info("Running the simulation ...")
animation, collisions = simulator.run(frame_start=0,
                                      frame_end=scene.frame_end+1)

# --- Rendering
if FLAGS.save_state:
  logging.info("Saving the renderer state to '%s' ",
               output_dir / "scene.blend")
  renderer.save_state(output_dir / "scene.blend")


logging.info("Rendering the scene ...")
data_stack = renderer.render()

# --- Postprocessing
kb.compute_visibility(data_stack["segmentation"], scene.assets)
visible_foreground_assets = [asset for asset in scene.foreground_assets
                             if np.max(asset.metadata["visibility"]) > 0]
visible_foreground_assets = sorted(  # sort assets by their visibility
    visible_foreground_assets,
    key=lambda asset: np.sum(asset.metadata["visibility"]),
    reverse=True)

data_stack["segmentation"] = kb.adjust_segmentation_idxs(
    data_stack["segmentation"],
    scene.assets,
    visible_foreground_assets)
scene.metadata["num_instances"] = len(visible_foreground_assets)

# Save to image files
kb.write_image_dict(data_stack, output_dir)
kb.post_processing.compute_bboxes(data_stack["segmentation"],
                                  visible_foreground_assets)

# --- Metadata
logging.info("Collecting and storing metadata for each object.")
kb.write_json(filename=output_dir / "metadata.json", data={
    "flags": vars(FLAGS),
    "metadata": kb.get_scene_metadata(scene),
    "camera": kb.get_camera_info(scene.camera),
    "instances": kb.get_instance_info(scene, visible_foreground_assets),
})
kb.write_json(filename=output_dir / "events.json", data={
    "collisions":  kb.process_collisions(
        collisions, scene, assets_subset=visible_foreground_assets),
})

kb.done()
