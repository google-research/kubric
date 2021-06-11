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
import logging
import pathlib

import bpy
import numpy as np
import kubric as kb
from kubric.renderer.blender import Blender as KubricRenderer
from kubric.simulator.pybullet import PyBullet as KubricSimulator

# --- Some configuration values
# the region in which to place objects [(min), (max)]
SPAWN_REGION = [(-4, -4, 0), (4, 4, 3)]
# the range of velocities from which to sample [(min), (max)]
VELOCITY_RANGE = [(-4, -4, 0), (4, 4, 0)]
# the names of the KuBasic assets to use


# --- CLI arguments
parser = kb.ArgumentParser()
parser.add_argument("--min_num_objects", type=int, default=3)
parser.add_argument("--max_num_objects", type=int, default=10)
parser.add_argument("--floor_friction", type=float, default=0.3)
parser.add_argument("--camera_jitter", type=float, default=1.0)
parser.add_argument("--object_scale", type=float, default=8.0)
parser.add_argument("--background", choices=["hdri", "clevr", "dome"], default="hdri")
parser.add_argument("--camera", choices=["fixed", "moving"], default="fixed")
parser.add_argument("--assets_dir", type=str, default="gs://kubric-public/GSO")
parser.add_argument("--hdri_dir", type=str, default="gs://kubric-public/hdri_haven/4k")
parser.add_argument("--objects_set", choices=["train", "test"], default="train")
parser.add_argument("--backgrounds_set", choices=["train", "test"], default="train")

parser.set_defaults(frame_end=24, frame_rate=12, width=128, height=128)

FLAGS = parser.parse_args()

# --- Common setups & resources
kb.setup_logging(FLAGS.logging_level)
kb.log_my_flags(FLAGS)
scratch_dir, output_dir = kb.setup_directories(FLAGS)
seed = FLAGS.seed if FLAGS.seed else np.random.randint(0, 2147483647)
rng = np.random.RandomState(seed=seed)
scene = kb.Scene.from_flags(FLAGS)
renderer = KubricRenderer(scene, scratch_dir)
simulator = KubricSimulator(scene, scratch_dir)

gso = kb.AssetSource(FLAGS.assets_dir)
hdris = kb.TextureSource(FLAGS.hdri_dir)


# --- Populate the scene
logging.info("Creating a large cube as the floor...")

floor = kb.Cube(scale=(100, 100, 1), position=(0, 0, -1.01),
                friction=FLAGS.floor_friction, static=True, background=True)
scene.add(floor)

# sample a background image (either from the train or held-out set)
held_out_backgrounds = list(hdris.db.sample(frac=0.1, replace=False, random_state=42)['id'])
train_backgrounds = [i for i in hdris.db["id"] if i not in held_out_backgrounds]
if FLAGS.backgrounds_set == "train":
  logging.info("Choosing one of the %d training backgrounds...", len(train_backgrounds))
  background_hdri = hdris.create(texture_name=rng.choice(train_backgrounds))
else:
  logging.info("Choosing one of the %d held-out backgrounds...", len(held_out_backgrounds))
  background_hdri = hdris.create(texture_name=rng.choice(held_out_backgrounds))

# set the background
renderer._set_ambient_light_hdri(background_hdri.filename)
renderer.use_denoising = False  # somehow leads to blurry HDRIs

if FLAGS.background == "hdri":
  renderer._set_background_hdri(background_hdri.filename)
  floor.linked_objects[renderer].cycles.is_shadow_catcher = True

elif FLAGS.background == "dome":
  # TODO: this is one large dirty hack!
  renderer._set_background_hdri(background_hdri.filename)
  dome_path = hdris.fetch("dome.blend")
  dome = kb.FileBasedObject(
      static=True, background=True,
      simulation_filename=None,
      render_filename=str(dome_path),
      render_import_kwargs={
          "filepath": str(dome_path / "Object" / "Dome"),
          "directory": str(dome_path / "Object"),
          "filename": "Dome",
      })
  scene.add(dome)
  dome_mat = dome.linked_objects[renderer].data.materials[0]
  texture_node = dome_mat.node_tree.nodes["Image Texture"]
  texture_node.image = bpy.data.images.load(background_hdri.filename)
else:
  floor.material = kb.PrincipledBSDFMaterial(color=kb.get_color("gray"), roughness=1., specular=0.)



# --- Place random objects
num_objects = rng.randint(FLAGS.min_num_objects, FLAGS.max_num_objects)
held_out_objects = list(gso.db.sample(frac=0.1, replace=False, random_state=42)['id'])
train_objects = [i for i in gso.db["id"] if i not in held_out_objects]

if FLAGS.objects_set == "train":
  logging.info("Randomly placing %d objects out of the set of %d training objects:",
               num_objects, len(train_objects))
  objects_set = train_objects
else:
  logging.info("Randomly placing %d objects out of the set of %d held-out objects:",
               num_objects, len(held_out_objects))
  objects_set = held_out_objects

object_info = []
for i in range(num_objects):
  obj = gso.create(asset_id=rng.choice(objects_set), scale=FLAGS.object_scale)
  scene.add(obj)
  obj.metadata = {
      "asset_id": obj.asset_id,
      "jft_category": gso.db[gso.db["id"] == obj.asset_id].iloc[0]['jft_category'],
  }
  kb.move_until_no_overlap(obj, simulator, spawn_region=SPAWN_REGION)
  # bias velocity towards center
  obj.velocity = (rng.uniform(*VELOCITY_RANGE) - [obj.position[0], obj.position[1], 0])
  logging.info("    Added %s", obj)


logging.info("Setting up the Camera...")
scene.camera = kb.PerspectiveCamera(focal_length=35., sensor_width=32,
                                    position=(7.5, -6.5, 4.5))
scene.camera.position += (rng.rand(3) - 0.5) * 2 * FLAGS.camera_jitter
scene.camera.look_at((0, 0, 2))

if FLAGS.camera == "moving":
  # sample two points in the cube ([-10, -10, 1], [10, 10, 3])
  camera_range = [[-10, -10, 1], [10, 10, 3]]
  camera_start = rng.uniform(*camera_range)
  camera_end = rng.uniform(*camera_range)
  # linearly interpolate the camera position between these two points
  # while keeping it focused on the center of the scene
  for frame in range(FLAGS.frame_start, FLAGS.frame_end + 1):
    interp = (frame - FLAGS.frame_start) / (FLAGS.frame_end - FLAGS.frame_start + 1)
    scene.camera.position = interp * camera_start + (1 - interp) * camera_end
    scene.camera.look_at((0, 0, 1))
    scene.camera.keyframe_insert("position", frame)
    scene.camera.keyframe_insert("quaternion", frame)

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
logging.info("Parse and post-process renderer-specific output...")
renderer.postprocess(from_dir=scratch_dir, to_dir=output_dir)

# --- Metadata
logging.info("Collecting and storing metadata for each object.")
kb.save_as_json(output_dir / "metadata.json", {
    "metadata": kb.get_scene_metadata(scene, seed=seed),
    "camera": kb.get_camera_info(scene.camera),
    "instances": kb.get_instance_info(scene),
    "events": {"collisions":  kb.process_collisions(collisions, scene)},
    "background": {
        "hdri": str(pathlib.Path(background_hdri.filename).name),
        "floor_friction": FLAGS.floor_friction,
    }
  })

kb.done()
