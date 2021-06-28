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

import numpy as np
import kubric as kb


# --- Some configuration values
# the region in which to place objects [(min), (max)]
SPAWN_REGION = [(-4, -4, 0), (4, 4, 3)]
# the range of velocities from which to sample [(min), (max)]
VELOCITY_RANGE = [(-4, -4, 0), (4, 4, 0)]
# the names of the KuBasic assets to use
OBJECT_TYPES = ["Cube", "Cylinder", "Sphere", "Cone", "Torus", "Gear", "TorusKnot", "Sponge",
                "Spot", "Teapot", "Suzanne"]


# --- CLI arguments
parser = kb.ArgumentParser()
parser.add_argument("--min_num_objects", type=int, default=3)
parser.add_argument("--max_num_objects", type=int, default=10)
parser.add_argument("--object_types", choices=["clevr", "kubasic", "gso"], default="clevr")
parser.add_argument("--background", choices=["clevr", "hdri"])
parser.add_argument("--camera", choices=["clevr", "moving"], default="clevr")
parser.add_argument("--assets_dir", type=str, default="gs://kubric-public/KuBasic")
parser.add_argument("--hdri_dir", type=str, default="gs://kubric-public/hdri_haven/4k")
parser.add_argument("--objects_set", choices=["train", "test"], default="train")
parser.add_argument("--backgrounds_set", choices=["train", "test"], default="train")
parser.set_defaults(frame_end=24, frame_rate=12, width=256, height=256)
FLAGS = parser.parse_args()
objects_types = OBJECT_TYPES if "ALL" in FLAGS.object_types else FLAGS.object_types


# --- Common setups & resources

scene, simulator, renderer, rng, output_dir = kb.setup(FLAGS)


logging.info("Loading assets from %s", FLAGS.assets_dir)
kubasic = kb.AssetSource(FLAGS.assets_dir)


# --- Populate the scene
if FLAGS.background == "clevr":
  logging.info("Adding four (studio) lights to the scene similar to the CLEVR setup...")
  scene.add(kb.assets.utils.get_clevr_lights(FLAGS.light_jitter), rng=rng)
  scene.ambient_illumination = kb.Color(0.05, 0.05, 0.05)

  logging.info("Creating a large gray floor...")
  floor_material = kb.PrincipledBSDFMaterial(color="gray", roughness=1., specular=0.)
  scene.add(kb.Cube(name="floor", scale=(100, 100, 1), position=(0, 0, -1),
                    material=floor_material, friction=FLAGS.floor_friction, static=True,
                    background=True))

elif FLAGS.background == "hdri":
  logging.info("Loading background HDRIs from %s", FLAGS.hdri_dir)
  hdri_source = kb.TextureSource(FLAGS.hdri_dir)
  train_backgrounds, held_out_backgrounds = hdri_source.get_test_split(fraction=0.1)
  if FLAGS.backgrounds_set == "train":
    logging.info("Choosing one of the %d training backgrounds...", len(train_backgrounds))
    background_hdri = hdri_source.create(texture_name=rng.choice(train_backgrounds))
  else:
    logging.info("Choosing one of the %d held-out backgrounds...", len(held_out_backgrounds))
    background_hdri = hdri_source.create(texture_name=rng.choice(held_out_backgrounds))

  renderer._set_background_hdri(background_hdri.filename)
  dome = kb.assets.utils.add_hdri_dome(hdri_source, background_hdri, scene)


logging.info("Setting up the Camera...")
if FLAGS.camera == "clevr":
  scene.camera = kb.PerspectiveCamera(focal_length=35., sensor_width=32,
                                      position=(7.48113, -6.50764, 5.34367))
  scene.camera.position += rng.rand(3) * FLAGS.camera_jitter
  scene.camera.look_at((0, 0, 0))
if FLAGS.camera == "moving":
  scene.camera = kb.PerspectiveCamera(focal_length=35., sensor_width=32)
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


# --- Place random objects
num_objects = rng.randint(FLAGS.min_num_objects, FLAGS.max_num_objects)
logging.info("Randomly placing %d objects:", num_objects)
for i in range(num_objects):
  obj = kb.assets.utils.get_random_clevr_object(kubasic, object_types=FLAGS.object_types,
                                                color_strategy=FLAGS.object_colors,
                                                rng=rng)
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
renderer.postprocess(from_dir=scratch_dir, to_dir=output_dir)

# --- Metadata
logging.info("Collecting and storing metadata for each object.")
kb.save_as_pkl(output_dir / "metadata.pkl", {
    "metadata": kb.get_scene_metadata(scene, seed=seed),
    "camera": kb.get_camera_info(scene.camera),
    "instances": kb.get_instance_info(scene),
    "events": {"collisions":  kb.process_collisions(collisions, scene)},
    "background": {
        "floor_color": floor_color.rgb,
        "floor_friction": FLAGS.floor_friction,
    }
})

kb.done()
