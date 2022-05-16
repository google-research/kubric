# Copyright 2021 The Kubric Authors
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
Worker file for the "Multi-view object matting" dataset.

This dataset creates a scene where a foreground object is to be distinguished
from the background. Foreground objects are borrowed from shapnet. Backgrounds
are from indoor scenes of polyhaven. All foreground objects are situated on top
of a "table" which is gernated to be random in color. Instead of background
removal with a single image. This dataset is special in that multiple images of
the foreground object (taken from different camera poses) are given. This
"multi-view" persepctive should be very helpful for background removal but is
currently underexplored in the literature.
"""

import math
from pathlib import Path
import logging
import numpy as np
import tensorflow as tf

import kubric as kb
from kubric.renderer import Blender as KubricRenderer
from kubric import file_io


# TODO: go to https://shapenet.org/ create an account and agree to the terms
#       then find the URL for the kubric preprocessed ShapeNet and put it here:
SHAPENET_PATH = "gs://KUBRIC_SHAPENET_PATH/ShapeNetCore.v2.json"

if SHAPENET_PATH == "gs://KUBRIC_SHAPENET_PATH/ShapeNetCore.v2.json":
  raise ValueError("Wrong ShapeNet path. Please visit https://shapenet.org/ "
                   "agree to terms and conditions, and find the correct path.")


def add_hdri_dome(scene, background_hdri=None):
  """ Adding HDRI dome. """

  # Download dome.blend
  dome_path = Path("/kubric/dome.blend")
  if not dome_path.exists():
    tf.io.gfile.copy(FLAGS.hdri_dir + "dome.blend", dome_path)

  dome = kb.FileBasedObject(
      name="BackgroundDome",
      position=(0, 0, 0),
      static=True, background=True,
      simulation_filename=None,
      render_filename=str(dome_path),
      render_import_kwargs={
          "filepath": str(Path(dome_path) / "Object" / "Dome"),
          "directory": str(Path(dome_path) / "Object"),
          "filename": "Dome",
      })
  scene.add(dome)
  # pylint: disable=import-outside-toplevel
  from kubric.renderer import Blender
  import bpy
  blender_renderer = [v for v in scene.views if isinstance(v, Blender)]
  if blender_renderer:
    dome_blender = dome.linked_objects[blender_renderer[0]]
    if bpy.app.version > (3, 0, 0):
      dome_blender.visible_shadow = False
    else:
      dome_blender.cycles_visibility.shadow = False
    if background_hdri is not None:
      dome_mat = dome_blender.data.materials[0]
      texture_node = dome_mat.node_tree.nodes["Image Texture"]
      texture_node.image = bpy.data.images.load(background_hdri.filename)
  return dome

manifest_path = file_io.as_path(SHAPENET_PATH)
manifest = file_io.read_json(manifest_path)
# import pdb; pdb.set_trace()
assets = manifest["assets"]
name = manifest.get("name", manifest_path.stem)  # default to filename
data_dir = manifest.get("data_dir", manifest_path.parent)  # default to manifest dir

# --- CLI arguments (and modified defaults)
parser = kb.ArgumentParser()
parser.set_defaults(
  seed=1,
  frame_start=1,
  frame_end=10,
  resolution=(128, 128),
)

parser.add_argument("--backgrounds_split",
                    choices=["train", "test"], default="train")
parser.add_argument("--dataset_mode",
                    choices=["easy", "hard"], default="hard")
parser.add_argument("--hdri_dir",
                    type=str, default="gs://mv_bckgr_removal/hdri_haven/4k/")
parser.add_argument("--hdri_assets", type=str,
                    default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json")
FLAGS = parser.parse_args()


if FLAGS.dataset_mode == "hard":
  add_distractors = True

# --- Common setups
kb.utils.setup_logging(FLAGS.logging_level)
kb.utils.log_my_flags(FLAGS)
job_dir = kb.as_path(FLAGS.job_dir)
rng = np.random.RandomState(FLAGS.seed)
scene = kb.Scene.from_flags(FLAGS)

# --- Add a renderer
renderer = KubricRenderer(scene,
  use_denoising=True,
  adaptive_sampling=False,
  background_transparency=True)

# --- Fetch a random asset
asset_source = kb.AssetSource.from_manifest(SHAPENET_PATH)
# all_ids = list(asset_source.db['id'])
all_ids = [name for name, unused_spec in asset_source._assets.items()]
num_total_objs = len(all_ids)
fraction = 0.1

rng_train_test_split = np.random.RandomState(1)
rng_train_test_split.shuffle(all_ids)
held_out_obj_ids = all_ids[:math.ceil(fraction * num_total_objs)]

# held_out_obj_ids = list(asset_source.db.sample(
#     frac=fraction, replace=False, random_state=42)["id"])
train_obj_ids = [id for id in all_ids if
                 id not in held_out_obj_ids]

if FLAGS.backgrounds_split == "train":
  asset_id = rng.choice(train_obj_ids)
else:
  asset_id = rng.choice(held_out_obj_ids)

obj = asset_source.create(asset_id=asset_id)
logging.info(f"selected '{asset_id}'")

# --- make object flat on X/Y and not penetrate floor
obj.quaternion = kb.Quaternion(axis=[1,0,0], degrees=90)
obj.position = obj.position - (0, 0, obj.aabbox[0][2])

obj_size = np.linalg.norm(obj.aabbox[1] - obj.aabbox[0])
if add_distractors:
  obj_radius = np.linalg.norm(obj.aabbox[1][:2] - obj.aabbox[0][:2])
obj_height = obj.aabbox[1][2] - obj.aabbox[0][2]

obj.metadata = {
    "asset_id": obj.asset_id,
    "category": [spec for name, spec in asset_source._assets.items()
                 if name == obj.asset_id][0]['metadata']["category"],
}
scene.add(obj)

size_multiple = 1.
if add_distractors:
  distractor_locs = []
  for i in range(4):
    asset_id_2 = rng.choice(train_obj_ids)
    obj2 = asset_source.create(asset_id=asset_id_2)
    logging.info(f"selected '{asset_id}'")

    # --- make object flat on X/Y and not penetrate floor
    obj2.quaternion = kb.Quaternion(axis=[1,0,0], degrees=90)
    obj_2_radius = np.linalg.norm(obj2.aabbox[1][:2] - obj2.aabbox[0][:2])

    position = rng.rand((2)) * 2 - 1
    position /= np.linalg.norm(position)
    position *= (obj_radius + obj_2_radius) / 2.

    distractor_locs.append(-position)
    obj2.position = obj2.position - (position[0], position[1], obj2.aabbox[0][2])

    obj_size_2 = np.linalg.norm(obj2.aabbox[1] - obj2.aabbox[0])

    obj_height_2 = obj2.aabbox[1][2] - obj2.aabbox[0][2]
    obj2.metadata = {
      "asset_id": obj2.asset_id,
      "category": [spec for name, spec in asset_source._assets.items()
                   if name == obj2.asset_id][0]['metadata']["category"],
    }
    scene.add(obj2)

  distractor_dir = np.vstack(distractor_locs)
  distractor_dir /= np.linalg.norm(distractor_dir, axis=-1, keepdims=True)

  size_multiple = 1.5

material = kb.PrincipledBSDFMaterial(
    color=kb.Color.from_hsv(rng.uniform(), 1, 1),
    metallic=1.0, roughness=0.2, ior=2.5)

table = kb.Cube(name="floor",
                scale=(obj_size*size_multiple, obj_size*size_multiple, 0.02),
                position=(0, 0, -0.02), material=material)
scene += table

logging.info("Loading background HDRIs from %s", FLAGS.hdri_dir)

hdri_source = kb.AssetSource.from_manifest(FLAGS.hdri_assets)

train_backgrounds, held_out_backgrounds = hdri_source.get_test_split(
    fraction=0.1)

if FLAGS.backgrounds_split == "train":
  logging.info("Choosing one of the %d training backgrounds...",
               len(train_backgrounds))
  background_hdri = hdri_source.create(asset_id=rng.choice(train_backgrounds))
else:
  logging.info("Choosing one of the %d held-out backgrounds...",
               len(held_out_backgrounds))
  background_hdri = hdri_source.create(
      asset_id=rng.choice(held_out_backgrounds))
dome = add_hdri_dome(scene, background_hdri)

renderer._set_ambient_light_hdri(background_hdri.filename)

# --- Add Klevr-like lights to the scene
scene += kb.assets.utils.get_clevr_lights(rng=rng)

def sample_point_in_half_sphere_shell(
    inner_radius: float,
    outer_radius: float,
    rng: np.random.RandomState
    ):
  """Uniformly sample points that are in a given distance
     range from the origin and with z >= 0."""

  while True:
    v = rng.uniform((-outer_radius, -outer_radius, obj_height/1.2),
                    (outer_radius, outer_radius, obj_height))
    len_v = np.linalg.norm(v)
    correct_angle = True
    if add_distractors:
      cam_dir = v[:2] / np.linalg.norm(v[:2])
      correct_angle = np.all(np.dot(distractor_dir, cam_dir) < np.cos(np.pi / 9.))
    if inner_radius <= len_v <= outer_radius and correct_angle:
      return tuple(v)

# --- Keyframe the camera
scene.camera = kb.PerspectiveCamera()
for frame in range(FLAGS.frame_start, FLAGS.frame_end + 1):
  # scene.camera.position = (1, 1, 1)  #< frozen camera
  scene.camera.position = sample_point_in_half_sphere_shell(
      obj_size*1.7, obj_size*2, rng)
  scene.camera.look_at((0, 0, obj_height/2))
  scene.camera.keyframe_insert("position", frame)
  scene.camera.keyframe_insert("quaternion", frame)

# --- Rendering
logging.info("Rendering the scene ...")
renderer.save_state(job_dir / "scene.blend")
data_stack = renderer.render()

# --- Postprocessing
kb.compute_visibility(data_stack["segmentation"], scene.assets)
data_stack["segmentation"] = kb.adjust_segmentation_idxs(
    data_stack["segmentation"],
    scene.assets,
    [obj]).astype(np.uint8)

kb.file_io.write_rgba_batch(data_stack["rgba"], job_dir)
kb.file_io.write_segmentation_batch(data_stack["segmentation"], job_dir)

# --- Collect metadata
logging.info("Collecting and storing metadata for each object.")
data = {
  "metadata": kb.get_scene_metadata(scene),
  "camera": kb.get_camera_info(scene.camera),
}
kb.file_io.write_json(filename=job_dir / "metadata.json", data=data)
kb.done()
