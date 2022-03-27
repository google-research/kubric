"""
Worker file for the Multi-View Background removal dataset.

This dataset creates a scene where a foreground object is to be distinguished
from the background. Foreground objects are borrowed from shapnet. Backgrounds
are from indoor scenes of polyhaven. All foreground objects are situated on top
of a "table" which is gernated to be random in color. Instead of background
removal with a single image. This dataset is special in that multiple images of
the foreground object (taken from different camera poses) are given. This
"multi-view" persepctive should be very helpful for background removal but is
currently underexplored in the literature.
"""
import logging
import numpy as np

import kubric as kb
from kubric.renderer import Blender as KubricRenderer

# --- WARNING: this path is not yet public
source_path = (
    "gs://tensorflow-graphics/public/60c9de9c410be30098c297ac/ShapeNetCore.v2")

# --- CLI arguments (and modified defaults)
parser = kb.ArgumentParser()
parser.set_defaults(
  seed=1,
  frame_start=1,
  frame_end=10,
  width=128,
  height=128,
)

parser.add_argument("--backgrounds_split",
                    choices=["train", "test"], default="train")
parser.add_argument("--dataset_mode",
                    choices=["easy", "hard"], default="hard")
parser.add_argument("--hdri_dir",
                    type=str, default="gs://mv_bckgr_removal/hdri_haven/4k/")
                    # "/mnt/mydata/images/"
FLAGS = parser.parse_args()


if FLAGS.dataset_mode == "hard":
  add_distractors = False

def add_hdri_dome(hdri_source, scene, background_hdri=None):
  dome_path = hdri_source.fetch("dome.blend")
  dome = kb.FileBasedObject(
      name="BackgroundDome",
      position=(0, 0, 0),
      static=True, background=True,
      simulation_filename=None,
      render_filename=str(dome_path),
      render_import_kwargs={
          "filepath": str(dome_path / "Object" / "Dome"),
          "directory": str(dome_path / "Object"),
          "filename": "Dome",
      })
  scene.add(dome)
  # pylint: disable=import-outside-toplevel
  from kubric.renderer import Blender
  import bpy
  blender_renderer = [v for v in scene.views if isinstance(v, Blender)]
  if blender_renderer:
    dome_blender = dome.linked_objects[blender_renderer[0]]
    dome_blender.cycles_visibility.shadow = False
    if background_hdri is not None:
      dome_mat = dome_blender.data.materials[0]
      texture_node = dome_mat.node_tree.nodes["Image Texture"]
      texture_node.image = bpy.data.images.load(background_hdri.filename)
  return dome

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
asset_source = kb.AssetSource(source_path)
all_ids = list(asset_source.db['id'])
fraction = 0.1
held_out_obj_ids = list(asset_source.db.sample(
    frac=fraction, replace=False, random_state=42)["id"])
train_obj_ids = [i for i in asset_source.db["id"] if
                 i not in held_out_obj_ids]

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
    "category": asset_source.db[
      asset_source.db["id"] == obj.asset_id].iloc[0]["category_name"],
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
        "asset_id": obj.asset_id,
        "category": asset_source.db[
          asset_source.db["id"] == obj2.asset_id].iloc[0]["category_name"],
    }
    scene.add(obj2)

  distractor_dir = np.vstack(distractor_locs)
  distractor_dir /= np.linalg.norm(distractor_dir, axis=-1, keepdims=True)

  size_multiple = 1.5

material = kb.PrincipledBSDFMaterial(
    color=kb.Color.from_hsv(rng.uniform(), 1, 1),
    metallic=1.0, roughness=0.2, ior=2.5)

table = kb.Cube(name="floor", scale=(obj_size*size_multiple, obj_size*size_multiple, 0.02),
                position=(0, 0, -0.02), material=material)
scene += table

logging.info("Loading background HDRIs from %s", FLAGS.hdri_dir)

hdri_source = kb.TextureSource(FLAGS.hdri_dir)
train_backgrounds, held_out_backgrounds = hdri_source.get_test_split(
    fraction=0.1)
if FLAGS.backgrounds_split == "train":
  logging.info("Choosing one of the %d training backgrounds...",
               len(train_backgrounds))
  background_hdri = hdri_source.create(texture_name=rng.choice(train_backgrounds))
else:
  logging.info("Choosing one of the %d held-out backgrounds...",
               len(held_out_backgrounds))
  background_hdri = hdri_source.create(
      texture_name=rng.choice(held_out_backgrounds))
dome = kb.assets.utils.add_hdri_dome(hdri_source, scene, background_hdri)

dome = add_hdri_dome(hdri_source, scene, background_hdri)
renderer._set_ambient_light_hdri(background_hdri.filename)
# table = add_table(hdri_source, scene, background_hdri)

# --- Add Klevr-like lights to the scene
scene += kb.assets.utils.get_clevr_lights(rng=rng)
# scene.ambient_illumination = kb.Color.from_hsv(np.random.uniform(), 1, 1)
# scene.ambient_illumination = kb.Color(0.05, 0.05, 0.05)

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

# --- Discard non-used information
del data_stack["uv"]
del data_stack["forward_flow"]
del data_stack["backward_flow"]
del data_stack["depth"]
del data_stack["normal"]

# --- Save to image files
kb.file_io.write_image_dict(data_stack, job_dir)

# --- Collect metadata
logging.info("Collecting and storing metadata for each object.")
data = {
  "metadata": kb.get_scene_metadata(scene),
  "camera": kb.get_camera_info(scene.camera),
}
kb.file_io.write_json(filename=job_dir / "metadata.json", data=data)
kb.done()
