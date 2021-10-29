import logging
import numpy as np

import kubric as kb
from kubric.renderer import Blender as KubricRenderer

# --- WARNING: this path is not yet public
source_path = "gs://kubric-public/ShapeNetCore.v2"

# --- CLI arguments (and modified defaults)
parser = kb.ArgumentParser()
parser.set_defaults(
  seed=1,
  frame_start=1,
  frame_end=5,
  width=256,
  height=256,
)
FLAGS = parser.parse_args()

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

# --- Add Klevr-like lights to the scene
scene += kb.assets.utils.get_clevr_lights(rng=rng)
scene.ambient_illumination = kb.Color(0.05, 0.05, 0.05)

# --- Add floor (~infinitely large sphere)
scene += kb.Sphere(name="floor", scale=1000, position=(0, 0, +1000), background=True, static=True)

# --- Keyframe the camera
scene.camera = kb.PerspectiveCamera()
for frame in range(FLAGS.frame_start, FLAGS.frame_end + 1):
  # scene.camera.position = (1, 1, 1)  #< frozen camera
  scene.camera.position = kb.sample_point_in_half_sphere_shell(1.1, 1.2)
  scene.camera.look_at((0, 0, 0))
  scene.camera.keyframe_insert("position", frame)
  scene.camera.keyframe_insert("quaternion", frame)

# --- Fetch a random (airplane) asset
asset_source = kb.AssetSource(source_path)
ids = list(asset_source.db.loc[asset_source.db['id'].str.startswith('02691156')]['id'])
asset_id = rng.choice(ids) #< e.g. 02691156_10155655850468db78d106ce0a280f87
obj = asset_source.create(asset_id=asset_id)
logging.info(f"selected '{asset_id}'")

# --- make object flat on X/Y and not penetrate floor
obj.quaternion = kb.Quaternion(axis=[1,0,0], degrees=90)
obj.position = obj.position - (0, 0, obj.aabbox[0][2])  

obj.metadata = {
    "asset_id": obj.asset_id,
    "category": asset_source.db[
      asset_source.db["id"] == obj.asset_id].iloc[0]["category_name"],
}
scene.add(obj)

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