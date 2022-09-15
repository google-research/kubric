"""ShapeNet kubric worker for Borg."""

import logging
import os

import kubric as kb
from kubric.core import color
from kubric.renderer import blender
import numpy as np
import tensorflow as tf

# import jax
# import pyquaternion
# from universal_diffusion import image_utils
# from universal_diffusion import scene_utils
# from universal_diffusion.google.nvs.rendering import shapenet_info

import ml_collections


def config(**kwargs):
  return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
  return config(
      airplanes=config(
          id='02691156',
          count=4044,
          orig_count=4045,
      ),
      bags=config(
          id='02773838',
          count=83,
          orig_count=83,
      ),
      baskets=config(
          id='02801938',
          count=113,
          orig_count=113,
      ),
      bathtubs=config(
          id='02808440',
          count=856,
          orig_count=856,
      ),
      beds=config(
          id='02818832',
          count=233,
          orig_count=233,
      ),
      benches=config(
          id='02828884',
          count=1813,
          orig_count=1813,
      ),
      birdhouses=config(
          id='02843684',
          count=73,
          orig_count=73,
      ),
      bookshelves=config(
          id='02871439',
          count=452,
          orig_count=452,
      ),
      bottles=config(
          id='02876657',
          count=498,
          orig_count=498,
      ),
      bowls=config(
          id='02880940',
          count=186,
          orig_count=186,
      ),
      buses=config(
          id='02924116',
          count=937,
          orig_count=939,
      ),
      cabinets=config(
          id='02933112',
          count=1571,
          orig_count=1571,
      ),
      cameras=config(
          id='02942699',
          count=113,
          orig_count=113,
      ),
      cans=config(
          id='02946921',
          count=108,
          orig_count=108,
      ),
      caps=config(
          id='02954340',
          count=56,
          orig_count=56,
      ),
      cars=config(
          id='02958343',
          count=3486,
          orig_count=3533,
      ),
      cellphones=config(
          id='02992529',
          count=831,
          orig_count=831,
      ),
      chairs=config(
          id='03001627',
          count=6767,
          orig_count=6778,
      ),
      clocks=config(
          id='03046257',
          count=650,
          orig_count=651,
      ),
      controllers=config(
          id='04074963',
          count=66,
          orig_count=66,
      ),
      dishwashers=config(
          id='03207941',
          count=93,
          orig_count=93,
      ),
      displays=config(
          id='03211117',
          count=1093,
          orig_count=1093,
      ),
      earphones=config(
          id='03261776',
          count=73,
          orig_count=73,
      ),
      faucets=config(
          id='03325088',
          count=744,
          orig_count=744,
      ),
      files=config(
          id='03337140',
          count=298,
          orig_count=298,
      ),
      guitars=config(
          id='03467517',
          count=797,
          orig_count=797,
      ),
      guns=config(
          id='03948459',
          count=307,
          orig_count=307,
      ),
      helmets=config(
          id='03513137',
          count=162,
          orig_count=162,
      ),
      jars=config(
          id='03593526',
          count=596,
          orig_count=596,
      ),
      keyboards=config(
          id='03085013',
          count=65,
          orig_count=65,
      ),
      knives=config(
          id='03624134',
          count=424,
          orig_count=424,
      ),
      lamps=config(
          id='03636649',
          count=2318,
          orig_count=2318,
      ),
      laptops=config(
          id='03642806',
          count=460,
          orig_count=460,
      ),
      mailboxes=config(
          id='03710193',
          count=94,
          orig_count=94,
      ),
      microphones=config(
          id='03759954',
          count=67,
          orig_count=67,
      ),
      microwaves=config(
          id='03761084',
          count=152,
          orig_count=152,
      ),
      motorcycles=config(
          id='03790512',
          count=337,
          orig_count=337,
      ),
      mugs=config(
          id='03797390',
          count=214,
          orig_count=214,
      ),
      pianos=config(
          id='03928116',
          count=239,
          orig_count=239,
      ),
      pillows=config(
          id='03938244',
          count=96,
          orig_count=96,
      ),
      pots=config(
          id='03991062',
          count=601,
          orig_count=602,
      ),
      printers=config(
          id='04004475',
          count=166,
          orig_count=166,
      ),
      rifles=config(
          id='04090263',
          count=2373,
          orig_count=2373,
      ),
      rockets=config(
          id='04099429',
          count=85,
          orig_count=85,
      ),
      skateboards=config(
          id='04225987',
          count=152,
          orig_count=152,
      ),
      sofas=config(
          id='04256520',
          count=3172,
          orig_count=3173,
      ),
      speakers=config(
          id='03691459',
          count=1597,
          orig_count=1597,
      ),
      stoves=config(
          id='04330267',
          count=218,
          orig_count=218,
      ),
      tables=config(
          id='04379243',
          count=8436,
          orig_count=8436,
      ),
      telephones=config(
          id='04401088',
          count=1088,
          orig_count=1089,
      ),
      towers=config(
          id='04460130',
          count=133,
          orig_count=133,
      ),
      trains=config(
          id='04468005',
          count=389,
          orig_count=389,
      ),
      trashcans=config(
          id='02747177',
          count=343,
          orig_count=343,
      ),
      vessels=config(
          id='04530566',
          count=1937,
          orig_count=1939,
      ),
      washers=config(
          id='04554684',
          count=169,
          orig_count=169,
      ),
  )


gfile = tf.io.gfile


SHAPENET_INFO = get_config()
# TODO(watsondaniel) hide.
SHAPENET_PATH = "gs://kubric-unlisted/assets/ShapeNetCore.v2.json"


def get_clevr_lights(rng: np.random.RandomState, light_jitter: float = 1.0):
  """Create lights that match the setup from the CLEVR dataset."""
  sun = kb.core.DirectionalLight(
      name="sun",
      color=color.Color.from_name("white"),
      shadow_softness=0.2,
      intensity=0.45 / 2,
      position=(11.6608, -6.62799, 25.8232))
  lamp_back = kb.core.RectAreaLight(
      name="lamp_back",
      color=color.Color.from_name("white"),
      intensity=50. / 2,
      position=(-1.1685, 2.64602, 5.81574))
  lamp_key = kb.core.RectAreaLight(
      name="lamp_key",
      color=color.Color.from_hexint(0xffedd0),
      intensity=100 / 2,
      width=0.5,
      height=0.5,
      position=(6.44671, -2.90517, 4.2584))
  lamp_fill = kb.core.RectAreaLight(
      name="lamp_fill",
      color=color.Color.from_hexint(0xc2d0ff),
      intensity=30 / 2,
      width=0.5,
      height=0.5,
      position=(-4.67112, -4.0136, 3.01122))
  lights = [sun, lamp_back, lamp_key, lamp_fill]

  # jitter lights
  for light in lights:
    light.position = light.position + rng.rand(3) * light_jitter
    light.look_at((0, 0, 0))

  return lights


def main(_):
  # CLI arguments (and modified defaults)
  parser = kb.ArgumentParser()
  parser.set_defaults(
      seed=0,  # NOTE: seed is the job replica id
      frame_end=5,
      resolution=(128, 128))

  parser.add_argument(
      "--shapenet_class",
      type=str,
      default="cars",
      help="ShapeNet object class to render.")

  parser.add_argument(
      "--object_start",
      type=int,
      default=0,
      help="Initial object id.")

  FLAGS = parser.parse_args()  # pylint: disable=invalid-name
  kb.utils.setup_logging(FLAGS.logging_level)
  kb.utils.log_my_flags(FLAGS)

  # Get the id of the object to load-- if already written, stop ASAP.
  asset_source = kb.AssetSource.from_manifest(SHAPENET_PATH)
  cls_id = SHAPENET_INFO[FLAGS.shapenet_class].id
  asset_ids = asset_source.db.loc[asset_source.db["id"].str.startswith(cls_id)]
  # Sorting for ensuring determinism across workers.
  asset_ids = list(sorted(asset_ids["id"]))
  asset_index = FLAGS.object_start + FLAGS.seed
  asset_id = asset_ids[asset_index]

  output_path = os.path.join(
      FLAGS.job_dir, FLAGS.shapenet_class, f"scene_{asset_index}.tfrecord")

  logging.info("output_path=%s", output_path)
  if tf.io.gfile.exists(output_path):
    logging.info("Already exists, will not re-render: %s", output_path)
    return
  if not gfile.exists(os.path.dirname(output_path)):
    gfile.makedirs(os.path.dirname(output_path))

  rng = np.random.RandomState(asset_index)
  scene = kb.Scene.from_flags(FLAGS)
  job_dir = kb.as_path(FLAGS.job_dir)
  renderer = blender.Blender(
      scene,
      samples_per_pixel=64,
      use_denoising=False,
      background_transparency=True)

  # Create the object.
  obj = asset_source.create(asset_id=asset_id)
  logging.info("shapenet core asset_id=%s", asset_id)
  logging.info("asset_index=%s", asset_index)
  # Make the object flat on X/Y.
  # obj.quaternion = tuple(pyquaternion.Quaternion(axis=[1, 0, 0], degrees=90))
  obj.quaternion = kb.Quaternion(axis=[1, 0, 0], degrees=90)
  scene.add(obj)

  # Determine the camera position dynamically, based on the object's bbox.
  bounding_sphere_radius = np.amax(
      np.square(obj.bbox_3d - obj.position[None]).sum(axis=-1))
  min_radius = 5. * bounding_sphere_radius
  max_radius = 5. * bounding_sphere_radius
  logging.info("bbox=%r", obj.bbox_3d)
  logging.info("min_radius=%.4f", min_radius)
  logging.info("max_radius=%.4f", max_radius)

  # Add Klevr-like lights to the scene.
  scene += get_clevr_lights(rng)
  scene.ambient_illumination = kb.Color(.05, .05, .05)

  # Create the pinhole camera model.
  focal_length = 128.
  sensor_width = 64.
  scene.camera = kb.PerspectiveCamera(
      focal_length=focal_length, sensor_width=sensor_width)

  # Keyframe the camera.
  for frame in range(FLAGS.frame_start, FLAGS.frame_end + 1):
    # Random point in sphere shell.
    position = rng.normal(0., 1., (3,))
    radius = min_radius + rng.uniform(0., 1.) * (max_radius - min_radius)
    position *= radius / np.linalg.norm(position)

    scene.camera.position = position
    # Always look at the origin (object center).
    scene.camera.look_at((0, 0, 0))

    scene.camera.keyframe_insert("position", frame)
    scene.camera.keyframe_insert("quaternion", frame)

  # Rendering happens here.
  logging.info("Rendering the scene...")
  output_stack = renderer.render()

  # Save files   
  kb.file_io.write_rgba_batch(output_stack["rgba"], job_dir)
  exit(0)

  # Now write out results as a tfrecord with canonicalized fields.
  image_rgba = output_stack["rgba"]
  # TMP
  print(image_rgba)
  print(image_rgba.shape, image_rgba.dtype, flush=True)
  # END TMP
  image = image_utils.rgba_to_rgb(image_rgba)

  num_frames = FLAGS.frame_end + 1 - FLAGS.frame_start
  label = np.array([FLAGS.shapenet_class] * num_frames)

  camera_info = kb.get_camera_info(scene.camera)
  camera_position = camera_info["positions"]

  so3_from_quaternion = jax.vmap(scene_utils.so3_from_quaternion)
  camera_rotation = np.asarray(so3_from_quaternion(camera_info["quaternions"]))

  camera_intrinsics = scene_utils.make_camera_intrinsics(
      fx=focal_length, fy=focal_length, h=sensor_width, w=sensor_width)
  camera_intrinsics = np.tile(camera_intrinsics[None], (num_frames, 1, 1))

  frame_id = np.array(
      list(map(str, range(FLAGS.frame_start - 1, FLAGS.frame_end))))
  scene_id = np.array([str(asset_index)] * num_frames)

  batch = scene_utils.canonical_3d_batch(
      image=image,
      label=label,
      camera_position=camera_position,
      camera_rotation=camera_rotation,
      camera_intrinsics=camera_intrinsics,
      frame_id=frame_id,
      scene_id=scene_id)
  scene_utils.write_scene_to_tfrecord(batch, output_path)

  # Gracefully close all opened assets and exit.
  kb.done()


if __name__ == "__main__":
  from absl import app  # pylint: disable=g-import-not-at-top
  app.run(main)