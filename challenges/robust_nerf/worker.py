# Copyright 2022 The Kubric Authors.
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
r"""Worker Script for generating scenes from the Semistatic-KLEVR dataset."""
# pylint: disable=logging-format-interpolation

import collections
import json
import logging
from typing import Dict, List, Optional, Tuple

import kubric as kb
import kubric.renderer.blender
import kubric.simulator.pybullet
import numpy as np
import tensorflow as tf

import bpy

# --- Some configuration values
# the region in which to place objects [(min), (max)]
# TODO(atagliasacchi): SPAWN_REGION should be specified by flags?
SPAWN_REGION = [(-4, -4, 0), (4, 4, 3)]
# the names of the KuBasic assets to use (WARNING: no "Suzanne" and "Sponge")
OBJECT_TYPES = [
    "cube",
    "cylinder",
    "sphere",
    "cone",
    "torus",
    "gear",
    "torusknot",
    "spot",
    "teapot",
]
# the set of colors to sample from
COLORS = collections.OrderedDict({
    "blue": kb.Color(42 / 255, 75 / 255, 215 / 255),
    "brown": kb.Color(129 / 255, 74 / 255, 25 / 255),
    "cyan": kb.Color(41 / 255, 208 / 255, 208 / 255),
    "gray": kb.Color(87 / 255, 87 / 255, 87 / 255),
    "green": kb.Color(29 / 255, 105 / 255, 20 / 255),
    "purple": kb.Color(129 / 255, 38 / 255, 192 / 255),
    "red": kb.Color(173 / 255, 35 / 255, 35 / 255),
    "yellow": kb.Color(255 / 255, 238 / 255, 5 / 255),
})
# Valid options when choosing colors by flags.
COLOR_STRATEGIES = ["clevr", "uniform", "black", "gray", "white"]

# the sizes of objects to sample from
SIZES = {
    "small": 0.7,
    "large": 1.4,
}
# Valid options when choosing sizes by flags.
SIZE_STRATEGIES = ["clevr", "uniform", "const"]

# Percentage of frames that fall into the training split.
_TRAIN_TEST_RATIO = 0.7


def parse_flags():
  """ArgumentParser flags parsing."""

  # --- CLI arguments
  parser = kb.ArgumentParser()
  # parser = argparse_flags.ArgumentParser()
  parser.add_argument("--num_objects_static", type=int, default=5)
  parser.add_argument("--num_objects_dynamic", type=int, default=3)
  parser.add_argument(
      "--object_types",
      nargs="+",
      default=["cube", "cylinder", "sphere", "torus", "gear"],
      choices=["ALL"] + OBJECT_TYPES)
  parser.add_argument(
      "--object_colors", choices=COLOR_STRATEGIES, default="clevr")
  parser.add_argument(
      "--object_sizes", choices=SIZE_STRATEGIES, default="clevr")
  parser.add_argument("--floor_color", choices=COLOR_STRATEGIES, default="gray")
  parser.add_argument("--floor_friction", type=float, default=0.3)
  parser.add_argument(
      "--background_color", choices=COLOR_STRATEGIES, default="white")
  parser.add_argument("--norender", action="store_true")
  parser.add_argument(
      "--assets_path", type=str, default="gs://kubric-public/assets/KuBasic.json")
  parser.add_argument(
      "--jitter", type=bool, default=False)
  parser.set_defaults(
      # Default is 1 for optical flow (unused here)
      frame_start=1,
      frame_end=301,  # Controls number of rendered frames per scene.
      frame_rate=12,
      resolution=(256, 256),
  )
  return parser.parse_args()


def _sample_color(
    strategy: str,
    rng: np.random.RandomState,
) -> Tuple[Optional[str], kb.Color]:
  """Samples an object's color.

  Args:
    strategy: Sampling strategy.
    rng: Source of randomness.

  Returns:
    color_label: Human-readable description of "color".
    color: Kubric color for this object.
  """
  # Note: If you add an option here, update COLOR_STRATEGIES as well.
  if strategy in ["black", "gray", "white"]:
    return strategy, kb.get_color(strategy)
  elif strategy == "clevr":
    color_label = rng.choice(list(COLORS.keys()))
    return color_label, COLORS[color_label]
  elif strategy == "uniform":
    return None, kb.random_hue_color(rng=rng)
  else:
    raise ValueError(f"Unknown color sampling strategy {strategy}")


def _sample_sizes(
    strategy: str,
    rng: np.random.RandomState,
) -> Tuple[Optional[str], float]:
  """Samples an object's size.

  Args:
    strategy: Sampling strategy.
    rng: Source of randomness.

  Returns:
    size_label: Human-readable description of "size".
    size: Size multiplier. Use as object scale.
  """
  if strategy == "clevr":
    size_label = rng.choice(list(SIZES.keys()))
    size = SIZES[size_label]
    return size_label, size
  elif strategy == "uniform":
    return None, rng.uniform(0.7, 1.4)
  elif strategy == "const":
    return None, 1
  else:
    raise ValueError(f"Unknown size sampling strategy {strategy}")


def _get_scene_boundaries(
    floor: kb.Cube,
    spawn_region: List[Tuple[float, float, float]],
) -> Dict[str, List[int]]:
  """Constructs bounding box for all scene-varying content.

  This bounding box ostensibly includes the objects themselves and their
  shadows. Due to complexity, a guarantee on this cannot be made. The logic
  here is best-effort.

  Args:
    floor: Cube representing the scene's floor. The cube is axis-aligned with
      its top surface lying below all objects in the scene.
    spawn_region: Lower and upper boundaries of a 3D axis-aligned box, inside
      which all foreground objects are guaranteed to lie within.

  Returns:
    Dict with two entries, "min" and "max". These entries are 3-element lists
      containing the lower and upper boundaries of a 3D axis-aligned bounding
      box.
  """
  spawn_min, spawn_max = np.asarray(spawn_region)

  if not np.allclose(floor.aabbox[1][2], spawn_min[2]):
    raise ValueError("Top of floor and bottom of bounding box be identical.")

  # Define some amount of buffer volume to the scene's bounding box to
  # encourage shadows on the floor to lie within the scene bounding box.
  boundary_epsilon = 0.1
  scene_min = spawn_min - boundary_epsilon
  scene_max = spawn_max + boundary_epsilon
  return {
      "min": list(scene_min),
      "max": list(scene_max),
  }


def _get_floor_scale_position_kwargs(
    spawn_region: List[Tuple[int, int, int]],
) -> Dict[str, Tuple[float, float, float]]:
  """Constructs scale, position of the cube representing the floor.

  Cube's scale and position are chosen such that the cube's top surface lies
  strictly inside of spawn_region.

  Args:
    spawn_region: Region of the floor.

  Returns:
    {'scale': <XYZ multipliers for [-1, 1] cube>,
     'position': <XYZ center of floor cube>}
  """
  spawn_min, spawn_max = np.array(spawn_region)

  # Center of spawn position.
  position = (spawn_max + spawn_min) / 2

  # Set center of floor to be equal to bottom of spawn region.
  position[-1] = spawn_min[-1]
  # Default cube has coordinates [[-1, -1, -1], [1, 1, 1]], so default cube
  # length is 2. So we scale floor by 2.
  scale = (spawn_max - spawn_min) / 2
  scale[-1] = 1e-8  # Floor height is minimal.
  return dict(
      position=tuple(position),
      scale=tuple(scale),
  )


def main(flags) -> None:
  if "ALL" in flags.object_types:
    objects_types = OBJECT_TYPES
  else:
    objects_types = flags.object_types

  # --- Common setups & resources
  kb.setup_logging(flags.logging_level)
  kb.log_my_flags(flags)
  scratch_dir, output_dir = kb.setup_directories(flags)

  if flags.seed is None:
    raise ValueError("You must specify --seed to proceed.")
  seed = flags.seed
  rng = np.random.RandomState(seed=seed)

  scene = kb.Scene.from_flags(flags)
  simulator = kubric.simulator.pybullet.PyBullet(scene, scratch_dir)
  renderer = kubric.renderer.blender.Blender(
      scene, scratch_dir,
      adaptive_sampling=False)  # Removes salt-and-pepper artifacts in shadows.

  logging.info("Loading assets from %s", flags.assets_path)
  kubasic = kb.AssetSource.from_manifest(flags.assets_path)

  # --- Populate the scene
  logging.info("Creating a large cube as the floor...")
  # Create the "floor" of the scene. This is a large, flat cube of
  # x_length=200, y_length=200 and z_length=2 centered at x=0, y=0, z=-1.
  # This means that the floor's elevation is z + z_length/2 == -1 + 1 == 0.
  _, floor_color = _sample_color(flags.floor_color, rng)
  floor_material = kb.PrincipledBSDFMaterial(
      color=floor_color, roughness=1., specular=0.)
  floor = kb.Cube(
      **_get_floor_scale_position_kwargs(SPAWN_REGION),
      material=floor_material,
      friction=flags.floor_friction,
      static=True,
      background=True)
  scene.add(floor)

  # Set background color.
  _, background_color = _sample_color(flags.background_color, rng)
  logging.info("Setting background color to %s...", background_color)
  scene.background = background_color

  logging.info("Adding several lights to the scene...")
  sun = kb.DirectionalLight(
      color=kb.get_color("white"),
      shadow_softness=0.2,
      intensity=3.,
      position=(11.6608, -6.62799, 25.8232))
  lamp_back = kb.RectAreaLight(
      color=kb.get_color("white"),
      intensity=50.,
      position=(-1.1685, 2.64602, 5.81574))
  lamp_key = kb.RectAreaLight(
      color=kb.get_color(0xffedd0),
      intensity=100.,
      width=0.5,
      height=0.5,
      position=(6.44671, -2.90517, 4.2584))
  lamp_fill = kb.RectAreaLight(
      color=kb.get_color("#c2d0ff"),
      intensity=30.,
      width=0.5,
      height=0.5,
      position=(-4.67112, -4.0136, 3.01122))
  lights = [sun, lamp_back, lamp_key, lamp_fill]
  for light in lights:
    light.look_at((0, 0, 0))
    scene.add(light)

  scene.ambient_illumination = kb.Color(0.05, 0.05, 0.05)

  logging.info("Setting up the Camera...")
  original_camera_position = (7.48113, -6.50764, 5.34367)
  scene.camera = kb.PerspectiveCamera(
      focal_length=35., sensor_width=32, position=original_camera_position)
  scene.camera.look_at((0, 0, 0))

  # TODO: Refactor this so that we can render both a "spherical view"
  # and a "lattice view" through the scene.

  # Render cameras at the same general distance from the origin, but at
  # different positions.
  #
  # We will use spherical coordinates (r, theta, phi) to do this.
  #   x = r * cos(theta) * sin(phi)
  #   y = r * sin(theta) * sin(phi)
  #   z = r * cos(phi)
  r = np.sqrt(sum(a * a for a in original_camera_position))
  phi = np.arccos(original_camera_position[2] / r)
  theta = np.arccos(original_camera_position[0] / (r * np.sin(phi)))

  # At each frame, we will modify theta so that we move in a circle.
  # At each given theta value, we will also render three values of phi so our
  # data "matches" LLFF (i.e., we have motion in z as well as x and y).
  num_frames_to_render = (scene.frame_end - scene.frame_start)

  # TODO: Either parameterize this or turn it into a constant
  # matching the style guide recommendation.
  num_phi_values_per_theta = 4

  def add_random_objects(num_objects, objects_types, rng):
    """Returns the list of created objects."""
    logging.info(f"Randomly placing {num_objects} objects")
    objects = list()
    for _ in range(num_objects):
      shape_name = rng.choice(objects_types)
      _, size = _sample_sizes("const", rng)
      _, color = _sample_color(flags.object_colors, rng)
      segmentation_id_val = objects_types.index(shape_name) + 1
      obj = kubasic.create(
          asset_id=shape_name, scale=size, segmentation_id=segmentation_id_val)

      material_name = "Rubber"
      obj.material = kb.PrincipledBSDFMaterial(
          color=color, metallic=0., ior=1.25, roughness=0.7, specular=0.33)

      scene.add(obj)
      obj.metadata = {
          "shape": shape_name.lower(),
          "size": size,
          "material": material_name.lower(),
          "color": color.rgb,
      }

      # Place object randomly within SPAWN_REGION
      kb.move_until_no_overlap(
          obj, simulator, spawn_region=SPAWN_REGION, rng=rng)
      objects += [obj]
      logging.info("Added object %s", obj)
    return objects

  # --- Place random objects (static in time)
  add_random_objects(
      flags.num_objects_static, objects_types=objects_types, rng=rng)
  # --- Place dynamic objects (will change in time)
  dyn_objects = add_random_objects(
      flags.num_objects_dynamic, objects_types=[
          "suzanne",
      ], rng=rng)

  # --- if in jitter mode shake the monkeys instead
  if flags.jitter:
    for dyn_object in dyn_objects:
      dyn_object.orig_position = dyn_object.position

  # import pdb; pdb.set_trace();
  # exit(0)

  theta_change = (2 * np.pi) / (num_frames_to_render / num_phi_values_per_theta)
  for frame in range(scene.frame_start, scene.frame_end + 1):
    i = (frame - scene.frame_start)
    theta_new = (i // num_phi_values_per_theta) * theta_change + theta

    # These values of (x, y, z) will lie on the same sphere as the original
    # camera.
    x = r * np.cos(theta_new) * np.sin(phi)
    y = r * np.sin(theta_new) * np.sin(phi)
    z = r * np.cos(phi)

    # To ensure have "roughly LLFF-style" data (multiple z values for the same
    # x and y), we will also adjust z. The camera is no longer guaranteed to
    # remain on the same sphere as the original camera.
    z_shift_direction = (i % num_phi_values_per_theta) - 1
    z = z + z_shift_direction * 1.2

    scene.camera.position = (x, y, z)
    scene.camera.look_at((0, 0, 0))
    scene.camera.keyframe_insert("position", frame)
    scene.camera.keyframe_insert("quaternion", frame)

    # --- randomly change the position of the dynamic objects in each frame
    for obj in dyn_objects:
      # --- hide dynamic objects once every 30 frames (evaluation set)
      if (i % 30) != 0:
        if not flags.jitter:
          kb.move_until_no_overlap(
              obj, simulator, spawn_region=SPAWN_REGION, rng=rng)
        else:
          obj.position = obj.orig_position + .1*np.random.randn(3)
      else:
        random_dir = np.random.randn(3)
        random_dir = random_dir / np.linalg.norm(random_dir)
        large_offset = 10e3
        position = (large_offset * random_dir).tolist()
        obj.position = position

      obj.keyframe_insert("position", frame)
      # print(f"moved from {old_position} to {obj.position}")

  # --- Rendering
  logging.info(f'Saving {output_dir / "scene.blend"}')
  renderer.save_state(output_dir / "scene.blend")

  print(f"jitter mode? {flags.jitter}")

  if flags.norender:
    logging.warning("rendering not executed")
    kb.done()
    exit(0)

  logging.info("Rendering the scene ...")
  render_data = renderer.render()
  # replace asset index (in scene.assets) with segmentation_id
  render_data["segmentation"] = kb.adjust_segmentation_idxs(
      render_data["segmentation"], scene.assets, scene.assets)

  # export all rendered images / segmentations / depth / ... to output dir
  kb.write_image_dict(render_data, output_dir)

  # --- Metadata
  logging.info("Collecting and storing metadata for each object.")
  kb.write_pkl(
      {
          "metadata": kb.get_scene_metadata(scene, seed=seed),
          "camera": kb.get_camera_info(scene.camera),
          "instances": kb.get_instance_info(scene),
          "background": {
              "background_color": background_color,
              "floor_color": floor_color.rgb,
              "floor_friction": flags.floor_friction,
          }
      }, output_dir / "metadata.pkl")

  # Save the scene camera in a JSON file.
  scene_metadata = kb.get_scene_metadata(scene, seed=seed)
  scene_camera = kb.get_camera_info(
      scene.camera,
      height=scene_metadata["resolution"][0],
      width=scene_metadata["resolution"][1],
  )
  scene_camera["positions"] = scene_camera["positions"].tolist()
  scene_camera["quaternions"] = scene_camera["quaternions"].tolist()
  scene_camera["K"] = scene_camera["K"].tolist()
  scene_camera["R"] = scene_camera["R"].tolist()

  # Generate the train/test ids
  train_ids, test_ids = _make_train_test_ids(rng, scene_metadata["num_frames"])
  logging.info(
      f"Split {num_frames_to_render} frames: {len(train_ids)} train frames, "
      f"{len(test_ids)} test frames.")

  metadata = {
      "metadata": scene_metadata,
      "camera": scene_camera,
      "segmentation_labels": objects_types,
      "scene_boundaries": _get_scene_boundaries(floor, SPAWN_REGION),
      "split_ids": {
          "train": train_ids,
          "test": test_ids,
      },
  }
  kb.write_json(metadata, output_dir / "metadata.json")

  scene_gltf_file = str(scratch_dir / "scene.glb")
  logging.info("Saving %s ", scene_gltf_file)
  bpy.ops.export_scene.gltf(filepath=scene_gltf_file)

  output_scene_gltf_file = str(output_dir / "scene.glb")
  tf.io.gfile.copy(scene_gltf_file, output_scene_gltf_file, overwrite=True)

  kb.done()


def _make_train_test_ids(
    rng: np.random.RandomState,
    num_frames: int,
) -> Tuple[List[int], List[int]]:
  """Randomly generate train / test ids."""
  all_idx = rng.permutation(num_frames)

  boundary = int(num_frames * _TRAIN_TEST_RATIO)

  train_ids = all_idx[:boundary]
  test_ids = all_idx[boundary:]
  # np.int64 is not serializable by Json, so need to convert to int first.
  return list(int(x) for x in train_ids), list(int(x) for x in test_ids)


if __name__ == "__main__":
  main(flags=parse_flags())
