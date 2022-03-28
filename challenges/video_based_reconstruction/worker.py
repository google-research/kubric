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

import os
import logging
import numpy as np
import kubric as kb
from kubric.assets import asset_source
from kubric.renderer.blender import Blender as KubricBlender
from kubric.simulator.pybullet import PyBullet as KubricSimulator
import sys
import imageio
import bpy
import pdb
import random
from scipy.spatial import transform
import pickle

logging.basicConfig(level="INFO")


parser = kb.ArgumentParser()
parser.add_argument("--rotate_camera", type=bool, default=False)
parser.add_argument("--camera_rot_range", type=float, default=2 * np.pi)
parser.add_argument("--object", choices=["cube", "torus", "car",
                    "airplane", "chair", "table", "pillow"], default="cube")
parser.add_argument("--extra_obj_texture", type=bool, default=False)
parser.add_argument("--obj_texture_path", type=str,
                    default="examples/tex/tex.jpg")
parser.add_argument("--no_texture", type=bool, default=False)

FLAGS = parser.parse_args()

POSITION = (0, 0, 1)
VELOCITY = (0.5, 0, -1)
ANGULAR_VELOCITY = (0, 0, 5)

out_dir = f"output/{FLAGS.object}"

# random.seed(0)

# --- create scene and attach a renderer and simulator
scene = kb.Scene(resolution=(256, 256))
scene.frame_end = 30  # < numbers of frames to render
scene.frame_rate = 24  # < rendering framerate
scene.step_rate = 240  # < simulation framerate
renderer = KubricBlender(scene)
simulator = KubricSimulator(scene)

# --- populate the scene with objects, lights, cameras
scene += kb.Cube(name="floor", scale=(10, 10, 0.1), position=(0, 0, -0.1),
                 static=True, background=True, segmentation_id=1)
scene += kb.DirectionalLight(name="sun", position=(-1, -0.5, 3),
                             look_at=(0, 0, 0), intensity=1.5)
scene.camera = kb.PerspectiveCamera(
    name="camera", position=(2, -2, 4), look_at=(0, 0, 0))


# color = kb.random_hue_color()
color = kb.Color(r=1, g=0.1, b=0.1, a=1.0)
# quaternion = [0.871342, 0.401984, -0.177436, 0.218378]
material = kb.PrincipledBSDFMaterial(color=color)

if FLAGS.object == "cube":
  obj = kb.Cube(
      name="cube",
      scale=0.3,
      velocity=VELOCITY,
      position=POSITION,
      mass=0.2,
      restitution=1,
      material=material,
      friction=1,
      segmentation_id=2
  )
  objname = "cube"
  # segmentation id doesn"t seem to be working -- the segmentation mask still uses object id

elif FLAGS.object == "torus":
  # set up assets
  asset_source = kb.AssetSource.from_manifest(
      "gs://kubric-public/assets/KuBasic/KuBasic.json")

  obj = asset_source.create(name="torus",
                            asset_id="torus", scale=0.5)
  objname = "torus"
  obj.material = material
  obj.position = POSITION
  obj.velocity = VELOCITY

else:

  # TODO: go to https://shapenet.org/ create an account and agree to the terms
  #       then find the URL for the kubric preprocessed ShapeNet and put it here:
  SHAPENET_PATH = "gs://KUBRIC_SHAPENET_PATH/ShapeNetCore.v2.json"

  if SHAPENET_PATH == "gs://KUBRIC_SHAPENET_PATH/ShapeNetCore.v2.json":
    raise ValueError("Wrong ShapeNet path. Please visit https://shapenet.org/ "
                     "agree to terms and conditions, and find the correct path.")
  asset_source = kb.AssetSource.from_manifest(SHAPENET_PATH)

  if FLAGS.object == "car":
    obj = asset_source.create(
        asset_id="02958343/d4d7d596cf08754e2dfac2620a0cf07b")
    obj.scale = 2
  elif FLAGS.object == "airplane":
    obj = asset_source.create(
        asset_id="02691156/a9b95631bcbefe9ad225a1c252daae25")
    obj.scale = 2
  elif FLAGS.object == "chair":
    obj = asset_source.create(
        asset_id="03001627/c375f006c3384e83c71d7f4874a478cb")
    obj.scale = 1.5
    scene.camera.position = (2, -2, 3)
    obj.angular_velocity = (0, 0, 5)
  elif FLAGS.object == "table":
    obj = asset_source.create(
        asset_id="04379243/d5978095ef90e63375dc74e2f2f50364")
    obj.scale = 2
    scene.camera.position = (2.5, -2.5, 2)
    scene.camera.look_at((0, 0, 0))
  elif FLAGS.object == "pillow":
    obj = asset_source.create(
        asset_id="03938244/b5cb58fb099204fea5c423249b53dbc4")
    obj.scale = 2
    POSITION = (0, 0, 0.2)
    VELOCITY = (0.5, 0, 0)
  else:
    raise NotImplementedError("Object not supported")
  obj.position = POSITION
  obj.velocity = VELOCITY
  obj.angular_velocity = ANGULAR_VELOCITY
  obj.metadata = {
      "asset_id": obj.asset_id,
  }
  obj.quaternion = kb.Quaternion(axis=[1, 0, 0], degrees=90)
  objname = obj.name

if FLAGS.extra_obj_texture:
  bpy_scene = bpy.context.scene
  obj.material = kb.PrincipledBSDFMaterial(name="material")
  obj.material.metallic = random.random()
  obj.material.roughness = random.random()**0.2

  scene += obj

  mat = bpy_scene.objects[objname].active_material
  tree = mat.node_tree

  mat_node = tree.nodes["Principled BSDF"]
  tex_image = mat.node_tree.nodes.new("ShaderNodeTexImage")
  tex_image.image = bpy.data.images.load(FLAGS.obj_texture_path)
  tree.links.new(mat_node.inputs["Base Color"], tex_image.outputs["Color"])
else:
  scene += obj

if FLAGS.no_texture:
  for material in bpy.data.materials:
    material.user_clear()
    bpy.data.materials.remove(material)


cam_params = []

if FLAGS.rotate_camera:
  # Render cameras at the same general distance from the origin, but at
  # different positions.
  #
  # We will use spherical coordinates (r, theta, phi) to do this.
  #   x = r * cos(theta) * sin(phi)
  #   y = r * sin(theta) * sin(phi)
  #   z = r * cos(phi)
  original_camera_position = scene.camera.position
  r = np.sqrt(sum(a * a for a in original_camera_position))
  phi = np.arccos(original_camera_position[2] / r)  # (180 - elevation)
  theta = np.arccos(original_camera_position[0] / (r * np.sin(phi)))  # azimuth
  num_phi_values_per_theta = 1
  theta_change = FLAGS.camera_rot_range / \
      ((scene.frame_end - scene.frame_start) / num_phi_values_per_theta)

  for frame in range(scene.frame_start, scene.frame_end + 1):
    i = (frame - scene.frame_start)
    theta_new = (i // num_phi_values_per_theta) * theta_change + theta

    # These values of (x, y, z) will lie on the same sphere as the original camera.
    x = r * np.cos(theta_new) * np.sin(phi)
    y = r * np.sin(theta_new) * np.sin(phi)
    z = r * np.cos(phi)

    scene.camera.position = (x, y, z)
    scene.camera.look_at((0, 0, 0))
    scene.camera.keyframe_insert("position", frame)
    scene.camera.keyframe_insert("quaternion", frame)

    cam_param = np.zeros([1, 8])
    quat = scene.camera.quaternion
    rot = transform.Rotation.from_quat(quat)
    inv_quat = rot.inv().as_quat()

    cam_param[0, 0] = scene.camera.focal_length
    cam_param[0, 1] = x
    cam_param[0, 2] = y
    cam_param[0, 3] = quat[3]
    cam_param[0, 4:7] = quat[:3]
    cam_param[0, 7] = z
    cam_params.append(cam_param)

else:
  x, y, z = scene.camera.position
  cam_param = np.zeros([1, 8])
  quat = scene.camera.quaternion
  rot = transform.Rotation.from_quat(quat)
  inv_quat = rot.inv().as_quat()

  cam_param[0, 0] = scene.camera.focal_length
  cam_param[0, 1] = x
  cam_param[0, 2] = y
  cam_param[0, 3] = quat[3]
  cam_param[0, 4:7] = quat[:3]
  cam_param[0, 7] = z

  for _ in range(scene.frame_end):
    cam_params.append(cam_param)


# --- executes the simulation (and store keyframes)
simulator.run()

# --- renders the output
kb.as_path("output").mkdir(exist_ok=True)
renderer.save_state(f"{out_dir}/{FLAGS.object}.blend")
frames_dict = renderer.render()


# with open(f"{out_dir}/frames.dict", "wb") as file:
#   pickle.dump(frames_dict, file)

# kb.write_image_dict(frames_dict, f"{out_dir}")


# convert segmentation mask to LASR style
palette = [[0, 0, 0], [0, 0, 0], [128, 128, 128], [
    128, 128, 128], [128, 128, 128], [128, 128, 128]]
kb.file_io.multi_write_image(
    frames_dict["segmentation"],
    str(kb.as_path(
        f"{out_dir}/LASR/Annotations/Full-Resolution/{FLAGS.object}") / "{:05d}.png"),
    write_fn=kb.write_palette_png,
    max_write_threads=16,
    palette=palette
)
kb.file_io.multi_write_image(
    frames_dict["segmentation"],
    str(kb.as_path(
        f"{out_dir}/LASR/Annotations/Full-Resolution/r{FLAGS.object}") / "{:05d}.png"),
    write_fn=kb.write_palette_png,
    max_write_threads=16,
    palette=[[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
)

kb.file_io.multi_write_image(
    frames_dict["rgba"],
    str(kb.as_path(
        f"{out_dir}/LASR/JPEGImages/Full-Resolution/{FLAGS.object}") / "{:05d}.png"),
    write_fn=kb.write_png,
    max_write_threads=16
)


# write optical flow and occlusion map in LASR format
def write_pfm(path, image, scale=1):
  """Write pfm file.

  Args:
      path (str): pathto file
      image (array): data
      scale (int, optional): Scale. Defaults to 1.
  """

  with open(path, "wb") as file:
    color = None

    if image.dtype.name != "float32":
      raise Exception("Image dtype must be float32.")

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
      color = True
    elif (
        len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1
    ):  # greyscale
      color = False
    else:
      raise Exception(
          "Image must have H x W x 3, H x W x 1 or H x W dimensions.")

    file.write("PF\n".encode() if color else "Pf\n".encode())
    file.write("%d %d\n".encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == "<" or endian == "=" and sys.byteorder == "little":
      scale = -scale

    file.write("%f\n".encode() % scale)

    image.tofile(file)


fw = frames_dict["forward_flow"][:-1, ...] * 256
bw = frames_dict["backward_flow"][1:, ...] * 256
imgs = frames_dict["rgba"]
M, N = imgs.shape[1:3]

occs = np.ones(fw.shape[:-1]).astype("float32")


os.makedirs(
    f"{out_dir}/LASR/FlowFW/Full-Resolution/{FLAGS.object}", exist_ok=True)
os.makedirs(
    f"{out_dir}/LASR/FlowBW/Full-Resolution/{FLAGS.object}", exist_ok=True)
os.makedirs(
    f"{out_dir}/LASR/FlowFW/Full-Resolution/r{FLAGS.object}", exist_ok=True)
os.makedirs(
    f"{out_dir}/LASR/FlowBW/Full-Resolution/r{FLAGS.object}", exist_ok=True)
os.makedirs(
    f"{out_dir}/LASR/Camera/Full-Resolution/{FLAGS.object}", exist_ok=True)
os.makedirs(
    f"{out_dir}/LASR/Camera/Full-Resolution/r{FLAGS.object}", exist_ok=True)

# write flows into pfm
for i in range(len(fw)):
  f = fw[i, ...]
  ones = np.ones_like(f[..., :1])
  f = np.concatenate([f[..., 1:], f[..., :1], ones], -1)
  b = np.concatenate([-bw[i, ..., 1:], -bw[i, ..., :1], ones], -1)

  f = np.flip(f, 0)
  b = np.flip(b, 0)

  write_pfm(
      f"{out_dir}/LASR/FlowFW/Full-Resolution/{FLAGS.object}/flo-{i:05d}.pfm", f)
  write_pfm(
      f"{out_dir}/LASR/FlowBW/Full-Resolution/{FLAGS.object}/flo-{i+1:05d}.pfm", b)
  write_pfm(f"{out_dir}/LASR/FlowFW/Full-Resolution/{FLAGS.object}/occ-{i:05d}.pfm",
            np.ones_like(occs[i, ...]))
  write_pfm(f"{out_dir}/LASR/FlowBW/Full-Resolution/{FLAGS.object}/occ-{i+1:05d}.pfm",
            np.ones_like(occs[i, ...]))

  write_pfm(
      f"{out_dir}/LASR/FlowFW/Full-Resolution/r{FLAGS.object}/flo-{i:05d}.pfm", f)
  write_pfm(
      f"{out_dir}/LASR/FlowBW/Full-Resolution/r{FLAGS.object}/flo-{i+1:05d}.pfm", b)
  write_pfm(f"{out_dir}/LASR/FlowFW/Full-Resolution/r{FLAGS.object}/occ-{i:05d}.pfm",
            np.ones_like(occs[i, ...]))
  write_pfm(f"{out_dir}/LASR/FlowBW/Full-Resolution/r{FLAGS.object}/occ-{i+1:05d}.pfm",
            np.ones_like(occs[i, ...]))

for i in range(len(cam_params)):
  # save camera parameters
  np.savetxt(
      f"{out_dir}/LASR/Camera/Full-Resolution/{FLAGS.object}/{i:05d}.txt", cam_params[i].T)
  np.savetxt(
      f"{out_dir}/LASR/Camera/Full-Resolution/r{FLAGS.object}/{i:05d}.txt", cam_params[i].T)

# write gif
imageio.mimsave(
    str(kb.as_path(f"{out_dir}/") / f"{FLAGS.object}.gif"), frames_dict["rgba"])
kb.file_io.write_flow_batch(
    frames_dict["forward_flow"],
    directory=f"{out_dir}/FlowFW", file_template="{:05d}.png", name="forward_flow",
    max_write_threads=16
)
kb.file_io.write_flow_batch(
    frames_dict["backward_flow"],
    directory=f"{out_dir}/FlowBW",
    file_template="{:05d}.png",
    name="backward_flow",
    max_write_threads=16
)

logging.info("Collecting and storing metadata for each object.")
kb.write_json(filename=f"{out_dir}/metadata.json", data={
    "flags": vars(FLAGS),
    "metadata": kb.get_scene_metadata(scene),
    "camera": kb.get_camera_info(scene.camera),
    "instances": kb.get_instance_info(scene, [obj]),
})

kb.done()
