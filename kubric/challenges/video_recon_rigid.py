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

logging.basicConfig(level="INFO")  # < CRITICAL, ERROR, WARNING, INFO, DEBUG

ROT_CAM = False
ROT_RANGE = 2 * np.pi  # np.pi / 4
OBJNAME = 'airplane-no-tex'
POSITION = (0, 0, 1)  # (0,0,0.2)
VELOCITY = (0.5, 0, -1)  # (4,-4,0)
OBJ_TYPE = 'shapenet'
TEXTURE = False
NO_MATERIAL = True

random.seed(0)

# --- create scene and attach a renderer and simulator
scene = kb.Scene(resolution=(256, 256))
scene.frame_end = 30   # < numbers of frames to render
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

if OBJ_TYPE == 'cube':
  obj = kb.Cube(name='cube', scale=0.3, velocity=VELOCITY, angular_velocity=[
    0, 0, 0], position=POSITION, mass=0.2, restitution=1, material=material, friction=1, segmentation_id=2)
  objname = 'cube'
  # segmentation id doesn't seem to be working -- the segmentation mask still uses object id

elif OBJ_TYPE == 'torus':
  # set up assets
  asset_source = kb.AssetSource.from_manifest("gs://kubric-public/KuBasic")

  obj = asset_source.create(name="torus",
                            asset_id='Torus', scale=0.5)
  objname = 'torus'
  obj.material = material
  obj.position = POSITION
  obj.velocity = VELOCITY

elif OBJ_TYPE == 'shapenet':

  # TODO: go to https://shapenet.org/ create an account and agree to the terms
  #       then find the URL for the kubric preprocessed ShapeNet and put it here:
  SHAPENET_PATH = "gs://KUBRIC_SHAPENET_PATH/ShapeNetCore.v2.json"

  if SHAPENET_PATH == "gs://KUBRIC_SHAPENET_PATH/ShapeNetCore.v2.json":
    raise ValueError("Wrong ShapeNet path. Please visit https://shapenet.org/ "
                     "agree to terms and conditions, and find the correct path.")
  asset_source = kb.AssetSource.from_manifest(SHAPENET_PATH)

  if OBJNAME == 'car' or OBJNAME == 'car-no-tex':
    obj = asset_source.create(
        asset_id='02958343/d4d7d596cf08754e2dfac2620a0cf07b')
    obj.scale = 2
    obj.angular_velocity = (0, 0, 5)
  elif OBJNAME == 'car-rot-obj':
    obj = asset_source.create(
        asset_id='02958343/d4d7d596cf08754e2dfac2620a0cf07b')
    obj.scale = 2
    obj.angular_velocity = (0, 0, 5)
  elif OBJNAME == 'airplane':
    obj = asset_source.create(
        asset_id='02691156/a9b95631bcbefe9ad225a1c252daae25')
    obj.scale = 2
  elif OBJNAME == 'airplane-rot-obj' or OBJNAME == 'airplane-no-tex':
    obj = asset_source.create(
        asset_id='02691156/a9b95631bcbefe9ad225a1c252daae25')
    obj.scale = 2
    obj.angular_velocity = (0, 0, 5)
  elif OBJNAME == 'chair':
    obj = asset_source.create(
        asset_id='03001627/c375f006c3384e83c71d7f4874a478cb')
    obj.scale = 1.5
    scene.camera.position = (2, -2, 3)
    scene.camera.look_at((0, 0, 0))
  elif OBJNAME == 'chair-rot-obj':
    obj = asset_source.create(
        asset_id='03001627/c375f006c3384e83c71d7f4874a478cb')
    obj.scale = 1.5
    scene.camera.position = (2, -2, 3)
    obj.angular_velocity = (0, 0, 5)
    scene.camera.look_at((0, 0, 0))
  elif OBJNAME == 'table-rot-obj':
    obj = asset_source.create(
        asset_id='04379243/d5978095ef90e63375dc74e2f2f50364')
    obj.scale = 2
    obj.angular_velocity = (0, 0, 5)
    scene.camera.position = (2.5, -2.5, 2)
    scene.camera.look_at((0, 0, 0))
  elif OBJNAME == 'pillow-rot-obj':
    obj = asset_source.create(
        asset_id='03938244/b5cb58fb099204fea5c423249b53dbc4')
    obj.scale = 2
    POSITION = (0, 0, 0.2)
    VELOCITY = (0.5, 0, 0)
    obj.angular_velocity = (0, 0, 5)
  else:
    raise NotImplementedError
  obj.position = POSITION
  obj.velocity = VELOCITY
  obj.metadata = {
      "asset_id": obj.asset_id,
  }
  obj.quaternion = kb.Quaternion(axis=[1, 0, 0], degrees=90)
  objname = obj.name
else:
  raise NotImplementedError

if TEXTURE:
  bpy_scene = bpy.context.scene
  obj.material = kb.PrincipledBSDFMaterial(name="material")
  obj.material.metallic = random.random()
  obj.material.roughness = random.random()**0.2

  scene += obj

  mat = bpy_scene.objects[objname].active_material
  tree = mat.node_tree

  mat_node = tree.nodes["Principled BSDF"]
  tex_image = mat.node_tree.nodes.new('ShaderNodeTexImage')
  tex_image.image = bpy.data.images.load('examples/tex/tex.jpg')
  tree.links.new(mat_node.inputs['Base Color'], tex_image.outputs['Color'])
else:
  scene += obj

if NO_MATERIAL:
  for material in bpy.data.materials:
    material.user_clear()
    bpy.data.materials.remove(material)


cam_params = []

if ROT_CAM:
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
  theta_change = ROT_RANGE / \
      ((scene.frame_end - scene.frame_start) / num_phi_values_per_theta)

  # pdb.set_trace()

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
renderer.save_state(f"output/{OBJNAME}/{OBJNAME}.blend")
frames_dict = renderer.render()


with open(f'output/{OBJNAME}/frames.dict', 'wb') as file:
  pickle.dump(frames_dict, file)

# kb.write_image_dict(frames_dict, f"output/{OBJNAME}")


# convert segmentation mask to LASR style
palette = [[0, 0, 0], [0, 0, 0], [128, 128, 128], [
    128, 128, 128], [128, 128, 128], [128, 128, 128]]
kb.file_io.multi_write_image(
    frames_dict['segmentation'],
    str(kb.as_path(f"output/{OBJNAME}/LASR/Annotations/Full-Resolution/{OBJNAME}") / "{:05d}.png"),
    write_fn=kb.write_palette_png,
    max_write_threads=16,
    palette=palette
    )
kb.file_io.multi_write_image(
    frames_dict['segmentation'],
    str(kb.as_path(f"output/{OBJNAME}/LASR/Annotations/Full-Resolution/r{OBJNAME}") / "{:05d}.png"),
    write_fn=kb.write_palette_png,
    max_write_threads=16,
    palette=[[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
    )

kb.file_io.multi_write_image(
    frames_dict['rgba'],
    str(kb.as_path(f"output/{OBJNAME}/LASR/JPEGImages/Full-Resolution/{OBJNAME}") / "{:05d}.png"),
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


fw = frames_dict['forward_flow'][:-1, ...] * 256
bw = frames_dict['backward_flow'][1:, ...] * 256
imgs = frames_dict['rgba']
M, N = imgs.shape[1:3]

occs = np.ones(fw.shape[:-1]).astype('float32')


os.makedirs(
    f'output/{OBJNAME}/LASR/FlowFW/Full-Resolution/{OBJNAME}', exist_ok=True)
os.makedirs(
    f'output/{OBJNAME}/LASR/FlowBW/Full-Resolution/{OBJNAME}', exist_ok=True)
os.makedirs(
    f'output/{OBJNAME}/LASR/FlowFW/Full-Resolution/r{OBJNAME}', exist_ok=True)
os.makedirs(
    f'output/{OBJNAME}/LASR/FlowBW/Full-Resolution/r{OBJNAME}', exist_ok=True)
os.makedirs(
    f'output/{OBJNAME}/LASR/Camera/Full-Resolution/{OBJNAME}', exist_ok=True)
os.makedirs(
    f'output/{OBJNAME}/LASR/Camera/Full-Resolution/r{OBJNAME}', exist_ok=True)

# write flows into pfm
for i in range(len(fw)):
  f = fw[i, ...]
  ones = np.ones_like(f[..., :1])
  f = np.concatenate([f[..., 1:], f[..., :1], ones], -1)
  b = np.concatenate([-bw[i, ..., 1:], -bw[i, ..., :1], ones], -1)

  f = np.flip(f, 0)
  b = np.flip(b, 0)

  write_pfm(
      f'output/{OBJNAME}/LASR/FlowFW/Full-Resolution/{OBJNAME}/flo-{i:05d}.pfm', f)
  write_pfm(
      f'output/{OBJNAME}/LASR/FlowBW/Full-Resolution/{OBJNAME}/flo-{i+1:05d}.pfm', b)
  write_pfm(f'output/{OBJNAME}/LASR/FlowFW/Full-Resolution/{OBJNAME}/occ-{i:05d}.pfm',
            np.ones_like(occs[i, ...]))
  write_pfm(f'output/{OBJNAME}/LASR/FlowBW/Full-Resolution/{OBJNAME}/occ-{i+1:05d}.pfm',
            np.ones_like(occs[i, ...]))

  write_pfm(
      f'output/{OBJNAME}/LASR/FlowFW/Full-Resolution/r{OBJNAME}/flo-{i:05d}.pfm', f)
  write_pfm(
      f'output/{OBJNAME}/LASR/FlowBW/Full-Resolution/r{OBJNAME}/flo-{i+1:05d}.pfm', b)
  write_pfm(f'output/{OBJNAME}/LASR/FlowFW/Full-Resolution/r{OBJNAME}/occ-{i:05d}.pfm',
            np.ones_like(occs[i, ...]))
  write_pfm(f'output/{OBJNAME}/LASR/FlowBW/Full-Resolution/r{OBJNAME}/occ-{i+1:05d}.pfm',
            np.ones_like(occs[i, ...]))

for i in range(len(cam_params)):
  # save camera parameters
  np.savetxt(
      f'output/{OBJNAME}/LASR/Camera/Full-Resolution/{OBJNAME}/{i:05d}.txt', cam_params[i].T)
  np.savetxt(
      f'output/{OBJNAME}/LASR/Camera/Full-Resolution/r{OBJNAME}/{i:05d}.txt', cam_params[i].T)

# write gif
imageio.mimsave(
    str(kb.as_path(f"output/{OBJNAME}/") / f"{OBJNAME}.gif"), frames_dict['rgba'])
kb.file_io.write_flow_batch(
    frames_dict['forward_flow'],
    directory=f"output/{OBJNAME}/FlowFW", file_template="{:05d}.png", name="forward_flow",
    max_write_threads=16
    )
kb.file_io.write_flow_batch(
    frames_dict['backward_flow'],
    directory=f"output/{OBJNAME}/FlowBW",
    file_template="{:05d}.png",
    name="backward_flow",
    max_write_threads=16
    )
