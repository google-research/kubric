# Copyright 2023 The Kubric Authors.
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

from kubric.safeimport.bpy import bpy

from kubric import core
from kubric.renderer import blender
from kubric.renderer import blender_utils


def test_prepare_blender_object():
  @blender_utils.prepare_blender_object
  def add_asset(self, asset):
    bpy.ops.mesh.primitive_cube_add()
    cube = bpy.context.active_object
    return cube

  cube_asset = core.Cube()
  cube_obj = add_asset(None, cube_asset)

  assert cube_obj.name == cube_asset.uid
  assert cube_obj.rotation_mode == "QUATERNION"
  assert cube_obj in bpy.context.scene.collection.objects.values()


def test_blender_scene_properties(tmp_path):
  scene = core.Scene(
      frame_start=2,
      frame_end=3,
      frame_rate=5,
      resolution=(7, 11),
  )
  renderer = blender.Blender(scene, tmp_path)
  assert renderer in scene.views
  assert renderer.scene == scene

  assert renderer.blender_scene.frame_start == 2
  assert renderer.blender_scene.frame_end == 3
  assert renderer.blender_scene.render.fps == 5
  assert renderer.blender_scene.render.resolution_x == 7
  assert renderer.blender_scene.render.resolution_y == 11


def test_blender_camera_on_init(tmp_path):
  cam = core.PerspectiveCamera(position=(1, 2, 3), quaternion=(0, 1, 0, 0), focal_length=3,
                               sensor_width=4)
  renderer = blender.Blender(core.Scene(camera=cam), tmp_path)

  assert renderer in cam.linked_objects
  blender_cam = cam.linked_objects[renderer]
  assert renderer.blender_scene.camera == blender_cam
  assert blender_cam in renderer.blender_scene.collection.objects.values()
  assert tuple(blender_cam.location) == (1, 2, 3)
  assert tuple(blender_cam.rotation_quaternion) == (0, 1, 0, 0)
  assert blender_cam.data.lens == 3
  assert blender_cam.data.sensor_width == 4


def test_blender_camera_assign_after_init(tmp_path):
  scene = core.Scene()
  renderer = blender.Blender(scene, tmp_path)

  cam = core.PerspectiveCamera(position=(1, 2, 3), quaternion=(0, 1, 0, 0), focal_length=3,
                               sensor_width=4)

  scene.camera = cam

  assert renderer in cam.linked_objects
  blender_cam = cam.linked_objects[renderer]
  assert renderer.blender_scene.camera == blender_cam
  assert blender_cam in renderer.blender_scene.collection.objects.values()
  assert tuple(blender_cam.location) == (1, 2, 3)
  assert tuple(blender_cam.rotation_quaternion) == (0, 1, 0, 0)
  assert blender_cam.data.lens == 3
  assert blender_cam.data.sensor_width == 4


def test_blender_adaptive_sampling_default(tmp_path):
  renderer = blender.Blender(core.Scene(), tmp_path)
  assert renderer.adaptive_sampling is False
  assert renderer.blender_scene.cycles.use_adaptive_sampling is False


def test_blender_set_adaptive_sampling(tmp_path):
  renderer = blender.Blender(core.Scene(), tmp_path)
  renderer.adaptive_sampling = False
  assert renderer.adaptive_sampling is False
  assert renderer.blender_scene.cycles.use_adaptive_sampling is False


def test_blender_init_adaptive_sampling(tmp_path):
  renderer = blender.Blender(core.Scene(), tmp_path, adaptive_sampling=False)
  assert renderer.adaptive_sampling is False
  assert renderer.blender_scene.cycles.use_adaptive_sampling is False


def test_blender_use_denoising_default(tmp_path):
  renderer = blender.Blender(core.Scene(), tmp_path)
  assert renderer.use_denoising is True
  assert renderer.blender_scene.cycles.use_denoising is True


def test_blender_set_use_denoising(tmp_path):
  renderer = blender.Blender(core.Scene(), tmp_path)
  renderer.use_denoising = False
  assert renderer.use_denoising is False
  assert renderer.blender_scene.cycles.use_denoising is False


def test_blender_use_denoising_init(tmp_path):
  renderer = blender.Blender(core.Scene(), tmp_path, use_denoising=False)
  assert renderer.use_denoising is False
  assert renderer.blender_scene.cycles.use_denoising is False


def test_blender_samples_per_pixel_default(tmp_path):
  renderer = blender.Blender(core.Scene(), tmp_path)
  assert renderer.samples_per_pixel == 128
  assert renderer.blender_scene.cycles.samples == 128


def test_blender_set_samples_per_pixel(tmp_path):
  renderer = blender.Blender(core.Scene(), tmp_path)
  renderer.samples_per_pixel = 64
  assert renderer.samples_per_pixel == 64
  assert renderer.blender_scene.cycles.samples == 64


def test_blender_samples_per_pixel_init(tmp_path):
  renderer = blender.Blender(core.Scene(), tmp_path, samples_per_pixel=256)
  assert renderer.samples_per_pixel == 256
  assert renderer.blender_scene.cycles.samples == 256
