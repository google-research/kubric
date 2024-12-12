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
import imageio
import logging
import numpy as np
import kubric as kb
from kubric.renderer.blender import Blender as KubricRenderer

logging.basicConfig(level="INFO")


# --- CLI arguments
parser = kb.ArgumentParser()
FLAGS = parser.parse_args()

# --- create scene and attach a renderer
image_size = (512, 512)
scene = kb.Scene(resolution=image_size)
scene.frame_start = 1
scene.frame_end = 180  # < numbers of frames to render
scene.frame_rate = 24 # < rendering framerate
camera_radius = 0.7
camera_height = 1.5
camera_look_at = (0, 0, camera_height)
use_static_scene = False 

if use_static_scene:
    blend_file = "examples/KuBasic/rain_v22/rain_v2.1.blend"
    scratch_dir = "output/static"
else:
    blend_file = "examples/KuBasic/rain_v22/rain_v2.1_face_animated.blend"
    scratch_dir = "output/dynamic"

scene += kb.DirectionalLight(name="sun", position=(10, -10, 1.5), look_at=(0, 0, 1), intensity=3.0)
scene += kb.DirectionalLight(name="sun", position=(-10, -10, 1.5), look_at=(0, 0, 1), intensity=3.0)
scene += kb.PerspectiveCamera(name="camera", position=(0, -camera_radius, camera_height), look_at=camera_look_at)
renderer = KubricRenderer(scene, scratch_dir=scratch_dir,
                          custom_scene=blend_file)

frames = []

for frame_nr in range(scene.frame_start - 1, scene.frame_end + 2):
    interp = float(frame_nr - scene.frame_start + 1) / float(scene.frame_end - scene.frame_start + 3)
    scene.camera.position = (-camera_radius*np.sin(interp*2*np.pi),
                            -camera_radius*np.abs(np.cos(interp*2*np.pi)),
                            camera_height)
    scene.camera.look_at(camera_look_at)
    scene.camera.keyframe_insert("position", frame_nr )
    scene.camera.keyframe_insert("quaternion", frame_nr )

    kb.write_json(filename=f"{scratch_dir}/camera/frame_{frame_nr:04d}.json", data=
        kb.get_camera_info(scene.camera))

    frames.append(frame_nr)

frames_dict = renderer.render(frames=frames, ignore_missing_textures=True)
kb.write_image_dict(frames_dict, f"{scratch_dir}/output/")

# --- render (and save the blender file)
renderer.save_state(f"{scratch_dir}/articulation.blend")

# -- generate a gif
gif_writer = imageio.get_writer(f"{scratch_dir}/summary.gif", mode="I")
mp4_writer = imageio.get_writer(f"{scratch_dir}/video.mp4", fps=scene.frame_rate)
all_files = os.listdir(f"{scratch_dir}/output/")
for frame in range(scene.frame_start - 1, scene.frame_end + 2):
    image_files = [os.path.join(f"{scratch_dir}/output", f) for f in all_files if str(frame).zfill(4) in f]
    image_files = [f for f in image_files if f.endswith(".png") and "coordinates" not in f]
    image_files = sorted(image_files)
    assert len(image_files) > 0
    images = [imageio.imread(f, "PNG") for f in image_files]
    images = [np.dstack(3*[im[..., None]]) if len(im.shape) == 2 else im for im in images]
    images = [im[..., :3] if im.shape[2]>3 else im for im in images]
    if len(images) % 2 == 1:
        images += [np.zeros_like(images[0])]
    images = [np.hstack(images[:len(images)//2]), np.hstack(images[len(images)//2:])]
    images = np.vstack(images)
    gif_writer.append_data(images)
    rgb_image_file = os.path.join(f"{scratch_dir}/images", f"frame_{frame:04d}.png")
    image = imageio.imread(rgb_image_file)
    mp4_writer.append_data(image)
gif_writer.close()
mp4_writer.close()