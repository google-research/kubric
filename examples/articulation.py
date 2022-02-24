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
scene = kb.Scene(resolution=(512, 512))
scene.frame_end = 60  # < numbers of frames to render
scene.frame_rate = 24  # < rendering framerate

scene += kb.DirectionalLight(name="sun", position=(10, -10, 1.5), look_at=(0, 0, 1), intensity=3.0)
scene += kb.DirectionalLight(name="sun", position=(-10, -10, 1.5), look_at=(0, 0, 1), intensity=3.0)
scene += kb.PerspectiveCamera(name="camera", position=(0, -0.7, 1.5), look_at=(0, 0, 1.5))

renderer = KubricRenderer(scene, scratch_dir="output_tmp", custom_scene="examples/KuBasic/rain_v22/rain_v2.1_face_animated.blend")

camera_radius = 0.7
frames = []
logging.info(scene.frame_end)
for frame in range(scene.frame_start - 1, scene.frame_end + 2):
    interp = float(frame - scene.frame_start + 1) / float(scene.frame_end - scene.frame_start + 3)
    scene.camera.position = (-camera_radius*np.sin(interp*2*np.pi),
                             -camera_radius*np.abs(np.cos(interp*2*np.pi)),
                              1.5)
    scene.camera.look_at((0, 0, 1.5))
    scene.camera.keyframe_insert("position", frame)
    scene.camera.keyframe_insert("quaternion", frame)
    frames.append(frame)
frames_dict = renderer.render(frames=frames, ignore_missing_textures=True)
kb.write_image_dict(frames_dict, "output")

# --- render (and save the blender file)
renderer.save_state("output/articulation.blend")

# -- generate a gif
with imageio.get_writer("output/summary.gif", mode="I") as writer:
    all_files = os.listdir("output/")
    for frame in range(scene.frame_start - 1, scene.frame_end + 2):
        image_files = [os.path.join("output", f) for f in all_files if str(frame).zfill(5) in f]
        image_files = [f for f in image_files if f.endswith(".png") and "coordinates" not in f]
        image_files = sorted(image_files)
        # image_files = image_files[1:]
        images = [imageio.imread(f, "PNG") for f in image_files]
        images = [np.dstack(3*[im[..., None]]) if len(im.shape) == 2 else im for im in images]
        images = [im[..., :3] if im.shape[2]>3 else im for im in images]
        images = [np.hstack(images[:len(images)//2]), np.hstack(images[len(images)//2:])]
        images = np.vstack(images)
        writer.append_data(images)



        # logging.info([im.shape for im in images])