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

import logging
import kubric as kb
from kubric.renderer.blender import Blender as KubricRenderer

logging.basicConfig(level="INFO")

# --- CLI arguments
parser = kb.ArgumentParser()
# --- create scene and attach a renderer
scene = kb.Scene(resolution=(256, 256))
scene.frame_end = 30   # < numbers of frames to render
scene.frame_rate = 24  # < rendering framerate

scene += kb.DirectionalLight(name="sun", position=(1, -1, 3), look_at=(0, 0, 1), intensity=3.0)
scene += kb.PerspectiveCamera(name="camera", position=(0, -0.7, 1.5), look_at=(0, 0, 1.5))
renderer = KubricRenderer(scene, scratch_dir="output_tmp", custom_scene="examples/KuBasic/rain_v22/rain_v2.1_face_animated.blend")


# --- render (and save the blender file)
renderer.save_state("output/articulation.blend")
frames_dict = renderer.render()
kb.write_image_dict(frames_dict, "output")