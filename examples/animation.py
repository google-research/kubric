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

"""
3D asset licenses:

Scene backgroup: CC0 licensed meshes and materials from PolyHaven.com.
Human model: CC0 licensed rigged model from Quaternius.com.

More background assets can be purchased on BlenderKit.com.
Photorealistic rigged human models can be purchased at RenderPeople.com.
"""

import json
import logging
import kubric as kb
import shutil
import tensorflow as tf

from typing import Dict, Sequence

from kubric.renderer.blender import Blender as KubricRenderer


def pose_rigid_body(rigid_body, pose: Dict[str, float], frame: float) -> None:
    rigid_body.rotation_mode = 'QUATERNION'
    for channel_name, channel_value in pose.items():
        if channel_name == 'x':
            rigid_body.location.x = channel_value
        if channel_name == 'y':
            rigid_body.location.y = channel_value
        if channel_name == 'z':
            rigid_body.location.z = channel_value
        rigid_body.keyframe_insert(data_path='location', frame=frame)
    
        if channel_name == 'qw':
            rigid_body.rotation_quaternion.w = channel_value
        if channel_name == 'qx':
            rigid_body.rotation_quaternion.x = channel_value
        if channel_name == 'qy':
            rigid_body.rotation_quaternion.y = channel_value
        if channel_name == 'qz':
            rigid_body.rotation_quaternion.z = channel_value
        rigid_body.keyframe_insert(data_path='rotation_quaternion', frame=frame)


def pose_armature(
    scene, armature_name: str, frame: float, pose: Dict[str, Dict[str, float]]) -> None:
    armature = scene.objects[armature_name]

    skeleton = {b.name: b for b in armature.pose.bones}

    for bone_name, bone_pose in pose.items():
        if bone_name not in skeleton:
            logging.info(f'Animation tried to pose unknown bone {bone_name}')
            continue

        target_bone = skeleton[bone_name]

        pose_rigid_body(target_bone, bone_pose, frame)


logging.basicConfig(level='INFO')

parser = kb.ArgumentParser()
FLAGS = parser.parse_args()

tf.io.gfile.mkdir('examples/KuBasic/')
tf.io.gfile.copy(
    'gs://kubric-public/data/pose_estimation/AnimationExample.zip', 
    'examples/KuBasic/AnimationExample.zip')
shutil.unpack_archive('examples/KuBasic/AnimationExample.zip', 'examples/KuBasic')

scene = kb.Scene(resolution=(512, 512))
scene.frame_end = 30
scene.frame_rate = 24

scene += kb.DirectionalLight(name='dir_light_0', position=(4, -4, 1.5), look_at=(0, 0, 1), intensity=2.5)
scene += kb.PointLight(name='point_light_0', position=(-4, -4, 1.5), intensity=400.0)
scene += kb.PointLight(name='point_light_0', position=(3, -3, 2.5), intensity=300.0)
scene += kb.PerspectiveCamera(name='camera', position=(0, -3.5, 2.5), look_at=(0, 0, 2.0))

renderer = KubricRenderer(
    scene, scratch_dir='output_tmp', custom_scene='examples/KuBasic/AnimationExample/tanktop_in_room.blend')

"""
To only animate the left arm:

pose_armature(
    renderer.blender_scene,
    'HumanArmature',
    1.0,
    {'UpperArm.L': {
        'qw': 1.0,
        'qx': 0.0,
        'qy': 0.0,
        'qz': 0.0
    }})

pose_armature(
    renderer.blender_scene,
    'HumanArmature',
    24.0,
    {'UpperArm.L': {
        'qw': 0.8341912627220154,
        'qx': -0.5452688932418823,
        'qy': -0.0407070517539978,
        'qz': -0.07176481932401657
    }})

pose_armature(
    renderer.blender_scene,
    'HumanArmature',
    60.0,
    {'UpperArm.L': {
        'qw': 1.0,
        'qx': 0.0,
        'qy': 0.0,
        'qz': 0.0
    }})
"""

with open('examples/KuBasic/AnimationExample/punch_0.json') as anim_file:
    animation: Dict[str, Dict[str, Dict[str, str]]] = json.load(anim_file)

for frame, pose in animation.items():
    pose_armature(renderer.blender_scene, 'HumanArmature', float(frame), pose)

frames = range(scene.frame_start - 1, scene.frame_end + 2)

scene.camera.position = (5.0, -5.8, 2.8)
scene.camera.look_at((0.0, 0.0, 2.2))
scene.camera.keyframe_insert('position', 1.0)
scene.camera.keyframe_insert('quaternion', 1.0)

scene.camera.position = (-5.2, -5.7, 3.2)
scene.camera.look_at((0.0, 0.0, 2.4))
scene.camera.keyframe_insert('position', 60.0)
scene.camera.keyframe_insert('quaternion', 60.0)

frames_dict = renderer.render(frames=frames, ignore_missing_textures=True)
kb.write_image_dict(frames_dict, 'output')

renderer.save_state('output/animation.blend')
