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

import random
import logging
import kubric as kb
from kubric.renderer.blender import Blender as KubricRenderer
import bpy

logging.basicConfig(level="INFO")

scene = kb.Scene(resolution=(1024, 1024), ambient_illumination=kb.Color(0.0, 0.1, 0.2), background=kb.Color(0.1, 0.4, 0.95))
renderer = KubricRenderer(scene, scratch_dir="output_tmp")

floor = kb.Cube(name="floor", scale=(1000, 1000, 0.1), position=(0, 0, -0.1))
floor.material = kb.PrincipledBSDFMaterial(name="material")
scene += floor

scene += kb.DirectionalLight(name="sun", position=(-1, -0.5, 3), look_at=(0, 0, 0), intensity=1.5)
scene += kb.PerspectiveCamera(name="camera", position=(1, 0, 10), look_at=(-5, 0, 1), focal_length=20)

bpy_scene = bpy.context.scene
for j in range(100):
    position = (
        -30.0 * ((j // 10) / 10),
        30.0 * ((j % 10) / 10 - 0.5) + 1.0,
        1.0,
    )
    instance = kb.Sphere(name=f"inst_{j}", scale=1, position=position)
    # instance = kb.Cube(name=f"inst_{j}", scale=1, position=position, quaternion=kb.random_rotation())
    instance.material = kb.PrincipledBSDFMaterial(name="material")
    instance.material.metallic = random.random()
    instance.material.roughness = random.random()**0.2
    scene += instance

    mat = bpy_scene.objects[f"inst_{j}"].active_material
    tree = mat.node_tree

    mat_node = tree.nodes["Principled BSDF"]
    ramp_node = tree.nodes.new(type="ShaderNodeValToRGB")
    tex_node = tree.nodes.new(type="ShaderNodeTexNoise")
    scaling_node = tree.nodes.new(type="ShaderNodeMapping")
    rotation_node = tree.nodes.new(type="ShaderNodeMapping")
    vector_node = tree.nodes.new(type="ShaderNodeNewGeometry")

    tree.links.new(vector_node.outputs["Position"], rotation_node.inputs["Vector"])
    tree.links.new(rotation_node.outputs["Vector"], scaling_node.inputs["Vector"])
    tree.links.new(scaling_node.outputs["Vector"], tex_node.inputs["Vector"])
    tree.links.new(tex_node.outputs["Fac"], ramp_node.inputs["Fac"])
    tree.links.new(ramp_node.outputs["Color"], mat_node.inputs["Base Color"])

    rotation_node.inputs["Rotation"].default_value = (
        random.random() * 3.141,
        random.random() * 3.141,
        random.random() * 3.141,
    )

    scaling_node.inputs["Scale"].default_value = (
        random.random()**2 * 2.0,
        random.random()**2 * 2.0,
        random.random()**2 * 2.0,
    )

    tex_node.inputs["Roughness"].default_value = random.random()
    tex_node.inputs["Detail"].default_value = 10.0 * random.random()

    for i in range(random.randint(3, 6)):
        ramp_node.color_ramp.elements.new(random.random())

    base_color = kb.random_hue_color()
    for element in ramp_node.color_ramp.elements:
        mult = random.random()**2
        element.color = (
            0.3 * random.random() + base_color.r * mult,
            0.3 * random.random() + base_color.g * mult,
            0.3 * random.random() + base_color.b * mult,
            1
        )

frame = renderer.render_still()
kb.write_png(frame["rgba"], "output/proc_texture.png")
