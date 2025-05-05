# Copyright 2024 The Kubric Authors.
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

import os
import logging
import numpy as np
import bpy
import sys

import kubric as kb
from kubric.renderer import Blender
import yaml

from kubric.utils import ArgumentParser

parser = ArgumentParser()

# Function to add a thin cylinder pointing up at the object's location
def add_cylinder_at_object(scene, obj, height=1.0, radius=0.05):
    cylinder = kubasic.create(asset_id="cylinder")
    cylinder.scale = (radius, radius, height)
    cylinder.position = obj.position - np.array([0, 0, obj.aabbox[0][2]]) + np.array([0, 0, height / 2])
    cylinder.material = kb.PrincipledBSDFMaterial(color=kb.Color(0.8, 0.2, 0.2))  # Red color for visibility
    scene.add(cylinder)
    
parser.set_defaults(
    frame_end=1,
    resolution=(512, 512),
)
FLAGS = parser.parse_args()



# --- Common setups & resources
scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
cam = scene.camera
renderer = Blender(scene, scratch_dir,
                   samples_per_pixel=64,
                   background_transparency=True)

# --- Fetch shapenet
source_path = os.getenv("SHAPENET_GCP_BUCKET", "gs://kubric-unlisted/assets/ShapeNetCore.v2.json")
shapenet = kb.AssetSource.from_manifest(source_path)

# --- Fetch KuBasic
kubasic_source_path = "gs://kubric-public/assets/KuBasic/KuBasic.json"
kubasic = kb.AssetSource.from_manifest(kubasic_source_path)

# --- Add Klevr-like lights to the scene
scene += kb.assets.utils.get_clevr_lights(rng=rng)
scene.ambient_illumination = kb.Color(0.2, 0.2, 0.2)
print("Added Klevr-like lights to the scene")


# Add walls at 5 meters
wall_material = kb.PrincipledBSDFMaterial(color=kb.Color(0.9, 0.9, 0.9))  # Brighter light grey color
# outdoor objects
all_categories = ["bench", "pot", "table", "chair"]
all_nonnative_categories = ["birdhouse", "trashcan", "tree", "rock"]  # need to manually set asset_id

# add the basic scene background
obj = kb.FileBasedObject(
    asset_id="custom", 
    render_filename="/kubric/examples/lts/assets/skydome/xrtlbp429gwv.obj", 
    bounds=((-1, -1, -1), (1, 1, 1)),
    simulation_filename=None
)
obj.scale = (1, 1, 1)
obj.quaternion = kb.Quaternion(axis=[1, 0, 0], degrees=90)
obj.position = np.array([-18.4, -25.8, 0])
scene += obj

# Add a sky blue wall at x=-10
wall_material = kb.PrincipledBSDFMaterial(color=kb.Color(0.5, 0.5, 1.0))
wall = kb.Cube(scale=(0.1, 10, 10), position=(-10, 0, 5), material=wall_material)
scene += wall

# Load assets from objects.yaml
with open("/kubric/examples/lts/objects.yaml", "r") as file: # mount path
    assets = yaml.safe_load(file)

# --- Keyframe the camera
scene.camera = kb.PerspectiveCamera()
print(scene.camera.intrinsics)
for frame in range(1):
    scene.camera.position = (7, 0, 2)
    scene.camera.look_at((0, 0, 0))
    scene.camera.keyframe_insert("position", frame)
    scene.camera.keyframe_insert("quaternion", frame)

    # Define the geometry dictionary
    geometry = {obj: {} for obj in all_categories + all_nonnative_categories}
    selected_objects = []

    all_categories = ["bench", "pot", "table", "chair"]
    all_nonnative_categories = ["birdhouse", "trashcan"]
    special_categories = ["tree", "rock"]

    all_objects = all_categories + all_nonnative_categories + special_categories

    angle_step = 2 * np.pi / len(all_objects)
    current_angle = 0

    selected_objects = {}

    for key in all_objects:
        position = [2 * np.cos(current_angle), 2 * np.sin(current_angle), 0]
        current_angle += angle_step

        if key == "tree":
            obj = kb.FileBasedObject(
                asset_id="custom", 
                render_filename="/kubric/examples/lts/assets/tree/ImageToStl.com_model.obj", 
                bounds=((-1, -1, -1), (1, 1, 1)),
                simulation_filename=None
            )
            obj.scale = (0.05, 0.05, 0.05)
            obj.position = position
            obj.position = obj.position - np.array([0, 0, obj.aabbox[0][2]]) 
            obj.position = obj.position - np.array([0, 0, 0.1]) # adjust the z value to make it sit on the floor
            scene += obj
            selected_objects[key] = obj
            add_cylinder_at_object(scene, obj)

        elif key == "rock":
            obj = kb.FileBasedObject(
                asset_id="custom", 
                render_filename="/kubric/examples/lts/assets/rock/qroz9y5c1c6e.obj", 
                bounds=((-1, -1, -1), (1, 1, 1)),
                simulation_filename=None
            )
            obj.scale = (0.1, 0.1, 0.1)
            obj.quaternion = kb.Quaternion(axis=[1, 0, 0], degrees=90)
            obj.position = position
            obj.position = obj.position - np.array([0, 0, obj.aabbox[0][2]]) 
            obj.position = obj.position - np.array([0, 0, 0.1]) # adjust the z value to make it sit on the floor
            scene += obj
            selected_objects[key] = obj
            add_cylinder_at_object(scene, obj)

        else:
            if key in all_categories:
                asset_info = assets["native_shapenet_assets"][key]
                asset_id = asset_info["asset_id"]
                obj = shapenet.create(asset_id=asset_id)
            else:
                asset_info = assets["non_native_shapenet_assets"][key]
                asset_id = asset_info["asset_id"]
                obj = shapenet.create(asset_id=asset_id)


            obj.quaternion = kb.Quaternion(axis=[1, 0, 0], degrees=90)
            obj.quaternion = kb.Quaternion(axis=[0, 0, 1], degrees=asset_info["yaw"]) * obj.quaternion
            obj.scale = asset_info["scale"]
            obj.position = np.array(position)
            obj.position = obj.position - np.array([0, 0, obj.aabbox[0][2]]) 

            if key == "birdhouse":
                obj.position = obj.position + np.array([0, 0, 1])
                column = kubasic.create(asset_id="cylinder")
                column.scale = (0.1, 0.1, 1.0)
                column.position = (position[0], position[1], 0.5)
                column.material = kb.PrincipledBSDFMaterial(color=kb.Color(0.2, 0.1, 0.05))
                scene.add(column)
            scene.add(obj)
            add_cylinder_at_object(scene, obj)
            selected_objects[key] = obj
            logging.info(f"selected '{asset_id}' from category '{key}'")

    # Add another bench rotated 90 degrees
    new_bench_info = assets["native_shapenet_assets"]["bench"]
    new_bench_id = new_bench_info["asset_id"]
    new_bench = shapenet.create(asset_id=new_bench_id)
    new_bench.scale = assets["native_shapenet_assets"]["bench"]["scale"]
    new_bench.quaternion = kb.Quaternion(axis=[1, 0, 0], degrees=90)
    new_bench.quaternion =  kb.Quaternion(axis=[0, 0, 1], degrees=90) * new_bench.quaternion 
    new_bench.position = np.array([0, 0, 0])
    new_bench.position = new_bench.position - np.array([0, 0, new_bench.aabbox[0][2]])
    scene += new_bench
    add_cylinder_at_object(scene, new_bench)



    for key, obj in selected_objects.items():
        adjusted_aabbox = obj.aabbox - obj.position
        # Center the lower and upper bound on zero
        extent = np.mean(np.abs(adjusted_aabbox), axis=0)
        print(f"Category ID: {key}, Extent: {extent}\n\n")

    high_intensity_light = kb.PointLight(color=kb.Color(1, 1, 1), intensity=10000)
    high_intensity_light.position = (-5, 5, 10)
    scene += high_intensity_light

    scene.camera.keyframe_insert("position", frame)
    scene.camera.keyframe_insert("quaternion", frame)

    logging.info("Rendering the scene ...")
    data_stack = renderer.render()

    kb.compute_visibility(data_stack["segmentation"], scene.assets)
    data_stack["segmentation"] = kb.adjust_segmentation_idxs(
        data_stack["segmentation"],
        scene.assets,
        selected_objects).astype(np.uint8)

    kb.file_io.write_rgba_batch(data_stack["rgba"], output_dir)
    kb.file_io.write_depth_batch(data_stack["depth"], output_dir)
    kb.file_io.write_segmentation_batch(data_stack["segmentation"], output_dir)

    logging.info("Collecting and storing metadata for each object.")
    data = {
        "metadata": kb.get_scene_metadata(scene),
        "camera": kb.get_camera_info(scene.camera),
        "object": kb.get_instance_info(scene, selected_objects)
    }
    kb.file_io.write_json(filename=output_dir / "metadata.json", data=data)
    kb.done()
