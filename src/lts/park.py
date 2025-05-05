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

import kubric as kb
from kubric.renderer import Blender
import yaml

# --- CLI arguments
parser = kb.ArgumentParser()
parser.set_defaults(
    frame_end=1,
    resolution=(512, 512),
)
FLAGS = parser.parse_args()

# --- Common setups & resources
scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
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
all_nonnative_categories = ["birdhouse", "trashcan"]  # need to manually set asset_id

# --- Keyframe the camera
scene.camera = kb.PerspectiveCamera()
for frame in range(1):
    scene.camera.position = (7, 0, 2)
    scene.camera.look_at((0, 0, 0))
    scene.camera.keyframe_insert("position", frame)
    scene.camera.keyframe_insert("quaternion", frame)

    # Load assets from objects.yaml
    with open("/kubric/examples/lts/objects.yaml", "r") as file: # mount path
        assets = yaml.safe_load(file)


    categories = np.random.choice(all_categories, 2, replace=False)  # randomly select two from each cateory
    nonnative_categories = np.random.choice(all_nonnative_categories, 2, replace=False)
    # randomly select two from each cateory

    selected_objects = []

    obj = kb.FileBasedObject(
        asset_id="custom", 
        render_filename="/kubric/examples/lts/ImageToStl.com_model/ImageToStl.com_model.obj", 
        bounds=((-1, -1, -1), (1, 1, 1)),
        simulation_filename=None
    )
    obj.scale = (0.05, 0.05, 0.05)
    obj.position = rng.uniform(-3, 3, size=3)
    obj.position = obj.position - np.array([0, 0, obj.position[2]]) 
    obj.position = obj.position - np.array([0, 0, 0.1]) # adjust the z value to make it sit on the floor
    scene += obj

    obj = kb.FileBasedObject(
        asset_id="custom", 
        render_filename="/kubric/examples/lts/model/qroz9y5c1c6e.obj", 
        bounds=((-1, -1, -1), (1, 1, 1)),
        simulation_filename=None
    )
    obj.scale = (0.1, 0.1, 0.1)
    obj.quaternion = kb.Quaternion(axis=[1, 0, 0], degrees=90)
    # put the centroid of the box in a random position on the ground using the aabbox
    random_position = rng.uniform(-3, 3, size=3)
    random_position[2] = 0
    obj.position = obj.position - np.array([0, 0, obj.aabbox[0][2]]) 
    obj.position = obj.position -  np.array([0, 0, 0.1]) # adjust the z value to make it sit on the floor
    # get the centroid of the box
    scene += obj

    # print(obj.aabbox )

    obj = kb.FileBasedObject(
        asset_id="custom", 
        render_filename="/kubric/examples/lts/sky2/xrtlbp429gwv.obj", 
        bounds=((-1, -1, -1), (1, 1, 1)),
        simulation_filename=None
    )
    obj.scale = (1, 1, 1)

    obj.quaternion = kb.Quaternion(axis=[1, 0, 0], degrees=90)
    # obj.quaternion = obj.quaternion * kb.Quaternion(axis=[0, 0, 1], degrees=180)
    # put the centroid of the box in a random position on the ground using the aabbox
    print(obj.aabbox)
    obj.position = np.array([-18.4, -25.8, 0])
    print(obj.aabbox)

    # get the centroid of the box
    scene += obj


    # Add a sky blue wall at x=-10
    wall_material = kb.PrincipledBSDFMaterial(color=kb.Color(0.5, 0.5, 1.0))
    wall = kb.Cube(scale=(0.1, 10, 10), position=(-10, 0, 5), material=wall_material)
    scene += wall

    for category in categories:
        asset_info = assets["native_shapenet_assets"][category]
        position = rng.uniform(-2, 2, size=3)
        rotation = asset_info["rotation"]

        asset_ids = [name for name, spec in shapenet._assets.items()
                    if spec["metadata"]["category"].lower() == category.lower()]

        asset_id = rng.choice(asset_ids)
        obj = shapenet.create(asset_id=asset_id)
        print(obj.aabbox)

        obj.quaternion = kb.Quaternion(axis=[1, 0, 0], degrees=90)
        obj.scale = asset_info["scale"]  # Scale the object
        print(obj.aabbox)
        obj.position = np.array(position)

        # subtract the z value of the bottom of the object to make it sit on the floor
        obj.position = obj.position - np.array([0, 0, obj.aabbox[0][2]]) 
        print(obj.aabbox)

        scene.add(obj)
        selected_objects.append(obj)
        logging.info(f"selected '{asset_id}' from category '{category}'")

    for category in nonnative_categories:
        asset_info = assets["non_native_shapenet_assets"][category]
        position = rng.uniform(-2, 2, size=3)
        rotation = asset_info["rotation"]

        asset_id = asset_info["asset_id"]
        obj = shapenet.create(asset_id=asset_id)
        obj.quaternion = kb.Quaternion(axis=[1, 0, 0], degrees=90)
        obj.scale = asset_info["scale"]  # Scale the object
        print(obj.aabbox)
        obj.position = np.array(position) 

        # subtract the z value of the bottom of the object to make it sit on the floor
        obj.position = obj.position - np.array([0, 0, obj.aabbox[0][2]]) 
        if category == "birdhouse":
            # add another meter
            obj.position = obj.position  + np.array([0, 0, 1])
        print(obj.aabbox)
            
        scene.add(obj)
        selected_objects.append(obj)
        logging.info(f"selected '{asset_id}' from category '{category}'")

        # Add a wooden cylinder column for birdhouse
        if category == "birdhouse":
            column = kubasic.create(asset_id="cylinder")
            column.scale = (0.1, 0.1, 1.0)  # Small radius
            column.position = (position[0], position[1], 0.5)  # Add 1 meter to height
            column.material = kb.PrincipledBSDFMaterial(color=kb.Color(0.2, 0.1, 0.05))  # Wood color
            scene.add(column)
    # Add a high intensity light at the origin
    high_intensity_light = kb.PointLight(color=kb.Color(1, 1, 1), intensity=10000)
    high_intensity_light.position = (-5, 5, 10)
    scene += high_intensity_light

    # # Ensure all objects are visible and adjust camera
    # min_x, min_y, min_z = np.min([obj.position for obj in selected_objects], axis=0)
    # max_x, max_y, max_z = np.max([obj.position for obj in selected_objects], axis=0)
    # scene.camera.position = kb.sample_point_in_half_sphere_shell(5.0, 6.0, 0.5)  # Zoom way out
    # scene.camera.look_at(((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2 - 0.5))  # Tilt downward
    scene.camera.keyframe_insert("position", frame)
    scene.camera.keyframe_insert("quaternion", frame)


    # --- Rendering
    logging.info("Rendering the scene ...")
    renderer.save_state(output_dir / "scene.blend")
    data_stack = renderer.render()

    # --- Postprocessing
    kb.compute_visibility(data_stack["segmentation"], scene.assets)
    data_stack["segmentation"] = kb.adjust_segmentation_idxs(
        data_stack["segmentation"],
        scene.assets,
        selected_objects).astype(np.uint8)

    kb.file_io.write_rgba_batch(data_stack["rgba"], output_dir)
    kb.file_io.write_depth_batch(data_stack["depth"], output_dir)
    kb.file_io.write_segmentation_batch(data_stack["segmentation"], output_dir)

    # --- Collect metadata
    logging.info("Collecting and storing metadata for each object.")
    data = {
        "metadata": kb.get_scene_metadata(scene),
        "camera": kb.get_camera_info(scene.camera),
        "object": kb.get_instance_info(scene, selected_objects)
    }
    kb.file_io.write_json(filename=output_dir / "metadata.json", data=data)
    kb.done()
