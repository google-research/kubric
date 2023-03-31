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
"""TODO(klausg): write a one liner of what this module contains."""

import contextlib
import copy
import json
import pathlib
import tarfile
import shutil

import numpy as np
import trimesh
from kubric.safeimport.bpy import bpy


URDF_TEMPLATE = """
<robot name="{id}">
    <link name="base">
        <contact>
            <lateral_friction value="{friction}" />  
        </contact>
        <inertial>
            <origin xyz="{center_mass[0]} {center_mass[1]} {center_mass[2]}" />
            <mass value="{mass}" />
            <inertia ixx="{inertia[0][0]}" ixy="{inertia[0][1]}" 
                     ixz="{inertia[0][2]}" iyy="{inertia[1][1]}" 
                     iyz="{inertia[1][2]}" izz="{inertia[2][2]}" />
        </inertial>
        <visual>
            <origin xyz="0 0 0" />
            <geometry>
                <mesh filename="visual_geometry.obj" />
            </geometry>
        </visual>
        <collision>
            <origin xyz="0 0 0" />
            <geometry>
                <mesh filename="collision_geometry.obj" />
            </geometry>
        </collision>
    </link>
</robot>
"""


def get_active_object():
  return bpy.context.object


def get_vertices_and_faces(obj):
  bmesh = obj.data  # TODO: verify that this is a mesh object?
  vertices = np.array([v.co for v in bmesh.vertices])
  faces = np.array([list(p.vertices) for p in bmesh.polygons if len(p.vertices) > 2])  # pylint: disable=unnecessary-comprehension
  return vertices, faces


def get_custom_property(obj, name, default):
  # try getting density from material
  mat = obj.active_material
  if mat is None:
    print(f"No {name} information found. Using default {name}={default}.")
    value = 1.0
  elif name in mat:
    value = mat[name]
    print(f"Using {name}={value} from material {mat}.")
  elif name in obj:
    value = obj[name]
    print(f"Using {name}={value} from object.")
  else:
    print(f"No {name} information found. Using default {name}={default}.")
    value = 1.0
  return value


def create_trimesh_from_obj(obj):
  vertices, faces = get_vertices_and_faces(obj)

  tmesh = trimesh.Trimesh(vertices=vertices, faces=faces)

  if tmesh.is_empty:
    raise ValueError("Mesh is empty!")
  if not tmesh.is_watertight:
    raise ValueError("Mesh is not watertight (has holes)!")
  if not tmesh.is_winding_consistent:
    raise ValueError("Mesh is not winding consistent!")
  if tmesh.body_count() > 1:
    raise ValueError("Mesh consists of more than one connected component (bodies)!")

  return tmesh


def get_object_properties(obj, density=None, friction=None, tmesh=None):
  if tmesh is None:
    tmesh = create_trimesh_from_obj(obj)

  if density is None:
    tmesh.density = get_custom_property(obj, "Density", 1.0)

  if friction is None:
    friction = get_custom_property(obj, "Friction", 1.0)

  rounda = lambda x: np.round(x, decimals=4).tolist()
  roundf = lambda x: float(np.round(x, decimals=4))

  properties = {
      "id": obj.name,
      "material": obj.active_material.name if obj.active_material else None,
      "density": roundf(tmesh.density),
      "friction": roundf(friction),
      "nr_vertices": len(tmesh.vertices),
      "nr_faces": len(tmesh.faces),
      "bounds": rounda(tmesh.bounds),
      "area": roundf(tmesh.area),
      "volume": roundf(tmesh.volume),
      "mass": roundf(tmesh.mass),
      "center_mass": rounda(tmesh.center_mass),
      "inertia": rounda(tmesh.moment_inertia),
      "is_convex": tmesh.is_convex,
      "euler_number": tmesh.euler_number,
  }
  return properties


def center_top(obj):
  # TODO: currently destroys the smooth-all info and probably even more.
  #       so we need to find a less destructive way to shift points around
  vertices, faces = get_vertices_and_faces(obj)
  # put center of mesh at bottom middle (center along x and y, above z=0)
  xmin, xmax = vertices[:, 0].min(), vertices[:, 0].max()
  ymin, ymax = vertices[:, 1].min(), vertices[:, 1].max()
  zmin = vertices[:, 2].min()
  vertices -= np.array([(xmin + xmax)/2, (ymin + ymax)/2, zmin])
  obj.data.clear_geometry()
  obj.data.from_pydata(vertices.tolist(), [], faces.tolist())


def center_mesh_around(obj, new_center):
  for vert in obj.data.vertices:
    vert.co[0] -= new_center[0]
    vert.co[1] -= new_center[1]
    vert.co[2] -= new_center[2]


@contextlib.contextmanager
def select(obj_list):
  if not isinstance(obj_list, (list, tuple)):
    obj_list = [obj_list]
  previous_selection = copy.copy(bpy.context.selected_objects)
  previous_active = bpy.context.active_object

  for obj in bpy.context.selected_objects:
    obj.select_set(False)  # deselect everything
  for obj in obj_list:
    obj.select_set(True)  # select target objects
  if obj_list:  # set the active object to the first obj in obj_list
    bpy.context.view_layer.objects.active = obj_list[0]

  yield

  for obj in bpy.context.selected_objects:
    obj.select_set(False)  # deselect everything
  for obj in previous_selection:
    obj.select_set(True)  # re-select previous selected objects
  if obj_list:  # re-activate previous object
    bpy.context.view_layer.objects.active = previous_active


@contextlib.contextmanager
def center(obj_list):
  if not isinstance(obj_list, (list, tuple)):
    obj_list = [obj_list]

  prev_pos = {obj: copy.copy(obj.location) for obj in obj_list}
  for obj in obj_list:
    obj.location = (0, 0, 0)

  yield

  for obj in obj_list:
    obj.location = prev_pos[obj]


def apply_transformations(objs, position=False, rotation=True, scale=True):
  with select(objs):
    bpy.ops.object.transform_apply(location=position, rotation=rotation, scale=scale)


def create_blender_object_from_tmesh(tmesh, name):
  # create a new blender mesh
  bmesh_new = bpy.data.meshes.new(name)
  bmesh_new.clear_geometry()
  bmesh_new.from_pydata(tmesh.vertices.tolist(), [], tmesh.faces.tolist())
  bobj = bpy.data.objects.new(name, bmesh_new)
  bpy.context.scene.collection.objects.link(bobj)
  return bobj


def kubricify(output_folder, obj=None, density=None, friction=None):
  if obj is None:
    obj = get_active_object()
  with select(obj):
    print(f"Kubricifying {obj.name}...")
    print("Applying scale and rotation transformations...")
    apply_transformations(obj)
    print("Converting and validating...")
    # first convert to trimesh just for computing center of mass
    tmesh = create_trimesh_from_obj(obj)
    # center the mesh around its center of mass (i.e. ensure center-of-mass = [0,0,0])
    center_mesh_around(obj, tmesh.center_mass)
    # re-convert to trimesh for computing the properties
    tmesh = create_trimesh_from_obj(obj)
    print("Computing properties...")
    properties = get_object_properties(obj, density=density, friction=friction, tmesh=tmesh)

    print(properties)
    print(json.dumps(properties, indent=4, sort_keys=True))

    # Export
    output_path = pathlib.Path(output_folder) / obj.name
    print(f"Exporting to {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)  # ensure path exists
    urdf_path = save_urdf(output_path, properties)
    vis_path = save_visual_geometry(obj, output_path)

    if tmesh.is_convex():
      coll_path = save_collision_geometry(obj, output_path)
    else:
      print("Creating collision geometry...")
      cobj = create_blender_object_from_tmesh(tmesh.convex_hull, name=obj.name + "_CVX")
      cobj.location = obj.location
      cobj.location[2] -= tmesh.extents[2] * 1.5
      coll_path = save_collision_geometry(cobj, output_path)

    properties["paths"] = {
        "visual_geometry": [str(vis_path.relative_to(output_path))],
        "collision_geometry": [str(coll_path.relative_to(output_path))],
        "urdf": [str(urdf_path.relative_to(output_path))]
    }
    save_properties(output_path, properties)
    compress_object_dir(output_path, obj.name)

  print("tidying up...")
  shutil.rmtree(output_path)
  return properties


def compress_object_dir(output_path, obj_name):
  tar_path = str(output_path) + ".tar.gz"
  print("Zipping into", tar_path)
  with tarfile.open(tar_path, "w:gz") as tar:
    tar.add(output_path, arcname=obj_name)


def save_collision_geometry(obj, output_path):
  collision_path = output_path / "collision_geometry.obj"
  print(collision_path)
  with select(obj), center(obj):
    bpy.ops.export_scene.obj(filepath=str(collision_path),
                             axis_forward="Y", axis_up="Z", use_selection=True,
                             use_materials=False)
  return collision_path


def save_visual_geometry(obj, output_path):
  # TODO: instead export as blend file to keep more material info
  # see e.g.: https://github.com/sybrenstuvel/splode/blob/master/splode/internal.py#L106

  vis_path = output_path / "visual_geometry.obj"
  print(vis_path)
  with select(obj), center(obj):
    bpy.ops.export_scene.obj(filepath=str(vis_path),
                             axis_forward="Y", axis_up="Z", use_selection=True,
                             use_materials=True)
  return vis_path


def save_urdf(output_path, properties):
  urdf_path = output_path / "object.urdf"
  print(urdf_path)
  with open(urdf_path, "w", encoding="utf-8") as f:
    f.write(URDF_TEMPLATE.format(**properties))
  return urdf_path


def save_properties(output_path, properties):
  json_path = output_path / "data.json"
  print(json_path)
  with open(json_path, "w", encoding="utf-8") as f:
    json.dump(properties, f, indent=4, sort_keys=True)
  return json_path


def export_collection(collection_name, output_folder):
  details_list = []
  output_folder = pathlib.Path(output_folder)
  for obj in bpy.data.collections[collection_name].all_objects:
    details_list.append(kubricify(output_folder, obj))

  with open(output_folder / "manifest.json", "w", encoding="utf-8") as f:
    json.dump(details_list, f, indent=4, sort_keys=True)
