"""Processes shapenet into assets that Kubric can import."""

import os
import trimesh
import glob
import numpy as np
import trimesh.exchange.obj as tri_obj
import re
import json
import shutil
import argparse
import pybullet as pb
import tarfile

from subprocess import PIPE, Popen
from pathlib import Path

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

def save_asset(asset_path, target_dir='output'):
    """
    Saves a kubric-compatible asset corresponding to a shapenet object
    """
    obj_path = os.path.join(asset_path, 'models', 'model_normalized.obj')

    # Ignore asset if there is no material
    assert os.path.exists(asset_path)
    # TODO: what to do when materials are missing?
    if not os.path.exists(obj_path.replace('.obj', '.mtl')):
        return
    name = os.path.split(asset_path)[-1]

    # Create target dir
    target_asset_dir = os.path.join(target_dir, name)
    if os.path.exists(target_asset_dir):
        shutil.rmtree(target_asset_dir)
    os.makedirs(target_asset_dir, exist_ok=True)

    # Use Manifold Plus to created waterfilled object
    tmesh_waterfilled, tmesh_center = get_tmesh(
        get_waterfilled_obj(obj_path), return_center_mass=True)
    waterfilled_path = os.path.join(
        target_asset_dir, 'visual_geometry_manifold_plus.obj')
    save_tmesh(waterfilled_path, tmesh_waterfilled)

    # Create collision mesh using pybullets VHACD
    coll_path = os.path.join(target_asset_dir, 'collision_geometry.obj')
    pb.vhacd(str(waterfilled_path),
             str(coll_path),
             str(os.path.join(target_asset_dir, 'pybullet_logs.txt')))

    # Translate original object using waterfilled object center
    tmesh, _ = get_tmesh(obj_path, return_center_mass=True)

    with open(obj_path, 'r') as f:
        lines = f.readlines()
    
    ## Save translated original object
    vis_path = os.path.join(target_asset_dir, 'visual_geometry.obj')
    with open(vis_path, 'a') as f:
        for l in lines:
            if 'v ' == l[:2] and len(l.split(' ')) == 4:
                r_list = l.split(' ')[-3:]
                r_list_new = []
                for i in range(3):
                    shifted = float(r_list[i]) - tmesh_center[i]
                    r_list_new += [str(shifted)]
                    
                f.write('v ' + ' '.join(r_list_new) + '\n')
            else:
                f.write(l)

    ## Save gltf object
    _, stderr = subprocess_call(f'obj2gltf -i {vis_path} -o {vis_path.replace(".obj", "")}')
    if stderr != '':
        print(f'warning .gltf was not saved, {stderr}')

    ## material
    # TODO: deal with missing materials
    mat_path = obj_path.replace('.obj', '.mtl')
    mtl_path = os.path.join(target_asset_dir, 'visual_geometry.mtl')
    if os.path.exists(mat_path):
        with open(mat_path, 'r') as f:
            mat_str = f.read()
        with open(mtl_path, 'w') as f:
            f.write(mat_str.replace('../images/', ''))

    ## textures
    tex_path = os.path.join(target_asset_dir, 'texture.png')
    tex_path_list = []
    for texture_path in glob.glob(os.path.join(asset_path, 'images', '*')):
        tex_path = os.path.join(
            target_asset_dir,  os.path.split(texture_path)[-1])
        shutil.copy(texture_path, tex_path)
        tex_path_list += [tex_path]

    ## properties as urdf
    properties = get_object_properties(tmesh, name)

    urdf_path = os.path.join(target_asset_dir, 'object.urdf')
    with open(urdf_path, 'w') as f:
        f.write(URDF_TEMPLATE.format(**properties))

    ## properties as json
    properties["paths"] = {
        "visual_geometry": [str(vis_path).replace(target_asset_dir, '')],
        "collision_geometry": [str(coll_path).replace(target_asset_dir, '')],
        "urdf": [str(urdf_path).replace(target_asset_dir, '')],
        "texture": [t.replace(target_asset_dir, '') for t in tex_path_list],
    }

    json_path = os.path.join(target_asset_dir, 'data.json')
    with open(json_path, "w") as f:
        json.dump(properties, f, indent=4, sort_keys=True)

    # Save asset at zip
    tar_path = os.path.join(target_dir, name + ".tar.gz")
    print("          saving as", tar_path)
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(target_asset_dir, arcname=name)

    shutil.rmtree(target_asset_dir)
    return properties


def verify_tmesh(tmesh):
    # Sanity checks
    if tmesh.is_empty:
        raise ValueError("Mesh is empty!")
    if not tmesh.is_watertight:
        # 2. find a way to make it watertight
        trimesh.repair.fix_winding(tmesh)
        raise ValueError("Mesh is not watertight (has holes)!")
    if not tmesh.is_winding_consistent:
        raise ValueError("Mesh is not winding consistent!")
    if tmesh.body_count > 1:
        raise ValueError(
            "Mesh consists of more than one connected component (bodies)!")

    return True


def merge_meshes(yourList):
    vertice_list = [mesh.vertices for mesh in yourList]
    faces_list = [mesh.faces for mesh in yourList]
    faces_offset = np.cumsum([v.shape[0] for v in vertice_list])
    faces_offset = np.insert(faces_offset, 0, 0)[:-1]

    vertices = np.vstack(vertice_list)
    faces = np.vstack(
        [face + offset for face, offset in zip(faces_list, faces_offset)])

    merged__meshes = trimesh.Trimesh(vertices, faces)
    return merged__meshes

def subprocess_call(command):
    p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p.communicate()

    # get outputs
    stdout = stdout.decode('utf-8')
    stderr = stderr.decode('utf-8')
    
    return stdout, stderr


def get_object_properties(tmesh, name, density=None, friction=None):
    if density is None:
        tmesh.density = 1000.0
    friction = 0.0

    def rounda(x): return np.round(x, decimals=6).tolist()
    def roundf(x): return float(np.round(x, decimals=6))

    properties = {
        "id": name,
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
        # used for topological analysis (see: http://max-limper.de/publications/Euler/index.html),
        "euler_number": tmesh.euler_number,
    }
    return properties


def get_tmesh(src_fname, return_center_mass=False):
    # Load the mesh
    # -------------
    scene_or_mesh = trimesh.load_mesh(src_fname, process=False)
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh_list = [trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                     for g in scene_or_mesh.geometry.values()]
        tmesh = merge_meshes(mesh_list)
    else:
        tmesh = scene_or_mesh

    # center the tmesh
    center_mass = tmesh.center_mass
    tmesh.apply_translation(-center_mass)

    if return_center_mass:
        return tmesh, center_mass

    return tmesh


def save_tmesh(fname, tmesh):
    # rename content
    obj_content = tri_obj.export_obj(tmesh)
    obj_content = re.sub('mtllib material0.mtl\nusemtl material0\n',
                         'mtllib visual_geometry.mtl\nusemtl material_0\n', obj_content)
    with open(fname, 'w') as f:
        f.write(obj_content)


def get_waterfilled_obj(obj_path):
    dst = obj_path.replace('.obj', '_manifold_plus.obj')
    # TODO: when does this exist? Is it just not to repeat work?
    if not os.path.exists(dst):
        command = f'manifold --input {obj_path} --output {dst} --depth 8'
        _, stderr = subprocess_call(command)
        if stderr != '':
            print(f'warning manifold error , {stderr}')
        assert Path(dst).is_file()
    return dst


if __name__ == "__main__":
    # Parse the dataset directory
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', required=True, help='The shapenet directory.')
    parser.add_argument('--outdir', required=True, help='The output directory.')
    args = parser.parse_args()
    
    # Get the categories of shapenet
    cat_dirs = os.listdir(args.datadir)

    # Loop over each category
    for cat_dir in cat_dirs:
        obj_id_list = os.listdir(os.path.join(args.datadir, cat_dir))
        # Loop over each object and save asset
        for i, obj_id in enumerate(obj_id_list):
            obj_path = os.path.join(args.datadir, cat_dir, obj_id)
            save_asset(obj_path, target_dir=os.path.join(args.outdir, cat_dir))
            # import pdb; pdb.set_trace()