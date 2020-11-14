import os
import trimesh
import glob
import numpy as np
import trimesh.exchange.obj as tri_obj
import os
import re
import json
import subprocess
import time
import shutil
import pprint
import pybullet as pb
from subprocess import PIPE, Popen
import trimesh.exchange.obj as tri_obj
import tarfile


def clean_stdout(stdout):
    return stdout.split(' ')[0].replace('\n', '')


def subprocess_call(command):
    p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p.communicate()

    # get outputs
    stdout = stdout.decode('utf-8')
    stderr = stderr.decode('utf-8')

    # if stdout is None or stdout == '':
    #     raise ValueError(stderr)
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


def get_tmesh(asset_path, which_mesh=0, target_dir='.tmp'):
    obj_path = os.path.join(asset_path, 'models', 'model_normalized.obj')
    name = os.path.split(asset_path)[-1]
    # Use ManifoldPlus to water fill
    # ------------------------------
    dst = obj_path.replace('.obj', '_ManifoldPlus.obj')
    if not os.path.exists(dst):
        command = './ManifoldPlus/build/manifold --input %s --output %s --depth 8' % (
            obj_path, dst)
        subprocess_call(command)

    # SOURCE PATHS
    # ------------
    texture_path_list = glob.glob(os.path.join(asset_path, 'images', '*'))
    mat_path = obj_path.replace('.obj', '.mtl')
    if not os.path.exists(mat_path):
      return
    
    obj_path = dst

    # TARGET PATHS
    # ------------
    target_asset_dir = os.path.join(target_dir, name)
    if os.path.exists(target_asset_dir):
        shutil.rmtree(target_asset_dir)
    os.makedirs(target_asset_dir, exist_ok=True)

    vis_path = os.path.join(target_asset_dir, 'visual_geometry.obj')
    coll_path = os.path.join(target_asset_dir, 'collision_geometry.obj')
    urdf_path = os.path.join(target_asset_dir, 'object.urdf')
    tex_path = os.path.join(target_asset_dir, 'texture.png')
    json_path = os.path.join(target_asset_dir, 'data.json')
    tar_path = os.path.join(target_dir, name + ".tar.gz")

    # Load the mesh
    # -------------
    scene_or_mesh = trimesh.load_mesh(obj_path, process=False)
    if isinstance(scene_or_mesh, trimesh.Scene):

        mesh_list = [trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                     for g in scene_or_mesh.geometry.values()]
        tmesh = merge_meshes(mesh_list)
    else:
        tmesh = scene_or_mesh

    # make sure tmesh is suitable
    verify_tmesh(tmesh)

    # center the tmesh
    tmesh.apply_translation(-tmesh.center_mass)
    obj_content = tri_obj.export_obj(tmesh)
    obj_content = re.sub('mtllib material0.mtl\nusemtl material0\n',
                         'mtllib visual_geometry.mtl\nusemtl material_0\n', obj_content)
    with open(vis_path, 'w') as f:
        f.write(obj_content)

    # compute a collision mesh using pybullets VHACD
    pb.vhacd(str(vis_path),
             str(coll_path),
             str(os.path.join(target_asset_dir, 'pybullet_logs.txt')))

    # move material and texture
    if os.path.exists(mat_path):
        shutil.copy(mat_path, os.path.join(
            target_asset_dir, 'visual_geometry.mtl'))
    tex_path_list = []
    for texture_path in texture_path_list:
        tex_path = os.path.join(target_asset_dir,  os.path.split(texture_path)[-1])
        shutil.copy(texture_path, tex_path)
        tex_path_list += [tex_path]

    properties = get_object_properties(tmesh, name)

    with open(urdf_path, 'w') as f:
        f.write(URDF_TEMPLATE.format(**properties))

    # save properties
    properties["paths"] = {
        "visual_geometry": [str(vis_path)],
        "collision_geometry": [str(coll_path)],
        "urdf": [str(urdf_path)],
        "texture": tex_path_list,
    }

    with open(json_path, "w") as f:
        json.dump(properties, f, indent=4, sort_keys=True)

    # save zip
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


def fix_tmesh(tmesh):
    print("Merging vertices closer than a pre-set constant...")
    tmesh.merge_vertices()
    print("Removing duplicate faces...")
    tmesh.remove_duplicate_faces()
    print("Scaling...")
    tmesh.apply_scale(scaling=1.0)
    print("Making the mesh watertight...")
    trimesh.repair.broken_faces(tmesh, color=None)
    flag = trimesh.repair.fill_holes(tmesh)
    assert flag
    print("Fixing inversion and winding...")
    trimesh.repair.fix_inversion(tmesh)
    trimesh.repair.fix_winding(tmesh)


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


def save_image(fname, src_fname):
    from stl import mesh
    from mpl_toolkits import mplot3d
    from matplotlib import pyplot

    # Create a new plot
    figure = pyplot.figure()
    axes = mplot3d.Axes3D(figure)

    # Load the STL files and add the vectors to the plot
    your_mesh = mesh.Mesh.from_file(src_fname)
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

    # Auto scale to the mesh size
    scale = your_mesh.points.flatten('F')
    axes.auto_scale_xyz(scale, scale, scale)
    pyplot.savefig(fname)
    print('Saved')


if __name__ == "__main__":

    datadir = '/mnt/public/datasets/ShapeNetCore.v2/02691156'

    obj_hash_list = os.listdir(datadir)
    for i, obj_hash in enumerate(obj_hash_list):
        obj_path = os.path.join(datadir, obj_hash)
        tmesh = get_tmesh(obj_path)

        # save png for visualization
        # src_fname = '.tmp/%s.stl' % obj_hash
        # out_fname = '.tmp/%s.png' % obj_hash
        # tmesh.export(src_fname)
        # save_image(out_fname, src_fname)

        # fix_tmesh(tmesh)
        # try:
        #     verify_tmesh(tmesh)
        #     print(i, '- good')
        # except ValueError as e:
        #     print(i, '-', e)
