import os
import trimesh
import glob
import numpy as np

import os
import subprocess
import time
import pprint
import pybullet as pb
from subprocess import PIPE, Popen

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

    rounda = lambda x: np.round(x, decimals=6).tolist()
    roundf = lambda x: float(np.round(x, decimals=6))

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
      "euler_number": tmesh.euler_number,  # used for topological analysis (see: http://max-limper.de/publications/Euler/index.html),
    }
    return properties

def get_tmesh(obj_path, which_mesh=0):
  # 1. use the whole scene as a single mesh
  # 1.1 use meshlab to view the scene
  # obj_path = "/mnt/home/projects/kubric/.tmp/shapenet_1.obj"
  dst = obj_path.replace('.obj', '_ManifoldPlus.obj')
  if not os.path.exists(dst):
    command = './ManifoldPlus/build/manifold --input %s --output %s --depth 8' % (obj_path, dst)
    subprocess_call(command)
    
  scene_or_mesh = trimesh.load_mesh(dst, process=False)
  
  if isinstance(scene_or_mesh, trimesh.Scene):
    # convert scene into a list of meshes
    mesh_list = [trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                        for g in scene_or_mesh.geometry.values()]
    tmesh = merge_meshes(mesh_list)
  else:
    tmesh = scene_or_mesh

  verify_tmesh(tmesh)
  tmesh.apply_translation(-tmesh.center_mass)
  # export to obj again
  import trimesh.exchange.obj as tri_obj
  # obj_content = tri_obj.export_obj(tmesh)
  # # obj_content = re.sub('mtllib material0.mtl\nusemtl material0\n', 'mtllib visual_geometry.mtl\nusemtl material_0\n', obj_content)
  
  # with open('visual_geometry.obj', 'w') as f:
  #     f.write(obj_content)

  # pb.vhacd('visual_geometry.obj', 'collision_geometry.obj', 'logs.txt')

  # move material and texture
  mat_path = obj_path.replace('.obj', '.mtl')
  import shutil
  shutil.move(mat_path, 'visual_geometry.mtl')

  # check if there is a texture file.
  shutil.move(texture_path, tex_path)
  urdf_path = 'object.urdf'
  # todo get unique name
  name = 'custom'
  properties = get_object_properties(tmesh, name=name)

  with open(urdf_path, 'w') as f:
        f.write(URDF_TEMPLATE.format(**properties))

  properties["paths"] = {
          "visual_geometry": [str(vis_path)],
          "collision_geometry": [str(coll_path)],
          "urdf": [str(urdf_path)],
          "texture": [str(tex_path)],
      }

  with open(json_path, "w") as f:
      json.dump(properties, f, indent=4, sort_keys=True)

  tar_path = name + ".tar.gz"
  print("          saving as", tar_path)
  with tarfile.open(tar_path, "w:gz") as tar:
      tar.add(target_asset_dir, arcname=name)

  return tmesh 

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
    raise ValueError("Mesh consists of more than one connected component (bodies)!")

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
  faces = np.vstack([face + offset for face, offset in zip(faces_list, faces_offset)])

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
        obj_path = os.path.join(datadir, obj_hash, 'models', 'model_normalized.obj')
        tmesh = get_tmesh(obj_path)

        # save png for visualization
        # src_fname = '.tmp/%s.stl' % obj_hash
        # out_fname = '.tmp/%s.png' % obj_hash
        # tmesh.export(src_fname)
        # save_image(out_fname, src_fname)

        fix_tmesh(tmesh)
        try:
            verify_tmesh(tmesh)
            print(i, '- good')
        except ValueError as e:
            print(i, '-', e)
        