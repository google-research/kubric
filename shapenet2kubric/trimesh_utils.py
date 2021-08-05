import trimesh
import numpy as np
from pathlib import Path
import json

def get_object_properties(obj_path:Path):
  def rounda(x): return np.round(x, decimals=6).tolist()
  def roundf(x): return float(np.round(x, decimals=6))
  tmesh = get_tmesh(str(obj_path))

  properties = {
    "bounds": rounda(tmesh.bounds),
    "mass": roundf(tmesh.mass),
    "center_mass": rounda(tmesh.center_mass),
    "inertia": rounda(tmesh.moment_inertia),
  }
  return properties

def get_tmesh(obj_fd):
  scene_or_mesh = trimesh.load_mesh(obj_fd, process=False)
  if isinstance(scene_or_mesh, trimesh.Scene):
    mesh_list = [trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                 for g in scene_or_mesh.geometry.values()]
    tmesh = merge_meshes(mesh_list)
  else:
    tmesh = scene_or_mesh

  # TODO: see https://github.com/google-research/kubric/issues/134
  # TL;DR: this solution was a bit of a hack, but you'd be able to change pivot in blender?
  # center_mass = tmesh.center_mass
  # tmesh.apply_translation(-center_mass)
  return tmesh

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

if __name__ == '__main__':
  # model = '/Users/atagliasacchi/datasets/ShapeNetCore.v2/04090263/18807814a9cefedcd957eaf7f4edb205/models/model_normalized.obj'
  # model = '/Users/atagliasacchi/datasets/ShapeNetCore.v2/04090263/18807814a9cefedcd957eaf7f4edb205/kubric/model_watertight.obj'
  model = '/Users/atagliasacchi/datasets/ShapeNetCore.v2/04090263/18807814a9cefedcd957eaf7f4edb205/kubric/collision_geometry.obj'
  print(f"properties computed from {model}")
  properties = get_object_properties(model)
  print(json.dumps(properties, indent=2))