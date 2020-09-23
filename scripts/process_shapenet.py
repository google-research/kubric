import os
import trimesh
import glob


def create_trimesh_from_shapenet_obj(obj_path, which_mesh=0):
  scene_or_mesh = trimesh.load_mesh(obj_path, process=False)

  if isinstance(scene_or_mesh, trimesh.Scene):
    # convert scene into a list of meshes
    mesh_list = [trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                        for g in scene_or_mesh.geometry.values()]
    tmesh = mesh_list[which_mesh]
  else:
    tmesh = scene_or_mesh
  
  # Sanity checks
  if tmesh.is_empty:
    raise ValueError("Mesh is empty!")
  if not tmesh.is_watertight:
    raise ValueError("Mesh is not watertight (has holes)!")
  if not tmesh.is_winding_consistent:
    raise ValueError("Mesh is not winding consistent!")
  if tmesh.body_count > 1:
    raise ValueError("Mesh consists of more than one connected component (bodies)!")

  return tmesh


if __name__ == "__main__":
    datadir = '/mnt/public/datasets/ShapeNetCore.v2/02691156'

    obj_hash_list = os.listdir(datadir)
    for i, obj_hash in enumerate(obj_hash_list):
        obj_path = os.path.join(datadir, obj_hash, 'models', 'model_normalized.obj')
        try:
            mesh = create_trimesh_from_shapenet_obj(obj_path, which_mesh=0)
            print('good!')
        except ValueError as e:
            print(i, '-', e)
