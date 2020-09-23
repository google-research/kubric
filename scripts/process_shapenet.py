import os
import trimesh
import glob

def get_tmesh(obj_path, which_mesh=0):
  scene_or_mesh = trimesh.load_mesh(obj_path, process=False)

  if isinstance(scene_or_mesh, trimesh.Scene):
    # convert scene into a list of meshes
    mesh_list = [trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                        for g in scene_or_mesh.geometry.values()]
    tmesh = mesh_list[which_mesh]
  else:
    tmesh = scene_or_mesh
  return tmesh 

def verify_tmesh(tmesh):
  # Sanity checks
  if tmesh.is_empty:
    raise ValueError("Mesh is empty!")
  if not tmesh.is_watertight:
    raise ValueError("Mesh is not watertight (has holes)!")
  if not tmesh.is_winding_consistent:
    raise ValueError("Mesh is not winding consistent!")
  if tmesh.body_count > 1:
    raise ValueError("Mesh consists of more than one connected component (bodies)!")


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
        src_fname = '.tmp/%s.stl' % obj_hash
        out_fname = '.tmp/%s.png' % obj_hash
        tmesh.export(src_fname)
        save_image(out_fname, src_fname)

        try:
            verify_tmesh(tmesh)
        except ValueError as e:
            print(i, '-', e)
        