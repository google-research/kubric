"""Blender script to load obj files into blender.

Usage:
  blender --python load_obj.py -- --datadir=/ShapeNetCore.v2 --model=04090263/18807814a9cefedcd957eaf7f4edb205
"""

import bpy
import argparse
import sys
from pathlib import Path

# --- parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, help='e.g. /ShapeNetCore.v2')
parser.add_argument('--model', type=str, help='e.g. 04090263/18807814a9cefedcd957eaf7f4edb205')
args = parser.parse_args(args=sys.argv[sys.argv.index("--") + 1:]) #< ignore anythign before "--"

# --- delete the default cube object
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.select_all(action='DESELECT')
bpy.data.objects['Cube'].select_set(True)
bpy.ops.object.delete()

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# --- original model
model = Path(args.datadir) / Path(args.model) / 'models/model_normalized.obj'
if Path(model).is_file():
  bpy.ops.import_scene.obj(filepath=str(model), use_split_objects=False)

# --- watertight model
model = Path(args.datadir) / Path(args.model) / "kubric/model_watertight.obj"
if Path(model).is_file():
  bpy.ops.import_scene.obj(filepath=str(model), use_split_objects=False)

# --- collision model
model = Path(args.datadir) / Path(args.model) / "kubric/collision_geometry.obj"
if Path(model).is_file():
  bpy.ops.import_scene.obj(filepath=str(model), use_split_objects=False)

# --- rendering model (GLTF binary)
model = Path(args.datadir) / Path(args.model) / "kubric/visual_geometry.glb"
if Path(model).is_file():
  bpy.ops.import_scene.gltf(filepath=str(model))
  # otherwise you get a {objects}, vs. a single object
  bpy.ops.object.join() 
  # WARNING: the object is not part of (the default) collection?
  # bpy.context.collection.objects.link(bpy.context.active_object)