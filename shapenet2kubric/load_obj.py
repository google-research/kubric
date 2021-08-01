"""Blender script to load an obj file.

Usage:
  blender --python load_obj.py -- --filepath=/path/to/file.obj
"""

import bpy
import argparse
import sys

# --- parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--filepath', type=str, help="e.g. /ShapeNetCore.v2/04090263/18807814a9cefedcd957eaf7f4edb205/models/model_normalized.obj")
args = parser.parse_args(args=sys.argv[sys.argv.index("--") + 1:]) #< ignore anythign before "--"

# --- delete the default cube object
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.select_all(action='DESELECT')
bpy.data.objects['Cube'].select_set(True)
bpy.ops.object.delete()

# --- load my object
bpy.ops.import_scene.obj(filepath=args.filepath)