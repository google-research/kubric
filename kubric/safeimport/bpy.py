"""Module to safely import blender's bpy and guide the user to a solution.

USAGE:
  from kubric.safeimport.bpy import bpy
"""
import sys

_ERROR_MESSAGE_ = """
Note: the `bpy` module used by kubric cannot be installed via pip. Most likely, you are
executing this script within a raw python environment rather than in our docker container.
Please refer to our documentation: https://readthedocs.org/projects/kubric
"""

try:
  import bpy  # pylint: disable=unused-import
except ImportError as err:
  print(err)
  print(_ERROR_MESSAGE_)
  sys.exit(1)
