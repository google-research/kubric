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
