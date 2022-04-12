#!/usr/bin/env python3
# Copyright 2022 The Kubric Authors.
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

# Copyright 2022 The Kubric Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import sys
import trimesh
import logging
import numpy as np
from pathlib import Path

_DEFAULT_LOGGER = logging.getLogger(__name__)


class ObjectPropertiesException(Exception):
  def __init__(self, message):
    super().__init__(message)

def get_object_properties(obj_path:Path, logger=_DEFAULT_LOGGER, density=1.0):
  # --- override the trimesh logger
  trimesh.util.log = logger

  tmesh = get_tmesh(str(obj_path))
  properties = {
    "bounds": tmesh.bounds.tolist(),
    "center_mass": tmesh.center_mass.tolist(),
    "inertia": tmesh.moment_inertia.tolist(),
    "mass": tmesh.volume * density,
  }
  return properties


def get_tmesh(obj_fd):
  scene_or_mesh = trimesh.load_mesh(obj_fd, process=False)
  if isinstance(scene_or_mesh, trimesh.Scene):
    mesh_list = [trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                 for g in scene_or_mesh.geometry.values()]
    tmesh = _merge_meshes(mesh_list)
  else:
    tmesh = scene_or_mesh

  # TODO: see https://github.com/google-research/kubric/issues/134
  # TL;DR: this solution was a bit of a hack, but you'd be able to change pivot in blender?
  # center_mass = tmesh.center_mass
  # tmesh.apply_translation(-center_mass)
  return tmesh


def _merge_meshes(your_list):
  vertice_list = [mesh.vertices for mesh in your_list]
  faces_list = [mesh.faces for mesh in your_list]
  faces_offset = np.cumsum([v.shape[0] for v in vertice_list])
  faces_offset = np.insert(faces_offset, 0, 0)[:-1]

  vertices = np.vstack(vertice_list)
  faces = np.vstack(
      [face + offset for face, offset in zip(faces_list, faces_offset)])

  merged__meshes = trimesh.Trimesh(vertices, faces)
  return merged__meshes


if __name__ == '__main__':
  model = 'ShapeNetCore.v2/02958343/b3ffbbb2e8a5376d4ed9aac513001a31/models/model_normalized.obj'

  # --- setup logger (→stdout)
  logger = logging.getLogger(__name__)
  logger.setLevel(logging.DEBUG)
  handler = logging.StreamHandler(sys.stdout)
  logger.addHandler(handler)

  print(f"properties computed from {model}")
  properties = get_object_properties(model, logger=logger)
  print(json.dumps(properties, indent=2))
