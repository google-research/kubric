
import logging
import pathlib
import functools
import sys
from typing import Any, Callable, Union


import bidict

import munch
import trimesh
import trimesh.creation

from kubric import core

__thismodule__ = sys.modules[__name__]
logger = logging.getLogger(__name__)


class Trimesh:
  def __init__(self, scene):
    self.trimesh_scene = trimesh.Scene()
    self.scene = scene

  def show(self):
    return self.trimesh_scene.show()


def prepare_trimesh_object(func: Callable[[core.Asset], Any]) -> Callable[[core.Asset], Any]:

  @functools.wraps(func)
  def _func(obj: core.Asset):
    # if obj has already been converted, then return the corresponding linked object
    if __thismodule__ in obj.linked_objects:
      return obj.linked_objects[__thismodule__]

    # else use func to create a new blender object
    trimesh_obj = func(obj)

    # store the blender_obj in the list of linked objects
    obj.linked_objects[__thismodule__] = trimesh_obj

    # trigger change notification for all fields (for initialization)
    for trait_name in obj.trait_names():
      value = getattr(obj, trait_name)
      obj.notify_change(munch.Munch(owner=obj, type="change", name=trait_name,
                                    new=value, old=value))

    return trimesh_obj

  return _func


@functools.singledispatch
def add_object(obj: core.Asset):
  raise NotImplementedError()

def register_object3d_setters(obj, blender_obj):
  assert isinstance(obj, core.Object3D), f"{type(obj)} is not an Object3D"

  obj.observe(AttributeSetter(blender_obj, 'location'), 'position')
  obj.observe(KeyframeSetter(blender_obj, 'location'), 'position', type="keyframe")

  obj.observe(AttributeSetter(blender_obj, 'rotation_quaternion'), 'quaternion')
  obj.observe(KeyframeSetter(blender_obj, 'rotation_quaternion'), 'quaternion', type="keyframe")


@add_object.register(core.Cube)
@prepare_trimesh_object
def _add_object(obj: core.Cube):
  trimesh.creation.box(extents=obj.scale)


  bpy.ops.mesh.primitive_cube_add()
  cube = bpy.context.active_object

  register_object3d_setters(obj, cube)
  obj.observe(AttributeSetter(cube, 'active_material'), 'material')

  obj.observe(AttributeSetter(cube, 'scale'), 'scale')
  obj.observe(KeyframeSetter(cube, 'scale'), 'scale', type="keyframe")

  return cube


def translate()


class AttributeSetter:
  def __init__(self, trimesh_obj, ):
  pass