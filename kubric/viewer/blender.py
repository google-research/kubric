# Copyright 2020 The Kubric Authors
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

import logging
import pathlib
from functools import singledispatch
from typing import Tuple, Dict

import bpy
from bidict import bidict
from munch import Munch

from kubric.core import Asset, AttributeSetter
from kubric import core

logger = logging.getLogger(__name__)


class Blender:
  def __init__(self, scene: core.Scene):
    self.objects_to_blend = bidict()

    self.illum_node = None
    self.illum_hdri_node = None
    self.illum_mapping_node = None
    self.bg_node = None
    self.bg_hdri_node = None
    self.bg_mapping_node = None

    self.clear_and_reset()  # as blender has a default scene on load
    # the ray-tracing engine is set here because it affects the availability of some features
    bpy.context.scene.render.engine = 'CYCLES'
    self.add(scene)
    self.set_up_scene_shading()

  def add(self, obj: Asset):
    if obj in self.objects_to_blend:
      return self.objects_to_blend[obj]
    blender_obj, setters = add_object(obj)

    # set the name of the object to the UID
    blender_obj.name = obj.uid
    # if it has a rotation mode, then make sure it is set to quaternions
    if hasattr(blender_obj, 'rotation_mode'):
      blender_obj.rotation_mode = 'QUATERNION'

    # remember object association
    self.objects_to_blend[obj] = blender_obj

    # if object is an actual Object (eg. not a Scene, or a Material)
    # then ensure that it is linked into (used by) the current scene collection
    if isinstance(blender_obj, bpy.types.Object):
      collection = bpy.context.scene.collection.objects
      if blender_obj not in collection.values():
        collection.link(blender_obj)

    for name, setter in setters.items():
      setter.mapping = self.objects_to_blend

      # recursively add sub-assets
      value = getattr(obj, name)
      print(obj, name, value)
      if isinstance(value, Asset):
        print("HAHA: ", value, type(value), obj, name)
        value = self.add(value)
      # Initialize values
      setter(Munch(owner=obj, new=value, type='init'))
      # Link values
      obj.observe(setter, names=[name])

    obj.destruction_callbacks.append(Destructor([blender_obj]))
    obj.keyframe_callbacks.append(Keyframer(setters))

    return blender_obj

  def get_blender_object(self, obj: core.Object3D) -> bpy.types.Object:
    if isinstance(obj, bpy.types.Object):
      return obj
    elif isinstance(obj, core.Object3D):
      return self.objects_to_blend[obj]
    else:
      raise ValueError('Not a valid object {}'.format(obj))

  def clear_and_reset(self):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.world = bpy.data.worlds.new('World')

  def set_up_exr_output(self, path):
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # clear existing nodes
    for node in tree.nodes:
      tree.nodes.remove(node)

    # the render node has outputs for all the rendered layers
    render_node = tree.nodes.new(type="CompositorNodeRLayers")

    # create a new FileOutput node
    out_node = tree.nodes.new(type='CompositorNodeOutputFile')
    # set the format to EXR (multilayer)
    out_node.format.file_format = 'OPEN_EXR_MULTILAYER'
    out_node.base_path = path  # output directory

    layers = ['Image', 'Depth', 'Vector', 'UV', 'Normal', 'CryptoObject00']

    out_node.file_slots.clear()
    for l in layers:
      out_node.file_slots.new(l)
      links.new(render_node.outputs.get(l), out_node.inputs.get(l))

  def set_up_scene_shading(self):
    bpy.context.scene.world.use_nodes = True
    tree = bpy.context.scene.world.node_tree
    links = tree.links

    # clear the tree
    for node in tree.nodes.values():
      tree.nodes.remove(node)

    # create nodes
    out_node = tree.nodes.new(type='ShaderNodeOutputWorld')
    out_node.location = 1100, 0

    mix_node = tree.nodes.new(type='ShaderNodeMixShader')
    mix_node.location = 900, 0
    lightpath_node = tree.nodes.new(type='ShaderNodeLightPath')
    lightpath_node.location = 700, 350
    self.illum_node = tree.nodes.new(type='ShaderNodeBackground')
    self.illum_node.inputs['Color'].default_value = (0., 0., 0., 1.)
    self.illum_node.location = 700, 0
    self.bg_node = tree.nodes.new(type='ShaderNodeBackground')
    self.bg_node.inputs['Color'].default_value = (0., 0., 0., 1.)
    self.bg_node.location = 700, -120

    links.new(lightpath_node.outputs.get('Is Camera Ray'), mix_node.inputs.get('Fac'))
    links.new(self.illum_node.outputs.get('Background'), mix_node.inputs[1])
    links.new(self.bg_node.outputs.get('Background'), mix_node.inputs[2])
    links.new(mix_node.outputs.get('Shader'), out_node.inputs.get('Surface'))

    # create nodes for HDRI images, but leave them disconnected until set_ambient_illumination or set_background
    coord_node = tree.nodes.new(type='ShaderNodeTexCoord')

    self.bg_mapping_node = tree.nodes.new(type='ShaderNodeMapping')
    self.bg_mapping_node.location = 200, 200
    self.bg_hdri_node = tree.nodes.new(type='ShaderNodeTexEnvironment')
    self.bg_hdri_node.location = 400, 200
    links.new(coord_node.outputs.get('Generated'), self.bg_mapping_node.inputs.get('Vector'))
    links.new(self.bg_mapping_node.outputs.get('Vector'), self.bg_hdri_node.inputs.get('Vector'))
    #links.new(bg_hdri_node.outputs.get('Color'), self.bg_node.inputs.get('Color'))

    self.illum_mapping_node = tree.nodes.new(type='ShaderNodeMapping')
    self.illum_mapping_node.location = 200, -200
    self.illum_hdri_node = tree.nodes.new(type='ShaderNodeTexEnvironment')
    self.illum_hdri_node.location = 400, -200
    links.new(coord_node.outputs.get('Generated'), self.illum_mapping_node.inputs.get('Vector'))
    links.new(self.illum_mapping_node.outputs.get('Vector'), self.illum_hdri_node.inputs.get('Vector'))
    # links.new(illum_hdri_node.outputs.get('Color'), self.illum_node.inputs.get('Color'))

  def set_ambient_illumination(self, hdri_filepath=None, color=(0., 0., 0., 1.0), hdri_rotation=(0., 0., 0.)):
    tree = bpy.context.scene.world.node_tree
    links = tree.links
    if hdri_filepath is None:
      # disconnect incoming links from hdri node (if any)
      for link in self.illum_node.inputs['Color'].links:
        links.remove(link)
      self.illum_node.inputs['Color'].default_value = color
    else:
      # ensure hdri_node is connected
      links.new(self.illum_hdri_node.outputs.get('Color'), self.illum_node.inputs.get('Color'))
      self.illum_hdri_node.image = bpy.data.images.load(hdri_filepath, check_existing=True)
      self.illum_mapping_node.inputs.get('Rotation').default_value = hdri_rotation

  def set_background(self, hdri_filepath=None, color=(0., 0., 0., 1.0), hdri_rotation=(0., 0., 0.)):
    tree = bpy.context.scene.world.node_tree
    links = tree.links
    if hdri_filepath is None:
      # disconnect incoming links from hdri node (if any)
      for link in self.bg_node.inputs['Color'].links:
        links.remove(link)
      self.bg_node.inputs['Color'].default_value = color
    else:
      # ensure hdri_node is connected
      links.new(self.bg_hdri_node.outputs.get('Color'), self.bg_node.inputs.get('Color'))
      self.bg_hdri_node.image = bpy.data.images.load(hdri_filepath, check_existing=True)
      self.bg_mapping_node.inputs.get('Rotation').default_value = hdri_rotation

  def activate_render_passes(self):
    view_layer = bpy.context.scene.view_layers[0]
    view_layer.use_pass_vector = True  # flow
    view_layer.use_pass_uv = True  # UV
    view_layer.use_pass_normal = True  # surface normals
    view_layer.cycles.use_pass_crypto_object = True  # segmentation
    view_layer.cycles.pass_crypto_depth = 2

  def set_size(self, width: int, height: int):
    bpy.context.scene.render.resolution_x = width
    bpy.context.scene.render.resolution_y = height

  def render(self, path):
    bpy.context.scene.cycles.use_adaptive_sampling = True  # speeds up rendering
    bpy.context.scene.view_layers[0].cycles.use_denoising = True  # improves the output quality
    bpy.context.scene.render.filepath = path

    self.activate_render_passes()
    self.set_up_exr_output(path)

    bpy.ops.wm.save_mainfile(filepath=str(pathlib.Path(path) / 'out.blend'))

    bpy.ops.render.render(animation=True, write_still=True)


# ########## Functions to import kubric objects into blender ###########


@singledispatch
def add_object(obj: Asset) -> Tuple[bpy.types.Object, Dict[str, AttributeSetter]]:
  raise NotImplementedError()


@add_object.register(core.FileBasedObject)
def _add_object(obj: core.FileBasedObject):
  # TODO: support other file-formats
  bpy.ops.import_scene.obj(filepath=str(obj.render_filename),
                           axis_forward=obj.front, axis_up=obj.up)
  assert len(bpy.context.selected_objects) == 1
  blender_obj = bpy.context.selected_objects[0]

  setters = {
      'position': AttributeSetter(blender_obj, 'location'),
      'quaternion': AttributeSetter(blender_obj, 'rotation_quaternion'),
      'scale': AttributeSetter(blender_obj, 'scale'),
      'material': AttributeSetter(blender_obj, 'active_material')
  }
  return blender_obj, setters


@add_object.register(core.DirectionalLight)
def _add_object(obj: core.DirectionalLight):
  sun = bpy.data.lights.new(obj.uid, 'SUN')
  sun_obj = bpy.data.objects.new(obj.uid, sun)

  setters = {
      'position': AttributeSetter(sun_obj, 'location'),
      'quaternion': AttributeSetter(sun_obj, 'rotation_quaternion'),
      'scale': AttributeSetter(sun_obj, 'scale'),
      'color': AttributeSetter(sun, 'color'),
      'intensity': AttributeSetter(sun, 'energy')}
  return sun_obj, setters


@add_object.register(core.RectAreaLight)
def _add_object(obj: core.RectAreaLight):
  area = bpy.data.lights.new(obj.uid, 'AREA')
  area_obj = bpy.data.objects.new(obj.uid, area)

  setters = {
      'position': AttributeSetter(area_obj, 'location'),
      'quaternion': AttributeSetter(area_obj, 'rotation_quaternion'),
      'scale': AttributeSetter(area_obj, 'scale'),
      'color': AttributeSetter(area, 'color'),
      'intensity': AttributeSetter(area, 'energy'),
      'width': AttributeSetter(area, 'size'),
      'height': AttributeSetter(area, 'size_y')}
  return area_obj, setters


@add_object.register(core.PerspectiveCamera)
def _add_object(obj: core.PerspectiveCamera):
  camera = bpy.data.cameras.new(obj.uid)
  camera.type = 'PERSP'
  camera_obj = bpy.data.objects.new(obj.uid, camera)

  setters = {
      'position': AttributeSetter(camera_obj, 'location'),
      'quaternion': AttributeSetter(camera_obj, 'rotation_quaternion'),
      'scale': AttributeSetter(camera_obj, 'scale'),
      'focal_length': AttributeSetter(camera, 'lens'),
      'sensor_width': AttributeSetter(camera, 'sensor_width')}
  return camera_obj, setters


@add_object.register(core.PrincipledBSDFMaterial)
def _add_object(obj: core.PrincipledBSDFMaterial):
  mat = bpy.data.materials.new(obj.uid)
  mat.use_nodes = True
  bsdf_node = mat.node_tree.nodes['Principled BSDF']
  setters = {
      'color': AttributeSetter(bsdf_node.inputs['Base Color'], 'default_value'),
      'metallic': AttributeSetter(bsdf_node.inputs['Metallic'], 'default_value'),
      'specular': AttributeSetter(bsdf_node.inputs['Specular'], 'default_value'),
      'specular_tint': AttributeSetter(bsdf_node.inputs['Specular Tint'], 'default_value'),
      'ior': AttributeSetter(bsdf_node.inputs['IOR'], 'default_value'),
      'transmission': AttributeSetter(bsdf_node.inputs['Transmission'], 'default_value'),
      'transmission_roughness': AttributeSetter(bsdf_node.inputs['Transmission Roughness'], 'default_value'),
      'emission': AttributeSetter(bsdf_node.inputs['Emission'], 'default_value'),
  }
  return mat, setters


@add_object.register(core.Scene)
def _add_scene(obj: core.Scene):
  blender_scene = bpy.context.scene

  setters = {
      'frame_start': AttributeSetter(blender_scene, 'frame_start'),
      'frame_end': AttributeSetter(blender_scene, 'frame_end'),
      'frame_rate': AttributeSetter(blender_scene.render, 'fps'),
      'resolution': AttributeSetter(blender_scene.render, ['resolution_x', 'resolution_y']),
      'camera': AttributeSetter(blender_scene, 'camera')
  }
  return blender_scene, setters


# ########### ########### ########### ########### ########### ########### ########### ##########

class Destructor:
  def __init__(self, blender_objects):
    self.blender_objects = blender_objects

  def __call__(self, owner=None):
    for obj in self.blender_objects:
      try:
        if isinstance(obj, bpy.types.Object):
          bpy.data.objects.remove(obj, do_unlink=True)
        elif isinstance(obj, bpy.types.Material):
          bpy.data.materials.remove(obj, do_unlink=True)
      except Exception as e:
        print(e)


class Keyframer:
  def __init__(self, setters):
    self.setters = setters

  def __call__(self, owner, member, frame):
    setter = self.setters[member]
    setter.target_obj.keyframe_insert(setter.target_name, frame=frame)
