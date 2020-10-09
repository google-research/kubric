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
import functools
import sys
from typing import Any, Callable, Dict, Union, Tuple


import bidict
import bpy
import munch
import traitlets as tl

from kubric import core

__thismodule__ = sys.modules[__name__]
logger = logging.getLogger(__name__)


class Blender:
  def __init__(self, scene: core.Scene):
    self.objects_to_blend = bidict.bidict()

    self.ambient_node = None
    self.ambient_hdri_node = None
    self.illum_mapping_node = None
    self.bg_node = None
    self.bg_hdri_node = None
    self.bg_mapping_node = None

    self.clear_and_reset()  # as blender has a default scene on load
    # the ray-tracing engine is set here because it affects the availability of some features
    bpy.context.scene.render.engine = "CYCLES"
    self.add(scene)
    self.set_up_scene_shading()

    bpy.context.scene.cycles.use_adaptive_sampling = True  # speeds up rendering
    bpy.context.scene.view_layers[0].cycles.use_denoising = True  # improves the output quality

  def add(self, obj: core.Asset):
    if obj in self.objects_to_blend:
      return self.objects_to_blend[obj]
    blender_obj, setters = add_object(obj)

    # set the name of the object to the UID
    blender_obj.name = obj.uid
    # if it has a rotation mode, then make sure it is set to quaternions
    if hasattr(blender_obj, "rotation_mode"):
      blender_obj.rotation_mode = "QUATERNION"

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
      if isinstance(value, core.Asset):
        value = self.add(value)
      # Initialize values
      setter(munch.Munch(owner=obj, new=value, type="init"))
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
      raise ValueError("Not a valid object {}".format(obj))

  def clear_and_reset(self):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.world = bpy.data.worlds.new("World")

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
    out_node = tree.nodes.new(type="CompositorNodeOutputFile")
    # set the format to EXR (multilayer)
    out_node.format.file_format = "OPEN_EXR_MULTILAYER"
    out_node.base_path = str(path)  # output directory

    layers = ["Image", "Depth", "Vector", "UV", "Normal", "CryptoObject00"]

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
    out_node = tree.nodes.new(type="ShaderNodeOutputWorld")
    out_node.location = 1100, 0

    mix_node = tree.nodes.new(type="ShaderNodeMixShader")
    mix_node.location = 900, 0
    lightpath_node = tree.nodes.new(type="ShaderNodeLightPath")
    lightpath_node.location = 700, 350
    self.ambient_node = tree.nodes.new(type="ShaderNodeBackground")
    self.ambient_node.inputs["Color"].default_value = (0., 0., 0., 1.)
    self.ambient_node.location = 700, 0
    self.bg_node = tree.nodes.new(type="ShaderNodeBackground")
    self.bg_node.inputs["Color"].default_value = (0., 0., 0., 1.)
    self.bg_node.location = 700, -120

    links.new(lightpath_node.outputs.get("Is Camera Ray"), mix_node.inputs.get("Fac"))
    links.new(self.ambient_node.outputs.get("Background"), mix_node.inputs[1])
    links.new(self.bg_node.outputs.get("Background"), mix_node.inputs[2])
    links.new(mix_node.outputs.get("Shader"), out_node.inputs.get("Surface"))

    # create nodes for HDRI images, but leave them disconnected until set_ambient_illumination or set_background
    coord_node = tree.nodes.new(type="ShaderNodeTexCoord")

    self.bg_mapping_node = tree.nodes.new(type="ShaderNodeMapping")
    self.bg_mapping_node.location = 200, 200
    self.bg_hdri_node = tree.nodes.new(type="ShaderNodeTexEnvironment")
    self.bg_hdri_node.location = 400, 200
    links.new(coord_node.outputs.get("Generated"), self.bg_mapping_node.inputs.get("Vector"))
    links.new(self.bg_mapping_node.outputs.get("Vector"), self.bg_hdri_node.inputs.get("Vector"))
    #links.new(bg_hdri_node.outputs.get("Color"), self.bg_node.inputs.get("Color"))

    self.illum_mapping_node = tree.nodes.new(type="ShaderNodeMapping")
    self.illum_mapping_node.location = 200, -200
    self.ambient_hdri_node = tree.nodes.new(type="ShaderNodeTexEnvironment")
    self.ambient_hdri_node.location = 400, -200
    links.new(coord_node.outputs.get("Generated"), self.illum_mapping_node.inputs.get("Vector"))
    links.new(self.illum_mapping_node.outputs.get("Vector"), self.ambient_hdri_node.inputs.get("Vector"))
    # links.new(illum_hdri_node.outputs.get("Color"), self.illum_node.inputs.get("Color"))

  def set_ambient_light(self, hdri_filepath=None, color=(0., 0., 0., 1.0), hdri_rotation=(0., 0., 0.)):
    tree = bpy.context.scene.world.node_tree
    links = tree.links
    if hdri_filepath is None:
      # disconnect incoming links from hdri node (if any)
      for link in self.ambient_node.inputs["Color"].links:
        links.remove(link)
      self.ambient_node.inputs["Color"].default_value = color
    else:
      # ensure hdri_node is connected
      links.new(self.ambient_hdri_node.outputs.get("Color"), self.ambient_node.inputs.get("Color"))
      self.ambient_hdri_node.image = bpy.data.images.load(hdri_filepath, check_existing=True)
      self.illum_mapping_node.inputs.get("Rotation").default_value = hdri_rotation

  def set_background(self, hdri_filepath=None, color=(0., 0., 0., 1.0), hdri_rotation=(0., 0., 0.)):
    tree = bpy.context.scene.world.node_tree
    links = tree.links
    if hdri_filepath is None:
      # disconnect incoming links from hdri node (if any)
      for link in self.bg_node.inputs["Color"].links:
        links.remove(link)
      self.bg_node.inputs["Color"].default_value = color
    else:
      # ensure hdri_node is connected
      links.new(self.bg_hdri_node.outputs.get("Color"), self.bg_node.inputs.get("Color"))
      self.bg_hdri_node.image = bpy.data.images.load(hdri_filepath, check_existing=True)
      self.bg_mapping_node.inputs.get("Rotation").default_value = hdri_rotation

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

  def save_state(self, path: Union[pathlib.Path, str], filename: str = "scene.blend",
                 pack_textures: bool = True):
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if pack_textures:
      bpy.ops.file.pack_all()  # embed all textures into the blend file
    bpy.ops.wm.save_mainfile(filepath=str(path / filename))

  def render(self, path: Union[pathlib.Path, str]):
    self.activate_render_passes()

    path = pathlib.Path(path)
    bpy.context.scene.render.filepath = str(path / "images" / "frame_")
    self.set_up_exr_output(path / "exr" / "frame_")

    bpy.ops.render.render(animation=True, write_still=True)


# ########## Functions to import kubric objects into blender ###########


class AttributeSetter:
  def __init__(self, blender_obj, attribute: str, converter=None):
    self.blender_obj = blender_obj
    self.attribute = attribute
    self.converter = converter

  def __call__(self, change):
    # change = {'type': 'change', 'new': (1., 1., 1.), 'owner': obj}
    # change = {'type': 'keyframe', 'frame': 15, 'owner': obj}

    if change.type == 'change':
      new_value = change.new

      if isinstance(new_value, core.Undefined):
        return  # ignore any Undefined values

      if isinstance(new_value, core.Asset):
        # Convert Assets to Blender objects before assignment
        if __thismodule__ in new_value.linked_objects:
          new_value = new_value.linked_objects[__thismodule__]
        else:
          new_value = add_object(new_value)

      if self.converter:
        # use converter if given
        new_value = self.converter(new_value)

      setattr(self.blender_obj, self.attribute, new_value)


class KeyframeSetter:
  def __init__(self, blender_obj, attribute_path: str):
    self.attribute_path = attribute_path
    self.blender_obj = blender_obj

  def __call__(self, change):
    if not change.type == 'keyframe':
      print("Mistakes were made.", self, change)
      return
    self.blender_obj.keyframe_insert(self.attribute_path, frame=change.frame)


def prepare_blender_object(func: Callable[[core.Asset], Any]) -> Callable[[core.Asset], Any]:

  @functools.wraps(func)
  def _func(obj: core.Asset):
    # if obj has already been converted, then return the corresponding linked object
    if __thismodule__ in obj.linked_objects:
      return obj.linked_objects[__thismodule__]

    # else use func to create a new blender object
    blender_obj = func(obj)

    # store the blender_obj in the list of linked objects
    obj.linked_objects[__thismodule__] = blender_obj

    # set the name of the object to the UID
    blender_obj.name = obj.uid
    # if it has a rotation mode, then make sure it is set to quaternions
    if hasattr(blender_obj, "rotation_mode"):
      blender_obj.rotation_mode = "QUATERNION"

    # if object is an actual Object (eg. not a Scene, or a Material)
    # then ensure that it is linked into (used by) the current scene collection
    if isinstance(blender_obj, bpy.types.Object):
      collection = bpy.context.scene.collection.objects
      if blender_obj not in collection.values():
        collection.link(blender_obj)

    # trigger change notification for all fields (for initialization)
    for trait_name in obj.trait_names():
      value = getattr(obj, trait_name)
      obj.notify_change(munch.Munch(owner=obj, type="change", name=trait_name,
                                    new=value, old=value))

    return blender_obj

  return _func


@functools.singledispatch
def add_object(obj: core.Asset) -> Tuple[bpy.types.Object, Dict[str, AttributeSetter]]:
  raise NotImplementedError()


def register_object3d_setters(obj, blender_obj):
  assert isinstance(obj, core.Object3D), f"{type(obj)} is not an Object3D"

  obj.observe(AttributeSetter(blender_obj, 'location'), 'position')
  obj.observe(KeyframeSetter(blender_obj, 'location'), 'position', type="keyframe")

  obj.observe(AttributeSetter(blender_obj, 'rotation_quaternion'), 'quaternion')
  obj.observe(KeyframeSetter(blender_obj, 'rotation_quaternion'), 'quaternion', type="keyframe")

  obj.observe(AttributeSetter(blender_obj, 'scale'), 'scale')
  obj.observe(KeyframeSetter(blender_obj, 'scale'), 'scale', type="keyframe")


@add_object.register(core.Cube)
@prepare_blender_object
def _add_object(obj: core.Cube):
  bpy.ops.mesh.primitive_cube_add()
  cube = bpy.context.active_object

  register_object3d_setters(obj, cube)
  obj.observe(AttributeSetter(cube, 'active_material'), 'material')
  return cube


@add_object.register(core.Sphere)
@prepare_blender_object
def _add_object(obj: core.Sphere):
  bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=5)
  bpy.ops.object.shade_smooth()
  sphere = bpy.context.active_object

  register_object3d_setters(obj, sphere)
  obj.observe(AttributeSetter(sphere, 'active_material'), 'material', type="change")

  return sphere


@add_object.register(core.FileBasedObject)
@prepare_blender_object
def _add_object(obj: core.FileBasedObject):
  # TODO: support other file-formats
  bpy.ops.import_scene.obj(filepath=str(obj.render_filename),
                           axis_forward=obj.front, axis_up=obj.up)
  assert len(bpy.context.selected_objects) == 1
  blender_obj = bpy.context.selected_objects[0]

  register_object3d_setters(obj, blender_obj)
  obj.observe(AttributeSetter(blender_obj, 'active_material'), 'material')
  # TODO: trigger error when changing filenames or asset-id after the fact
  return blender_obj


@add_object.register(core.DirectionalLight)
@prepare_blender_object
def _add_object(obj: core.DirectionalLight):
  sun = bpy.data.lights.new(obj.uid, "SUN")
  sun_obj = bpy.data.objects.new(obj.uid, sun)

  register_object3d_setters(obj, sun_obj)
  obj.observe(AttributeSetter(sun, 'color'), 'color')
  obj.observe(KeyframeSetter(sun, 'color'), 'color', type="keyframe")
  obj.observe(AttributeSetter(sun, 'energy'), 'intensity')
  obj.observe(KeyframeSetter(sun, 'energy'), 'intensity', type="keyframe")
  return sun_obj


@add_object.register(core.RectAreaLight)
@prepare_blender_object
def _add_object(obj: core.RectAreaLight):
  area = bpy.data.lights.new(obj.uid, "AREA")
  area_obj = bpy.data.objects.new(obj.uid, area)

  register_object3d_setters(obj, area_obj)
  obj.observe(AttributeSetter(area, 'color'), 'color')
  obj.observe(KeyframeSetter(area, 'color'), 'color', type="keyframe")
  obj.observe(AttributeSetter(area, 'energy'), 'intensity')
  obj.observe(KeyframeSetter(area, 'energy'), 'intensity', type="keyframe")
  obj.observe(AttributeSetter(area, 'size'), 'width')
  obj.observe(KeyframeSetter(area, 'size'), 'width', type="keyframe")
  obj.observe(AttributeSetter(area, 'size_y'), 'height')
  obj.observe(KeyframeSetter(area, 'size_y'), 'height', type="keyframe")

  return area_obj


@add_object.register(core.PointLight)
@prepare_blender_object
def _add_object(obj: core.PointLight):
  point_light = bpy.data.lights.new(obj.uid, "POINT")
  point_light_obj = bpy.data.objects.new(obj.uid, point_light)

  register_object3d_setters(obj, point_light_obj)
  obj.observe(AttributeSetter(point_light, 'color'), 'color')
  obj.observe(KeyframeSetter(point_light, 'color'), 'color', type="keyframe")
  obj.observe(AttributeSetter(point_light, 'energy'), 'intensity')
  obj.observe(KeyframeSetter(point_light, 'energy'), 'intensity', type="keyframe")
  return point_light_obj


@add_object.register(core.PerspectiveCamera)
@prepare_blender_object
def _add_object(obj: core.PerspectiveCamera):
  camera = bpy.data.cameras.new(obj.uid)
  camera.type = "PERSP"
  camera_obj = bpy.data.objects.new(obj.uid, camera)

  register_object3d_setters(obj, camera_obj)
  obj.observe(AttributeSetter(camera, 'lens'), 'focal_length')
  obj.observe(KeyframeSetter(camera, 'lens'), 'focal_length', type="keyframe")
  obj.observe(AttributeSetter(camera, 'senor_width'), 'sensor_width')
  obj.observe(KeyframeSetter(camera, 'senor_width'), 'sensor_width', type="keyframe")

  return camera_obj


@add_object.register(core.OrthographicCamera)
@prepare_blender_object
def _add_object(obj: core.OrthographicCamera):
  camera = bpy.data.cameras.new(obj.uid)
  camera.type = 'ORTHO'
  camera_obj = bpy.data.objects.new(obj.uid, camera)

  register_object3d_setters(obj, camera_obj)
  obj.observe(AttributeSetter(camera, 'ortho_scale'), 'orthographic_scale')
  obj.observe(KeyframeSetter(camera, 'ortho_scale'), 'orthographic_scale', type="keyframe")

  return camera_obj


@add_object.register(core.PrincipledBSDFMaterial)
@prepare_blender_object
def _add_object(obj: core.PrincipledBSDFMaterial):
  mat = bpy.data.materials.new(obj.uid)
  mat.use_nodes = True
  bsdf_node = mat.node_tree.nodes["Principled BSDF"]

  obj.observe(AttributeSetter(bsdf_node.inputs["Base Color"], "default_value"), "color")
  obj.observe(KeyframeSetter(bsdf_node.inputs["Base Color"], "default_value"), "color", type="keyframe")
  obj.observe(AttributeSetter(bsdf_node.inputs["Roughness"], "default_value"), "roughness")
  obj.observe(KeyframeSetter(bsdf_node.inputs["Roughness"], "default_value"), "roughness", type="keyframe")
  obj.observe(AttributeSetter(bsdf_node.inputs["Metallic"], "default_value"), "metallic")
  obj.observe(KeyframeSetter(bsdf_node.inputs["Metallic"], "default_value"), "metallic", type="keyframe")
  obj.observe(AttributeSetter(bsdf_node.inputs["Specular"], "default_value"), "specular")
  obj.observe(KeyframeSetter(bsdf_node.inputs["Specular"], "default_value"), "specular", type="keyframe")
  obj.observe(AttributeSetter(bsdf_node.inputs["Specular Tint"], "default_value"), "specular_tint")
  obj.observe(KeyframeSetter(bsdf_node.inputs["Specular Tint"], "default_value"), "specular_tint", type="keyframe")
  obj.observe(AttributeSetter(bsdf_node.inputs["IOR"], "default_value"), "ior")
  obj.observe(KeyframeSetter(bsdf_node.inputs["IOR"], "default_value"), "ior", type="keyframe")
  obj.observe(AttributeSetter(bsdf_node.inputs["Transmission"], "default_value"), "transmission")
  obj.observe(KeyframeSetter(bsdf_node.inputs["Transmission"], "default_value"), "transmission", type="keyframe")
  obj.observe(AttributeSetter(bsdf_node.inputs["Transmission Roughness"], "default_value"), "transmission_roughness")
  obj.observe(KeyframeSetter(bsdf_node.inputs["Transmission Roughness"], "default_value"), "transmission_roughness", type="keyframe")
  obj.observe(AttributeSetter(bsdf_node.inputs["Emission"], "default_value"), "emission")
  obj.observe(KeyframeSetter(bsdf_node.inputs["Emission"], "default_value"), "emission", type="keyframe")

  return mat


@add_object.register(core.FlatMaterial)
@prepare_blender_object
def _add_object(obj: core.FlatMaterial):
  # --- Create node-based material
  mat = bpy.data.materials.new('Holdout')
  mat.use_nodes = True
  tree = mat.node_tree
  tree.nodes.remove(tree.nodes['Principled BSDF'])  # remove the default shader

  output_node = tree.nodes['Material Output']

  # This material is constructed from three different shaders:
  #  1. if holdout=False then emission_node is responsible for giving the object a uniform color
  #  2. if holdout=True, then the holdout_node is responsible for making the object transparent
  #  3. if indirect_visibility=False then transparent_node makes the node invisible for indirect
  #     effects such as shadows or reflections

  light_path_node = tree.nodes.new(type="ShaderNodeLightPath")
  holdout_node = tree.nodes.new(type="ShaderNodeHoldout")
  transparent_node = tree.nodes.new(type="ShaderNodeBsdfTransparent")
  holdout_mix_node = tree.nodes.new(type="ShaderNodeMixShader")
  indirect_mix_node = tree.nodes.new(type="ShaderNodeMixShader")
  overall_mix_node = tree.nodes.new(type="ShaderNodeMixShader")

  emission_node = tree.nodes.new(type="ShaderNodeEmission")

  tree.links.new(transparent_node.outputs['BSDF'], indirect_mix_node.inputs[1])
  tree.links.new(emission_node.outputs['Emission'], indirect_mix_node.inputs[2])

  tree.links.new(emission_node.outputs['Emission'], holdout_mix_node.inputs[1])
  tree.links.new(holdout_node.outputs['Holdout'], holdout_mix_node.inputs[2])

  tree.links.new(light_path_node.outputs['Is Camera Ray'], overall_mix_node.inputs['Fac'])
  tree.links.new(indirect_mix_node.outputs['Shader'], overall_mix_node.inputs[1])
  tree.links.new(holdout_mix_node.outputs['Shader'], overall_mix_node.inputs[2])

  tree.links.new(overall_mix_node.outputs['Shader'], output_node.inputs['Surface'])

  obj.observe(AttributeSetter(emission_node.inputs['Color'], 'default_value'), "color")
  obj.observe(KeyframeSetter(emission_node.inputs['Color'], 'default_value'), "color", type="keyframe")

  obj.observe(AttributeSetter(holdout_mix_node.inputs['Fac'], 'default_value'), "holdout")
  obj.observe(AttributeSetter(indirect_mix_node.inputs['Fac'], 'default_value'), "indirect_visibility")


@add_object.register(core.Scene)
@prepare_blender_object
def _add_scene(obj: core.Scene):
  blender_scene = bpy.context.scene

  setters = {
      "frame_start": core.AttributeSetter(blender_scene, "frame_start"),
      "frame_end": core.AttributeSetter(blender_scene, "frame_end"),
      "frame_rate": core.AttributeSetter(blender_scene.render, "fps"),
      "resolution": core.AttributeSetter(blender_scene.render, ["resolution_x", "resolution_y"]),
      "camera": core.AttributeSetter(blender_scene, "camera")
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
      except ReferenceError:
        pass  # In this case the object is already gone


class Keyframer:
  def __init__(self, setters):
    self.setters = setters

  def __call__(self, owner, member, frame):
    setter = self.setters[member]
    setter.target_obj.keyframe_insert(setter.target_name, frame=frame)
