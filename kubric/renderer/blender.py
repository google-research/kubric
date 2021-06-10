# Copyright 2021 The Kubric Authors.
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

import logging
import sys
import tempfile
from typing import Any, Optional

import bpy
import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
from singledispatchmethod import singledispatchmethod

import kubric as kb
import kubric.post_processing
from kubric import core
from kubric.redirect_io import RedirectStream
from kubric.renderer import blender_utils
from kubric import utils
from kubric.utils import PathLike

logger = logging.getLogger(__name__)


class Blender(core.View):
  def __init__(self,
               scene: core.Scene,
               scratch_dir=tempfile.mkdtemp(),
               adaptive_sampling=True,
               use_denoising=True,
               samples_per_pixel=128,
               background_transparency=False):
    self.ambient_node = None
    self.ambient_hdri_node = None
    self.illum_mapping_node = None
    self.bg_node = None
    self.bg_hdri_node = None
    self.bg_mapping_node = None
    self.scratch_dir = kb.str2path(scratch_dir)
    self.log_file = self.scratch_dir / "blender.log"
    logger.info("Blender rendering folder: '%s'", self.scratch_dir)

    self._clear_and_reset()  # as blender has a default scene on load
    self.blender_scene = bpy.context.scene

    # the ray-tracing engine is set here because it affects the availability of some features
    bpy.context.scene.render.engine = "CYCLES"
    self._setup_scene_shading()

    self.adaptive_sampling = adaptive_sampling  # speeds up rendering
    self.use_denoising = use_denoising  # improves the output quality
    self.samples_per_pixel = samples_per_pixel
    self.background_transparency = background_transparency
    self.activate_render_passes()
    bpy.context.scene.render.filepath = str(self.scratch_dir / "images" / "frame_")
    self.exr_output_node = blender_utils.set_up_exr_output_node()
    self.set_exr_output_path(self.scratch_dir / "exr" / "frame_")

    super().__init__(scene, scene_observers={
        "frame_start": [AttributeSetter(self.blender_scene, "frame_start")],
        "frame_end": [AttributeSetter(self.blender_scene, "frame_end")],
        "frame_rate": [AttributeSetter(self.blender_scene.render, "fps")],
        "resolution": [AttributeSetter(self.blender_scene.render, "resolution_x",
                                       converter=lambda x: x[0]),
                       AttributeSetter(self.blender_scene.render, "resolution_y",
                                       converter=lambda x: x[1])],
        "camera": [AttributeSetter(self.blender_scene, "camera",
                                   converter=self._convert_to_blender_object)],
        "ambient_illumination": [lambda change: self._set_ambient_light_color(change.new)],
        "background": [lambda change: self._set_background_color(change.new)],
    })

  @property
  def adaptive_sampling(self) -> bool:
    return self.blender_scene.cycles.use_adaptive_sampling

  @adaptive_sampling.setter
  def adaptive_sampling(self, value: bool):
    self.blender_scene.cycles.use_adaptive_sampling = value

  @property
  def use_denoising(self) -> bool:
    return self.blender_scene.cycles.use_denoising

  @use_denoising.setter
  def use_denoising(self, value: bool):
    self.blender_scene.cycles.use_denoising = value
    self.blender_scene.cycles.denoiser = "NLM"

  @property
  def samples_per_pixel(self) -> int:
    return self.blender_scene.cycles.samples

  @samples_per_pixel.setter
  def samples_per_pixel(self, nr: int):
    self.blender_scene.cycles.samples = nr

  @property
  def background_transparency(self) -> bool:
    return self.blender_scene.render.film_transparent

  @background_transparency.setter
  def background_transparency(self, value: bool):
    self.blender_scene.render.film_transparent = value

  def set_exr_output_path(self, path_prefix: Optional[PathLike]):
    """Set the target path prefix for EXR output.

    The final filename for a frame will be "{path_prefix}{frame_nr:04d}.exr".
    If path_prefix is None then EXR output is disabled.
    """
    if path_prefix is None:
      self.exr_output_node.mute = True
    else:
      self.exr_output_node.mute = False
      self.exr_output_node.base_path = str(path_prefix)

  def save_state(self, path: PathLike = "scene.blend", pack_textures: bool = True):
    """Saves the '.blend' blender file to disk.""" 
    filepath = str(self.scratch_dir / "scene.blend")
    logger.debug("copying '%s' → '%s'", filepath, path)

    with RedirectStream(stream=sys.stdout, filename=self.log_file):
      # --- save file to the local scratch directory
      bpy.ops.wm.save_mainfile(filepath=filepath)
      # --- embed all textures into the blend file
      if pack_textures: bpy.ops.file.pack_all()

    # --- copy to the target directory (might be bucket)
    tf.io.gfile.copy(filepath, path, overwrite=True)

  def render(self, verbose=False):
    """Renders the animation to `self.scratch_dir`; likely postprocessed by self.postprocess."""
    with RedirectStream(stream=sys.stdout, filename=self.log_file, disabled=verbose):
        bpy.ops.render.render(animation=True, write_still=False) #TODO: Issue #95
      
  def render_still(self, filepath: PathLike = "kubric.png", verbose=False):
    """Renders a single frame of the scene to filepath."""
    assert filepath.endswith(".png")
    
    # --- temporarily modify the blender render.filepath 
    render_filepath_backup = bpy.context.scene.render.filepath
    bpy.context.scene.render.filepath = str(self.scratch_dir / "stillframe.png")
    # --- render
    with RedirectStream(stream=sys.stdout, filename=self.log_file, disabled=verbose):
      bpy.ops.render.render(animation=False, write_still=True)
    # --- copy to desired output path and restore blender render.filepath
    tf.io.gfile.copy(bpy.context.scene.render.filepath, filepath, overwrite=True)
    bpy.context.scene.render.filepath = render_filepath_backup

  def postprocess(self, from_dir: PathLike, to_dir=PathLike):
    from_dir = tfds.core.as_path(from_dir)
    to_dir = tfds.core.as_path(to_dir)

    # --- split objects into foreground and background sets
    fg_objects = [obj for obj in self.scene.assets if obj.background is False]
    bg_objects = [obj for obj in self.scene.assets if obj.background is True]

    # --- collect all layers for all frames
    data_stack = {}
    for frame_id in range(self.scene.frame_start, self.scene.frame_end + 1):
      exr_filename = from_dir / "exr" / f"frame_{frame_id:04d}.exr"
      png_filename = from_dir / "images" / f"frame_{frame_id:04d}.png"

      # TODO(klausg): this is blender specific, should not be IN the blender module?
      layers = kubric.post_processing.get_render_layers_from_exr(exr_filename,
                                                                 bg_objects,
                                                                 fg_objects)
      data = {k: layers[k] for k in
              ["backward_flow", "forward_flow", "depth", "uv", "normal"]}
      # Use the contrast-normalized PNG instead of the EXR for RGBA.
      data["rgba"] = utils.read_png(png_filename)
      data["segmentation"] = layers["segmentation_indices"][:, :, :1]
      for key in data:
        if key in data_stack:
          data_stack[key].append(data[key])
        else:
          data_stack[key] = [data[key]]
    for key in data_stack:
      data_stack[key] = np.stack(data_stack[key], axis=0)
    # Save to image files
    utils.write_image_dict(data_stack, to_dir)
    # compute bounding boxes
    instance_bboxes = kubric.post_processing.compute_bboxes(data_stack["segmentation"])
    utils.save_as_json(to_dir / "bboxes.json", instance_bboxes)

  @singledispatchmethod
  def add_asset(self, asset: core.Asset) -> Any:
    raise NotImplementedError(f"Cannot add {asset!r}")

  @singledispatchmethod
  def remove_asset(self, asset: core.Asset) -> None:
    if self in asset.linked_objects:
      blender_obj = asset.linked_objects[self]
      try:
        if isinstance(blender_obj, bpy.types.Object):
          bpy.data.objects.remove(blender_obj, do_unlink=True)
        elif isinstance(blender_obj, bpy.types.Material):
          bpy.data.materials.remove(blender_obj, do_unlink=True)
        else:
          raise NotImplementedError(f"Cannot remove {asset!r}")
      except ReferenceError:
        pass  # In this case the object is already gone

  @add_asset.register(core.Cube)
  @blender_utils.prepare_blender_object
  def _add_asset(self, asset: core.Cube):
    bpy.ops.mesh.primitive_cube_add()
    cube = bpy.context.active_object

    register_object3d_setters(asset, cube)
    asset.observe(AttributeSetter(cube, "active_material",
                                  converter=self._convert_to_blender_object), "material")
    asset.observe(AttributeSetter(cube, "scale"), "scale")
    asset.observe(KeyframeSetter(cube, "scale"), "scale", type="keyframe")
    return cube

  @add_asset.register(core.Sphere)
  @blender_utils.prepare_blender_object
  def _add_asset(self, obj: core.Sphere):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=5)
    bpy.ops.object.shade_smooth()
    sphere = bpy.context.active_object

    register_object3d_setters(obj, sphere)
    obj.observe(AttributeSetter(sphere, "active_material",
                                converter=self._convert_to_blender_object), "material")
    obj.observe(AttributeSetter(sphere, "scale"), "scale")
    obj.observe(KeyframeSetter(sphere, "scale"), "scale", type="keyframe")
    return sphere

  @add_asset.register(core.FileBasedObject)
  @blender_utils.prepare_blender_object
  def _add_asset(self, obj: core.FileBasedObject):
    if obj.render_filename is None:
      return None  # if there is no render file, then ignore this object
    _, _, extension = obj.render_filename.rpartition(".")
    with RedirectStream(stream=sys.stdout, filename=self.log_file):  # reduce the logging noise
      if extension == "obj":
        bpy.ops.import_scene.obj(filepath=obj.render_filename,
                                 **obj.render_import_kwargs)
      elif extension in ["glb", "gltf"]:
        bpy.ops.import_scene.gltf(filepath=obj.render_filename,
                                  **obj.render_import_kwargs)
        # gltf files often contain "Empty" objects as placeholders for camera / lights etc.
        # here we are interested only in the meshes, so delete everything else
        non_mesh_objects = [obj for obj in bpy.context.selected_objects if obj.type != "MESH"]
        bpy.ops.object.delete({"selected_objects": non_mesh_objects})
        bpy.ops.object.join()
      elif extension == "fbx":
        bpy.ops.import_scene.fbx(filepath=obj.render_filename,
                                 **obj.render_import_kwargs)
      elif extension in ["x3d", "wrl"]:
        bpy.ops.import_scene.x3d(filepath=obj.render_filename,
                                 **obj.render_import_kwargs)

      elif extension == "blend":
        # for now we require the paths to be encoded in the render_import_kwargs. That is:
        # - filepath = dir / "Object" / object_name
        # - directory = dir / "Object"
        # - filename = object_name

        bpy.ops.wm.append(**obj.render_import_kwargs)
      else:
        raise ValueError(f"Unknown file-type: '{extension}' for {obj}")

    assert len(bpy.context.selected_objects) == 1
    blender_obj = bpy.context.selected_objects[0]

    # deactivate auto_smooth because for some reason it lead to no smoothing at all
    # TODO: make smoothing configurable
    blender_obj.data.use_auto_smooth = False

    register_object3d_setters(obj, blender_obj)
    obj.observe(AttributeSetter(blender_obj, "active_material",
                                converter=self._convert_to_blender_object), "material")
    obj.observe(AttributeSetter(blender_obj, "scale"), "scale")
    obj.observe(KeyframeSetter(blender_obj, "scale"), "scale", type="keyframe")
    return blender_obj

  @add_asset.register(core.DirectionalLight)
  @blender_utils.prepare_blender_object
  def _add_asset(self, obj: core.DirectionalLight):
    sun = bpy.data.lights.new(obj.uid, "SUN")
    sun_obj = bpy.data.objects.new(obj.uid, sun)

    register_object3d_setters(obj, sun_obj)
    obj.observe(AttributeSetter(sun, "color"), "color")
    obj.observe(KeyframeSetter(sun, "color"), "color", type="keyframe")
    obj.observe(AttributeSetter(sun, "energy"), "intensity")
    obj.observe(KeyframeSetter(sun, "energy"), "intensity", type="keyframe")
    return sun_obj

  @add_asset.register(core.RectAreaLight)
  @blender_utils.prepare_blender_object
  def _add_asset(self, obj: core.RectAreaLight):
    area = bpy.data.lights.new(obj.uid, "AREA")
    area_obj = bpy.data.objects.new(obj.uid, area)

    register_object3d_setters(obj, area_obj)
    obj.observe(AttributeSetter(area, "color"), "color")
    obj.observe(KeyframeSetter(area, "color"), "color", type="keyframe")
    obj.observe(AttributeSetter(area, "energy"), "intensity")
    obj.observe(KeyframeSetter(area, "energy"), "intensity", type="keyframe")
    obj.observe(AttributeSetter(area, "size"), "width")
    obj.observe(KeyframeSetter(area, "size"), "width", type="keyframe")
    obj.observe(AttributeSetter(area, "size_y"), "height")
    obj.observe(KeyframeSetter(area, "size_y"), "height", type="keyframe")
    return area_obj

  @add_asset.register(core.PointLight)
  @blender_utils.prepare_blender_object
  def _add_asset(self, obj: core.PointLight):
    point_light = bpy.data.lights.new(obj.uid, "POINT")
    point_light_obj = bpy.data.objects.new(obj.uid, point_light)

    register_object3d_setters(obj, point_light_obj)
    obj.observe(AttributeSetter(point_light, "color"), "color")
    obj.observe(KeyframeSetter(point_light, "color"), "color", type="keyframe")
    obj.observe(AttributeSetter(point_light, "energy"), "intensity")
    obj.observe(KeyframeSetter(point_light, "energy"), "intensity", type="keyframe")
    return point_light_obj

  @add_asset.register(core.PerspectiveCamera)
  @blender_utils.prepare_blender_object
  def _add_asset(self, obj: core.PerspectiveCamera):
    camera = bpy.data.cameras.new(obj.uid)
    camera.type = "PERSP"
    # fix sensor width and determine sensor height by the aspect ratio of the image:
    camera.sensor_fit = "HORIZONTAL"
    camera_obj = bpy.data.objects.new(obj.uid, camera)

    register_object3d_setters(obj, camera_obj)
    obj.observe(AttributeSetter(camera, "lens"), "focal_length")
    obj.observe(KeyframeSetter(camera, "lens"), "focal_length", type="keyframe")
    obj.observe(AttributeSetter(camera, "sensor_width"), "sensor_width")
    obj.observe(KeyframeSetter(camera, "sensor_width"), "sensor_width", type="keyframe")
    return camera_obj

  @add_asset.register(core.OrthographicCamera)
  @blender_utils.prepare_blender_object
  def _add_asset(self, obj: core.OrthographicCamera):
    camera = bpy.data.cameras.new(obj.uid)
    camera.type = "ORTHO"
    camera_obj = bpy.data.objects.new(obj.uid, camera)

    register_object3d_setters(obj, camera_obj)
    obj.observe(AttributeSetter(camera, "ortho_scale"), "orthographic_scale")
    obj.observe(KeyframeSetter(camera, "ortho_scale"), "orthographic_scale", type="keyframe")
    return camera_obj

  @add_asset.register(core.PrincipledBSDFMaterial)
  @blender_utils.prepare_blender_object
  def _add_asset(self, obj: core.PrincipledBSDFMaterial):
    mat = bpy.data.materials.new(obj.uid)
    mat.use_nodes = True
    bsdf_node = mat.node_tree.nodes["Principled BSDF"]

    obj.observe(AttributeSetter(bsdf_node.inputs["Base Color"], "default_value"), "color")
    obj.observe(KeyframeSetter(bsdf_node.inputs["Base Color"], "default_value"), "color",
                type="keyframe")
    obj.observe(AttributeSetter(bsdf_node.inputs["Roughness"], "default_value"), "roughness")
    obj.observe(KeyframeSetter(bsdf_node.inputs["Roughness"], "default_value"), "roughness",
                type="keyframe")
    obj.observe(AttributeSetter(bsdf_node.inputs["Metallic"], "default_value"), "metallic")
    obj.observe(KeyframeSetter(bsdf_node.inputs["Metallic"], "default_value"), "metallic",
                type="keyframe")
    obj.observe(AttributeSetter(bsdf_node.inputs["Specular"], "default_value"), "specular")
    obj.observe(KeyframeSetter(bsdf_node.inputs["Specular"], "default_value"), "specular",
                type="keyframe")
    obj.observe(AttributeSetter(bsdf_node.inputs["Specular Tint"], "default_value"), "specular_tint")
    obj.observe(KeyframeSetter(bsdf_node.inputs["Specular Tint"], "default_value"), "specular_tint",
                type="keyframe")
    obj.observe(AttributeSetter(bsdf_node.inputs["IOR"], "default_value"), "ior")
    obj.observe(KeyframeSetter(bsdf_node.inputs["IOR"], "default_value"), "ior",
                type="keyframe")
    obj.observe(AttributeSetter(bsdf_node.inputs["Transmission"], "default_value"), "transmission")
    obj.observe(KeyframeSetter(bsdf_node.inputs["Transmission"], "default_value"), "transmission",
                type="keyframe")
    obj.observe(AttributeSetter(bsdf_node.inputs["Transmission Roughness"], "default_value"),
                "transmission_roughness")
    obj.observe(KeyframeSetter(bsdf_node.inputs["Transmission Roughness"], "default_value"),
                "transmission_roughness", type="keyframe")
    obj.observe(AttributeSetter(bsdf_node.inputs["Emission"], "default_value"), "emission")
    obj.observe(KeyframeSetter(bsdf_node.inputs["Emission"], "default_value"), "emission",
                type="keyframe")
    return mat

  @add_asset.register(core.FlatMaterial)
  @blender_utils.prepare_blender_object
  def _add_asset(self, obj: core.FlatMaterial):
    # --- Create node-based material
    mat = bpy.data.materials.new("Holdout")
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.remove(tree.nodes["Principled BSDF"])  # remove the default shader

    output_node = tree.nodes["Material Output"]

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

    tree.links.new(transparent_node.outputs["BSDF"], indirect_mix_node.inputs[1])
    tree.links.new(emission_node.outputs["Emission"], indirect_mix_node.inputs[2])
    tree.links.new(emission_node.outputs["Emission"], holdout_mix_node.inputs[1])
    tree.links.new(holdout_node.outputs["Holdout"], holdout_mix_node.inputs[2])
    tree.links.new(light_path_node.outputs["Is Camera Ray"], overall_mix_node.inputs["Fac"])
    tree.links.new(indirect_mix_node.outputs["Shader"], overall_mix_node.inputs[1])
    tree.links.new(holdout_mix_node.outputs["Shader"], overall_mix_node.inputs[2])
    tree.links.new(overall_mix_node.outputs["Shader"], output_node.inputs["Surface"])

    obj.observe(AttributeSetter(emission_node.inputs["Color"], "default_value"), "color")
    obj.observe(KeyframeSetter(emission_node.inputs["Color"], "default_value"), "color",
                type="keyframe")
    obj.observe(AttributeSetter(holdout_mix_node.inputs["Fac"], "default_value"), "holdout")
    obj.observe(KeyframeSetter(holdout_mix_node.inputs["Fac"], "default_value"), "holdout",
                type="keyframe")
    obj.observe(AttributeSetter(indirect_mix_node.inputs["Fac"], "default_value"),
                "indirect_visibility")
    obj.observe(KeyframeSetter(indirect_mix_node.inputs["Fac"], "default_value"),
                "indirect_visibility", type="keyframe")
    return mat

  def _clear_and_reset(self):
    with RedirectStream(stream=sys.stdout, filename=self.log_file):
      bpy.ops.wm.read_factory_settings(use_empty=True)
      bpy.context.scene.world = bpy.data.worlds.new("World")

  def _setup_scene_shading(self):
    self.blender_scene.world.use_nodes = True
    tree = self.blender_scene.world.node_tree
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

    self.illum_mapping_node = tree.nodes.new(type="ShaderNodeMapping")
    self.illum_mapping_node.location = 200, -200
    self.ambient_hdri_node = tree.nodes.new(type="ShaderNodeTexEnvironment")
    self.ambient_hdri_node.location = 400, -200
    links.new(coord_node.outputs.get("Generated"), self.illum_mapping_node.inputs.get("Vector"))
    links.new(self.illum_mapping_node.outputs.get("Vector"), self.ambient_hdri_node.inputs.get("Vector"))

  def _set_ambient_light_color(self, color=(0., 0., 0., 1.0)):
    # disconnect incoming links from hdri node (if any)
    for link in self.ambient_node.inputs["Color"].links:
      self.blender_scene.world.node_tree.links.remove(link)
    self.ambient_node.inputs["Color"].default_value = color

  def _set_ambient_light_hdri(self, hdri_filepath=None, hdri_rotation=(0., 0., 0.)):
    # ensure hdri_node is connected
    self.blender_scene.world.node_tree.links.new(self.ambient_hdri_node.outputs.get("Color"),
                                                 self.ambient_node.inputs.get("Color"))
    self.ambient_hdri_node.image = bpy.data.images.load(hdri_filepath, check_existing=True)
    self.illum_mapping_node.inputs.get("Rotation").default_value = hdri_rotation

  def _set_background_color(self, color=core.get_color("black")):
    # disconnect incoming links from hdri node (if any)
    for link in self.bg_node.inputs["Color"].links:
      self.blender_scene.world.node_tree.links.remove(link)
    # set color
    self.bg_node.inputs["Color"].default_value = color

  def _set_background_hdri(self, hdri_filepath=None, hdri_rotation=(0., 0., 0.)):
    # ensure hdri_node is connected
    self.blender_scene.world.node_tree.links.new(self.bg_hdri_node.outputs.get("Color"),
                                                 self.bg_node.inputs.get("Color"))
    self.bg_hdri_node.image = bpy.data.images.load(hdri_filepath, check_existing=True)
    self.bg_mapping_node.inputs.get("Rotation").default_value = hdri_rotation

  def activate_render_passes(self):
    view_layer = self.blender_scene.view_layers[0]
    view_layer.use_pass_vector = True  # flow
    view_layer.use_pass_uv = True  # UV
    view_layer.use_pass_normal = True  # surface normals
    view_layer.cycles.use_pass_crypto_object = True  # segmentation
    view_layer.cycles.pass_crypto_depth = 2

  def _convert_to_blender_object(self, asset: core.Asset):
    return asset.linked_objects[self]


class AttributeSetter:
  def __init__(self, blender_obj, attribute: str, converter=None):
    self.blender_obj = blender_obj
    self.attribute = attribute
    self.converter = converter

  def __call__(self, change):
    # change = {"type": "change", "new": (1., 1., 1.), "owner": obj}
    new_value = change.new

    if isinstance(new_value, core.Undefined):
      return  # ignore any Undefined values

    if self.converter:
      # use converter if given
      new_value = self.converter(new_value)

    setattr(self.blender_obj, self.attribute, new_value)


class KeyframeSetter:
  def __init__(self, blender_obj, attribute_path: str):
    self.attribute_path = attribute_path
    self.blender_obj = blender_obj

  def __call__(self, change):
    self.blender_obj.keyframe_insert(self.attribute_path, frame=change.frame)


def register_object3d_setters(obj, blender_obj):
  assert isinstance(obj, core.Object3D), f"{obj!r} is not an Object3D"

  obj.observe(AttributeSetter(blender_obj, "location"), "position")
  obj.observe(KeyframeSetter(blender_obj, "location"), "position", type="keyframe")

  obj.observe(AttributeSetter(blender_obj, "rotation_quaternion"), "quaternion")
  obj.observe(KeyframeSetter(blender_obj, "rotation_quaternion"), "quaternion", type="keyframe")
