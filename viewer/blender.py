# Copyright 2020 Google LLC, Derek Liu
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
"""Implementation of blender backend."""


import bpy
import numpy as np
from viewer import interface


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


class NotImplementableError(NotImplementedError):
  """When a method in the interface cannot be realized in a particular implementation."""
  pass


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


class Object3D(interface.Object3D):

  # Mapping from interface properties to blender properties (used in keyframing).
  _member_to_blender_data_path = {
    "position": "location"
  }

  def __init__(self, blender_object):  # , name=None):
    super().__init__(self)
    self._blender_object = blender_object
    # if self.name: self._blender_object.name = self.name

  def _set_position(self, value):
    # (UI: click mesh > Transform > Location)
    super()._set_position(value)
    self._blender_object.location = self.position

  def _set_scale(self, value):
    # (UI: click mesh > Transform > Scale)
    super()._set_scale(value)
    self._blender_object.scale = self.scale

  def _set_quaternion(self, value):
    # (UI: click mesh > Transform > Rotation)
    super()._set_quaternion(value)
    self._blender_object.rotation_euler = self.quaternion.to_euler()

  def keyframe_insert(self, member: str, frame: int):
    assert hasattr(self, member), "cannot keyframe an undefined property"
    data_path = Object3D._member_to_blender_data_path[member]
    self._blender_object.keyframe_insert(data_path=data_path, frame=frame)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


class Scene(interface.Scene):
  # TODO: look at API scene.objects.link(blender_object)
  # TODO: create a named scene, and refer via bpy.data.scenes['Scene']
  
  def __init__(self):
    super().__init__()
    bpy.context.scene.render.fps = 24
    bpy.context.scene.render.fps_base = 1.0

  def _set_frame_start(self, value):
    super()._set_frame_start(value)
    bpy.context.scene.frame_start = value

  def _set_frame_end(self, value):
    super()._set_frame_start(value)
    bpy.context.scene.frame_end = value

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------                                       


class Camera(interface.Camera, Object3D):
  def __init__(self, name=None):
    bpy.ops.object.camera_add()  # Created camera with name 'Camera'
    Object3D.__init__(self, bpy.context.object)


class OrthographicCamera(interface.OrthographicCamera, Camera):
  def __init__(self, left=-1, right=+1, top=+1, bottom=-1, near=.1, far=2000):
    interface.OrthographicCamera.__init__(self, left, right, top, bottom, near, far)
    Camera.__init__(self)
    # --- extra things to set
    self._blender_object.data.type = 'ORTHO'
    self._blender_object.data.ortho_scale = (right-left)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


class AmbientLight(interface.AmbientLight):
  def __init__(self, color=0x030303, intensity=1):
    bpy.context.scene.world.use_nodes = True  # TODO: shouldn't use_nodes be moved to scene?
    interface.AmbientLight.__init__(self, color=color, intensity=intensity)

  def _set_color(self, value):
    super()._set_color(value)
    bpy.context.scene.world.node_tree.nodes["Background"].inputs[
      'Color'].default_value = self.color

  def _set_intensity(self, value):
    super()._set_intensity(value)
    bpy.context.scene.world.node_tree.nodes["Background"].inputs[
      'Strength'].default_value = self.intensity


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


class DirectionalLight(interface.DirectionalLight, Object3D):
  def __init__(self, color=0xffffff, intensity=1, shadow_softness=.1):
    bpy.ops.object.light_add(type='SUN')  # Creates light with name 'Sun'
    Object3D.__init__(self, bpy.context.object)
    self._blender_object.data.use_nodes = True
    interface.DirectionalLight.__init__(self, color=color, intensity=intensity,
                                        shadow_softness=shadow_softness)

  def _set_color(self, value):
    super()._set_color(value)
    self._blender_object.data.node_tree.nodes["Emission"].inputs[
      'Color'].default_value = self.color

  def _set_intensity(self, value):
    super()._set_intensity(value)
    self._blender_object.data.node_tree.nodes["Emission"].inputs[
      'Strength'].default_value = self.intensity

  def _set_shadow_softness(self, value):
    super()._set_shadow_softness(value)
    self._blender_object.data.angle = self.shadow_softness


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


class BufferAttribute(interface.BufferAttribute):
  pass


class Float32BufferAttribute(interface.Float32BufferAttribute):
  def __init__(self, array, itemSize, normalized=None):
    self.array = array  # TODO: @property


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


class Geometry():
  # NOTE: this should not inherit from Object3D!
  pass


class BoxGeometry(interface.BoxGeometry, Geometry):
  def __init__(self, width=1.0, height=1.0, depth=1.0):
    assert width == height and width == depth, "blender only creates unit cubes"
    interface.BoxGeometry.__init__(self, width=width, height=height, depth=depth)
    bpy.ops.mesh.primitive_cube_add(size=width)
    self._blender_object = bpy.context.object


class PlaneGeometry(interface.Geometry, Geometry):
  def __init__(self, width: float = 1, height: float = 1,
      widthSegments: int = 1, heightSegments: int = 1):
    assert widthSegments == 1 and heightSegments == 1, "not implemented"
    bpy.ops.mesh.primitive_plane_add()
    self._blender_object = bpy.context.object


class BufferGeometry(interface.BufferGeometry, Geometry):
  def __init__(self):
    interface.BufferGeometry.__init__(self)

  def set_index(self, nparray):
    interface.BufferGeometry.set_index(self, nparray)

  def set_attribute(self, name, attribute: interface.BufferAttribute):
    interface.BufferGeometry.set_attribute(self, name, attribute)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


class Material(interface.Material):
  def __init__(self, specs={}):
    # TODO: is this the same as object3D? guess not?
    self._blender_material = bpy.data.materials.new('Material')

  def blender_apply(self, blender_object):
    """Used by materials that need to access the blender object."""
    pass


class MeshBasicMaterial(interface.MeshBasicMaterial, Material):
  def __init__(self, specs={}):
    Material.__init__(self, specs)
    interface.MeshBasicMaterial.__init__(self, specs)


class MeshPhongMaterial(interface.MeshPhongMaterial, Material):
  def __init__(self, specs={}):
    Material.__init__(self, specs)
    interface.Material.__init__(self, specs=specs)
    # TODO: apply specs

  def blender_apply(self, blender_object):
    bpy.context.view_layer.objects.active = blender_object
    bpy.ops.object.shade_smooth()


class MeshFlatMaterial(interface.MeshFlatMaterial, Material):
  def __init__(self, specs={}):
    Material.__init__(self, specs)
    interface.Material.__init__(self, specs=specs)

  def blender_apply(self, blender_object):
    bpy.context.view_layer.objects.active = blender_object
    bpy.ops.object.shade_flat()


class ShadowMaterial(interface.ShadowMaterial, Material):
  def __init__(self, specs={}):
    Material.__init__(self, specs=specs)
    interface.ShadowMaterial.__init__(self, specs=specs)

  def blender_apply(self, blender_object):
    if self.receive_shadow:
      blender_object.cycles.is_shadow_catcher = True


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


class Mesh(interface.Mesh, Object3D):
  def __init__(self, geometry: Geometry, material: Material):
    interface.Mesh.__init__(self, geometry, material)

    # --- Create the blender object
    # WARNING: differently from threejs, blender creates an object when
    # primivitives are created, so we need to make sure we do not duplicate it
    if hasattr(geometry, "_blender_object"):
      # TODO: is there a better way to achieve this?
      Object3D.__init__(self, geometry._blender_object)
    else:
      bpy.ops.object.add(type="MESH")
      Object3D.__init__(self, bpy.context.object)

    # --- Assigns the buffers to the object
    # TODO: is there a better way to achieve this?
    if isinstance(self.geometry, BufferGeometry):
      vertices = self.geometry.attributes["position"].array.tolist()
      faces = self.geometry.index.tolist()
      self._blender_object.data.from_pydata(vertices, [], faces)

    # --- Adds the material to the object
    self._blender_object.data.materials.append(material._blender_material)
    self.material.blender_apply(self._blender_object)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


class Renderer(interface.Renderer):

  def __init__(self, useBothCPUGPU=False):
    super().__init__()
    self.clear_scene()  # as blender has a default scene on load
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 128
    bpy.context.scene.cycles.max_bounces = 6
    bpy.context.scene.cycles.film_exposure = 1.5
    bpy.context.scene.view_layers['View Layer']['cycles']['use_denoising'] = 1

    # --- transparency
    bpy.context.scene.render.film_transparent = True
    # bpy.context.scene.cycles.film_transparent = True  # TODO: derek?

    # --- compute devices # TODO derek?
    cyclePref = bpy.context.preferences.addons['cycles'].preferences
    cyclePref.compute_device_type = 'CUDA'
    for dev in cyclePref.devices:
      if dev.type == "CPU" and useBothCPUGPU is False:
        dev.use = False
      else:
        dev.use = True
    bpy.context.scene.cycles.device = 'GPU'
    for dev in cyclePref.devices:
      print(dev)
      print(dev.use)

  def set_size(self, width: int, height: int):
    super().set_size(width, height)
    bpy.context.scene.render.resolution_x = self.width
    bpy.context.scene.render.resolution_y = self.height

  def set_clear_color(self, color: int, alpha: float):
    raise NotImplementableError()

  def clear_scene(self):
    bpy.ops.wm.read_homefile()
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

  def default_camera_view(self):
    """Changes the UI so that the default view is from the camera POW."""
    view3d = next(area for area in bpy.context.screen.areas if area.type == 'VIEW_3D')
    view3d.spaces[0].region_3d.view_perspective = 'CAMERA'

  def postprocess_solidbackground(self, color=0xFFFFFF):
    # TODO: why in the composited output the color is not exactly the specified one? HDR?
    bpy.context.scene.use_nodes = True  #TODO: should this rather be an assert?
    tree = bpy.context.scene.node_tree
    input_node = tree.nodes["Render Layers"]
    output_node  = tree.nodes["Composite"]  # output_node = tree.nodes.new("CompositorNodeComposite")
    alphaover = tree.nodes.new("CompositorNodeAlphaOver") 
    tree.links.new(input_node.outputs["Alpha"], alphaover.inputs[0]) # fac
    alphaover.inputs[1].default_value = interface.hex_to_rgba(color, 1.0) # image 1
    tree.links.new(input_node.outputs["Image"], alphaover.inputs[2]) # image 2
    tree.links.new(alphaover.outputs["Image"], output_node.inputs["Image"])

  def postprocess_remove_weakalpha(self, threshold=0.05): 
    bpy.context.scene.use_nodes = True  #TODO: should this rather be an assert?
    tree = bpy.context.scene.node_tree
    input_node = tree.nodes["Render Layers"]
    output_node  = tree.nodes["Composite"]  # output_node = tree.nodes.new("CompositorNodeComposite")
    ramp = tree.nodes.new('CompositorNodeValToRGB')
    ramp.color_ramp.elements[0].color[3] = 0
    ramp.color_ramp.elements[0].position = threshold
    ramp.color_ramp.interpolation = "CARDINAL"
    tree.links.new(input_node.outputs["Alpha"], ramp.inputs["Fac"])
    tree.links.new(ramp.outputs["Alpha"], output_node.inputs["Alpha"])

  def render(self, scene: Scene, camera: Camera, path: str):
    # --- adjusts resolution according to threejs style camera
    if isinstance(camera, OrthographicCamera):
      aspect = (camera.right - camera.left)*1.0 / (camera.top - camera.bottom)
      new_y_res = int(bpy.context.scene.render.resolution_x / aspect)
      if new_y_res != bpy.context.scene.render.resolution_y:
        print("WARNING: blender renderer adjusted the film resolution", end="")
        print(new_y_res, bpy.context.scene.render.resolution_y)
        bpy.context.scene.render.resolution_y = new_y_res

    # --- Sets the default camera
    bpy.context.scene.camera = camera._blender_object

    if not path.endswith(".blend"):
      bpy.context.scene.render.filepath = path

    # --- creates blender file
    if path.endswith(".blend"):
      self.default_camera_view()  # TODO: not saved... why?
      bpy.ops.wm.save_mainfile(filepath=path)
    
    # --- renders a movie
    elif path.endswith(".mov"):
      assert bpy.context.scene.render.film_transparent == False
      bpy.context.scene.render.image_settings.file_format = "FFMPEG"
      bpy.context.scene.render.image_settings.color_mode = "RGB"
      bpy.context.scene.render.ffmpeg.format = "QUICKTIME"
      bpy.context.scene.render.ffmpeg.codec = "H264"

    # --- renders one frame directly to a png file
    elif path.endswith(".png"):
      bpy.context.scene.render.film_transparent = True
      self.postprocess_solidbackground(color=0xFF0000)
      # TODO: add capability bpy.context.scene.frame_set(frame_number)
      bpy.ops.render.render(write_still=True, animation=False)

    # --- creates a movie as a image sequence {png}
    else:      
      # Then convert to gif with ImageMagick: `convert -delay 8 -loop 0 *.png output.gif`
      bpy.ops.render.render(write_still=True, animation=True)  # movies do not support transparency    