# Copyright 2020 Google LLC
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


import mathutils


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def hex_to_rgba(hexint: int, alpha=1.0):
  b = hexint & 255
  g = (hexint >> 8) & 255
  r = (hexint >> 16) & 255
  return [r / 255.0, g / 255.0, b / 255.0, alpha]

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# noinspection PyPropertyDefinition
class Object3D(object):
  # TODO .rotation and .quaternion fields are coupled; see https://github.com/mrdoob/three.js/blob/master/src/core/Object3D.js
  position = property(lambda self: self._get_position(),
                      lambda self, value: self._set_position(value))
  scale = property(lambda self: self._get_scale(),
                   lambda self, value: self._set_scale(value))
  quaternion = property(lambda self: self._get_quaternion(),
                        lambda self, value: self._set_quaternion(value))

  def __init__(self, name=None):
    self.name = name
    self._position = mathutils.Vector((0, 0, 0))
    self._quaternion = mathutils.Quaternion()
    self._scale = (1, 1, 1)
    self.parent = None  # TODO: parent management
    self.up = (0, 1, 0)

  def _get_position(self):
    return self._position

  def _set_position(self, value):
    self._position = mathutils.Vector(value)

  def _get_scale(self):
    return self._scale

  def _set_scale(self, value):
    self._scale = value

  def _get_quaternion(self):
    return self._quaternion

  def _set_quaternion(self, value):
    self._quaternion = value

  def keyframe_insert(self, member: str, frame: int):
    raise NotImplementedError

  def look_at(self, x, y, z):
    direction = mathutils.Vector((x, y, z)) - self.position
    # TODO: shouldn't we be using self.up here?
    self.quaternion = direction.to_track_quat('-Z', 'Y')


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class Scene(object):
  # TODO: environment maps https://threejs.org/docs/#api/en/scenes/Scene.background
  frame_start = property(lambda self: self._frame_start, 
                         lambda self, value: self._set_frame_start(value))
  frame_end = property(lambda self: self._frame_end,
                       lambda self, value: self._set_frame_end(value))

  def __init__(self):
    self._objects3d = list()
    self.frame_start = 0
    self.frame_end = 250 # blender's default

  def _set_frame_start(self, value):
    self._frame_start = value

  def _set_frame_end(self, value):
    self._frame_end = value

  def add(self, obj):
    # TODO: node? check the threejs API
    self._objects3d.append(obj)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


class Camera(Object3D):
  def __init__(self):
    Object3D.__init__(self)


class OrthographicCamera(Camera):
  def __init__(self,
               left: float=-1,right: float=+1,
               top: float=+1, bottom: float=-1,
               near=.1, far=2000):
    Camera.__init__(self)
    assert (right > left) and (top > bottom) and (far > near) and (near > 0.0)
    self.left = left
    self.right = right
    self.top = top
    self.bottom = bottom
    self.near = near
    self.far = far


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


class Renderer(object):
  """Superclass of all renderers."""

  # TODO: convert parameters to "specs" dictionary? (like THREEJS)
  def __init__(self, width: int = 320, height: int = 240):
    self.set_size(width, height)

  def set_size(self, width: int, height: int):
    self.width = width
    self.height = height

  def set_clear_color(self, color: int, alpha: float):
    # https://threejs.org/docs/#api/en/renderers/WebGLRenderer.setClearColor
    self._clear_color = hex_to_rgba(color, alpha)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


# noinspection PyPropertyDefinition
class Light(Object3D):
  color = property(lambda self: self._get_color(),
                   lambda self, value: self._set_color(value))
  intensity = property(lambda self: self._get_intensity(),
                       lambda self, value: self._set_intensity(value))

  def __init__(self, color=0xffffff, intensity=1):
    Object3D.__init__(self)
    self.color = color
    self.intensity = intensity

  def _get_color(self):
    return self._color

  def _set_color(self, value):
    self._color = hex_to_rgba(value)

  def _get_intensity(self):
    return self._intensity

  def _set_intensity(self, val):
    self._intensity = val


class AmbientLight(Light):
  def __init__(self, color=0x030303, intensity=1):
    Light.__init__(self, color=color, intensity=intensity)


# noinspection PyPropertyDefinition
class DirectionalLight(Light):
  shadow_softness = property(lambda self: self._get_shadow_softness(),
                             lambda self, value: self._set_shadow_softness(
                               value))

  """Slight difference from THREEJS: uses position+lookat."""

  def __init__(self, color=0xffffff, intensity=1, shadow_softness=.1):
    Light.__init__(self, color=color, intensity=intensity)
    self.shadow_softness = shadow_softness

  def _get_shadow_softness(self):
    return self._shadow_softness

  def _set_shadow_softness(self, value):
    self._shadow_softness = value


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


class BufferAttribute(object):
  """https://threejs.org/docs/#api/en/core/BufferAttribute"""
  pass


class Float32BufferAttribute(BufferAttribute):
  def __init__(self, array, itemSize, normalized=None):
    self.array = array  # TODO: @property


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class Geometry(object):
  """See: https://threejs.org/docs/#api/en/core/BufferGeometry"""
  pass


class BoxGeometry(Geometry):
  def __init__(self, width=1.0, height=1.0, depth=1.0):
    Geometry.__init__(self)
    self.width = width
    self.height = height
    self.depth = depth


class BufferGeometry(Geometry):
  """https://threejs.org/docs/#api/en/core/BufferGeometry"""

  def __init__(self):
    Geometry.__init__(self)
    self.index = None
    self.attributes = dict()

  def set_index(self, nparray):
    self.index = nparray  # TODO: checks

  def set_attribute(self, name, attribute: BufferAttribute):
    self.attributes[name] = attribute  # TODO: checks


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class Material(object):
  def __init__(self, specs={}):
    # TODO: apply specs
    self.receive_shadow = False


class MeshBasicMaterial(Material):
  def __init__(self, specs={}):
    Material.__init__(self, specs=specs)


class MeshFlatMaterial(Material):
  def __init__(self, specs={}):
    Material.__init__(self, specs=specs)
    # TODO: apply specs


class MeshPhongMaterial(Material):
  def __init__(self, specs={}):
    Material.__init__(self, specs=specs)
    # TODO: apply specs


class ShadowMaterial(Material):
  def __init__(self, specs={}):
    Material.__init__(self, specs=specs)
    # TODO: apply specs
    self.receive_shadow = True


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


class Mesh(Object3D):
  def __init__(self, geometry: Geometry, material: Material):
    Object3D.__init__(self)
    self.geometry = geometry
    self.material = material
