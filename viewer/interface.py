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

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

import mathutils
class Object3D(object):
  # TODO .rotation and .quaternion fields are coupled; see https://github.com/mrdoob/three.js/blob/master/src/core/Object3D.js
  position = property(lambda self: self._get_position(), lambda self, value: self._set_position(value))
  scale = property(lambda self: self._get_scale(), lambda self, value: self._set_scale(value))
  quaternion = property(lambda self: self._get_quaternion(), lambda self, value: self._set_quaternion(value))

  def __init__(self, name=None):
    self.name = name
    self._position = mathutils.Vector((0,0,0)) 
    self._quaternion = mathutils.Quaternion() 
    self._scale = (1,1,1) 
    self.parent = None  # TODO: parent management
    self.up = (0,1,0) 

  def _get_position(self): return self._position
  def _set_position(self, value): self._position = mathutils.Vector(value)
  def _get_scale(self): return self._scale
  def _set_scale(self, value): self._scale = value
  def _get_quaternion(self): return self._quaternion
  def _set_quaternion(self, value): self._quaternion = value

  def look_at(self, x, y, z):
    direction = mathutils.Vector((x,y,z)) - self.position
    # TODO: shouldn't we be using self.up here?
    self.quaternion = direction.to_track_quat('-Z', 'Y')

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class Scene(object):
  def __init__(self):
    self._objects3d = list()

  def add(self, object):  #< node? check 3js API
    self._objects3d.append(object)

  def blender(self):
    for object3d in self._objects3d:
      object3d.blender()

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

from tensorflow_graphics.viewer import interface
class Camera(interface.Object3D):
  def __init__(self):
    interface.Object3D.__init__(self)
    self.zoom = 1

class OrthographicCamera(interface.Camera):
  def __init__(self, left=-1, right=+1, top=+1, bottom=-1, near=.1, far=2000):
    # TODO(Derek): how to set camera limits in blender?
    interface.Camera.__init__(self)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class Renderer(object):
  """Superclass of all renderers."""
  def __init__(self):
    self.set_size()

  def set_size(self, width: int=320, height: int=240):
    self.width = width
    self.height = height

# ------------------------------------------------------------------------------
# -------------------------------  lights.py   ---------------------------------
# ------------------------------------------------------------------------------

from tensorflow_graphics.viewer import interface
class Light(interface.Object3D):
  color = property(lambda self: self._get_color(), lambda self, value: self._set_color(value))
  intensity = property(lambda self: self._get_intensity(), lambda self, value: self._set_intensity(value))

  def __init__(self, color=0xffffff, intensity=1):
    interface.Object3D.__init__(self)
    self.color = color
    self.intensity = intensity

  def _get_color(self):
    return self._color
  def _set_color(self, value):
    def hex_to_rgba(hexint: int):
      b = hexint & 255 
      g = (hexint>>8) & 255
      r = (hexint>>16) & 255
      return (r/255.0, g/255.0, b/255.0, 1.0)
    self._color = hex_to_rgba(value)

  def _get_intensity(self):
    return self._intensity
  def _set_intensity(self, val):
    self._intensity = val

class AmbientLight(interface.Light):
  def __init__(self, color=0x030303, intensity=1):
    interface.Light.__init__(self, color=color, intensity=intensity)

class DirectionalLight(interface.Light):
  shadow_softness = property(lambda self: self._get_shadow_softness(), lambda self, value: self._set_shadow_softness(value))

  """Slight difference from THREEJS: uses position+lookat."""
  def __init__(self, color=0xffffff, intensity=1, shadow_softness=.1):
    interface.Light.__init__(self, color=color, intensity=intensity)
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

from tensorflow_graphics.viewer import interface
class Geometry(object):
  """See: https://threejs.org/docs/#api/en/core/BufferGeometry"""
  pass
  
class BoxGeometry(interface.Geometry):
  def __init__(self, width=1.0, height=1.0, depth=1.0):
    interface.Geometry.__init__(self)
    self.width = width
    self.height = height
    self.depth = depth

class BufferGeometry(interface.Geometry):
  """https://threejs.org/docs/#api/en/core/BufferGeometry"""
  def __init__(self):
    interface.Geometry.__init__(self)
    self.index = None
    self.attributes = dict()
  
  def set_index(self, nparray):
    self.index = nparray # TODO: checks

  def set_attribute(self, name, attribute: interface.BufferAttribute):
    self.attributes[name] = attribute # TODO: checks

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

from tensorflow_graphics.viewer import interface
class Material(object):
  def __init__(self, specs={}):
    # TODO: apply specs
    self.receive_shadow = False

class MeshBasicMaterial(interface.Material):
  def __init__(self, specs={}):
    interface.Material.__init__(self, specs=specs)

class MeshFlatMaterial(interface.Material):
  def __init__(self, specs={}):
    interface.Material.__init__(self, specs=specs)
    # TODO: apply specs

class MeshPhongMaterial(interface.Material):
  def __init__(self, specs={}):
    interface.Material.__init__(self, specs=specs)
    # TODO: apply specs

class ShadowMaterial(interface.Material):
  def __init__(self, specs={}):
    interface.Material.__init__(self, specs=specs)
    # TODO: apply specs
    self.receive_shadow = True

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

from tensorflow_graphics.viewer import interface
class Mesh(interface.Object3D):
  def __init__(self, geometry: interface.Geometry, material: interface.Material):
    interface.Object3D.__init__(self)
    self.geometry = geometry
    self.material = material