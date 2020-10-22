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
""" This file defines the basic object hierarchy that forms the center of Kubrics interface.

The root classes are Scene and Asset, which further specializes into:
 * Material
   - PrincipledBSDFMaterial
   - FlatMaterial
   - UndefinedMaterial
 * Object3D
   - PhysicalObject
     > FileBasedObject
     > Cube
     > Sphere
 * Light
   - DirectionalLight
   - RectAreaLight
   - PointLight
 * Camera
   - PerspectiveCamera
   - OrthographicCamera
   - UndefinedCamera

"""


import collections
import multiprocessing
from typing import Tuple

import mathutils
import munch
import traitlets as tl
from traitlets import default, validate

import kubric.traits as ktl
from kubric.color import Color

__all__ = (
    "Asset", "Scene", "Undefined",
    "Material", "UndefinedMaterial", "PrincipledBSDFMaterial", "FlatMaterial",
    "Object3D", "PhysicalObject", "FileBasedObject", "Sphere", "Cube",
    "Light", "DirectionalLight", "RectAreaLight", "PointLight",
    "Camera", "PerspectiveCamera", "OrthographicCamera",
)


def next_global_count(reset=False):
  if reset or not hasattr(next_global_count, "counter"):
    next_global_count.counter = 0
    next_global_count.lock = multiprocessing.Lock()

  with next_global_count.lock:
    next_global_count.counter += 1

  return next_global_count.counter


# ## ### ####  Materials  #### ### ## #

class Handler:
  def __init__(self, view):
    self.view = view

  def __call__(self, change):
    if isinstance(change.new, Asset):
      if self.view not in change.new.linked_objects:
        new_value = self.view.add()
        new_value = change.new.linked_objects[self.view]
      else:
        pass

    if change.type == 'change':
      self.change_callback(change)
    elif change.type == 'keyframe':
      self.keyframe_callback(change)


  def change_callback(self, change):
    pass

  def keyframe_callback(self, change):
    pass



class Asset(tl.HasTraits):
  """Base class for the entire OO interface in Kubric.
  All objects, materials, lights, and cameras inherit from Asset.

  Assets
   * have a UID
   * are hashable,
   * have traits that can be observed (by external objects)
   * support inserting keyframes for certain traits
   * track linked (external) objects
  """

  uid = tl.Unicode(read_only=True)

  def __init__(self, **kwargs):
    initializable_traits = self.trait_names()
    initializable_traits.remove("uid")
    unknown_traits = [kwarg for kwarg in kwargs
                      if kwarg not in initializable_traits]

    if unknown_traits:
      raise KeyError(f"Cannot initialize unknown trait(s) {unknown_traits}. "
                     f"Possible traits are: {initializable_traits}")

    super().__init__(**kwargs)

    self.linked_objects = {}
    self.keyframes = collections.defaultdict(dict)

  @default("uid")
  def _uid(self):
    return f"{next_global_count():03d}:{self.__class__.__name__}"
    # return str(uuid.uuid4())

  def keyframe_insert(self, member: str, frame: int):
    if not self.has_trait(member):
      raise KeyError("Unknown member \"{}\".".format(member))
    self.keyframes[member][frame] = getattr(self, member)

    # use the traitlets observer system to notify all the AttributeSetters about the new keyframe
    self.notify_change(munch.Munch(name=member,
                                   owner=self,
                                   frame=frame,
                                   type='keyframe'))


  def observe(self, handler, names=tl.All, type='change'):
    def _change_call(change):
      if isinstance(change.new, Asset):
        change.new = change.new.linked_objects[handler.view]
      handler(change)



  def __hash__(self):
    return hash(self.uid)

  def __eq__(self, other):
    return id(self) == id(other)

  def __repr__(self):
    traits = ["{}={!r}".format(k, getattr(self, k))
              for k in self.trait_names() if k != "uid"]
    if traits:
      return f"<{self.uid} {' '.join(traits)}>"
    else:
      return f"<{self.uid}>"


class Undefined:
  """Base class for all Asset types that denote properties which are not set by Kubric."""
  @default("uid")
  def _uid(self):
    return f"<{self.__class__.__name__}>"


# ## ### ####  Materials  #### ### ## #

class Material(Asset):
  """Base class for all materials."""
  pass


class UndefinedMaterial(Material, Undefined):
  """Marker class to indicate that Kubric should not interfere with this material."""
  pass


class PrincipledBSDFMaterial(Material):
  """A physically based material suited for uniform colored plastic, rubber, metal, glass, etc..."""
  color = ktl.RGBA(default_value=Color.from_name("white"))
  metallic = tl.Float(0.)
  specular = tl.Float(0.5)
  specular_tint = tl.Float(0.)
  roughness = tl.Float(0.4)
  ior = tl.Float(1.45)
  transmission = tl.Float(0)
  transmission_roughness = tl.Float(0)
  emission = ktl.RGBA(default_value=Color.from_name("black"))


class FlatMaterial(Material):
  """Renders the object as a uniform color without any shading.
  If holdout is true, then the pixels of the object will be transparent in the final image (alpha=0).
  (Note, that this is not the same as a transparent object. It still "occludes" other objects)

  The indirect_visibility flag controls if the object casts shadows, can be seen in reflections and
  emits light.
  """
  color = ktl.RGBA(default_value=Color.from_name('white'))
  holdout = tl.Bool(False)
  indirect_visibility = tl.Bool(True)


# ## ### ####  3D Objects  #### ### ## #


class Object3D(Asset):
  position = ktl.Vector3D(default_value=(0., 0., 0.))
  quaternion = ktl.Quaternion(default_value=(1., 0., 0., 0.))

  up = tl.CaselessStrEnum(["X", "Y", "Z", "-X", "-Y", "-Z"], default_value="Y")
  front = tl.CaselessStrEnum(["X", "Y", "Z", "-X", "-Y", "-Z"], default_value="-Z")

  def __init__(self, position=(0., 0., 0.), quaternion=None,
               up="Y", front="-Z", look_at=None, euler=None, **kwargs):
    if look_at is not None:
      assert quaternion is None and euler is None
      direction = mathutils.Vector(look_at) - mathutils.Vector(self.position)
      quaternion = direction.to_track_quat(self.front.upper(), self.up.upper())
    elif euler is not None:
      assert look_at is None and quaternion is None
      quaternion = mathutils.Euler(euler).to_quaternion()
    elif quaternion is None:
      quaternion = (1., 0., 0., 0.)

    super().__init__(position=position, quaternion=quaternion, up=up,
                     front=front, **kwargs)

  def look_at(self, target):
    direction = mathutils.Vector(target) - mathutils.Vector(self.position)
    self.quaternion = direction.to_track_quat(self.front.upper(), self.up.upper())


class PhysicalObject(Object3D):
  scale = ktl.Vector3D(default_value=(1., 1., 1.))

  velocity = ktl.Vector3D(default_value=(0., 0., 0.))
  angular_velocity = ktl.Vector3D(default_value=(0., 0., 0.))

  static = tl.Bool(False)
  mass = tl.Float(1.0)
  friction = tl.Float(0.0)
  restitution = tl.Float(0.5)

  bounds = tl.Tuple(ktl.Vector3D(), ktl.Vector3D(),
                    default_value=((0., 0., 0.), (0., 0, 0.)))

  material = tl.Instance(Material, default_value=UndefinedMaterial())

  @validate("mass")
  def _valid_mass(self, proposal):
    mass = proposal["value"]
    if mass < 0:
      raise tl.TraitError(f"mass cannot be negative ({mass})")
    return mass

  @validate("friction")
  def _valid_friction(self, proposal):
    friction = proposal["value"]
    if friction < 0:
      raise tl.TraitError(f"friction cannot be negative ({friction})")
    return friction

  @validate("friction")
  def _valid_friction(self, proposal):
    friction = proposal["value"]
    if friction < 0:
      raise tl.TraitError(f"friction cannot be negative ({friction})")
    return friction

  @validate("bounds")
  def _valid_bounds(self, proposal):
    lower, upper = proposal["value"]
    for l, u in zip(lower, upper):
      if l > u:
        raise tl.TraitError(f"lower bound cannot be larger than upper bound ({lower} !<= {upper})")
    return lower, upper


class Cube(PhysicalObject):
  pass


class Sphere(PhysicalObject):
  pass


class FileBasedObject(PhysicalObject):
  asset_id = tl.Unicode()

  simulation_filename = tl.Unicode()   # TODO: use pathlib.Path instead
  render_filename = tl.Unicode()       # TODO: use pathlib.Path instead

  # TODO: trigger error when changing filenames or asset-id after the fact


# ## ### ####  Lights  #### ### ## #

class Light(Object3D):
  color = ktl.RGB(default_value=Color.from_name("white").rgb)
  intensity = tl.Float(1.)


class DirectionalLight(Light):
  shadow_softness = tl.Float(0.2)


class RectAreaLight(Light):
  width = tl.Float(1)
  height = tl.Float(1)


class PointLight(Light):
  pass


# ## ### ####  Cameras  #### ### ## #

class Camera(Object3D):
  pass


class UndefinedCamera(Camera, Undefined):
  pass


class PerspectiveCamera(Camera):
  focal_length = tl.Float(50)
  sensor_width = tl.Float(36)


class OrthographicCamera(Camera):
  orthographic_scale = tl.Float(6.0)


# ## ### ####  View  #### ### ## #

class View:
  def add(self, asset):
    raise NotImplementedError()

  def remove(self, asset):
    raise NotImplementedError()


# ## ### ####  Scene  #### ### ## #
class Scene(tl.HasTraits):
  """ Scenes hold Assets and are the main interface used by Views (such as Renderers).

  Each scene has global properties:
    * frame_start
    * frame_end
    * frame_rate
    * step_rate
    * resolution
    * gravity
    * camera
    * global_illumination
    * background

  The scene also links to views such as the simulator or the renderer.
  Whenever an Asset is added via `scene.add(asset)` it is also added to all
  linked views.
  """

  def __init__(self, frame_start: int = 1, frame_end: int = 48, frame_rate: int = 24,
               step_rate: int = 240, resolution: Tuple[int, int] = (512, 512),
               gravity: Tuple[float, float, float] = (0, 0, -10.),
               camera: Camera = UndefinedCamera(),
               global_illumination: Color = Color.from_name("black"),
               background: Color = Color.from_name("black")):
    super().__init__(frame_start=frame_start, frame_end=frame_end, frame_rate=frame_rate,
                     step_rate=step_rate, resolution=resolution, gravity=gravity, camera=camera,
                     global_illumination=global_illumination, background=background)
    self._assets = []
    self._views = []

  frame_start = tl.Integer()
  frame_end = tl.Integer()

  frame_rate = tl.Integer()
  step_rate = tl.Integer()

  camera = tl.Instance("Camera")
  resolution = tl.Tuple(tl.Integer(), tl.Integer())

  gravity = ktl.Vector3D()

  # TODO: Union[RGB, HDRI]
  global_illumination = ktl.RGB()
  background = ktl.RGB()

  @property
  def assets(self):
    return tuple(self._assets)

  @property
  def views(self):
    return tuple(self._views)

  def link_view(self, view: View):
    if view in self._views:
      raise ValueError("View already registered")
    self._views.append(view)

    for asset in self._assets:
      self._add_to_view(asset, view)

  def unlink_view(self, view: View):
    if view not in self._views:
      raise ValueError("View not linked")

    self._views.remove(view)
    for asset in self._assets:
      self._remove_from_view(asset, view)

  def add(self, asset: Asset):
    if asset not in self._assets:
      self._assets.append(asset)

    for view in self._views:
      self._add_to_view(asset, view)

  def remove(self, asset: Asset):
    if asset not in self._assets:
      raise ValueError(f"{asset} cannot be removed, because it is not part of this scene.")
    self._assets.remove(asset)

    for view in self._views:
      self._remove_from_view(asset, view)

  def _add_to_view(self, asset: Asset, view: View):
    if view in asset.linked_objects:
      return asset.linked_objects[view]

    external_obj = view.add(asset)
    asset.linked_objects[view] = external_obj
    for trait_name in asset.trait_names():
      value = getattr(asset, trait_name)
      asset.notify_change(munch.Munch(owner=asset, type="change",
                                      name=trait_name, new=value, old=value))

  def _remove_from_view(self, asset: Asset, view: View):
    if view in asset.linked_objects:
      del asset.linked_objects[view]
    view.remove(asset)

  @tl.observe("camera", type="change")
  def _observe_camera(self, change):
    new_camera = change.new
    if new_camera not in self._assets:
      self.add(new_camera)
