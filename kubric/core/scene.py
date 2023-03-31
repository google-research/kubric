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

from typing import Tuple, Union, List

import traitlets as tl

import kubric  # pylint: disable=unused-import
from kubric.utils import next_global_count
from kubric.core import color
from kubric.core import traits as ktl
from kubric.core.assets import Asset  #< avoids self.assets property name clash
from kubric.core.assets import UndefinedAsset  #< avoids self.assets property name clash
from kubric.core.cameras import Camera
from kubric.core.cameras import UndefinedCamera
from kubric.core.objects import Object3D

class Scene(tl.HasTraits):
  """ Scenes hold Assets and are the main interface used by Views (such as Renderers).

  Each scene has global properties:
    * frame_start
    * frame_end
    * frame_rate
    * step_rate
    * resolution  (width, height)
    * gravity
    * camera
    * global_illumination
    * background

  The scene also links to views such as the simulator or the renderer.
  Whenever an Asset is added via `scene.add(asset)` it is also added to all
  linked views.
  """

  uid = tl.Unicode(read_only=True)

  frame_start = tl.Integer()
  frame_end = tl.Integer()

  frame_rate = tl.Integer()
  step_rate = tl.Integer()

  camera = tl.Instance(Camera)
  resolution = tl.Tuple(tl.Integer(), tl.Integer())

  gravity = ktl.Vector3D()

  # TODO: Union[RGB, HDRI]
  ambient_illumination = ktl.RGBA()
  background = ktl.RGBA()

  def __init__(self, frame_start: int = 1, frame_end: int = 48, frame_rate: int = 24,
               step_rate: int = 240, resolution: Tuple[int, int] = (512, 512),
               gravity: Tuple[float, float, float] = (0, 0, -10.),
               camera: Camera = UndefinedCamera(),
               ambient_illumination: color.Color = color.get_color("black"),
               background: color.Color = color.get_color("black")):
    self._assets = []
    self._views = []
    self.metadata = {}
    super().__init__(frame_start=frame_start, frame_end=frame_end, frame_rate=frame_rate,
                     step_rate=step_rate, resolution=resolution, gravity=gravity, camera=camera,
                     ambient_illumination=ambient_illumination, background=background)

  @tl.default("uid")
  def _uid(self):
    name = self.__class__.__name__
    return f"{name}.{next_global_count(name):03d}"

  @tl.validate("step_rate")
  def _valid_step_rate(self, proposal):
    proposed_step_rate = proposal["value"]
    if proposed_step_rate <= 0:
      raise tl.TraitError(f"step_rate should be > 0, but was {proposed_step_rate}")
    if proposed_step_rate % self.frame_rate != 0:
      raise tl.TraitError(f"step_rate should be a multiple of frame_rate, \
        but {proposed_step_rate} % {self.frame_rate} != 0")
    return proposed_step_rate

  @tl.validate("frame_rate")
  def _valid_frame_rate(self, proposal):
    proposed_frame_rate = proposal["value"]
    if proposed_frame_rate <= 0:
      raise tl.TraitError(f"frame_rate should be > 0, but was {proposed_frame_rate}")
    if self.step_rate % proposed_frame_rate != 0:
      raise tl.TraitError(f"step_rate should be a multiple of frame_rate, but \
        {self.step_rate} % {proposed_frame_rate} != 0")
    return proposed_frame_rate

  @property
  def assets(self):
    return tuple(self._assets)

  @property
  def foreground_assets(self):
    return tuple(a for a in self._assets if isinstance(a, Object3D) and not a.background)

  @property
  def background_assets(self):
    return tuple(a for a in self._assets if a.background)

  @property
  def views(self):
    return tuple(self._views)

  def link_view(self, view: "kubric.core.view.View"):
    if view in self._views:
      raise ValueError("View already registered")
    self._views.append(view)

    for asset in self._assets:
      if not isinstance(asset, UndefinedAsset):
        view.add(asset)

  def unlink_view(self, view: "kubric.core.view.View"):
    if view not in self._views:
      raise ValueError("View not linked")

    self._views.remove(view)
    for asset in self._assets:
      view.remove(asset)

  def add(self, asset: Union[Asset, List[Asset]]):
    # --- if list, unroll list and call itself on elements
    if isinstance(asset, (list, tuple)):
      for a in asset:
        self.add(a)
      return

    if isinstance(asset, UndefinedAsset):
      return

    if asset in self._assets:
      return

    self._assets.append(asset)
    assert self not in asset.scenes
    asset.scenes.append(self)

    for view in self._views:
      view.add(asset)

    # --- if is a camera object, and none is set, set as the camera
    if isinstance(asset, Camera):
      self.camera = asset

  def __iadd__(self, asset: Union[Asset, List[Asset]]):
    """Adds assets to the scene with a 'scene+=asset' coding pattern."""
    self.add(asset)
    return self

  def remove(self, asset: Asset):
    if asset not in self._assets:
      raise ValueError(f"{asset} cannot be removed, because it is not part of this scene.")
    self._assets.remove(asset)
    assert self in asset.scenes
    asset.scenes.remove(self)

    for view in self._views:
      view.remove(asset)

  @staticmethod
  def from_flags(flags):
    if isinstance(flags.resolution, str):
      resolution = tuple(int(x) for x in flags.resolution.split("x"))
      if len(resolution) == 1:
        resolution = (resolution[0], resolution[0])
    elif isinstance(flags.resolution, int):
      resolution = (flags.resolution, flags.resolution)
    else:
      resolution = tuple(flags.resolution)
    assert len(resolution) == 2, flags.resolution
    assert all(isinstance(x, int) for x in resolution)

    return Scene(frame_start=flags.frame_start,
                 frame_end=flags.frame_end,
                 frame_rate=flags.frame_rate,
                 step_rate=flags.step_rate,
                 resolution=resolution)

  @tl.observe("camera", type="change")
  def _observe_camera(self, change):
    new_camera = change.new
    if new_camera not in self._assets:
      self.add(new_camera)

  def __hash__(self):
    return hash(self.uid)

  def __eq__(self, other):
    if not isinstance(other, Scene):
      return NotImplemented
    return self.uid == other.uid
