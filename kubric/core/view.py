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

import abc
from typing import Any, Callable, Dict, List
from types import MappingProxyType
import munch

from kubric.core.scene import Scene
from kubric.core.assets import Asset
from kubric.core.assets import UndefinedAsset


empty_dict = MappingProxyType({})  # immutable empty dict for use in default arguments


class View(abc.ABC):
  """ Interface implemented by simulators and renderers. """

  def __init__(self, scene: Scene, scene_observers: Dict[str, List[Callable]] = empty_dict):
    super().__init__()
    self.scene_observers = scene_observers
    self._scene = None
    self.scene = scene

  @property
  def scene(self) -> Scene:
    return self._scene

  @scene.setter
  def scene(self, scene: Scene):
    old_scene = self._scene
    if old_scene:
      old_scene.unlink_view(self)

    self._scene = scene
    scene.link_view(self)

    for trait_name, setters in self.scene_observers.items():
      assert isinstance(setters, (list, tuple))
      for setter in setters:
        if old_scene:
          old_scene.unobserve(setter, trait_name)
        self._scene.observe(setter, trait_name)
        setter(munch.Munch(new=getattr(scene, trait_name),
                           name=trait_name,
                           owner=scene,
                           type="change"))

  def add(self, asset: Asset) -> None:
    # if asset has already been converted, then do nothing
    if self in asset.linked_objects:
      return

    if isinstance(asset, UndefinedAsset):
      return

    # else use add_asset to create a new view-object and store it
    view_obj = self.add_asset(asset)
    if view_obj is None:
      return

    asset.linked_objects[self] = view_obj

    # trigger change notification for all fields (for initialization)
    # FIXME: This notifies all the views, but should only notify self.
    #        Unfortunately traitlets doesn't have a solution ready for this,
    #        so we will have to hack it in ourselves.
    for trait_name in asset.trait_names():
      value = getattr(asset, trait_name)
      if isinstance(value, Asset):  # recursively add assets to the
        self.add(value)
      asset.notify_change(munch.Munch(owner=asset, type="change",
                                      name=trait_name, new=value, old=value))

  def remove(self, asset: Asset) -> None:
    # use the view-specific remove function to delete the view-object
    self.remove_asset(asset)

    # remove the view-object from the dict of linked objects
    if self in asset.linked_objects:
      del asset.linked_objects[self]

  @abc.abstractmethod
  def add_asset(self, asset: Asset) -> Any:
    pass  # pragma: no cover

  @abc.abstractmethod
  def remove_asset(self, asset: Asset) -> None:
    pass  # pragma: no cover
