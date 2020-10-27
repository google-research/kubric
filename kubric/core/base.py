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

import abc
import collections
import multiprocessing
from typing import Any

import munch
import traitlets as tl

__all__ = ("Asset", "Undefined", "UndefinedAsset", "View")


def next_global_count(name, reset=False):
  """ Return the total number of times this function has been called with the given name.
   Used to create the increasing UID counts for each class (e.g. "Sphere:07").
   When passing reset=True, then all counts are reset to 0.
   """
  if reset or not hasattr(next_global_count, "counter"):
    next_global_count.counter = collections.defaultdict(int)
    next_global_count.lock = multiprocessing.Lock()

  with next_global_count.lock:
    next_global_count.counter[name] += 1
    return next_global_count.counter[name]


class Asset(tl.HasTraits):
  """ Base class for the entire OO interface in Kubric.
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
    self.scenes = []
    self.keyframes = collections.defaultdict(dict)

  @tl.default("uid")
  def _uid(self):
    """ Use the class name and a global count separated by a colon as the UID.
    Counting starts at 1 and always has at least two digits.
    UIDs are thus of the form Scene:01 or Cube:04.
    That way the first 99 instances of each class can be sorted alphabetically.
    """
    name = self.__class__.__name__
    if isinstance(self, Undefined):
      return f"{name}"
    else:
      return f"{name}:{next_global_count(name):02d}"

  def keyframe_insert(self, member: str, frame: int):
    if not self.has_trait(member):
      raise KeyError("Unknown member \"{}\".".format(member))
    self.keyframes[member][frame] = getattr(self, member)

    # use the traitlets observer system to notify all the AttributeSetters about the new keyframe
    self.notify_change(munch.Munch(name=member,
                                   owner=self,
                                   frame=frame,
                                   type='keyframe'))

  def __hash__(self):
    return hash(self.uid)

  def __eq__(self, other):
    if not isinstance(other, Asset):
      return False
    return self.uid == other.uid

  def __repr__(self):
    traits = sorted(["{}={!r}".format(k, getattr(self, k))
                     for k in self.trait_names() if k != "uid"])
    if traits:
      return f"<{self.uid} {' '.join(traits)}>"
    else:
      return f"<{self.uid}>"


class Undefined:
  pass


class UndefinedAsset(Asset, Undefined):
  pass


# ## ### ####  View  #### ### ## #

class View(abc.ABC):
  """ Interface implemented by simulators and renderers. """

  def add(self, asset: Asset) -> None:
    # if asset has already been converted, then return the corresponding linked object
    if self in asset.linked_objects:
      return asset.linked_objects[self]

    # else use add_asset to create a new view-object and store it
    view_obj = self.add_asset(asset)
    asset.linked_objects[self] = view_obj

    # trigger change notification for all fields (for initialization)
    # FIXME: This notifies all the views, but should only notify self.
    #        Unfortunately traitlets doesn't have a solution ready for this,
    #        so we will have to hack it in ourselves.
    for trait_name in asset.trait_names():
      value = getattr(asset, trait_name)
      asset.notify_change(munch.Munch(owner=asset, type="change",
                                      name=trait_name, new=value, old=value))

  def remove(self, asset: Asset) -> None:
    # remove the view-object from the dict of linked objects
    if self in asset.linked_objects:
      del asset.linked_objects[self]

    # use the view-specific remove function to delete the view-object
    self.remove_asset(asset)

  @abc.abstractmethod
  def add_asset(self, asset: Asset) -> Any:
    pass

  @abc.abstractmethod
  def remove_asset(self, asset: Asset) -> None:
    pass
