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

import collections
import multiprocessing

import munch
import traitlets as tl

__all__ = ("Asset", "Undefined", "UndefinedAsset", "View")


def next_global_count(reset=False):
  if reset or not hasattr(next_global_count, "counter"):
    next_global_count.counter = 0
    next_global_count.lock = multiprocessing.Lock()

  with next_global_count.lock:
    next_global_count.counter += 1

  return next_global_count.counter


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
    self.scenes = []
    self.keyframes = collections.defaultdict(dict)

  @tl.default("uid")
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

  def __hash__(self):
    return hash(self.uid)

  def __eq__(self, other):
    if not isinstance(other, Asset):
      return False
    return self.uid == other.uid

  def __repr__(self):
    traits = ["{}={!r}".format(k, getattr(self, k))
              for k in self.trait_names() if k != "uid"]
    if traits:
      return f"<{self.uid} {' '.join(traits)}>"
    else:
      return f"<{self.uid}>"


class Undefined:
  """Base class for all Asset types that denote properties which are not set by Kubric."""
  @tl.default("uid")
  def _uid(self):
    return f"<{self.__class__.__name__}>"


class UndefinedAsset(Asset, Undefined):
  pass


# ## ### ####  View  #### ### ## #

class View:
  def add(self, asset):
    raise NotImplementedError()

  def remove(self, asset):
    raise NotImplementedError()

