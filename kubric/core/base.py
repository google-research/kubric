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

import collections
import contextlib
import multiprocessing

import munch
import numpy as np
import traitlets as tl

__all__ = ("Asset", "Undefined", "UndefinedAsset")


def next_global_count(name, reset=False):
  """ Return the total number of times this function has been called with the given name.
   Used to create the increasing UID counts for each class (e.g. "Sphere.007").
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
  background = tl.Bool(default_value=False)
  metadata = tl.Dict(key_trait=tl.ObjectName())

  def __init__(self, **kwargs):
    initializable_traits = self.trait_names()
    initializable_traits.remove("uid")
    unknown_traits = [kwarg for kwarg in kwargs
                      if kwarg not in initializable_traits]

    if unknown_traits:
      raise KeyError(f"Cannot initialize unknown trait(s) {unknown_traits}. "
                     f"Possible traits are: {initializable_traits}")

    self.linked_objects = {}
    self.scenes = []
    self.keyframes = collections.defaultdict(dict)

    super().__init__(**kwargs)

  @property
  def active_scene(self):
    # TODO: this is currently just a hack to avoid the ambiguity when dealing with multiple scenes
    return self.scenes[0] if self.scenes else None

  @tl.default("uid")
  def _uid(self):
    """ Use the class name and a global count separated by a period as the UID.
    Counting starts at 1 and always has at least three digits.
    UIDs are thus of the form Scene.001 or Cube.004.
    That way the first 999 instances of each class can be sorted alphabetically.
    """
    name = self.__class__.__name__
    if isinstance(self, Undefined):
      return f"{name}"
    else:
      return f"{name}.{next_global_count(name):03d}"

  def keyframe_insert(self, member: str, frame: int):
    if not self.has_trait(member):
      raise KeyError("Unknown member \"{}\".".format(member))
    self.keyframes[member][frame] = getattr(self, member)

    # use the traitlets observer system to notify all the AttributeSetters about the new keyframe
    self.notify_change(munch.Munch(name=member,
                                   owner=self,
                                   frame=frame,
                                   type='keyframe'))

  @contextlib.contextmanager
  def at_frame(self, frame, interpolation="linear"):
    if frame is None:
      try:
        yield self
      finally:
        return

    old_values = {}
    try:
      for key, value in self.keyframes.items():
        old_values[key] = getattr(self, key)
        setattr(self, key, self.get_value_at(name=key, frame=frame,
                                             interpolation=interpolation))
      yield self
    finally:
      for key, value in old_values.items():
        setattr(self, key, value)

  def get_value_at(self, name, frame, interpolation="linear"):
    if name not in self.keyframes:
      # no animation data found, try retrieving static value
      return getattr(self, name)
    keyframes = self.keyframes[name]

    if frame in keyframes:
      return keyframes[frame]

    available_frames = sorted(keyframes.keys())
    right_idx = np.searchsorted(available_frames, frame)
    if right_idx == 0:
      return keyframes[available_frames[0]]
    if right_idx == len(available_frames):
      return keyframes[available_frames[-1]]
    left_frame, right_frame = available_frames[right_idx-1], available_frames[right_idx]

    if interpolation == "const":
      return keyframes[left_frame]
    elif interpolation == "nearest":
      if abs(frame - left_frame) <= abs(frame - right_frame):
        return keyframes[left_frame]
      else:
        return keyframes[right_frame]
    elif interpolation == "linear":
      mixing = (frame - left_frame) / (right_frame - left_frame)
      left_val = np.array(keyframes[left_frame])
      right_val = np.array(keyframes[right_frame])
      return (1-mixing) * left_val + mixing * right_val

  def get_values_over_time(self, name, frames=None, interpolation="linear"):
    if frames is None:
      frames = list(range(self.active_scene.frame_start,
                          self.active_scene.frame_end+1))
    return np.array([self.get_value_at(name, frame=f, interpolation=interpolation)
                     for f in frames], dtype=np.float32)


  def __hash__(self):
    return hash(self.uid)

  def __eq__(self, other):
    if not isinstance(other, Asset):
      return NotImplemented
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

