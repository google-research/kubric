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
from kubric import color
import numpy as np
import mathutils
from kubric import core
import itertools

# TODO: what is this line for?
# mathutils.Quaternion()
# TODO: this was inconsistent across kubric
# default_random_state = np.random.RandomState()


def random_hue_color(saturation: float = 1., value: float = 1, rnd=np.random.RandomState()):
  print(rnd.get_state()[1][0])
  return color.Color.from_hsv(rnd.random_sample(), saturation, value)


def random_rotation(axis=None, rnd=np.random.RandomState()):
  """ Compute a random rotation as a quaternion.
  If axis is None the rotation is sampled uniformly over all possible orientations.
  Otherwise it corresponds to a random rotation around the given axis."""

  if axis is None:
    # uniform over all possible orientations
    z = 2
    while z > 1:
      x, y = rnd.rand(2)
      z = x*x + y*y

    w = 2
    while w > 1:
      u, v = rnd.rand(2)
      w = u*u + v*v

    s = np.sqrt((1-z) / w)
    return x, y, s*u, s*v
  else:
    if isinstance(axis, str) and axis.upper() in ["X", "Y", "Z"]:
      axis = {"X": (1., 0., 0., 0.),
              "Y": (0., 1., 0.),
              "Z": (0., 0., 1.)}[axis.upper()]

    quat = mathutils.Quaternion(axis, rnd.uniform(0, 2*np.pi))
    return tuple(quat)


def rotation_sampler(axis=None):
  def _sampler(obj: core.PhysicalObject, rnd):
    obj.quaternion = random_rotation(axis=axis, rnd=rnd)
  return _sampler


def position_sampler(region):
  region = np.array(region, dtype=np.float)

  def _sampler(obj: core.PhysicalObject, rnd):
    # make a copy of the bbox points in the matutils.Vector format
    bounds = np.array(obj.bounds, dtype=np.float)
    bbox_points = [mathutils.Vector(x)
                   for x in itertools.product(bounds[:, 0], bounds[:, 1], bounds[:, 2])]
    obj_orientation = mathutils.Quaternion(obj.quaternion)
    for x in bbox_points:
      x.rotate(obj_orientation)  # rotates x in place by the object orientation
    bbox_points = np.array([tuple(x) for x in bbox_points])  # convert to np.array
    rotated_bounds = np.array([bbox_points.min(axis=0), bbox_points.max(axis=0)])

    effective_region = np.array(region) - rotated_bounds
    obj.position = rnd.uniform(*effective_region)

  return _sampler