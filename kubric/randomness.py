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
"""Utilities to generate randomly generated quantities (rotations, positions, colors)."""

import numpy as np
import pyquaternion as pyquat
from typing import Optional, Tuple

from kubric.core import color
from kubric.core import objects


CLEVR_COLORS = {
    "blue": color.Color(42/255, 75/255, 215/255),
    "brown": color.Color(129/255, 74/255, 25/255),
    "cyan": color.Color(41/255, 208/255, 208/255),
    "gray": color.Color(87/255, 87/255, 87/255),
    "green": color.Color(29/255, 105/255, 20/255),
    "purple": color.Color(129/255, 38/255, 192/255),
    "red": color.Color(173/255, 35/255, 35/255),
    "yellow": color.Color(255/255, 238/255, 5/255),
}


# the sizes of objects to sample from
CLEVR_SIZES = {
    "small": 0.7,
    "large": 1.4,
}


def default_rng():
  return np.random.RandomState()


def random_hue_color(saturation: float = 1., value: float = 1, rng=default_rng()):
  return color.Color.from_hsv(rng.uniform(), saturation, value)


def random_rotation(axis=None, rng=default_rng()):
  """ Compute a random rotation as a quaternion.
  If axis is None the rotation is sampled uniformly over all possible orientations.
  Otherwise it corresponds to a random rotation around the given axis."""

  if axis is None:
    # uniform across rotation space
    # copied from pyquat.Quaternion.random to be able to use a custom rng
    r1, r2, r3 = rng.random(3)

    q1 = np.sqrt(1.0 - r1) * (np.sin(2 * np.pi * r2))
    q2 = np.sqrt(1.0 - r1) * (np.cos(2 * np.pi * r2))
    q3 = np.sqrt(r1) * (np.sin(2 * np.pi * r3))
    q4 = np.sqrt(r1) * (np.cos(2 * np.pi * r3))

    return q1, q2, q3, q4

  else:
    if isinstance(axis, str) and axis.upper() in ["X", "Y", "Z"]:
      axis = {"X": (1., 0., 0.),
              "Y": (0., 1., 0.),
              "Z": (0., 0., 1.)}[axis.upper()]

    quat = pyquat.Quaternion(axis=axis, angle=rng.uniform(0, 2*np.pi))
    return tuple(quat)


def rotation_sampler(axis=None):
  def _sampler(obj: objects.PhysicalObject, rng):
    obj.quaternion = random_rotation(axis=axis, rng=rng)
  return _sampler


def bottom_sampler(region):
  """Sample positions at the bottom of a region"""
  region = np.array(region, dtype=np.float)

  def _sampler(obj: objects.PhysicalObject, rng):
    obj.position = (0, 0, 0)  # reset position to origin
    effective_region = region - obj.aabbox
    effective_region[1, 2] = effective_region[0, 2]  # only consider lowest Z
    obj.position = rng.uniform(*effective_region)

  return _sampler


def position_sampler(region):
  region = np.array(region, dtype=np.float)

  def _sampler(obj: objects.PhysicalObject, rng):
    obj.position = (0, 0, 0)  # reset position to origin
    effective_region = np.array(region) - obj.aabbox
    obj.position = rng.uniform(*effective_region)

  return _sampler


def resample_while(asset, samplers, condition, max_trials=100, rng=default_rng()):
  for _ in range(max_trials):
    for sampler in samplers:
      sampler(asset, rng)
    if not condition(asset):
      return
  else:
    raise RuntimeError("Failed to place", asset)


def move_until_no_overlap(asset, simulator, spawn_region=((-1, -1, -1), (1, 1, 1)), max_trials=100,
                          rng=default_rng()):
  return resample_while(asset,
                        samplers=[rotation_sampler(), position_sampler(spawn_region)],
                        condition=simulator.check_overlap,
                        max_trials=max_trials,
                        rng=rng)


def sample_color(
    strategy: str,
    rng: np.random.RandomState = default_rng()
    ) -> Tuple[Optional[str], color.Color]:
  """Sample a random color according to a given strategy.

  Args:
    strategy (str): One of
    * "clevr": Sample one of the 8 colors used in the CLEVR dataset.
      (blue, brown, cyan, gray, green, purple, red, and yellow)
    * "uniform_hue": Sample a color with value and saturation equal to one and a random hue.
    * specific color name: Do not sample randomly; Instead return the requested color directly.
      Can be one of aqua, black, blue, fuchsia, green, gray, lime, maroon, navy, olive, purple,
      red, silver, teal, white, or yellow.

  """
  if strategy == "gray":
    return "gray", color.get_color("gray")
  elif strategy == "clevr":
    color_label = rng.choice(list(CLEVR_COLORS.keys()))
    return color_label, CLEVR_COLORS[color_label]
  elif strategy == "uniform_hue":
    return None, random_hue_color(rng=rng)
  else:
    raise ValueError(f"Unknown color sampling strategy {strategy}")


def sample_sizes(
    strategy: str,
    rng: np.random.RandomState = default_rng()
  ) -> Tuple[Optional[str], float]:
  """Sample a random (asset) size according to a given strategy."""
  if strategy == "clevr":
    size_label = rng.choice(list(CLEVR_SIZES.keys()))
    size = CLEVR_SIZES[size_label]
    return size_label, size
  elif strategy == "uniform":
    return None, rng.uniform(0.7, 1.4)
  elif strategy == "const":
    return None, 1
  else:
    raise ValueError(f"Unknown size sampling strategy {strategy}")


def sample_point_in_half_sphere_shell(
    inner_radius: float,
    outer_radius: float,
    offset: float = 0.,
    rng: np.random.RandomState = default_rng()
  ) -> Tuple[float, float, float]:
  """Uniformly sample points that are in a given distance range from the origin
     and with z >= offset."""
  while True:
    # normalize(3-dim standard normal) is distributed on the unit sphere surface
    xyz = rng.normal(0, 1, (3, ))
    if xyz[2] < offset:  # if z is less than offset, rejection.
      continue
    xyz = xyz / np.linalg.norm(xyz)  # unit vector on the unit sphere surface
    # radius follows surface area of the sphere of radius r
    radius = rng.uniform(inner_radius**3, outer_radius**3) ** (1/3.)
    xyz = xyz * radius  # projected to the sphere surface of radius r
    return tuple(xyz.tolist())
