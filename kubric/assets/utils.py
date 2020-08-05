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

import numpy as np
import sklearn.utils


def mm3hash(name):
  """ Compute the uint32 hash that Blenders Cryptomatte uses.
  https://github.com/Psyop/Cryptomatte/blob/master/specification/cryptomatte_specification.pdf
  """
  hash_32 = sklearn.utils.murmurhash3_32(name, positive=True)
  exp = hash_32 >> 23 & 255
  if (exp == 0) or (exp == 255):
    hash_32 ^= 1 << 23
  return hash_32


def random_rotation(rnd=np.random.RandomState()):
  """ Compute a random rotation as a quaternion that is uniform over orientations."""

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
