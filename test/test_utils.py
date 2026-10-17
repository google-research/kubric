# Copyright 2026 The Kubric Authors.
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

"""Tests for kubric.utils, in particular the seed handling of utils.setup."""

import argparse

import numpy as np

from kubric import utils


def make_flags(seed, tmp_path):
  return argparse.Namespace(
      logging_level="WARNING",
      seed=seed,
      resolution=(32, 32),
      frame_start=1,
      frame_end=2,
      frame_rate=2,
      step_rate=24,
      scratch_dir=str(tmp_path / "scratch"),
      job_dir=str(tmp_path / "output"),
  )


def test_setup_with_seed_zero_is_deterministic(tmp_path):
  """Regression test for https://github.com/google-research/kubric/issues/181.

  An explicit `--seed=0` used to be discarded by a truthiness check and was
  silently replaced by a random seed.
  """
  scene, rng, _, _ = utils.setup(make_flags(seed=0, tmp_path=tmp_path))
  assert scene.metadata["seed"] == 0
  expected_rng = np.random.RandomState(seed=0)
  for _ in range(10):
    assert rng.randint(0, 2**31) == expected_rng.randint(0, 2**31)


def test_setup_with_explicit_seed(tmp_path):
  scene, rng, _, _ = utils.setup(make_flags(seed=42, tmp_path=tmp_path))
  assert scene.metadata["seed"] == 42
  expected_rng = np.random.RandomState(seed=42)
  for _ in range(10):
    assert rng.randint(0, 2**31) == expected_rng.randint(0, 2**31)


def test_setup_without_seed_samples_random_seed(tmp_path):
  scene, _, _, _ = utils.setup(make_flags(seed=None, tmp_path=tmp_path))
  seed = scene.metadata["seed"]
  assert isinstance(seed, (int, np.integer))
  assert 0 <= seed < 2147483647
