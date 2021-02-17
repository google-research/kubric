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


import argparse
import logging
import pathlib
import pprint
import shutil
import sys
import tempfile

import tensorflow_datasets.public_api as tfds

logger = logging.getLogger(__name__)


class ArgumentParser(argparse.ArgumentParser):
  def __init__(self, *args, **kwargs):
    argparse.ArgumentParser.__init__(self, *args, **kwargs)

    # --- default arguments for kubric
    self.add_argument("--frame_rate", type=int, default=24)
    self.add_argument("--step_rate", type=int, default=240)
    self.add_argument("--frame_start", type=int, default=1)
    self.add_argument("--frame_end", type=int, default=24)  # 1 second
    self.add_argument("--logging_level", type=str, default="INFO")
    self.add_argument("--random_seed", type=int, default=0)
    self.add_argument("--width", type=int, default=512)
    self.add_argument("--height", type=int, default=512)
    self.add_argument("--scratch_dir", type=str, default=None)
    self.add_argument("--job-dir", type=str, default="output")

  def parse_args(self, args=None, namespace=None):
    # --- parse argument in a way compatible with blender's REPL
    if args is not None and "--" in sys.argv:
      args = sys.argv[sys.argv.index("--")+1:]
      flags = super(ArgumentParser, self).parse_args(args=args, namespace=namespace)
    else:
      flags = super(ArgumentParser, self).parse_args(args=args)
    return flags


def setup_directories(FLAGS):
  if FLAGS.scratch_dir is None:
    scratch_dir = tfds.core.as_path(tempfile.mkdtemp())
  else:
    scratch_dir = tfds.core.as_path(FLAGS.scratch_dir)
    if scratch_dir.exists():
      logging.info("Deleting content of old scratch-dir: %s", scratch_dir)
      shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True)
  logging.info("Using scratch directory: %s", scratch_dir)

  output_dir = tfds.core.as_path(FLAGS.job_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  logging.info("Using output directory: %s", output_dir)
  return scratch_dir, output_dir


def is_local_path(path):
  """ Determine if a given path is local or remote. """
  first_part = pathlib.Path(path).parts[0]
  if first_part.endswith(':') and len(first_part) > 2:
    return False
  else:
    return True


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

def setup_logging(logging_level):
  logging.basicConfig(level=logging_level)


def log_my_flags(flags):
  flags_string = pprint.pformat(vars(flags), indent=2, width=100)
  logger.info(flags_string)
