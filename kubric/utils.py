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

import sys
import os
import argparse
import logging
import pprint
import pathlib
import numpy as np

from kubric.simulator.pybullet import _pybullet_logs
from kubric.renderer.blender import _blender_logs

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

  def parse_args(self, args=None, namespace=None):
    # --- parse argument in a way compatible with blender's REPL
    if args is not None and "--" in sys.argv:
      args=sys.argv[sys.argv.index("--")+1:]
      flags = super(ArgumentParser, self).parse_args(args=args, namespace=namespace)
    else:
      flags = super(ArgumentParser, self).parse_args(args=args)
    return flags


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

def setup_logging(logging_level):
  logging.basicConfig(level=logging_level)
  logger.info(f"PyBullet stdout redirected to: {_pybullet_logs}")
  logger.info(f"Blender stdout redirected to:  {_blender_logs}")


def log_my_flags(flags):
  flags_string = pprint.pformat(vars(flags), indent=2, width=100)
  logger.debug(flags_string)


