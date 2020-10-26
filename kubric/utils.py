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
import argparse
import pprint
import logging
import numpy as np


class ArgumentParser(argparse.ArgumentParser):
  def __init__(self, *args, **kwargs):
    argparse.ArgumentParser.__init__(self, *args, **kwargs)

    # --- default arguments for kubric
    self.add_argument("--frame_rate", type=int, default=24)
    self.add_argument("--step_rate", type=int, default=240)
    self.add_argument("--frame_start", type=int, default=1)
    self.add_argument("--frame_end", type=int, default=24)  # 1 second
    self.add_argument("--logging_level", type=str, default="INFO")
    self.add_argument("--output_dir", type=str, default="./output")
    self.add_argument("--random_seed", type=int, default=0)
    self.add_argument("--width", type=int, default=512)
    self.add_argument("--height", type=int, default=512)
    self.add_argument("--norender", type=int, default=240)

  def parse_args(self):
    # --- parse argument in a way compatible with blender's REPL
    if "--" in sys.argv:
      flags = super(ArgumentParser, self).parse_args(args=sys.argv[sys.argv.index("--")+1:])
    else:
      flags = super(ArgumentParser, self).parse_args(args=[])
    return flags

# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

import logging
def setup_logging(logging_level):
  logging.basicConfig(level=logging_level)
  # log = logging.getLogger(__name__)  #TODO: why is this necessary?

def log_my_flags(flags):
  flags_string = pprint.pformat(vars(flags), indent=2, width=100)
  logging.info(flags_string)

# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

