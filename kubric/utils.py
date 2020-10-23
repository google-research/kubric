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

class ArgumentParser(argparse.ArgumentParser):
  def __init__(self, *args, **kwargs):
    argparse.ArgumentParser.__init__(self, *args, **kwargs)

    # --- default arguments for kubric
    self.add_argument("--frame_rate", type=int, default=24)
    self.add_argument("--step_rate", type=int, default=240)
    self.add_argument("--frame_start", type=int, default=1)
    self.add_argument("--frame_end", type=int, default=24)  # 1 second
    self.add_argument("--logging_level", type=str, default="INFO")
    self.add_argument("--work_dir", type=str, default="./output/work_dir")
    self.add_argument("--output_dir", type=str, default="./output")
    self.add_argument("--seed", type=int, default=0)
    self.add_argument("--width", type=int, default=512)
    self.add_argument("--height", type=int, default=512)
    self.add_argument("--max_placement_trials", type=int, default=100)
    self.add_argument("--asset_source", action="append", default=[],
                        help="add an additonal source of assets using a URI "
                             "e.g. '.Assets/KLEVR' or 'gs://kubric/GSO'."
                             "Can be passed multiple times.")

  def parse_args(self):
    # --- parse argument in a way compatible with blender's REPL
    if "--" in sys.argv:
      flags = super(ArgumentParser, self).parse_args(args=sys.argv[sys.argv.index("--")+1:])
    else:
      flags = super(ArgumentParser, self).parse_args(args=[])
    return flags

def setup_loggin(flags):
  pass