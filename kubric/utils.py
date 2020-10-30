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
import ctypes
import argparse
import logging
import pprint
import pathlib
import numpy as np

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

def log_my_flags(flags):
  flags_string = pprint.pformat(vars(flags), indent=2, width=100)
  logger.info(flags_string)

# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

class RedirectStream(object):
  """Usage:
  with RedirectStream(sys.stdout, filename="stdout.txt"):
    print("commands will have stdout directed to file")
  """

  @staticmethod
  def _flush_c_stream(stream):
    streamname = stream.name[1:-1]
    libc = ctypes.CDLL(None)
    libc.fflush(ctypes.c_void_p.in_dll(libc, streamname))

  def __init__(self, stream=sys.stdout, filename=os.devnull):
    self.stream = stream
    self.filename = filename

  def __enter__(self):
    self.stream.flush()  # ensures python stream unaffected 
    self.fd = open(self.filename, "w+")
    self.dup_stream = os.dup(self.stream.fileno())
    os.dup2(self.fd.fileno(), self.stream.fileno()) # replaces stream
  
  def __exit__(self, type, value, traceback):
    RedirectStream._flush_c_stream(self.stream)  # ensures C stream buffer empty
    os.dup2(self.dup_stream, self.stream.fileno()) # restores stream
    os.close(self.dup_stream)
    self.fd.close()