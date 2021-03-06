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

import ctypes
import os


class RedirectStream(object):
  """
  Usage:
    with RedirectStream(sys.stdout, filename="stdout.txt"):
      print("commands will have stdout directed to file")
  """

  @staticmethod
  def _flush_c_stream():
    libc = ctypes.CDLL(None)
    libc.fflush(None)

  def __init__(self, stream, filename=os.devnull):
    self.stream = stream
    self.filename = filename

  def __enter__(self):
    try:
      self.stream.flush()  # ensures python stream unaffected
      self.fd = open(self.filename, "w+")
      self.dup_stream = os.dup(self.stream.fileno())
      os.dup2(self.fd.fileno(), self.stream.fileno())  # replaces stream
    except Exception as e:
      # TODO: redirect stream breaks in jupyter notebooks.
      #       This try except is a hacky workaround
      print(e)
  
  def __exit__(self, type, value, traceback):
    try:
      RedirectStream._flush_c_stream()  # ensures C stream buffer empty
      os.dup2(self.dup_stream, self.stream.fileno())  # restores stream
      os.close(self.dup_stream)
      self.fd.close()
    except:
      # TODO: redirect stream breaks in jupyter notebooks.
      #       This try except is a hacky workaround
      pass
