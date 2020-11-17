import ctypes
import os
import sys


class RedirectStream(object):
  """Usage:
  with RedirectStream(sys.stdout, filename="stdout.txt"):
    print("commands will have stdout directed to file")
  """

  @staticmethod
  def _flush_c_stream():
    libc = ctypes.CDLL(None)
    libc.fflush(None)

  def __init__(self, stream=sys.stdout, filename=os.devnull):
    self.stream = stream
    self.filename = filename

  def __enter__(self):
    self.stream.flush()  # ensures python stream unaffected 
    self.fd = open(self.filename, "w+")
    self.dup_stream = os.dup(self.stream.fileno())
    os.dup2(self.fd.fileno(), self.stream.fileno()) # replaces stream
  
  def __exit__(self, type, value, traceback):
    RedirectStream._flush_c_stream()  # ensures C stream buffer empty
    os.dup2(self.dup_stream, self.stream.fileno()) # restores stream
    os.close(self.dup_stream)
    self.fd.close()
