# pylint: disable=logging-fstring-interpolation

import sys
import argparse
from pathlib import Path
from shapenet_denylist import invalid_model
import subprocess
import logging
import multiprocessing

def functor(object_folder:str, logger=multiprocessing.get_logger()):
  object_folder = Path(object_folder)

  logger.debug(f'obj2manifold running on "{object_folder}"')
  source_path = object_folder / 'models' / 'model_normalized.obj'
  target_path = object_folder / 'models' / 'model_watertight.obj'
  cmd = f'manifold --input {source_path} --output {target_path}'

  try:
    retobj = subprocess.run(cmd, capture_output=True, check=True, shell=True, text=True)
    if retobj.stderr != "": logger.debug(retobj.stderr)
    if retobj.stdout != "": logger.debug(retobj.stdout)
  except subprocess.CalledProcessError:
    logger.error(f'FAILED: "{cmd}"')

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--datadir', default='/ShapeNetCore.v2')
  parser.add_argument('--model', default='02933112/718f8fe82bc186a086d53ab0fe94e911')
  args = parser.parse_args()

  # --- setup logger
  stdout_logger = logging.getLogger(__name__)
  stdout_logger.setLevel(logging.DEBUG)
  handler = logging.StreamHandler(sys.stdout)
  stdout_logger.addHandler(handler)

  # --- execute functor
  functor(Path(args.datadir)/args.model, stdout_logger)