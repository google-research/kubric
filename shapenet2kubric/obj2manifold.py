# pylint disable=logging-fstring-interpolation

import sys
import argparse
from pathlib import Path
from shapenet_denylist import invalid_model
import subprocess
import logging
import multiprocessing

def functor(object_folder:str, logger=multiprocessing.get_logger()):
  object_folder = Path(object_folder)
  logger.info(f'Dummy run of obj2manifold on {object_folder}')

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--datadir', default='/ShapeNetCore.v2')
  parser.add_argument('--model', default='03046257/5972bc07e59371777bcb070cc655f13a')
  args = parser.parse_args()

  # --- setup logger
  logger = logging.getLogger(__name__)
  logger.setLevel(logging.DEBUG)
  handler = logging.StreamHandler(sys.stdout)
  logger.addHandler(handler)

  # --- execute functor
  functor(Path(args.datadir)/args.model, logger)