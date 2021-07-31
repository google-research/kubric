#!/usr/local/bin/python3
import sys
import argparse
from pathlib import Path
from shapenet_denylist import invalid_model
import subprocess
import logging
import multiprocessing

def functor(object_folder:str, logger=multiprocessing.get_logger()):
  object_folder = Path(object_folder)

  # --- check object folder is appropriate
  if invalid_model(object_folder): 
    logger.debug(f'skipping denylist model "{object_folder}"')
    return

  # TODO: add option to delete file if exists?
  # see: https://docs.python.org/3/library/subprocess.html
  source_path = object_folder / 'models' / 'model_normalized.obj'
  target_path = object_folder / 'models' / 'model_normalized.glb'
  if not source_path.is_file():
    logger.error(f'The source path "{source_path}" is not a file?')
  logger.debug(f'conversion of "{object_folder}"')
  cmd = f'obj2gltf -i {source_path} -o {target_path}'.split(' ')
  try:
    retobj = subprocess.run(cmd, capture_output=True, check=True)

    # --- catch (silent) errors in stdout
    stdout = retobj.stdout.decode('utf-8')
    silent_error = not (stdout.startswith('Total') and stdout.endswith('ms\n'))
    if silent_error:
      logger.error(f'{stdout}')
    else:
      logger.debug(f'{stdout}')

    # --- if there is stdout, log it
    if retobj.stderr != "":
      logger.debug(retobj.stderr.decode('utf-8'))
    

  except subprocess.CalledProcessError:
    logger.error(f'obj2gltf on "{target_path}" failed.')

  # --- verify post-hypothesis
  if not target_path.is_file():
    logger.error(f'The output "{target_path}" was not written.')

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--datadir', default='/ShapeNetCore.v2')
  parser.add_argument('--model')
  args = parser.parse_args()

  # --- setup logger
  logger = logging.getLogger(__name__)
  logger.setLevel(logging.DEBUG)
  handler = logging.StreamHandler(sys.stdout)
  logger.addHandler(handler)

  # --- execute functor
  functor(Path(args.datadir)/args.model, logger)