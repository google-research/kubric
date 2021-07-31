#!/usr/local/bin/python3
from pathlib import Path
from shapenet_denylist import invalid_model
import subprocess
import logging
import multiprocessing

def functor(object_folder:str, logger=multiprocessing.get_logger()):
  # --- check object folder is appropriate
  object_folder = Path(object_folder)
  if invalid_model(object_folder): 
    logger.debug(f'skipping denylist model "{object_folder}"')

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
  logger = logging.getLogger(__name__)
  functor('/ShapeNetCore.v2/03046257/5972bc07e59371777bcb070cc655f13a', logger)