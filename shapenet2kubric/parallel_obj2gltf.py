# pylint disable=logging-fstring-interpolation
import argparse
import pathlib
import multiprocessing
import subprocess
import logging
import sys
import tqdm

from shapenet_denylist import invalid_model

# --- python3.7 needed by suprocess 'capture output'
assert sys.version_info.major>=3 and sys.version_info.minor>=7

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

logger = multiprocessing.get_logger()
logger.setLevel(logging.DEBUG)


def setup_logging(datadir=None):
  # see: see https://docs.python.org/3/library/multiprocessing.html#logging
  formatter = logging.Formatter('[%(levelname)s/%(processName)s] %(message)s')

  # --- sends DEBUG+ logs to file
  if datadir is not None:
    logpath = pathlib.Path(datadir)/'shapenet2kubric.log'
    fh = logging.FileHandler(logpath)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

  # --- send WARNING+ logs to console
  sh = logging.StreamHandler()
  sh.setLevel((logging.WARNING, logging.DEBUG)[datadir is None])
  sh.setFormatter(formatter)
  logger.addHandler(sh)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def shapenet_objects_dirs(datadir: pathlib.Path):
  """Returns a list of pathlib.Path folders, one per object."""
  object_folders = list()
  categories = [x for x in pathlib.Path(datadir).iterdir() if x.is_dir()]
  for category in categories:
    object_folders += [x for x in category.iterdir() if x.is_dir()]
  # [print(folder) for folder in object_folders]
  return object_folders

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def process_object(object_folder):
  object_folder = pathlib.Path(object_folder)
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

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def parallel_launch(datadir, num_processes):
  # --- gathers collection over which threads will be executed
  object_folders = shapenet_objects_dirs(datadir)

  # --- launches jobs in parallel
  with tqdm.tqdm(total=len(object_folders)) as pbar:
    with multiprocessing.Pool(num_processes) as pool:
      for counter, _ in enumerate(pool.imap(process_object, object_folders)):
        pbar.update(1)
        logger.debug(f"Processed {counter}/{len(object_folders)}")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--datadir', default='/ShapeNetCore.v2')
  parser.add_argument('--num_processes', default=8, type=int)
  args = parser.parse_args()

  # --- choose what to execute
  if True:
    setup_logging(args.datadir)
    parallel_launch(args.datadir, args.num_processes)
 
  if False:
    setup_logging(datadir=None)
    process_object(pathlib.Path(args.datadir)/'03046257/5972bc07e59371777bcb070cc655f13a')
    
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# NOTE: if you want even more performance, this could be used
# import fcntl
# import time
# try:
#   with open('foo.txt', 'w+') as fd:
#     fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
#     time.sleep(5)
#     fcntl.flock(fd, fcntl.LOCK_UN)
# except BlockingIOError:
#   print("file was busy")