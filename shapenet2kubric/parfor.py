# pylint disable=logging-fstring-interpolation
import argparse
from pathlib import Path
import multiprocessing
import logging
import sys
import tqdm

# --- python3.7 needed by suprocess 'capture output'
assert sys.version_info.major>=3 and sys.version_info.minor>=7

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

logger = multiprocessing.get_logger()
logger.setLevel(logging.DEBUG)


def setup_logging(datadir:Path, functor_module):
  # see: see https://docs.python.org/3/library/multiprocessing.html#logging
  formatter = logging.Formatter('[%(levelname)s/%(processName)s] %(message)s')

  # --- sends DEBUG+ logs to file
  logpath = datadir/'shapenet2kubric.log'
  fh = logging.FileHandler(logpath)
  fh.setLevel(logging.DEBUG)
  fh.setFormatter(formatter)
  logger.addHandler(fh)

  # --- send WARNING+ logs to console
  sh = logging.StreamHandler()
  sh.setLevel(logging.WARNING)
  sh.setFormatter(formatter)
  logger.addHandler(sh)

  # TODO: remove
  # --- bind logger to the functor
  # functor_module.set_logger(logger)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def shapenet_objects_dirs(datadir: Path):
  """Returns a list of pathlib.Path folders, one per object."""
  object_folders = list()
  categories = [x for x in Path(datadir).iterdir() if x.is_dir()]
  for category in categories:
    object_folders += [x for x in category.iterdir() if x.is_dir()]
  # [print(folder) for folder in object_folders]
  return object_folders

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def parfor(collection, functor, num_processes):
  # --- launches jobs in parallel
  with tqdm.tqdm(total=len(collection)) as pbar:
    with multiprocessing.Pool(num_processes) as pool:
      for counter, _ in enumerate(pool.imap(functor, collection)):
        logger.debug(f"Processed {counter}/{len(collection)}")
        pbar.update(1)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--datadir', default='/ShapeNetCore.v2')
  parser.add_argument('--num_processes', default=8, type=int)
  parser.add_argument('--functor_module', default='obj2gltf')
  args = parser.parse_args()

  datadir = Path(args.datadir)
  
  # TODO: importlib
  from obj2gltf import functor
  # functor = lambda x: functor_module(x, logger)
  # setup_logging(datadir, functor_module)

  collection = shapenet_objects_dirs(args.datadir)
  parfor(collection, functor, args.num_processes)

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