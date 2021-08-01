# pylint: disable=logging-fstring-interpolation
import argparse
from pathlib import Path
import multiprocessing
import logging
import sys
import tqdm
import importlib
from shapenet_denylist import invalid_model
from shapenet_denylist import __shapenet_list__

# --- python3.7 needed by suprocess 'capture output'
assert sys.version_info.major>=3 and sys.version_info.minor>=7

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

logger = multiprocessing.get_logger()
logger.setLevel(logging.DEBUG)


def setup_logging(datadir:str, functor_module:str):
  # see: see https://docs.python.org/3/library/multiprocessing.html#logging
  formatter = logging.Formatter('[%(levelname)s/%(processName)s] %(message)s')

  # --- sends DEBUG+ logs to file
  datadir = Path(args.datadir)
  logpath = datadir/f'shapenet2kubric.{functor_module}.log'
  fh = logging.FileHandler(logpath)
  fh.setLevel(logging.DEBUG)
  fh.setFormatter(formatter)
  logger.addHandler(fh)

  # --- send WARNING+ logs to console
  sh = logging.StreamHandler()
  sh.setLevel(logging.WARNING)
  sh.setFormatter(formatter)
  logger.addHandler(sh)

  # --- inform
  logging.warning(f'logging DEBUG+ to "{logpath}"')
  logging.warning(f'logging WARNING+ to stderr')

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def shapenet_objects_dirs(datadir: Path):
  """Returns a list of pathlib.Path folders, one per object."""

  logging.info("gathering shapenet folders: {__shapenet_list__}")
  object_folders = list()
  categories = [x for x in Path(datadir).iterdir() if x.is_dir()]
  for category in categories:
    object_folders += [x for x in category.iterdir() if x.is_dir()]
  logging.debug(f"gathering folders: {object_folders}")

  # --- remove invalid folders
  logging.debug(f"dropping problemantic folders: {__shapenet_list__}")
  object_folders = [folder for folder in object_folders if not invalid_model(folder) ]

  # TODO: add objects w/o materials to the denylist?
  # if not os.path.exists(obj_path.replace('.obj', '.mtl')):

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
  parser.add_argument('--functor_module', default='obj2gltf')
  parser.add_argument('--datadir', default='/ShapeNetCore.v2')
  parser.add_argument('--num_processes', default=8, type=int)
  parser.add_argument('--stop_after', default=0, type=int)
  args = parser.parse_args()
  
  # TODO: importlib
  functor_module = importlib.import_module(args.functor_module)
  setup_logging(args.datadir, args.functor_module)
  collection = shapenet_objects_dirs(args.datadir)
  if args.stop_after != 0: 
    collection = collection[0:args.stop_after]
  parfor(collection, functor_module.functor, args.num_processes)

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