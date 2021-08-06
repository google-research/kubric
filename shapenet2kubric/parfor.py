# pylint: disable=logging-fstring-interpolation
import argparse
from pathlib import Path
import multiprocessing
import logging
import sys
import tqdm
from shapenet_denylist import invalid_model
from shapenet_denylist import __shapenet_list__
from convert2 import functor
from datetime import datetime

# --- python3.7 needed by suprocess 'capture output'
assert sys.version_info.major>=3 and sys.version_info.minor>=7

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

logger = multiprocessing.get_logger()
logger.setLevel(logging.DEBUG)

def setup_logging(datadir:str):
  # see: see https://docs.python.org/3/library/multiprocessing.html#logging
  formatter = logging.Formatter('[%(levelname)s/%(processName)s] %(message)s')

  # --- sends DEBUG+ logs to file
  datadir = Path(args.datadir)
  logpath = datadir / 'shapenet2kubric.log'
  fh = logging.FileHandler(logpath, mode='w')
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

def shapenet_objects_dirs(datadir: str):
  """Returns a list of pathlib.Path folders, one per object."""
  taxonomy_path = Path(datadir) / 'taxonomy.json'
  if not taxonomy_path.is_file():
    logging.fatal(f'Verify that "{str(datadir)}" is a valid shapenet folder, as the taxonomy file was not found at "{str(taxonomy_path)}"')

  logging.info(f"gathering shapenet folders: {datadir}")
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

class Functor(object):
  def __init__(self, stages):
    self.stages = stages
    self.logger = multiprocessing.get_logger()
  def __call__(self, object_folder):
    functor(object_folder, self.stages, self.logger)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--datadir', default='/ShapeNetCore.v2')
  parser.add_argument('--num_processes', default=8, type=int)
  parser.add_argument('--stop_after', default=0, type=int)
  parser.add_argument('--stages', nargs='+', default=["0", "1", "2", "3", "4", "5"])
  args = parser.parse_args()

  # --- specify and communicate logging policy
  setup_logging(args.datadir)

  # --- fetch which stages to execute
  stages = [int(stage) for stage in args.stages]
  functor_with_stages = Functor(stages)

  # --- collect folders over which parfor will be executed
  collection = shapenet_objects_dirs(args.datadir)

  # --- trim the parfor collection (for quick dry-run)
  if args.stop_after != 0: 
    collection = collection[0:args.stop_after]

  # --- launch
  logger.info(f'starting parfor on {args.datadir} at {str(datetime.now())}')
  parfor(collection, functor_with_stages, args.num_processes)

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