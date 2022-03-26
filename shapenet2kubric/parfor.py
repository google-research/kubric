# Copyright 2022 The Kubric Authors.
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

# Copyright 2022 The Kubric Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=logging-fstring-interpolation
import argparse
from pathlib import Path
import multiprocessing
import logging
import sys
import tqdm
import json
from datetime import datetime
from shapenet_denylist import invalid_model
from shapenet_denylist import __shapenet_set__

from convert import stage0
from convert import stage1
from convert import stage2
from convert import stage3
from convert import stage4
from convert import stage5
from convert import stage6

# --- python3.7 needed by subprocess 'capture output'
assert sys.version_info.major >= 3 and sys.version_info.minor >= 7

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

logger = multiprocessing.get_logger()
logger.setLevel(logging.DEBUG)


def setup_logging(datadir: str):
  # see: see https://docs.python.org/3/library/multiprocessing.html#logging
  formatter = logging.Formatter('[%(levelname)s/%(processName)s] %(message)s')

  # --- sends DEBUG+ logs to file
  datadir = Path(datadir)
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
    raise RuntimeError(f'Verify that "{str(datadir)}" is a valid shapenet '
                       f'folder, as the taxonomy file was not found at '
                       f'"{str(taxonomy_path)}"')

  logger.info(f"gathering shapenet folders: {datadir}")
  object_folders = list()
  categories = [x for x in Path(datadir).iterdir() if x.is_dir()]
  for category in categories:
    object_folders += [x for x in category.iterdir() if x.is_dir()]
  logging.debug(f"gathered object folders: {object_folders}")

  # --- remove invalid folders
  logger.debug(f"dropping problematic folders: {__shapenet_set__}")
  object_folders = [folder for folder in object_folders if not invalid_model(folder)]

  # TODO: add objects w/o materials to the denylist?
  # if not os.path.exists(obj_path.replace('.obj', '.mtl')):

  return object_folders

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


def parfor(collection, functor, num_processes, manifest_path):
  # --- launches jobs in parallel
  dataset_properties = list()
  with tqdm.tqdm(total=len(collection)) as pbar:
    with multiprocessing.Pool(num_processes, maxtasksperchild=1) as pool:
      for counter, properties in enumerate(pool.imap_unordered(functor, collection)):
        logger.debug(f"Processed {counter}/{len(collection)}")
        dataset_properties.append(properties)
        pbar.update(1)

  # --- writes cumulative properties to the manifest
  invalid_manifest = any([properties is None for properties in dataset_properties])
  if invalid_manifest:
    logger.error(f'Cannot compile manifest from invalid properties')
    logger.debug(f'properties dump: \n {dataset_properties}')
  else:
    logger.info(f"Dumping aggregated information to {manifest_path}")
    with open(manifest_path, 'w') as fp:
      json.dump(dataset_properties, fp, indent=4, sort_keys=True)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


class Functor(object):
  def __init__(self, stages):
    self.stages = stages
    self.logger = multiprocessing.get_logger()
    
  def __call__(self, object_folder:str):
    self.logger.debug(f'pipeline running on "{object_folder}"')
    object_folder = Path(object_folder)
  
    stages = self.stages
    logger = self.logger
    properties = None
    
    try:
      if 0 in stages: stage0(object_folder, logger)
      if 1 in stages: stage1(object_folder, logger)
      if 2 in stages: stage2(object_folder, logger)
      if 3 in stages: stage3(object_folder, logger)
      if 4 in stages: properties = stage4(object_folder, logger)
      if 5 in stages: stage5(object_folder, logger)
      if 6 in stages: stage6(object_folder, logger)
      return properties

    except Exception as e:
      import traceback

      logger.error(f'Pipeline exception on "{object_folder}"')
      logger.error(f'Exception details: {str(e)} {traceback.format_tb(e.__traceback__)}')

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--datadir', default='/ShapeNetCore.v2')
  parser.add_argument('--num_processes', default=48, type=int)
  parser.add_argument('--stop_after', default=0, type=int)
  parser.add_argument('--stages', nargs='+', default=["0", "1", "2", "3", "4", "5", "6"])
  args = parser.parse_args()

  # --- specify and communicate logging policy
  setup_logging(args.datadir)
  logging.getLogger("trimesh").setLevel(logging.ERROR)

  # --- fetch which stages to execute
  stages = [int(stage) for stage in args.stages]
  functor_with_stages = Functor(stages)

  # --- collect folders over which parfor will be executed
  collection = shapenet_objects_dirs(args.datadir)
  manifest_path = Path(args.datadir) / 'manifest.json'
  
  # --- trim the parfor collection (for quick dry-run)
  if args.stop_after != 0: 
    collection = collection[0:args.stop_after]
  
  # --- launch
  logger.info(f'starting parfor on {args.datadir} at {str(datetime.now())}')
  parfor(collection, functor_with_stages, args.num_processes, manifest_path)
