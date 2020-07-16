import argparse
import os
import glob
import random
import logging
import tempfile
from google.cloud import storage
from google.cloud.storage.blob import Blob
import pathlib
from simulator import Simulator

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class AssetSource(object):
  # see: https://googleapis.dev/python/storage/latest

  def download_manifest(self, path):
    logging.info("Downloading manifest: {}".format(path))
    blob = Blob.from_string(path)
    string = blob.download_as_string(client=self.client)
    lines = [line.decode('utf-8') for line in string.splitlines()]
    return lines

  def copy_folder(self, source: str):
    remote_folder = self.bucket_name+"/"+source
    local_folder = os.path.join(self.tempdir, source)
    logging.info("Copying '{}' to '{}'".format(remote_folder, local_folder))
    remote_blobs = self.bucket.list_blobs(prefix=source)
    for remote_blob in remote_blobs:
      local_blob = os.path.join(self.tempdir, remote_blob.name)  #< where to download
      pathlib.Path(local_blob).parent.mkdir(parents=True, exist_ok=True)  #< parents must exist 
      remote_blob.download_to_filename(local_blob)
    return local_folder

  def __init__(self, bucket_name:str, path: str):
    assert bucket_name.startswith("gs://")
    self.path = path
    self.bucket_name = bucket_name
    self.tempdir = tempfile.mkdtemp()
    self.client = storage.Client()
    self.bucket = self.client.get_bucket(bucket_name[5:])
    self.manifest = self.download_manifest(bucket_name+"/"+path+"/manifest.txt")
    
  def take(self, num_objects: int):
    # --- pick a few models
    remote_folders = random.sample(self.manifest, num_objects)
    # --- mirror samples locally
    local_folders = [self.copy_folder(self.path+"/"+folder) for folder in remote_folders]
    # --- fetch URDF files in the copied folders
    urdf_paths = [glob.glob(folder+"/*.urdf") for folder in local_folders]
    # --- TODO: unchecked assumption one URDF per folder!!!
    urdf_paths = [urdf_path[0] for urdf_path in urdf_paths]
    return urdf_paths

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class Placer(object):
  # TODO: rename to "Initializer?"
  def __init__(self, source: AssetSource, template: str):
    assert template == "sphereworld"

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class Renderer(object):
  def __init__(self, framerate: int):
    pass

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--template", type=str, default="sphereworld")
parser.add_argument("--num_objects", type=int, default=3)
parser.add_argument("--frame_rate", type=int, default=24)
parser.add_argument("--step_rate", type=int, default=240)
parser.add_argument("--logging_level", type=str, default="INFO")
FLAGS = parser.parse_args()

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

logging.basicConfig(level=FLAGS.logging_level)

# --- Download a few models locally
source = AssetSource(bucket_name="gs://kubric", path="katamari")
urdf_paths = source.take(FLAGS.num_objects)

# --- load models in the simulator
simulator = Simulator(frame_rate=FLAGS.frame_rate, step_rate=FLAGS.step_rate)
for urdf_path in urdf_paths:
  simulator.load_object(urdf_path)

# placer = Placer(template=FLAGS.template)
# renderer = Renderer(FLAGS.frame_rate)