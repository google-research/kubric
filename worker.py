import argparse
import os
import re
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

  def __init__(self, path):
    self.path = path

    if self.path.startswith("gs://"):
      bucket_name, prefix = re.findall("gs://(.*)/(.*)", self.path)[0]
      self.bucket_name, self.prefix = bucket_name, prefix
      self.local_temp_folder = tempfile.mkdtemp()
      self.client = storage.Client()
      self.bucket = self.client.get_bucket(self.bucket_name)
      self.manifest = self.download_manifest("gs://"+bucket_name+"/"+prefix+"/manifest.txt")
      self.manifest = ["gs://"+bucket_name+"/"+prefix+"/"+line for line in self.manifest]
    else:
      # TODO: if manifest does not exist
      # self.manifest = glob.glob(path+"/**/*.urdf")
      with open(os.path.join(path, "manifest.txt")) as f:
        self.manifest = f.read().splitlines()

    
  def take(self, num_objects: int):
    if self.path.startswith("gs://"):
      # --- pick a few, and mirror locally
      remote_folders = random.sample(self.manifest, num_objects)
      local_folders = [self.copy_folder(folder) for folder in remote_folders]
    else:
      # --- pick a few local models
      local_folders = random.sample(self.manifest, num_objects)
      local_folders = [os.path.join(self.path,folder) for folder in local_folders]
    
    # --- fetch URDF files in the folders
    urdf_paths = [glob.glob(folder+"/*.urdf") for folder in local_folders]
    # --- TODO: unchecked assumption one URDF per folder!!!
    urdf_paths = [urdf_path[0] for urdf_path in urdf_paths]
    return urdf_paths

  def download_manifest(self, remote_path):
    assert remote_path.startswith("gs://")
    logging.info("Downloading manifest: {}".format(remote_path))
    blob = Blob.from_string(remote_path)
    string = blob.download_as_string(client=self.client)
    lines = [line.decode('utf-8') for line in string.splitlines()]
    return lines

  def copy_folder(self, remote_path: str):
    assert remote_path.startswith("gs://")
    remote_subfolder_name = remote_path.replace(self.path+"/","")
    local_folder = os.path.join(self.local_temp_folder, remote_subfolder_name)
    logging.info("Copying '{}' to '{}'".format(remote_path, local_folder))
    remote_blobs = self.bucket.list_blobs(prefix=self.prefix+"/"+remote_subfolder_name)
    for remote_blob in remote_blobs:
      local_blob_name = remote_blob.name.replace(self.prefix+"/","")
      local_blob_path = os.path.join(self.local_temp_folder, local_blob_name)  #< where to download
      pathlib.Path(local_blob_path).parent.mkdir(parents=True, exist_ok=True)  #< parents must exist 
      remote_blob.download_to_filename(local_blob_path)
    return local_folder

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class Placer(object):
  # TODO: rename to "Initializer?"
  def __init__(self, template: str=None, simulator: Simulator=None):
    assert template == "sphereworld"
    self.simulator = simulator
    # TODO: where to store planar geometry?
    # self.simulator.load_object("urdf/plane.urdf")

  def place(self, object_id: int):
    # TODO: brutally hardcoded implementation
    if object_id==0: self.simulator.place_object(object_id, position=(-.2, 0, 1))
    if object_id==1: self.simulator.place_object(object_id, position=(+.0, 0, 1))
    if object_id==2: self.simulator.place_object(object_id, position=(+.2, 0, 1))
    # TODO: self.simlator.set_velocity(...)

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
# TODO: improve to handle both gs:// and local folders transparently
asset_source = AssetSource(path="gs://kubric/katamari")
# asset_source = AssetSource(path="katamari")
urdf_paths = asset_source.take(FLAGS.num_objects)

# --- load models & place them in the simulator
simulator = Simulator(frame_rate=FLAGS.frame_rate, step_rate=FLAGS.step_rate)
placer = Placer(template=FLAGS.template, simulator=simulator)
for urdf_path in urdf_paths:
  object_id = simulator.load_object(urdf_path)
  placer.place(object_id)

# --- run the simulation
animation = simulator.run(1)
# renderer = Renderer(FLAGS.frame_rate)