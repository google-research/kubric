# Copyright 2024 The Kubric Authors.
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

# pylint: disable=line-too-long, unexpected-keyword-arg
import dataclasses
import json
import logging
from typing import List, Dict, Union

from etils import epath
import imageio
import numpy as np
import png
import tensorflow as tf
import tensorflow_datasets.public_api as tfds


_DESCRIPTION = """
Very similar to MOVi-E, except that it adds a random amount of motion blur.
A simple rigid-body simulation with GSO objects and an HDRI background.
The scene consists of a dome (half-sphere) onto which a random HDRI is projected, 
which acts as background, floor and lighting.
The scene contains between 10 and 20 random static objects, and between 1 and 3
dynamic objects (tossed onto the others).
The camera moves on a straight line with constant velocity.
The starting point is sampled randomly in a half-sphere shell around the scene,
and from there the camera moves into a random direction with a random speed between 0 and 4.
This sampling process is repeated until a trajectory is found that starts and 
ends within the specified half-sphere shell around the center of the scene. 
The camera always points towards the origin.

Static objects are spawned without overlap in the region [(-7, -7, 0), (7, 7, 10)],
and are simulated to fall and settle before the first frame of the scene.
Dynamic objects are spawned without overlap in the region [(-5, -5, 1), (5, 5, 5)], and
initialized with a random velocity from the range [(-4, -4, 0), (4, 4, 0)]
minus the position of the object to bias their trajectory towards the center of
the scene.

The scene is simulated for 2 seconds, with the physical properties of the
objects kept at the default of friction=0.5, restitution=0.5 and density=1.0.

The dataset contains approx 6k videos rendered at 512x512 pixels and 12fps.

Each sample contains the following video-format data:
(s: sequence length, h: height, w: width)

- "video": (s, h, w, 3) [uint8]
  The RGB frames.
- "segmentations": (s, h, w, 1) [uint8]
  Instance segmentation as per-pixel object-id with background=0.
  Note: because of this the instance IDs used here are one higher than their
  corresponding index in sample["instances"].
- "depth": (s, h, w, 1) [uint16]
  Distance of each pixel from the center of the camera.
  (Note this is different from the z-value sometimes used, which measures the
  distance to the camera *plane*.)
  The values are stored as uint16 and span the range specified in
  sample["metadata"]["depth_range"]. To convert them back to world-units
  use:
    minv, maxv = sample["metadata"]["depth_range"]
    depth = sample["depth"] / 65535 * (maxv - minv) + minv
- "forward_flow": (s, h, w, 2) [uint16]
  Forward optical flow in the form (delta_row, delta_column).
  The values are stored as uint16 and span the range specified in
  sample["metadata"]["forward_flow_range"]. To convert them back to pixels use:
    minv, maxv = sample["metadata"]["forward_flow_range"]
    depth = sample["forward_flow"] / 65535 * (maxv - minv) + minv
- "backward_flow": (s, h, w, 2) [uint16]
  Backward optical flow in the form (delta_row, delta_column).
  The values are stored as uint16 and span the range specified in
  sample["metadata"]["backward_flow_range"]. To convert them back to pixels use:
    minv, maxv = sample["metadata"]["backward_flow_range"]
    depth = sample["backward_flow"] / 65535 * (maxv - minv) + minv
- "normal": (s, h, w, 3) [uint16]
  Surface normals for each pixel in world coordinates.
- "object_coordinates": (s, h, w, 3) [uint16]
  Object coordinates encode the position of each point relative to the objects
  bounding box (i.e. back-left-top (X=Y=Z=1) corner is white,
  while front-right-bottom (X=Y=Z=0) corner is black.)

Additionally there is rich instance-level information in sample["instances"]:
- "mass": [float32]
  Mass of the object used for simulation.
- "friction": [float32]
  Friction coefficient used for simulation.
- "restitution": [float32]
  Restitution coefficient (bounciness) used for simulation.
- "positions": (s, 3) [float32]
  Position of the object for each frame in world-coordinates.
- "quaternions": (s, 4) [float32]
  Rotation of the object for each frame as quaternions.
- "velocities": (s, 3) [float32]
  Velocity of the object for each frame.
- "angular_velocities": (s, 3) [float32]
  Angular velocity of the object for each frame.
- "bboxes_3d": (s, 8, 3) [float32]
  World-space corners of the 3D bounding box around the object.
- "image_positions": (s, 2) [float32]
  Normalized (0, 1) image-space (2D) coordinates of the center of mass of the
  object for each frame.
- "bboxes": (None, 4) [float32]
   The normalized image-space (2D) coordinates of the bounding box
   [ymin, xmin, ymax, xmax] for all the frames in which the object is visible
   (as specified in bbox_frames).
- "bbox_frames": (None,) [int]
   A list of all the frames the object is visible.
- "visibility": (s,) [uint16]
  Visibility of the object in number of pixels for each frame (can be 0).
- "asset_id": [str] Asset id from Google Scanned Objects dataset. 
- "category": ["Action Figures", "Bag", "Board Games", 
               "Bottles and Cans and Cups", "Camera", "Car Seat", 
               "Consumer Goods", "Hat", "Headphones", "Keyboard", "Legos", 
               "Media Cases", "Mouse", "None", "Shoe", "Stuffed Toys", "Toys"]
- "scale": float 
- "is_dynamic": bool indicating whether (at the start of the scene) the object 
                is sitting on the floor or is being tossed.  

Information about the camera in sample["camera"]
(given for each frame eventhough the camera is static, so as to stay
consistent with other variants of the dataset):

- "focal_length": [float32]
- "sensor_width": [float32]
- "field_of_view": [float32]
- "positions": (s, 3) [float32]
- "quaternions": (s, 4) [float32]


And finally information about collision events in sample["events"]["collisions"]:

- "instances": (2,)[uint16]
  Indices of the two instance between which the collision happened.
  Note that collisions with the floor/background objects are marked with 65535
- "frame": tf.int32,
  Frame in which the collision happenend.
- "force": tf.float32,
  The force (strength) of the collision.
- "position": tfds.features.Tensor(shape=(3,), dtype=tf.float32),
  Position of the collision event in 3D world coordinates.
- "image_position": tfds.features.Tensor(shape=(2,), dtype=tf.float32),
  Position of the collision event projected onto normalized 2D image coordinates.
- "contact_normal": tfds.features.Tensor(shape=(3,), dtype=tf.float32),
  The normal-vector of the contact (direction of the force).
"""

_CITATION = """\
@inproceedings{greff2022kubric,
title = {Kubric: a scalable dataset generator}, 
    author = {Klaus Greff and Francois Belletti and Lucas Beyer and Carl Doersch and
              Yilun Du and Daniel Duckworth and David J Fleet and Dan Gnanapragasam and
              Florian Golemo and Charles Herrmann and Thomas Kipf and Abhijit Kundu and
              Dmitry Lagun and Issam Laradji and Hsueh-Ti (Derek) Liu and Henning Meyer and
              Yishu Miao and Derek Nowrouzezahrai and Cengiz Oztireli and Etienne Pot and
              Noha Radwan and Daniel Rebain and Sara Sabour and Mehdi S. M. Sajjadi and Matan Sela and
              Vincent Sitzmann and Austin Stone and Deqing Sun and Suhani Vora and Ziyu Wang and
              Tianhao Wu and Kwang Moo Yi and Fangcheng Zhong and Andrea Tagliasacchi},
    booktitle = {{IEEE} Conference on Computer Vision and Pattern Recognition, {CVPR}},
    year = {2022},
    publisher = {Computer Vision Foundation / {IEEE}},
}"""

@dataclasses.dataclass
class MoviFConfig(tfds.core.BuilderConfig):
  """"Configuration for Multi-Object Video (MoviF) dataset."""
  height: int = 256
  width: int = 256
  num_frames: int = 24
  validation_ratio: float = 0.1
  train_val_path: str = None
  test_split_paths: Dict[str, str] = dataclasses.field(default_factory=dict)


class MoviF(tfds.core.BeamBasedBuilder):
  """DatasetBuilder for MOVi-F dataset."""
  VERSION = tfds.core.Version("1.0.0")
  RELEASE_NOTES = {
      "1.0.0": "initial release",
  }

  BUILDER_CONFIGS = [
      MoviFConfig(
          name="512x512",
          description="Full resolution of 512x512",
          height=512,
          width=512,
          validation_ratio=0.025,
          train_val_path="gs://research-brain-kubric-xgcp/jobs/movid_flow_v2_600_1/",
          test_split_paths={
          }
      ),
      MoviFConfig(
          name="256x256",
          description="Downscaled to  256x256",
          height=256,
          width=256,
          validation_ratio=0.025,
          train_val_path="gs://research-brain-kubric-xgcp/jobs/movid_flow_v2_600_1/",
          test_split_paths={
          }
      ),
      MoviFConfig(
          name="128x128",
          description="Downscaled to 128x128",
          height=128,
          width=128,
          validation_ratio=0.025,
          train_val_path="gs://research-brain-kubric-xgcp/jobs/movid_flow_v2_600_1/",
          test_split_paths={
          }
      ),
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""

    h = self.builder_config.height
    w = self.builder_config.width
    s = self.builder_config.num_frames

    def get_movi_f_instance_features(seq_length: int):
      features = get_instance_features(seq_length)
      features.update({
          "asset_id": tfds.features.Text(),
          "category": tfds.features.ClassLabel(
              names=["Action Figures", "Bag", "Board Games",
                     "Bottles and Cans and Cups", "Camera",
                     "Car Seat", "Consumer Goods", "Hat",
                     "Headphones", "Keyboard", "Legos",
                     "Media Cases", "Mouse", "None", "Shoe",
                     "Stuffed Toys", "Toys"]),
          "scale": tf.float32,
          "is_dynamic": tf.bool,
      })
      return features

    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            "metadata": {
                "video_name": tfds.features.Text(),
                "width": tf.int32,
                "height": tf.int32,
                "num_frames": tf.int32,
                "num_instances": tf.uint16,

                "depth_range": tfds.features.Tensor(shape=(2,),
                                                    dtype=tf.float32),
                "forward_flow_range": tfds.features.Tensor(shape=(2,),
                                                           dtype=tf.float32),
                "backward_flow_range": tfds.features.Tensor(shape=(2,),
                                                            dtype=tf.float32),
                "motion_blur": tf.float32,
            },
            "background": tfds.features.Text(),
            "instances": tfds.features.Sequence(
                feature=get_movi_f_instance_features(seq_length=s)),
            "camera": get_camera_features(s),
            "events": get_events_features(),
            # -----
            "video":  tfds.features.Video(shape=(s, h, w, 3)),
            "segmentations": tfds.features.Sequence(
                tfds.features.Image(shape=(h, w, 1), dtype=tf.uint8),
                length=s),
            "forward_flow": tfds.features.Sequence(
                tfds.features.Tensor(shape=(h, w, 2), dtype=tf.uint16),
                length=s),
            "backward_flow": tfds.features.Sequence(
                tfds.features.Tensor(shape=(h, w, 2), dtype=tf.uint16),
                length=s),
            "depth": tfds.features.Sequence(
                tfds.features.Image(shape=(h, w, 1), dtype=tf.uint16),
                length=s),
            "normal": tfds.features.Video(shape=(s, h, w, 3), dtype=tf.uint16),
            "object_coordinates": tfds.features.Video(shape=(s, h, w, 3),
                                                      dtype=tf.uint16),
        }),
        supervised_keys=None,
        homepage="https://github.com/google-research/kubric",
        citation=_CITATION)

  def _split_generators(self, unused_dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    del unused_dl_manager
    path = as_path(self.builder_config.train_val_path)
    all_subdirs = [str(d) for d in path.iterdir()]
    logging.info("Found %d sub-folders in master path: %s",
                 len(all_subdirs), path)

    # shuffle
    rng = np.random.RandomState(seed=42)
    rng.shuffle(all_subdirs)

    validation_ratio = self.builder_config.validation_ratio
    validation_examples = max(1, round(len(all_subdirs) * validation_ratio))
    training_examples = len(all_subdirs) - validation_examples
    logging.info("Using %f of examples for validation for a total of %d",
                 validation_ratio, validation_examples)
    logging.info("Using the other %d examples for training", training_examples)

    splits = {
        tfds.Split.TRAIN: self._generate_examples(all_subdirs[:training_examples]),
        tfds.Split.VALIDATION: self._generate_examples(all_subdirs[training_examples:]),
    }

    for key, path in self.builder_config.test_split_paths.items():
      path = as_path(path)
      split_dirs = [d for d in path.iterdir()]
      # sort the directories by their integer number
      split_dirs = sorted(split_dirs, key=lambda x: int(x.name))
      logging.info("Found %d sub-folders in '%s' path: %s",
                   len(split_dirs), key, path)
      splits[key] = self._generate_examples([str(d) for d in split_dirs])

    return splits

  def _generate_examples(self, directories: List[str]):
    """Yields examples."""

    target_size = (self.builder_config.height, self.builder_config.width)

    def _process_example(video_dir):
      key, result, metadata = load_scene_directory(video_dir, target_size)

      # add MoviF-D specific instance information:
      for i, obj in enumerate(result["instances"]):
        obj["asset_id"] = metadata["instances"][i]["asset_id"]
        scale_factor, category = get_scale_and_category(obj["asset_id"] )
        obj["category"] = category
        obj["scale"] = metadata["instances"][i]["scale"] * scale_factor
        obj["is_dynamic"] = metadata["instances"][i]["is_dynamic"]

      return key, result

    beam = tfds.core.lazy_imports.apache_beam
    return (beam.Create(directories) |
            beam.Filter(is_complete_dir) |
            beam.Map(_process_example))


DEFAULT_LAYERS = ("rgba", "segmentation", "forward_flow", "backward_flow",
                  "depth", "normal", "object_coordinates")


def load_scene_directory(scene_dir, target_size, layers=DEFAULT_LAYERS):
  scene_dir = as_path(scene_dir)
  example_key = f"{scene_dir.name}"

  with tf.io.gfile.GFile(str(scene_dir / "data_ranges.json"), "r") as fp:
    data_ranges = json.load(fp)

  with tf.io.gfile.GFile(str(scene_dir / "metadata.json"), "r") as fp:
    metadata = json.load(fp)

  with tf.io.gfile.GFile(str(scene_dir / "events.json"), "r") as fp:
    events = json.load(fp)

  num_frames = metadata["metadata"]["num_frames"]

  result = {
      "metadata": {
          "video_name": example_key,
          "width": target_size[1],
          "height": target_size[0],
          "num_frames": num_frames,
          "num_instances": metadata["metadata"]["num_instances"],
          "motion_blur": metadata["metadata"]["motion_blur"]
      },
      "background": metadata["metadata"]["background"],
      "instances": [format_instance_information(obj)
                    for obj in metadata["instances"]],
      "camera": format_camera_information(metadata),
      "events": format_events_information(events),
  }

  resolution = metadata["metadata"]["height"], metadata["metadata"]["width"]

  assert resolution[1] / target_size[0] == resolution[0] / target_size[1]
  scale = resolution[1] / target_size[0]
  assert scale == resolution[1] // target_size[0]

  paths = {
      key: [scene_dir / f"{key}_{f:05d}.png" for f in range(num_frames)]
      for key in layers if key != "depth"
  }

  if "depth" in layers:
    depth_paths = [scene_dir / f"depth_{f:05d}.tiff" for f in range(num_frames)]
    depth_frames = np.array([
        subsample_nearest_neighbor(read_tiff(frame_path), target_size)
        for frame_path in depth_paths])
    depth_min, depth_max = np.min(depth_frames), np.max(depth_frames)
    result["depth"] = convert_float_to_uint16(depth_frames, depth_min, depth_max)
    result["metadata"]["depth_range"] = [depth_min, depth_max]

  if "forward_flow" in layers:
    result["metadata"]["forward_flow_range"] = [
        data_ranges["forward_flow"]["min"] / scale * 512,
        data_ranges["forward_flow"]["max"] / scale * 512]
    result["forward_flow"] = [
        subsample_nearest_neighbor(read_png(frame_path)[..., :2],
                                   target_size)
        for frame_path in paths["forward_flow"]]

  if "backward_flow" in layers:
    result["metadata"]["backward_flow_range"] = [
        data_ranges["backward_flow"]["min"] / scale * 512,
        data_ranges["backward_flow"]["max"] / scale * 512]
    result["backward_flow"] = [
        subsample_nearest_neighbor(read_png(frame_path)[..., :2],
                                   target_size)
        for frame_path in paths["backward_flow"]]

  for key in ["normal", "object_coordinates", "uv"]:
    if key in layers:
      result[key] = [
          subsample_nearest_neighbor(read_png(frame_path),
                                     target_size)
          for frame_path in paths[key]]

  if "segmentation" in layers:
    # somehow we ended up calling this "segmentations" in TFDS and
    # "segmentation" in kubric. So we have to treat it separately.
    result["segmentations"] = [
        subsample_nearest_neighbor(read_png(frame_path),
                                   target_size)
        for frame_path in paths["segmentation"]]

  if "rgba" in layers:
    result["video"] = [
        subsample_avg(read_png(frame_path), target_size)[..., :3]
        for frame_path in paths["rgba"]]

  return example_key, result, metadata


def get_camera_features(seq_length):
  return {
      "focal_length": tf.float32,
      "sensor_width": tf.float32,
      "field_of_view": tf.float32,
      "positions": tfds.features.Tensor(shape=(seq_length, 3),
                                        dtype=tf.float32),
      "quaternions": tfds.features.Tensor(shape=(seq_length, 4),
                                          dtype=tf.float32),
  }


def format_camera_information(metadata):
  return {
      "focal_length": metadata["camera"]["focal_length"],
      "sensor_width": metadata["camera"]["sensor_width"],
      "field_of_view": metadata["camera"]["field_of_view"],
      "positions": np.array(metadata["camera"]["positions"], np.float32),
      "quaternions": np.array(metadata["camera"]["quaternions"], np.float32),
  }


def get_events_features():
  return {
      "collisions": tfds.features.Sequence({
          "instances": tfds.features.Tensor(shape=(2,), dtype=tf.uint16),
          "frame": tf.int32,
          "force": tf.float32,
          "position": tfds.features.Tensor(shape=(3,), dtype=tf.float32),
          "image_position": tfds.features.Tensor(shape=(2,), dtype=tf.float32),
          "contact_normal": tfds.features.Tensor(shape=(3,), dtype=tf.float32),
      })
  }


def format_events_information(events):
  return {
      "collisions": [{
          "instances": np.array(c["instances"], dtype=np.uint16),
          "frame": c["frame"],
          "force": c["force"],
          "position": np.array(c["position"], dtype=np.float32),
          "image_position": np.array(c["image_position"], dtype=np.float32),
          "contact_normal": np.array(c["contact_normal"], dtype=np.float32),
      } for c in events["collisions"]],
  }


def get_instance_features(seq_length: int):
  return {
      "mass": tf.float32,
      "friction": tf.float32,
      "restitution": tf.float32,

      "positions": tfds.features.Tensor(shape=(seq_length, 3),
                                        dtype=tf.float32),
      "quaternions": tfds.features.Tensor(shape=(seq_length, 4),
                                          dtype=tf.float32),
      "velocities": tfds.features.Tensor(shape=(seq_length, 3),
                                         dtype=tf.float32),
      "angular_velocities": tfds.features.Tensor(shape=(seq_length, 3),
                                                 dtype=tf.float32),
      "bboxes_3d": tfds.features.Tensor(shape=(seq_length, 8, 3),
                                        dtype=tf.float32),

      "image_positions": tfds.features.Tensor(shape=(seq_length, 2),
                                              dtype=tf.float32),
      "bboxes": tfds.features.Sequence(
          tfds.features.BBoxFeature()),
      "bbox_frames": tfds.features.Sequence(
          tfds.features.Tensor(shape=(), dtype=tf.int32)),
      "visibility": tfds.features.Tensor(shape=(seq_length,), dtype=tf.uint16),
  }


def format_instance_information(obj):
  return {
      "mass": obj["mass"],
      "friction": obj["friction"],
      "restitution": obj["restitution"],
      "positions": np.array(obj["positions"], np.float32),
      "quaternions": np.array(obj["quaternions"], np.float32),
      "velocities": np.array(obj["velocities"], np.float32),
      "angular_velocities": np.array(obj["angular_velocities"], np.float32),
      "bboxes_3d": np.array(obj["bboxes_3d"], np.float32),
      "image_positions": np.array(obj["image_positions"], np.float32),
      "bboxes": [tfds.features.BBox(*bbox) for bbox in obj["bboxes"]],
      "bbox_frames": np.array(obj["bbox_frames"], dtype=np.uint16),
      "visibility": np.array(obj["visibility"], dtype=np.uint16),
  }


def subsample_nearest_neighbor(arr, size):
  src_height, src_width, _ = arr.shape
  dst_height, dst_width = size
  height_step = src_height // dst_height
  width_step = src_width // dst_width
  assert height_step * dst_height == src_height
  assert width_step * dst_width == src_width

  height_offset = int(np.floor((height_step-1)/2))
  width_offset = int(np.floor((width_step-1)/2))
  subsampled = arr[height_offset::height_step, width_offset::width_step, :]
  return subsampled


def convert_float_to_uint16(array, min_val, max_val):
  return np.round((array - min_val) / (max_val - min_val) * 65535
                  ).astype(np.uint16)


def subsample_avg(arr, size):
  src_height, src_width, channels = arr.shape
  dst_height, dst_width = size
  height_bin = src_height // dst_height
  width_bin = src_width // dst_width
  return np.round(arr.reshape((dst_height, height_bin,
                               dst_width, width_bin,
                               channels)).mean(axis=(1, 3))).astype(np.uint8)


def is_complete_dir(video_dir, layers=DEFAULT_LAYERS):
  video_dir = as_path(video_dir)
  filenames = [d.name for d in video_dir.iterdir()]
  if not ("data_ranges.json" in filenames and
          "metadata.json" in filenames and
          "events.json" in filenames):
    return False
  nr_frames_per_category = {
      key: len([fn for fn in filenames if fn.startswith(key)])
      for key in layers}

  nr_expected_frames = nr_frames_per_category["rgba"]
  if nr_expected_frames == 0:
    return False
  if not all(nr_frames == nr_expected_frames
             for nr_frames in nr_frames_per_category.values()):
    return False

  return True


PathLike = Union[str, epath.Path]


def as_path(path: PathLike) -> epath.Path:
  """Convert str or pathlike object to epath.Path.

  Instead of pathlib.Paths, we use the TFDS path because they transparently
  support paths to GCS buckets such as "gs://kubric-public/GSO".
  """
  return tfds.core.as_path(path)


def read_png(filename, rescale_range=None) -> np.ndarray:
  filename = as_path(filename)
  png_reader = png.Reader(bytes=filename.read_bytes())
  width, height, pngdata, info = png_reader.read()
  del png_reader

  bitdepth = info["bitdepth"]
  if bitdepth == 8:
    dtype = np.uint8
  elif bitdepth == 16:
    dtype = np.uint16
  else:
    raise NotImplementedError(f"Unsupported bitdepth: {bitdepth}")

  plane_count = info["planes"]
  pngdata = np.vstack(list(map(dtype, pngdata)))
  if rescale_range is not None:
    minv, maxv = rescale_range
    pngdata = pngdata / 2**bitdepth * (maxv - minv) + minv

  return pngdata.reshape((height, width, plane_count))


def write_tiff(data: np.ndarray, filename: PathLike):
  """Save data as as tif image (which natively supports float values)."""
  assert data.ndim == 3, data.shape
  assert data.shape[2] in [1, 3, 4], "Must be grayscale, RGB, or RGBA"

  img_as_bytes = imageio.imwrite("<bytes>", data, format="tiff")
  filename = as_path(filename)
  filename.write_bytes(img_as_bytes)


def read_tiff(filename: PathLike) -> np.ndarray:
  filename = as_path(filename)
  img = imageio.imread(filename.read_bytes(), format="tiff")
  if img.ndim == 2:
    img = img[:, :, None]
  return img

def get_scale_and_category(asset_id):
  conversion_dict = {
      '11pro_SL_TRX_FG': {'scale_factor': 0.290936, 'category': 'Shoe'},
      '2_of_Jenga_Classic_Game': {'scale_factor': 0.292098, 'category': 'Consumer Goods'},
      '30_CONSTRUCTION_SET': {'scale_factor': 0.26941899999999996, 'category': 'Toys'},
      '3D_Dollhouse_Happy_Brother': {'scale_factor': 0.0955, 'category': 'Consumer Goods'},
      '3D_Dollhouse_Lamp': {'scale_factor': 0.166821, 'category': 'Toys'},
      '3D_Dollhouse_Refrigerator': {'scale_factor': 0.209969, 'category': 'Toys'},
      '3D_Dollhouse_Sink': {'scale_factor': 0.131078, 'category': 'Toys'},
      '3D_Dollhouse_Sofa': {'scale_factor': 0.209633, 'category': 'Toys'},
      '3D_Dollhouse_Swing': {'scale_factor': 0.104819, 'category': 'Toys'},
      '3D_Dollhouse_TablePurple': {'scale_factor': 0.09798799999999999, 'category': 'Toys'},
      '3M_Antislip_Surfacing_Light_Duty_White': {'scale_factor': 0.154701, 'category': 'None'},
      '3M_Vinyl_Tape_Green_1_x_36_yd': {'scale_factor': 0.110786, 'category': 'None'},
      '45oz_RAMEKIN_ASST_DEEP_COLORS': {'scale_factor': 0.089988, 'category': 'Consumer Goods'},
      '50_BLOCKS': {'scale_factor': 0.355794, 'category': 'Toys'},
      '5_HTP': {'scale_factor': 0.08882299999999999, 'category': 'Bottles and Cans and Cups'},
      '60_CONSTRUCTION_SET': {'scale_factor': 0.343594, 'category': 'Toys'},
      'ACE_Coffee_Mug_Kristen_16_oz_cup': {'scale_factor': 0.134936, 'category': 'Consumer Goods'},
      'ALPHABET_AZ_GRADIENT': {'scale_factor': 0.332328, 'category': 'Toys'},
      'ALPHABET_AZ_GRADIENT_WQb1ufEycSj': {'scale_factor': 0.331944, 'category': 'Toys'},
      'AMBERLIGHT_UP_W': {'scale_factor': 0.245312, 'category': 'Shoe'},
      'ASICS_GEL1140V_WhiteBlackSilver': {'scale_factor': 0.280768, 'category': 'Shoe'},
      'ASICS_GEL1140V_WhiteRoyalSilver': {'scale_factor': 0.28071500000000005, 'category': 'Shoe'},
      'ASICS_GELAce_Pro_Pearl_WhitePink': {'scale_factor': 0.290163, 'category': 'Shoe'},
      'ASICS_GELBlur33_20_GS_BlackWhiteSafety_Orange': {'scale_factor': 0.254278, 'category': 'Shoe'},
      'ASICS_GELBlur33_20_GS_Flash_YellowHot_PunchSilver': {'scale_factor': 0.25366999999999995, 'category': 'Shoe'},
      'ASICS_GELChallenger_9_Royal_BlueWhiteBlack': {'scale_factor': 0.298461, 'category': 'Shoe'},
      'ASICS_GELDirt_Dog_4_SunFlameBlack': {'scale_factor': 0.284277, 'category': 'Shoe'},
      'ASICS_GELLinksmaster_WhiteCoffeeSand': {'scale_factor': 0.299169, 'category': 'Shoe'},
      'ASICS_GELLinksmaster_WhiteRasberryGunmetal': {'scale_factor': 0.27285000000000004, 'category': 'Shoe'},
      'ASICS_GELLinksmaster_WhiteSilverCarolina_Blue': {'scale_factor': 0.272868, 'category': 'Shoe'},
      'ASICS_GELResolution_5_Flash_YellowBlackSilver': {'scale_factor': 0.2993, 'category': 'Shoe'},
      'ASICS_GELTour_Lyte_WhiteOrchidSilver': {'scale_factor': 0.26630600000000004, 'category': 'Shoe'},
      'ASICS_HyperRocketgirl_SP_5_WhiteMalibu_BlueBlack': {'scale_factor': 0.24914599999999998, 'category': 'Shoe'},
      'ASSORTED_VEGETABLE_SET': {'scale_factor': 0.230192, 'category': 'Toys'},
      'Adrenaline_GTS_13_Color_DrkDenimWhtBachlorBttnSlvr_Size_50_yfK40TNjq0V': {'scale_factor': 0.288056, 'category': 'Shoe'},
      'Adrenaline_GTS_13_Color_WhtObsdianBlckOlmpcSlvr_Size_70': {'scale_factor': 0.30678300000000003, 'category': 'Shoe'},
      'Air_Hogs_Wind_Flyers_Set_Airplane_Red': {'scale_factor': 0.272032, 'category': 'None'},
      'AllergenFree_JarroDophilus': {'scale_factor': 0.095997, 'category': 'Bottles and Cans and Cups'},
      'Android_Figure_Chrome': {'scale_factor': 0.081451, 'category': 'Consumer Goods'},
      'Android_Figure_Orange': {'scale_factor': 0.080687, 'category': 'Consumer Goods'},
      'Android_Figure_Panda': {'scale_factor': 0.075739, 'category': 'Consumer Goods'},
      'Android_Lego': {'scale_factor': 0.106349, 'category': 'Legos'},
      'Animal_Crossing_New_Leaf_Nintendo_3DS_Game': {'scale_factor': 0.13718000000000002, 'category': 'Media Cases'},
      'Animal_Planet_Foam_2Headed_Dragon': {'scale_factor': 0.40190299999999995, 'category': 'Toys'},
      'Apples_to_Apples_Kids_Edition': {'scale_factor': 0.269371, 'category': 'Consumer Goods'},
      'Arm_Hammer_Diaper_Pail_Refills_12_Pack_MFWkmoweejt': {'scale_factor': 0.169219, 'category': 'Consumer Goods'},
      'Aroma_Stainless_Steel_Milk_Frother_2_Cup': {'scale_factor': 0.157677, 'category': 'Consumer Goods'},
      'Asus_80211ac_DualBand_Gigabit_Wireless_Router_RTAC68R': {'scale_factor': 0.318751, 'category': 'None'},
      'Asus_M5A78LMUSB3_Motherboard_Micro_ATX_Socket_AM3': {'scale_factor': 0.276686, 'category': 'None'},
      'Asus_M5A99FX_PRO_R20_Motherboard_ATX_Socket_AM3': {'scale_factor': 0.334631, 'category': 'None'},
      'Asus_Sabertooth_990FX_20_Motherboard_ATX_Socket_AM3': {'scale_factor': 0.350201, 'category': 'None'},
      'Asus_Sabertooth_Z97_MARK_1_Motherboard_ATX_LGA1150_Socket': {'scale_factor': 0.350557, 'category': 'None'},
      'Asus_X99Deluxe_Motherboard_ATX_LGA2011v3_Socket': {'scale_factor': 0.350274, 'category': 'None'},
      'Asus_Z87PRO_Motherboard_ATX_LGA1150_Socket': {'scale_factor': 0.33416500000000005, 'category': 'None'},
      'Asus_Z97AR_LGA_1150_Intel_ATX_Motherboard': {'scale_factor': 0.33413499999999996, 'category': 'None'},
      'Asus_Z97IPLUS_Motherboard_Mini_ITX_LGA1150_Socket': {'scale_factor': 0.231536, 'category': 'None'},
      'Avengers_Gamma_Green_Smash_Fists': {'scale_factor': 0.360015, 'category': 'Toys'},
      'Avengers_Thor_PLlrpYniaeB': {'scale_factor': 0.28297300000000003, 'category': 'Action Figures'},
      'Azure_Snake_Tieks_Leather_Snake_Print_Ballet_Flats': {'scale_factor': 0.242954, 'category': 'Shoe'},
      'BABY_CAR': {'scale_factor': 0.096854, 'category': 'Toys'},
      'BAGEL_WITH_CHEESE': {'scale_factor': 0.133645, 'category': 'Toys'},
      'BAKING_UTENSILS': {'scale_factor': 0.32602400000000004, 'category': 'Toys'},
      'BALANCING_CACTUS': {'scale_factor': 0.263536, 'category': 'Toys'},
      'BATHROOM_CLASSIC': {'scale_factor': 0.185214, 'category': 'Toys'},
      'BATHROOM_FURNITURE_SET_1': {'scale_factor': 0.210252, 'category': 'Toys'},
      'BEDROOM_CLASSIC': {'scale_factor': 0.314955, 'category': 'Toys'},
      'BEDROOM_CLASSIC_Gi22DjScTVS': {'scale_factor': 0.343402, 'category': 'Toys'},
      'BEDROOM_NEO': {'scale_factor': 0.262714, 'category': 'Toys'},
      'BIA_Cordon_Bleu_White_Porcelain_Utensil_Holder_900028': {'scale_factor': 0.175195, 'category': 'None'},
      'BIA_Porcelain_Ramekin_With_Glazed_Rim_35_45_oz_cup': {'scale_factor': 0.08867900000000001, 'category': 'None'},
      'BIRD_RATTLE': {'scale_factor': 0.11119000000000001, 'category': 'Toys'},
      'BRAILLE_ALPHABET_AZ': {'scale_factor': 0.333027, 'category': 'Toys'},
      'BREAKFAST_MENU': {'scale_factor': 0.25312999999999997, 'category': 'Toys'},
      'BUILD_A_ROBOT': {'scale_factor': 0.292531, 'category': 'Toys'},
      'BUILD_A_ZOO': {'scale_factor': 0.218314, 'category': 'Toys'},
      'BUNNY_RACER': {'scale_factor': 0.11154800000000001, 'category': 'Toys'},
      'BUNNY_RATTLE': {'scale_factor': 0.11466300000000001, 'category': 'Toys'},
      'Baby_Elements_Stacking_Cups': {'scale_factor': 0.346337, 'category': 'None'},
      'Balderdash_Game': {'scale_factor': 0.268966, 'category': 'Board Games'},
      'Beetle_Adventure_Racing_Nintendo_64': {'scale_factor': 0.116505, 'category': 'Consumer Goods'},
      'Beta_Glucan': {'scale_factor': 0.09761600000000001, 'category': 'Bottles and Cans and Cups'},
      'Beyonc_Life_is_But_a_Dream_DVD': {'scale_factor': 0.188517, 'category': 'Consumer Goods'},
      'Bifidus_Balance_FOS': {'scale_factor': 0.096467, 'category': 'Bottles and Cans and Cups'},
      'Big_Dot_Aqua_Pencil_Case': {'scale_factor': 0.208131, 'category': 'Bag'},
      'Big_Dot_Pink_Pencil_Case': {'scale_factor': 0.21375, 'category': 'Bag'},
      'Big_O_Sponges_Assorted_Cellulose_12_pack': {'scale_factor': 0.12575399999999998, 'category': 'Consumer Goods'},
      'BlackBlack_Nintendo_3DSXL': {'scale_factor': 0.15554, 'category': 'None'},
      'Black_Decker_CM2035B_12Cup_Thermal_Coffeemaker': {'scale_factor': 0.336937, 'category': 'None'},
      'Black_Decker_Stainless_Steel_Toaster_4_Slice': {'scale_factor': 0.318299, 'category': 'None'},
      'Black_Elderberry_Syrup_54_oz_Gaia_Herbs': {'scale_factor': 0.145106, 'category': 'Consumer Goods'},
      'Black_Forest_Fruit_Snacks_28_Pack_Grape': {'scale_factor': 0.257416, 'category': 'Consumer Goods'},
      'Black_Forest_Fruit_Snacks_Juicy_Filled_Centers_10_pouches_9_oz_total': {'scale_factor': 0.212087, 'category': 'Consumer Goods'},
      'Black_and_Decker_PBJ2000_FusionBlade_Blender_Jars': {'scale_factor': 0.299714, 'category': 'None'},
      'Black_and_Decker_TR3500SD_2Slice_Toaster': {'scale_factor': 0.295277, 'category': 'None'},
      'Blackcurrant_Lutein': {'scale_factor': 0.097312, 'category': 'Bottles and Cans and Cups'},
      'BlueBlack_Nintendo_3DSXL': {'scale_factor': 0.156599, 'category': 'None'},
      'Blue_Jasmine_Includes_Digital_Copy_UltraViolet_DVD': {'scale_factor': 0.192975, 'category': 'Media Cases'},
      'Borage_GLA240Gamma_Tocopherol': {'scale_factor': 0.11094899999999999, 'category': 'Bottles and Cans and Cups'},
      'Bradshaw_International_11642_7_Qt_MP_Plastic_Bowl': {'scale_factor': 0.28509, 'category': 'None'},
      'Breyer_Horse_Of_The_Year_2015': {'scale_factor': 0.23114099999999999, 'category': 'None'},
      'Brisk_Iced_Tea_Lemon_12_12_fl_oz_355_ml_cans_144_fl_oz_426_lt': {'scale_factor': 0.40285, 'category': 'Consumer Goods'},
      'Brother_Ink_Cartridge_Magenta_LC75M': {'scale_factor': 0.14715899999999998, 'category': 'Consumer Goods'},
      'Brother_LC_1053PKS_Ink_Cartridge_CyanMagentaYellow_1pack': {'scale_factor': 0.14492, 'category': 'Consumer Goods'},
      'Brother_Printing_Cartridge_PC501': {'scale_factor': 0.25587499999999996, 'category': 'Consumer Goods'},
      'CARSII': {'scale_factor': 0.122813, 'category': 'Toys'},
      'CAR_CARRIER_TRAIN': {'scale_factor': 0.22295399999999999, 'category': 'Toys'},
      'CASTLE_BLOCKS': {'scale_factor': 0.302403, 'category': 'Toys'},
      'CHICKEN_NESTING': {'scale_factor': 0.19498900000000002, 'category': 'Toys'},
      'CHICKEN_RACER': {'scale_factor': 0.099457, 'category': 'Toys'},
      'CHILDRENS_ROOM_NEO': {'scale_factor': 0.2017, 'category': 'Toys'},
      'CHILDREN_BEDROOM_CLASSIC': {'scale_factor': 0.224682, 'category': 'Toys'},
      'CITY_TAXI_POLICE_CAR': {'scale_factor': 0.126338, 'category': 'Toys'},
      'CLIMACOOL_BOAT_BREEZE_IE6CyqSaDwN': {'scale_factor': 0.293272, 'category': 'Shoe'},
      'COAST_GUARD_BOAT': {'scale_factor': 0.116176, 'category': 'Toys'},
      'CONE_SORTING': {'scale_factor': 0.269663, 'category': 'Toys'},
      'CONE_SORTING_kg5fbARBwts': {'scale_factor': 0.390906, 'category': 'Toys'},
      'CREATIVE_BLOCKS_35_MM': {'scale_factor': 0.281776, 'category': 'Toys'},
      'California_Navy_Tieks_Italian_Leather_Ballet_Flats': {'scale_factor': 0.24167100000000002, 'category': 'Shoe'},
      'Calphalon_Kitchen_Essentials_12_Cast_Iron_Fry_Pan_Black': {'scale_factor': 0.323628, 'category': 'None'},
      'Canon_225226_Ink_Cartridges_BlackColor_Cyan_Magenta_Yellow_6_count': {'scale_factor': 0.169147, 'category': 'Consumer Goods'},
      'Canon_Ink_Cartridge_Green_6': {'scale_factor': 0.146077, 'category': 'Consumer Goods'},
      'Canon_Pixma_Chromalife_100_Magenta_8': {'scale_factor': 0.11405399999999999, 'category': 'Consumer Goods'},
      'Canon_Pixma_Ink_Cartridge_251_M': {'scale_factor': 0.111867, 'category': 'Consumer Goods'},
      'Canon_Pixma_Ink_Cartridge_8': {'scale_factor': 0.11460300000000001, 'category': 'Consumer Goods'},
      'Canon_Pixma_Ink_Cartridge_8_Green': {'scale_factor': 0.114856, 'category': 'Consumer Goods'},
      'Canon_Pixma_Ink_Cartridge_8_Red': {'scale_factor': 0.11458399999999999, 'category': 'Consumer Goods'},
      'Canon_Pixma_Ink_Cartridge_Cyan_251': {'scale_factor': 0.111039, 'category': 'Consumer Goods'},
      'Cascadia_8_Color_AquariusHibscsBearingSeaBlk_Size_50': {'scale_factor': 0.254417, 'category': 'Shoe'},
      'Central_Garden_Flower_Pot_Goo_425': {'scale_factor': 0.115311, 'category': 'Consumer Goods'},
      'Chef_Style_Round_Cake_Pan_9_inch_pan': {'scale_factor': 0.246, 'category': 'None'},
      'Chefmate_8_Frypan': {'scale_factor': 0.35577499999999995, 'category': 'None'},
      'Chelsea_BlkHeelPMP_DwxLtZNxLZZ': {'scale_factor': 0.24076000000000003, 'category': 'Shoe'},
      'Chelsea_lo_fl_rdheel_nQ0LPNF1oMw': {'scale_factor': 0.256529, 'category': 'Shoe'},
      'Chelsea_lo_fl_rdheel_zAQrnhlEfw8': {'scale_factor': 0.256361, 'category': 'Shoe'},
      'Circo_Fish_Toothbrush_Holder_14995988': {'scale_factor': 0.154691, 'category': 'Consumer Goods'},
      'ClimaCool_Aerate_2_W_Wide': {'scale_factor': 0.268432, 'category': 'Shoe'},
      'Clorox_Premium_Choice_Gloves_SM_1_pair': {'scale_factor': 0.326326, 'category': 'None'},
      'Closetmaid_Premium_Fabric_Cube_Red': {'scale_factor': 0.32167, 'category': 'None'},
      'Clue_Board_Game_Classic_Edition': {'scale_factor': 0.496927, 'category': 'Board Games'},
      'CoQ10': {'scale_factor': 0.084399, 'category': 'Bottles and Cans and Cups'},
      'CoQ10_BjTLbuRVt1t': {'scale_factor': 0.083729, 'category': 'Bottles and Cans and Cups'},
      'CoQ10_wSSVoxVppVD': {'scale_factor': 0.08402000000000001, 'category': 'Bottles and Cans and Cups'},
      'Cole_Hardware_Antislip_Surfacing_Material_White': {'scale_factor': 0.154879, 'category': 'None'},
      'Cole_Hardware_Antislip_Surfacing_White_2_x_60': {'scale_factor': 0.157827, 'category': 'None'},
      'Cole_Hardware_Bowl_Scirocco_YellowBlue': {'scale_factor': 0.114898, 'category': 'None'},
      'Cole_Hardware_Butter_Dish_Square_Red': {'scale_factor': 0.115217, 'category': 'Consumer Goods'},
      'Cole_Hardware_Deep_Bowl_Good_Earth_1075': {'scale_factor': 0.284281, 'category': 'None'},
      'Cole_Hardware_Dishtowel_Blue': {'scale_factor': 0.23432399999999998, 'category': 'None'},
      'Cole_Hardware_Dishtowel_BlueWhite': {'scale_factor': 0.235778, 'category': 'None'},
      'Cole_Hardware_Dishtowel_Multicolors': {'scale_factor': 0.24363599999999996, 'category': 'None'},
      'Cole_Hardware_Dishtowel_Red': {'scale_factor': 0.236506, 'category': 'None'},
      'Cole_Hardware_Dishtowel_Stripe': {'scale_factor': 0.234402, 'category': 'None'},
      'Cole_Hardware_Electric_Pot_Assortment_55': {'scale_factor': 0.142984, 'category': 'None'},
      'Cole_Hardware_Electric_Pot_Cabana_55': {'scale_factor': 0.141934, 'category': 'None'},
      'Cole_Hardware_Flower_Pot_1025': {'scale_factor': 0.249862, 'category': 'None'},
      'Cole_Hardware_Hammer_Black': {'scale_factor': 0.291707, 'category': 'None'},
      'Cole_Hardware_Mini_Honey_Dipper': {'scale_factor': 0.11050700000000001, 'category': 'Consumer Goods'},
      'Cole_Hardware_Mug_Classic_Blue': {'scale_factor': 0.165662, 'category': 'None'},
      'Cole_Hardware_Orchid_Pot_85': {'scale_factor': 0.21143299999999998, 'category': 'None'},
      'Cole_Hardware_Plant_Saucer_Brown_125': {'scale_factor': 0.310668, 'category': 'None'},
      'Cole_Hardware_Plant_Saucer_Glazed_9': {'scale_factor': 0.228736, 'category': 'None'},
      'Cole_Hardware_Saucer_Electric': {'scale_factor': 0.124293, 'category': 'None'},
      'Cole_Hardware_Saucer_Glazed_6': {'scale_factor': 0.15989799999999998, 'category': 'None'},
      'Cole_Hardware_School_Bell_Solid_Brass_38': {'scale_factor': 0.15964899999999999, 'category': 'Consumer Goods'},
      'Colton_Wntr_Chukka_y4jO0I8JQFW': {'scale_factor': 0.32694599999999996, 'category': 'Shoe'},
      'Connect_4_Launchers': {'scale_factor': 0.275444, 'category': 'Board Games'},
      'Cootie_Game': {'scale_factor': 0.341183, 'category': 'Consumer Goods'},
      'Cootie_Game_tDhURNbfU5J': {'scale_factor': 0.269659, 'category': 'Consumer Goods'},
      'Copperhead_Snake_Tieks_Brown_Snake_Print_Ballet_Flats': {'scale_factor': 0.24607, 'category': 'Shoe'},
      'Corningware_CW_by_Corningware_3qt_Oblong_Casserole_Dish_Blue': {'scale_factor': 0.368499, 'category': 'None'},
      'Court_Attitude': {'scale_factor': 0.290111, 'category': 'Shoe'},
      'Craftsman_Grip_Screwdriver_Phillips_Cushion': {'scale_factor': 0.260432, 'category': 'None'},
      'Crayola_Bonus_64_Crayons': {'scale_factor': 0.147247, 'category': 'None'},
      'Crayola_Crayons_120_crayons': {'scale_factor': 0.23077799999999998, 'category': 'Consumer Goods'},
      'Crayola_Crayons_24_count': {'scale_factor': 0.116595, 'category': 'Consumer Goods'},
      'Crayola_Crayons_Washable_24_crayons': {'scale_factor': 0.11608500000000002, 'category': 'Consumer Goods'},
      'Crayola_Model_Magic_Modeling_Material_Single_Packs_6_pack_05_oz_packs': {'scale_factor': 0.217536, 'category': 'Consumer Goods'},
      'Crayola_Model_Magic_Modeling_Material_White_3_oz': {'scale_factor': 0.217501, 'category': 'Consumer Goods'},
      'Crayola_Washable_Fingerpaint_Red_Blue_Yellow_3_count_8_fl_oz_bottes_each': {'scale_factor': 0.177037, 'category': 'None'},
      'Crayola_Washable_Sidewalk_Chalk_16_pack': {'scale_factor': 0.14836699999999997, 'category': 'Consumer Goods'},
      'Crayola_Washable_Sidewalk_Chalk_16_pack_wDZECiw7J6s': {'scale_factor': 0.148148, 'category': 'Consumer Goods'},
      'Crazy_8': {'scale_factor': 0.305855, 'category': 'Shoe'},
      'Crazy_Shadow_2': {'scale_factor': 0.296205, 'category': 'Shoe'},
      'Crazy_Shadow_2_oW4Jd10HFFr': {'scale_factor': 0.296153, 'category': 'Shoe'},
      'Cream_Tieks_Italian_Leather_Ballet_Flats': {'scale_factor': 0.245575, 'category': 'Shoe'},
      'Creatine_Monohydrate': {'scale_factor': 0.183814, 'category': 'Bottles and Cans and Cups'},
      'Crosley_Alarm_Clock_Vintage_Metal': {'scale_factor': 0.162287, 'category': 'Consumer Goods'},
      'Crunch_Girl_Scouts_Candy_Bars_Peanut_Butter_Creme_78_oz_box': {'scale_factor': 0.163966, 'category': 'Consumer Goods'},
      'Curver_Storage_Bin_Black_Small': {'scale_factor': 0.285864, 'category': 'None'},
      'DANCING_ALLIGATOR': {'scale_factor': 0.294873, 'category': 'Toys'},
      'DANCING_ALLIGATOR_zoWBjc0jbTs': {'scale_factor': 0.322298, 'category': 'Toys'},
      'DIM_CDG': {'scale_factor': 0.08427199999999999, 'category': 'Bottles and Cans and Cups'},
      'DINING_ROOM_CLASSIC': {'scale_factor': 0.234514, 'category': 'Toys'},
      'DINING_ROOM_CLASSIC_UJuxQ0hv5XU': {'scale_factor': 0.227929, 'category': 'Toys'},
      'DINNING_ROOM_FURNITURE_SET_1': {'scale_factor': 0.190768, 'category': 'Toys'},
      'DOLL_FAMILY': {'scale_factor': 0.231017, 'category': 'Toys'},
      'DPC_Handmade_Hat_Brown': {'scale_factor': 0.35390299999999997, 'category': 'Hat'},
      'DPC_Thinsulate_Isolate_Gloves_Brown': {'scale_factor': 0.301094, 'category': 'None'},
      'DPC_tropical_Trends_Hat': {'scale_factor': 0.403856, 'category': 'Hat'},
      'DRAGON_W': {'scale_factor': 0.262775, 'category': 'Shoe'},
      'D_ROSE_45': {'scale_factor': 0.29872, 'category': 'Shoe'},
      'D_ROSE_773_II_Kqclsph05pE': {'scale_factor': 0.297332, 'category': 'Shoe'},
      'D_ROSE_773_II_hvInJwJ5HUD': {'scale_factor': 0.297851, 'category': 'Shoe'},
      'D_ROSE_ENGLEWOOD_II': {'scale_factor': 0.307426, 'category': 'Shoe'},
      'Dell_Ink_Cartridge': {'scale_factor': 0.139935, 'category': 'Consumer Goods'},
      'Dell_Ink_Cartridge_Yellow_31': {'scale_factor': 0.140117, 'category': 'Consumer Goods'},
      'Dell_Series_9_Color_Ink_Cartridge_MK993_High_Yield': {'scale_factor': 0.134106, 'category': 'Consumer Goods'},
      'Design_Ideas_Drawer_Store_Organizer': {'scale_factor': 0.307115, 'category': 'None'},
      'Deskstar_Desk_Top_Hard_Drive_1_TB': {'scale_factor': 0.199376, 'category': 'Consumer Goods'},
      'Diamond_Visions_Scissors_Red': {'scale_factor': 0.20956899999999998, 'category': 'Consumer Goods'},
      'Diet_Pepsi_Soda_Cola12_Pack_12_oz_Cans': {'scale_factor': 0.406395, 'category': 'Consumer Goods'},
      'Digital_Camo_Double_Decker_Lunch_Bag': {'scale_factor': 0.260741, 'category': 'Bag'},
      'Dino_3': {'scale_factor': 0.288729, 'category': 'Action Figures'},
      'Dino_4': {'scale_factor': 0.13681500000000002, 'category': 'Action Figures'},
      'Dino_5': {'scale_factor': 0.21971000000000002, 'category': 'Action Figures'},
      'Dixie_10_ounce_Bowls_35_ct': {'scale_factor': 0.152169, 'category': 'None'},
      'Dog': {'scale_factor': 0.231991, 'category': 'None'},
      'Don_Franciscos_Gourmet_Coffee_Medium_Decaf_100_Colombian_12_oz_340_g': {'scale_factor': 0.141041, 'category': 'Bottles and Cans and Cups'},
      'Down_To_Earth_Ceramic_Orchid_Pot_Asst_Blue': {'scale_factor': 0.13273400000000002, 'category': 'None'},
      'Down_To_Earth_Orchid_Pot_Ceramic_Lime': {'scale_factor': 0.140098, 'category': 'Consumer Goods'},
      'Down_To_Earth_Orchid_Pot_Ceramic_Red': {'scale_factor': 0.131359, 'category': 'Consumer Goods'},
      'ENFR_MID_ENFORCER': {'scale_factor': 0.293078, 'category': 'Shoe'},
      'Eat_to_Live_The_Amazing_NutrientRich_Program_for_Fast_and_Sustained_Weight_Loss_Revised_Edition_Book': {'scale_factor': 0.210368, 'category': 'Consumer Goods'},
      'Ecoforms_Cup_B4_SAN': {'scale_factor': 0.09998599999999999, 'category': 'None'},
      'Ecoforms_Garden_Pot_GP16ATurquois': {'scale_factor': 0.16426800000000003, 'category': 'None'},
      'Ecoforms_Plant_Bowl_Atlas_Low': {'scale_factor': 0.32199900000000004, 'category': 'None'},
      'Ecoforms_Plant_Bowl_Turquoise_7': {'scale_factor': 0.181971, 'category': 'None'},
      'Ecoforms_Plant_Container_12_Pot_Nova': {'scale_factor': 0.297374, 'category': 'None'},
      'Ecoforms_Plant_Container_B4_Har': {'scale_factor': 0.100927, 'category': 'None'},
      'Ecoforms_Plant_Container_FB6_Tur': {'scale_factor': 0.16166000000000003, 'category': 'None'},
      'Ecoforms_Plant_Container_GP16AMOCHA': {'scale_factor': 0.162711, 'category': 'None'},
      'Ecoforms_Plant_Container_GP16A_Coral': {'scale_factor': 0.164742, 'category': 'None'},
      'Ecoforms_Plant_Container_QP6CORAL': {'scale_factor': 0.19124999999999998, 'category': 'None'},
      'Ecoforms_Plant_Container_QP6HARVEST': {'scale_factor': 0.190191, 'category': 'None'},
      'Ecoforms_Plant_Container_QP_Harvest': {'scale_factor': 0.08998899999999999, 'category': 'None'},
      'Ecoforms_Plant_Container_QP_Turquoise': {'scale_factor': 0.089812, 'category': 'None'},
      'Ecoforms_Plant_Container_Quadra_Sand_QP6': {'scale_factor': 0.190355, 'category': 'None'},
      'Ecoforms_Plant_Container_Quadra_Turquoise_QP12': {'scale_factor': 0.282694, 'category': 'None'},
      'Ecoforms_Plant_Container_S14Turquoise': {'scale_factor': 0.147529, 'category': 'None'},
      'Ecoforms_Plant_Container_S24NATURAL': {'scale_factor': 0.241269, 'category': 'None'},
      'Ecoforms_Plant_Container_S24Turquoise': {'scale_factor': 0.241894, 'category': 'None'},
      'Ecoforms_Plant_Container_SB9Turquoise': {'scale_factor': 0.29974500000000004, 'category': 'None'},
      'Ecoforms_Plant_Container_URN_NAT': {'scale_factor': 0.15993200000000002, 'category': 'Consumer Goods'},
      'Ecoforms_Plant_Container_URN_SAN': {'scale_factor': 0.159359, 'category': 'Consumer Goods'},
      'Ecoforms_Plant_Container_Urn_55_Avocado': {'scale_factor': 0.159915, 'category': 'None'},
      'Ecoforms_Plant_Container_Urn_55_Mocha': {'scale_factor': 0.159915, 'category': 'None'},
      'Ecoforms_Plant_Plate_S11Turquoise': {'scale_factor': 0.118147, 'category': 'None'},
      'Ecoforms_Plant_Pot_GP9AAvocado': {'scale_factor': 0.094238, 'category': 'None'},
      'Ecoforms_Plant_Pot_GP9_SAND': {'scale_factor': 0.094213, 'category': 'None'},
      'Ecoforms_Plant_Saucer_S14MOCHA': {'scale_factor': 0.147928, 'category': 'None'},
      'Ecoforms_Plant_Saucer_S14NATURAL': {'scale_factor': 0.14758700000000002, 'category': 'None'},
      'Ecoforms_Plant_Saucer_S17MOCHA': {'scale_factor': 0.17692, 'category': 'None'},
      'Ecoforms_Plant_Saucer_S20MOCHA': {'scale_factor': 0.206587, 'category': 'None'},
      'Ecoforms_Plant_Saucer_SQ1HARVEST': {'scale_factor': 0.086505, 'category': 'None'},
      'Ecoforms_Plant_Saucer_SQ8COR': {'scale_factor': 0.180842, 'category': 'None'},
      'Ecoforms_Planter_Bowl_Cole_Hardware': {'scale_factor': 0.18074400000000002, 'category': 'None'},
      'Ecoforms_Planter_Pot_GP12AAvocado': {'scale_factor': 0.118599, 'category': 'None'},
      'Ecoforms_Planter_Pot_QP6Ebony': {'scale_factor': 0.189654, 'category': 'None'},
      'Ecoforms_Plate_S20Avocado': {'scale_factor': 0.207011, 'category': 'None'},
      'Ecoforms_Pot_Nova_6_Turquoise': {'scale_factor': 0.18079499999999998, 'category': 'None'},
      'Ecoforms_Quadra_Saucer_SQ1_Avocado': {'scale_factor': 0.087083, 'category': 'None'},
      'Ecoforms_Saucer_SQ3_Turquoise': {'scale_factor': 0.26184, 'category': 'None'},
      'Elephant': {'scale_factor': 0.270548, 'category': 'None'},
      'Embark_Lunch_Cooler_Blue': {'scale_factor': 0.295574, 'category': 'None'},
      'Envision_Home_Dish_Drying_Mat_Red_6_x_18': {'scale_factor': 0.40988199999999997, 'category': 'None'},
      'Epson_273XL_Ink_Cartridge_Magenta': {'scale_factor': 0.160327, 'category': 'Consumer Goods'},
      'Epson_DURABrite_Ultra_786_Black_Ink_Cartridge_T786120S': {'scale_factor': 0.181693, 'category': 'Consumer Goods'},
      'Epson_Ink_Cartridge_126_Yellow': {'scale_factor': 0.11465800000000001, 'category': 'Consumer Goods'},
      'Epson_Ink_Cartridge_Black_200': {'scale_factor': 0.11465800000000001, 'category': 'Consumer Goods'},
      'Epson_LabelWorks_LC4WBN9_Tape_reel_labels_047_x_295_Roll_Black_on_White': {'scale_factor': 0.123004, 'category': 'Consumer Goods'},
      'Epson_LabelWorks_LC5WBN9_Tape_reel_labels_071_x_295_Roll_Black_on_White': {'scale_factor': 0.123356, 'category': 'Consumer Goods'},
      'Epson_T5803_Ink_Cartridge_Magenta_1pack': {'scale_factor': 0.102123, 'category': 'Consumer Goods'},
      'Epson_UltraChrome_T0543_Ink_Cartridge_Magenta_1pack': {'scale_factor': 0.115055, 'category': 'Consumer Goods'},
      'Epson_UltraChrome_T0548_Ink_Cartridge_Matte_Black_1pack': {'scale_factor': 0.11535699999999999, 'category': 'Consumer Goods'},
      'F10_TRX_FG_ssscuo9tGxb': {'scale_factor': 0.282471, 'category': 'Shoe'},
      'F10_TRX_TF_rH7tmKCdUJq': {'scale_factor': 0.289659, 'category': 'Shoe'},
      'F5_TRX_FG': {'scale_factor': 0.28303, 'category': 'Shoe'},
      'FAIRY_TALE_BLOCKS': {'scale_factor': 0.26503899999999997, 'category': 'Toys'},
      'FARM_ANIMAL': {'scale_factor': 0.176346, 'category': 'Toys'},
      'FARM_ANIMAL_9GyfdcPyESK': {'scale_factor': 0.182681, 'category': 'Toys'},
      'FIRE_ENGINE': {'scale_factor': 0.137119, 'category': 'Toys'},
      'FIRE_TRUCK': {'scale_factor': 0.124043, 'category': 'Toys'},
      'FISHING_GAME': {'scale_factor': 0.393482, 'category': 'Toys'},
      'FOOD_BEVERAGE_SET': {'scale_factor': 0.267115, 'category': 'Toys'},
      'FRACTION_FUN_n4h4qte23QR': {'scale_factor': 0.189145, 'category': 'Toys'},
      'FRUIT_VEGGIE_DOMINO_GRADIENT': {'scale_factor': 0.347707, 'category': 'Toys'},
      'FRUIT_VEGGIE_MEMO_GRADIENT': {'scale_factor': 0.268224, 'category': 'Toys'},
      'FYW_ALTERNATION': {'scale_factor': 0.306532, 'category': 'Shoe'},
      'FYW_DIVISION': {'scale_factor': 0.306284, 'category': 'Shoe'},
      'FemDophilus': {'scale_factor': 0.114582, 'category': 'Consumer Goods'},
      'Final_Fantasy_XIV_A_Realm_Reborn_60Day_Subscription': {'scale_factor': 0.190407, 'category': 'Media Cases'},
      'Firefly_Clue_Board_Game': {'scale_factor': 0.404104, 'category': 'Consumer Goods'},
      'FisherPrice_Make_A_Match_Game_Thomas_Friends': {'scale_factor': 0.267802, 'category': 'Consumer Goods'},
      'Fisher_price_Classic_Toys_Buzzy_Bee': {'scale_factor': 0.205903, 'category': 'None'},
      'Focus_8643_Lime_Squeezer_10x35x188_Enamelled_Aluminum_Light': {'scale_factor': 0.33557800000000004, 'category': 'None'},
      'Folic_Acid': {'scale_factor': 0.083858, 'category': 'Bottles and Cans and Cups'},
      'Footed_Bowl_Sand': {'scale_factor': 0.161543, 'category': 'Consumer Goods'},
      'Fresca_Peach_Citrus_Sparkling_Flavored_Soda_12_PK': {'scale_factor': 0.40788599999999997, 'category': 'Consumer Goods'},
      'Frozen_Olafs_In_Trouble_PopOMatic_Game': {'scale_factor': 0.27336, 'category': 'Board Games'},
      'Frozen_Olafs_In_Trouble_PopOMatic_Game_OEu83W9T8pD': {'scale_factor': 0.26062399999999997, 'category': 'Board Games'},
      'Frozen_Scrabble_Jr': {'scale_factor': 0.271804, 'category': 'Board Games'},
      'Fruity_Friends': {'scale_factor': 0.172669, 'category': 'Consumer Goods'},
      'Fujifilm_instax_SHARE_SP1_10_photos': {'scale_factor': 0.12373300000000001, 'category': 'None'},
      'Full_Circle_Happy_Scraps_Out_Collector_Gray': {'scale_factor': 0.213046, 'category': 'None'},
      'GARDEN_SWING': {'scale_factor': 0.16647, 'category': 'Toys'},
      'GEARS_PUZZLES_STANDARD_gcYxhNHhKlI': {'scale_factor': 0.311234, 'category': 'Toys'},
      'GEOMETRIC_PEG_BOARD': {'scale_factor': 0.174989, 'category': 'Toys'},
      'GEOMETRIC_SORTING_BOARD': {'scale_factor': 0.174722, 'category': 'Toys'},
      'GEOMETRIC_SORTING_BOARD_MNi4Rbuz9vj': {'scale_factor': 0.36226499999999995, 'category': 'Toys'},
      'GIRLS_DECKHAND': {'scale_factor': 0.25479300000000005, 'category': 'Shoe'},
      'GRANDFATHER_DOLL': {'scale_factor': 0.129042, 'category': 'Toys'},
      'GRANDMOTHER': {'scale_factor': 0.12684800000000002, 'category': 'Toys'},
      'Germanium_GE132': {'scale_factor': 0.077376, 'category': 'Bottles and Cans and Cups'},
      'Ghost_6_Color_BlckWhtLavaSlvrCitrus_Size_80': {'scale_factor': 0.298198, 'category': 'Shoe'},
      'Ghost_6_Color_MdngtDenmPomBrtePnkSlvBlk_Size_50': {'scale_factor': 0.284107, 'category': 'Shoe'},
      'Ghost_6_GTX_Color_AnthBlckSlvrFernSulphSprng_Size_80': {'scale_factor': 0.29041799999999995, 'category': 'Shoe'},
      'Gigabyte_GA78LMTUSB3_50_Motherboard_Micro_ATX_Socket_AM3': {'scale_factor': 0.271574, 'category': 'None'},
      'Gigabyte_GA970AUD3P_10_Motherboard_ATX_Socket_AM3': {'scale_factor': 0.334386, 'category': 'None'},
      'Gigabyte_GAZ97XSLI_10_motherboard_ATX_LGA1150_Socket_Z97': {'scale_factor': 0.334685, 'category': 'None'},
      'Glycerin_11_Color_AqrsDrsdnBluBlkSlvShckOrng_Size_50': {'scale_factor': 0.28794600000000004, 'category': 'Shoe'},
      'Glycerin_11_Color_BrllntBluSkydvrSlvrBlckWht_Size_80': {'scale_factor': 0.304977, 'category': 'Shoe'},
      'GoPro_HERO3_Composite_Cable': {'scale_factor': 0.098623, 'category': 'None'},
      'Google_Cardboard_Original_package': {'scale_factor': 0.214329, 'category': 'Consumer Goods'},
      'Grand_Prix': {'scale_factor': 0.285006, 'category': 'Shoe'},
      'Granimals_20_Wooden_ABC_Blocks_Wagon': {'scale_factor': 0.190035, 'category': 'None'},
      'Granimals_20_Wooden_ABC_Blocks_Wagon_85VdSftGsLi': {'scale_factor': 0.031189, 'category': 'None'},
      'Granimals_20_Wooden_ABC_Blocks_Wagon_g2TinmUGGHI': {'scale_factor': 0.191824, 'category': 'None'},
      'Great_Dinos_Triceratops_Toy': {'scale_factor': 0.079184, 'category': 'None'},
      'Great_Jones_Wingtip': {'scale_factor': 0.31214, 'category': 'Shoe'},
      'Great_Jones_Wingtip_j5NV8GRnitM': {'scale_factor': 0.316897, 'category': 'Shoe'},
      'Great_Jones_Wingtip_kAqSg6EgG0I': {'scale_factor': 0.312017, 'category': 'Shoe'},
      'Great_Jones_Wingtip_wxH3dbtlvBC': {'scale_factor': 0.316946, 'category': 'Shoe'},
      'Grreat_Choice_Dog_Double_Dish_Plastic_Blue': {'scale_factor': 0.338102, 'category': 'None'},
      'Grreatv_Choice_Dog_Bowl_Gray_Bones_Plastic_20_fl_oz_total': {'scale_factor': 0.183525, 'category': 'None'},
      'Guardians_of_the_Galaxy_Galactic_Battlers_Rocket_Raccoon_Figure': {'scale_factor': 0.101443, 'category': 'Action Figures'},
      'HAMMER_BALL': {'scale_factor': 0.28392, 'category': 'Toys'},
      'HAMMER_PEG': {'scale_factor': 0.252371, 'category': 'Toys'},
      'HAPPY_ENGINE': {'scale_factor': 0.26964600000000005, 'category': 'Toys'},
      'HELICOPTER': {'scale_factor': 0.173153, 'category': 'Toys'},
      'HP_1800_Tablet_8GB_7': {'scale_factor': 0.19339, 'category': 'None'},
      'HP_Card_Invitation_Kit': {'scale_factor': 0.19645200000000002, 'category': 'Consumer Goods'},
      'Hasbro_Cranium_Performance_and_Acting_Game': {'scale_factor': 0.272031, 'category': 'Consumer Goods'},
      'Hasbro_Dont_Wake_Daddy_Board_Game': {'scale_factor': 0.40583199999999997, 'category': 'Consumer Goods'},
      'Hasbro_Dont_Wake_Daddy_Board_Game_NJnjGna4u1a': {'scale_factor': 0.500453, 'category': 'Consumer Goods'},
      'Hasbro_Life_Board_Game': {'scale_factor': 0.404874, 'category': 'Consumer Goods'},
      'Hasbro_Monopoly_Hotels_Game': {'scale_factor': 0.27975300000000003, 'category': 'Board Games'},
      'Hasbro_Trivial_Pursuit_Family_Edition_Game': {'scale_factor': 0.271042, 'category': 'Board Games'},
      'HeavyDuty_Flashlight': {'scale_factor': 0.229205, 'category': 'None'},
      'Hefty_Waste_Basket_Decorative_Bronze_85_liter': {'scale_factor': 0.300217, 'category': 'None'},
      'Hey_You_Pikachu_Nintendo_64': {'scale_factor': 0.116535, 'category': 'Consumer Goods'},
      'Hilary': {'scale_factor': 0.23549899999999999, 'category': 'Shoe'},
      'Home_Fashions_Washcloth_Linen': {'scale_factor': 0.17948799999999998, 'category': 'None'},
      'Home_Fashions_Washcloth_Olive_Green': {'scale_factor': 0.176647, 'category': 'None'},
      'Horse_Dreams_Pencil_Case': {'scale_factor': 0.21861999999999998, 'category': 'Bag'},
      'Horses_in_Pink_Pencil_Case': {'scale_factor': 0.21417, 'category': 'Bag'},
      'House_of_Cards_The_Complete_First_Season_4_Discs_DVD': {'scale_factor': 0.19093, 'category': 'Media Cases'},
      'Hyaluronic_Acid': {'scale_factor': 0.097981, 'category': 'Bottles and Cans and Cups'},
      'HyperX_Cloud_II_Headset_Gun_Metal': {'scale_factor': 0.32954399999999995, 'category': 'Headphones'},
      'HyperX_Cloud_II_Headset_Red': {'scale_factor': 0.33128, 'category': 'Headphones'},
      'INTERNATIONAL_PAPER_Willamette_4_Brown_Bag_500Count': {'scale_factor': 0.27007499999999995, 'category': 'None'},
      'Imaginext_Castle_Ogre': {'scale_factor': 0.395655, 'category': 'Action Figures'},
      'In_Green_Company_Surface_Saver_Ring_10_Terra_Cotta': {'scale_factor': 0.25343499999999997, 'category': 'None'},
      'Inositol': {'scale_factor': 0.110232, 'category': 'Bottles and Cans and Cups'},
      'InterDesign_Over_Door': {'scale_factor': 0.291241, 'category': 'None'},
      'IsoRich_Soy': {'scale_factor': 0.16883900000000002, 'category': 'Bottles and Cans and Cups'},
      'JA_Henckels_International_Premio_Cutlery_Block_Set_14Piece': {'scale_factor': 0.35232399999999997, 'category': 'None'},
      'JBL_Charge_Speaker_portable_wireless_wired_Green': {'scale_factor': 0.175699, 'category': 'Consumer Goods'},
      'JS_WINGS_20_BLACK_FLAG': {'scale_factor': 0.35365800000000003, 'category': 'Shoe'},
      'JUICER_SET': {'scale_factor': 0.286507, 'category': 'Toys'},
      'JUNGLE_HEIGHT': {'scale_factor': 0.372783, 'category': 'Toys'},
      'Jansport_School_Backpack_Blue_Streak': {'scale_factor': 0.440148, 'category': 'Bag'},
      'JarroDophilusFOS_Value_Size': {'scale_factor': 0.12851200000000002, 'category': 'Bottles and Cans and Cups'},
      'JarroSil_Activated_Silicon': {'scale_factor': 0.10902200000000001, 'category': 'Consumer Goods'},
      'JarroSil_Activated_Silicon_5exdZHIeLAp': {'scale_factor': 0.09232399999999999, 'category': 'Consumer Goods'},
      'Jarrow_Formulas_Glucosamine_Hci_Mega_1000_100_ct': {'scale_factor': 0.11046, 'category': 'Bottles and Cans and Cups'},
      'Jarrow_Glucosamine_Chondroitin_Combination_120_Caps': {'scale_factor': 0.11624699999999999, 'category': 'Bottles and Cans and Cups'},
      'Jawbone_UP24_Wireless_Activity_Tracker_Pink_Coral_L': {'scale_factor': 0.08122499999999999, 'category': 'None'},
      'Just_For_Men_Mustache_Beard_Brushin_Hair_Color_Gel_Kit_Jet_Black_M60': {'scale_factor': 0.143814, 'category': 'Consumer Goods'},
      'Just_For_Men_Mustache_Beard_Brushin_Hair_Color_Gel_MediumDark_Brown_M40': {'scale_factor': 0.145106, 'category': 'Consumer Goods'},
      'Just_For_Men_ShampooIn_Haircolor_Jet_Black_60': {'scale_factor': 0.156393, 'category': 'Consumer Goods'},
      'Just_For_Men_ShampooIn_Haircolor_Light_Brown_25': {'scale_factor': 0.157072, 'category': 'Consumer Goods'},
      'Just_For_Men_Shampoo_In_Haircolor_Darkest_Brown_50': {'scale_factor': 0.156759, 'category': 'Consumer Goods'},
      'Justified_The_Complete_Fourth_Season_3_Discs_DVD': {'scale_factor': 0.19405300000000003, 'category': 'Media Cases'},
      'KID_ROOM_FURNITURE_SET_1': {'scale_factor': 0.233539, 'category': 'Toys'},
      'KITCHEN_FURNITURE_SET_1': {'scale_factor': 0.180027, 'category': 'Toys'},
      'KITCHEN_SET_CLASSIC_40HwCHfeG0H': {'scale_factor': 0.18543500000000002, 'category': 'Toys'},
      'KS_Chocolate_Cube_Box_Assortment_By_Neuhaus_2010_Ounces': {'scale_factor': 0.184474, 'category': 'Consumer Goods'},
      'Kanex_MultiSync_Wireless_Keyboard': {'scale_factor': 0.44181899999999996, 'category': 'Keyboard'},
      'Kid_Icarus_Uprising_Nintendo_3DS_Game': {'scale_factor': 0.142098, 'category': 'Media Cases'},
      'Kingston_DT4000MR_G2_Management_Ready_USB_64GB': {'scale_factor': 0.07947699999999999, 'category': 'None'},
      'Kong_Puppy_Teething_Rubber_Small_Pink': {'scale_factor': 0.071657, 'category': 'Consumer Goods'},
      'Kotex_U_Barely_There_Liners_Thin_60_count': {'scale_factor': 0.130919, 'category': 'Consumer Goods'},
      'Kotex_U_Tween_Pads_16_pads': {'scale_factor': 0.154596, 'category': 'Consumer Goods'},
      'Kotobuki_Saucer_Dragon_Fly': {'scale_factor': 0.09667500000000001, 'category': 'None'},
      'Krill_Oil': {'scale_factor': 0.09757, 'category': 'Consumer Goods'},
      'LACING_SHEEP': {'scale_factor': 0.135763, 'category': 'Toys'},
      'LADYBUG_BEAD': {'scale_factor': 0.08421100000000001, 'category': 'Toys'},
      'LEGO_5887_Dino_Defense_HQ': {'scale_factor': 0.451717, 'category': 'Legos'},
      'LEGO_Bricks_More_Creative_Suitcase': {'scale_factor': 0.381096, 'category': 'Legos'},
      'LEGO_City_Advent_Calendar': {'scale_factor': 0.383465, 'category': 'Legos'},
      'LEGO_Creationary_Game': {'scale_factor': 0.282487, 'category': 'Consumer Goods'},
      'LEGO_Creationary_Game_ZJa163wlWp2': {'scale_factor': 0.435707, 'category': 'Consumer Goods'},
      'LEGO_Duplo_Build_and_Play_Box_4629': {'scale_factor': 0.43963399999999997, 'category': 'Legos'},
      'LEGO_Duplo_Creative_Animals_10573': {'scale_factor': 0.389224, 'category': 'Legos'},
      'LEGO_Fusion_Set_Town_Master': {'scale_factor': 0.35358, 'category': 'Board Games'},
      'LEGO_Star_Wars_Advent_Calendar': {'scale_factor': 0.385183, 'category': 'Legos'},
      'LEUCIPPUS_ADIPURE': {'scale_factor': 0.269211, 'category': 'Shoe'},
      'LTyrosine': {'scale_factor': 0.11058, 'category': 'Bottles and Cans and Cups'},
      'Lactoferrin': {'scale_factor': 0.099589, 'category': 'Bottles and Cans and Cups'},
      'Lalaloopsy_Peanut_Big_Top_Tricycle': {'scale_factor': 0.11840300000000001, 'category': 'Toys'},
      'Lavender_Snake_Tieks_Snake_Print_Ballet_Flats': {'scale_factor': 0.241577, 'category': 'Shoe'},
      'Leap_Frog_Paint_Dabber_Dot_Art_5_paint_bottles': {'scale_factor': 0.235717, 'category': 'Consumer Goods'},
      'Lego_Friends_Advent_Calendar': {'scale_factor': 0.385037, 'category': 'Legos'},
      'Lego_Friends_Mia': {'scale_factor': 0.198884, 'category': 'Legos'},
      'Lenovo_Yoga_2_11': {'scale_factor': 0.319921, 'category': 'None'},
      'Little_Big_Planet_3_Plush_Edition': {'scale_factor': 0.179867, 'category': 'Consumer Goods'},
      'Little_Debbie_Chocolate_Cupcakes_8_ct': {'scale_factor': 0.29843, 'category': 'Consumer Goods'},
      'Little_Debbie_Cloud_Cakes_10_ct': {'scale_factor': 0.24258000000000002, 'category': 'Consumer Goods'},
      'Little_Debbie_Donut_Sticks_6_cake_donuts_10_oz_total': {'scale_factor': 0.31245599999999996, 'category': 'Consumer Goods'},
      'Little_House_on_the_Prairie_Season_Two_5_Discs_Includes_Digital': {'scale_factor': 0.193604, 'category': 'Media Cases'},
      'Logitech_Ultimate_Ears_Boom_Wireless_Speaker_Night_Black': {'scale_factor': 0.260666, 'category': 'None'},
      'Lovable_Huggable_Cuddly_Boutique_Teddy_Bear_Beige': {'scale_factor': 0.370612, 'category': 'Stuffed Toys'},
      'Lovestruck_Tieks_Glittery_Rose_Gold_Italian_Leather_Ballet_Flats': {'scale_factor': 0.24568, 'category': 'Shoe'},
      'Luigis_Mansion_Dark_Moon_Nintendo_3DS_Game': {'scale_factor': 0.137881, 'category': 'Media Cases'},
      'Lutein': {'scale_factor': 0.09719900000000001, 'category': 'Bottles and Cans and Cups'},
      'MARTIN_WEDGE_LACE_BOOT': {'scale_factor': 0.29689299999999996, 'category': 'Shoe'},
      'MEAT_SET': {'scale_factor': 0.256775, 'category': 'Toys'},
      'MINI_EXCAVATOR': {'scale_factor': 0.171634, 'category': 'Toys'},
      'MINI_FIRE_ENGINE': {'scale_factor': 0.176744, 'category': 'Toys'},
      'MINI_ROLLER': {'scale_factor': 0.143661, 'category': 'Toys'},
      'MIRACLE_POUNDING': {'scale_factor': 0.29329700000000003, 'category': 'Toys'},
      'MK7': {'scale_factor': 0.08407, 'category': 'Bottles and Cans and Cups'},
      'MODERN_DOLL_FAMILY': {'scale_factor': 0.21216600000000002, 'category': 'Toys'},
      'MONKEY_BOWLING': {'scale_factor': 0.324229, 'category': 'Toys'},
      'MOSAIC': {'scale_factor': 0.276606, 'category': 'Toys'},
      'MOVING_MOUSE_PW_6PCSSET': {'scale_factor': 0.178139, 'category': 'Toys'},
      'MY_MOOD_MEMO': {'scale_factor': 0.33877100000000004, 'category': 'Toys'},
      'Mad_Gab_Refresh_Card_Game': {'scale_factor': 0.156004, 'category': 'Board Games'},
      'Magnifying_Glassassrt': {'scale_factor': 0.15183600000000003, 'category': 'None'},
      'Marc_Anthony_Skip_Professional_Oil_of_Morocco_Conditioner_with_Argan_Oil': {'scale_factor': 0.192133, 'category': 'Bottles and Cans and Cups'},
      'Marc_Anthony_Strictly_Curls_Curl_Envy_Perfect_Curl_Cream_6_fl_oz_bottle': {'scale_factor': 0.223464, 'category': 'Consumer Goods'},
      'Marc_Anthony_True_Professional_Oil_of_Morocco_Argan_Oil_Treatment': {'scale_factor': 0.162384, 'category': 'Consumer Goods'},
      'Marc_Anthony_True_Professional_Strictly_Curls_Curl_Defining_Lotion': {'scale_factor': 0.195286, 'category': 'Bottles and Cans and Cups'},
      'Mario_Luigi_Dream_Team_Nintendo_3DS_Game': {'scale_factor': 0.13748, 'category': 'Media Cases'},
      'Mario_Party_9_Wii_Game': {'scale_factor': 0.191257, 'category': 'Media Cases'},
      'Markings_Desk_Caddy': {'scale_factor': 0.293344, 'category': 'None'},
      'Markings_Letter_Holder': {'scale_factor': 0.19837300000000002, 'category': 'None'},
      'Marvel_Avengers_Titan_Hero_Series_Doctor_Doom': {'scale_factor': 0.291686, 'category': 'Action Figures'},
      'Mastic_Gum': {'scale_factor': 0.09726399999999999, 'category': 'Bottles and Cans and Cups'},
      'Matte_Black_Tieks_Italian_Leather_Ballet_Flats': {'scale_factor': 0.24087699999999998, 'category': 'Shoe'},
      'Mattel_SKIP_BO_Card_Game': {'scale_factor': 0.204601, 'category': 'Consumer Goods'},
      'Melissa_Doug_Cart_Turtle_Block': {'scale_factor': 0.352165, 'category': 'None'},
      'Melissa_Doug_Chunky_Puzzle_Vehicles': {'scale_factor': 0.304855, 'category': 'None'},
      'Melissa_Doug_Felt_Food_Pizza_Set': {'scale_factor': 0.27700199999999997, 'category': 'None'},
      'Melissa_Doug_Jumbo_Knob_Puzzles_Barnyard_Animals': {'scale_factor': 0.295261, 'category': 'Toys'},
      'Melissa_Doug_Pattern_Blocks_and_Boards': {'scale_factor': 0.381367, 'category': 'Consumer Goods'},
      'Melissa_Doug_Pound_and_Roll': {'scale_factor': 0.229482, 'category': 'Consumer Goods'},
      'Melissa_Doug_See_Spell': {'scale_factor': 0.361282, 'category': 'Consumer Goods'},
      'Melissa_Doug_Shape_Sorting_Clock': {'scale_factor': 0.306452, 'category': 'None'},
      'Melissa_Doug_Traffic_Signs_and_Vehicles': {'scale_factor': 0.34940499999999997, 'category': 'Consumer Goods'},
      'Mens_ASV_Billfish_Boat_Shoe_in_Dark_Brown_Leather_zdHVHXueI3w': {'scale_factor': 0.29263, 'category': 'Shoe'},
      'Mens_ASV_Billfish_Boat_Shoe_in_Tan_Leather_wmUJ5PbwANc': {'scale_factor': 0.28664500000000004, 'category': 'Shoe'},
      'Mens_ASV_Shock_Light_Bungee_in_Light_Grey_xGCOvtLDnQJ': {'scale_factor': 0.287325, 'category': 'Shoe'},
      'Mens_Authentic_Original_Boat_Shoe_in_Navy_Leather_NHHQddDLQys': {'scale_factor': 0.288783, 'category': 'Shoe'},
      'Mens_Authentic_Original_Boat_Shoe_in_Navy_Leather_RpT4GvUXRRP': {'scale_factor': 0.24981599999999998, 'category': 'Shoe'},
      'Mens_Authentic_Original_Boat_Shoe_in_Navy_Leather_xgoEcZtRNmH': {'scale_factor': 0.25005, 'category': 'Shoe'},
      'Mens_Bahama_in_Black_b4ADzYywRHl': {'scale_factor': 0.286945, 'category': 'Shoe'},
      'Mens_Bahama_in_Khaki_Oyster_xU2jeqYwhQJ': {'scale_factor': 0.28716600000000003, 'category': 'Shoe'},
      'Mens_Bahama_in_White_vSwvGMCo32f': {'scale_factor': 0.287218, 'category': 'Shoe'},
      'Mens_Billfish_3Eye_Boat_Shoe_in_Dark_Tan_wyns9HRcEuH': {'scale_factor': 0.29232199999999997, 'category': 'Shoe'},
      'Mens_Billfish_Slip_On_in_Coffee_e8bPKE9Lfgo': {'scale_factor': 0.28627400000000003, 'category': 'Shoe'},
      'Mens_Billfish_Slip_On_in_Coffee_nK6AJJAHOae': {'scale_factor': 0.286358, 'category': 'Shoe'},
      'Mens_Billfish_Slip_On_in_Tan_Beige_aaVUk0tNTv8': {'scale_factor': 0.291067, 'category': 'Shoe'},
      'Mens_Billfish_Ultra_Lite_Boat_Shoe_in_Dark_Brown_Blue_c6zDZTtRJr6': {'scale_factor': 0.291693, 'category': 'Shoe'},
      'Mens_Gold_Cup_ASV_2Eye_Boat_Shoe_in_Cognac_Leather': {'scale_factor': 0.29330300000000004, 'category': 'Shoe'},
      'Mens_Gold_Cup_ASV_Capetown_Penny_Loafer_in_Black_EjPnk3E8fCs': {'scale_factor': 0.290352, 'category': 'Shoe'},
      'Mens_Gold_Cup_ASV_Capetown_Penny_Loafer_in_Black_GkQBKqABeQN': {'scale_factor': 0.290134, 'category': 'Shoe'},
      'Mens_Gold_Cup_ASV_Dress_Casual_Venetian_in_Dark_Brown_Leather': {'scale_factor': 0.294626, 'category': 'Shoe'},
      'Mens_Largo_Slip_On_in_Taupe_gooyS417q4R': {'scale_factor': 0.29527000000000003, 'category': 'Shoe'},
      'Mens_Mako_Canoe_Moc_2Eye_Boat_Shoe_in_Coffee_9d05GG33QQQ': {'scale_factor': 0.290971, 'category': 'Shoe'},
      'Mens_Mako_Canoe_Moc_2Eye_Boat_Shoe_in_Coffee_K9e8FoV73uZ': {'scale_factor': 0.287227, 'category': 'Shoe'},
      'Mens_Mako_Canoe_Moc_2Eye_Boat_Shoe_in_OysterTaupe_otyRrfvPMiA': {'scale_factor': 0.287822, 'category': 'Shoe'},
      'Mens_RR_Moc_in_Navy_Suede_vmFfijhBzL3': {'scale_factor': 0.286655, 'category': 'Shoe'},
      'Mens_Santa_Cruz_Thong_in_Chocolate_La1fo2mAovE': {'scale_factor': 0.292392, 'category': 'Shoe'},
      'Mens_Santa_Cruz_Thong_in_Chocolate_lvxYW7lek6B': {'scale_factor': 0.29192399999999996, 'category': 'Shoe'},
      'Mens_Santa_Cruz_Thong_in_Tan_r59C69daRPh': {'scale_factor': 0.29359999999999997, 'category': 'Shoe'},
      'Mens_Striper_Sneaker_in_White_rnp8HUli59Y': {'scale_factor': 0.285544, 'category': 'Shoe'},
      'Mens_Tremont_Kiltie_Tassel_Loafer_in_Black_Amaretto': {'scale_factor': 0.29502700000000004, 'category': 'Shoe'},
      'Mens_Tremont_Kiltie_Tassel_Loafer_in_Black_Amaretto_FT0I9OjSA6O': {'scale_factor': 0.295296, 'category': 'Shoe'},
      'Mens_Tremont_Kiltie_Tassel_Loafer_in_Black_Amaretto_rCdzRZqgCnI': {'scale_factor': 0.29519, 'category': 'Shoe'},
      'Mens_Wave_Driver_Kiltie_Moc_in_Tan_Leather': {'scale_factor': 0.27839400000000003, 'category': 'Shoe'},
      'Metallic_Gold_Tieks_Italian_Leather_Ballet_Flats': {'scale_factor': 0.246286, 'category': 'Shoe'},
      'Metallic_Pewter_Tieks_Italian_Leather_Ballet_Flats': {'scale_factor': 0.246666, 'category': 'Shoe'},
      'Mist_Wipe_Warmer': {'scale_factor': 0.264191, 'category': 'None'},
      'My_First_Animal_Tower': {'scale_factor': 0.353131, 'category': 'Toys'},
      'My_First_Rolling_Lion': {'scale_factor': 0.15856599999999998, 'category': 'Toys'},
      'My_First_Wiggle_Crocodile': {'scale_factor': 0.197243, 'category': 'Toys'},
      'My_Little_Pony_Princess_Celestia': {'scale_factor': 0.231977, 'category': 'Toys'},
      'My_Monopoly_Board_Game': {'scale_factor': 0.404929, 'category': 'Board Games'},
      'NAPA_VALLEY_NAVAJO_SANDAL': {'scale_factor': 0.23256100000000002, 'category': 'Shoe'},
      'NESCAFE_NESCAFE_TC_STKS_DECAF_6_CT': {'scale_factor': 0.131175, 'category': 'Consumer Goods'},
      'NUTS_BOLTS': {'scale_factor': 0.18021500000000001, 'category': 'Toys'},
      'NattoMax': {'scale_factor': 0.088837, 'category': 'Bottles and Cans and Cups'},
      'Neat_Solutions_Character_Bib_2_pack': {'scale_factor': 0.205155, 'category': 'None'},
      'Nescafe_16Count_Dolce_Gusto_Cappuccino_Capsules': {'scale_factor': 0.123972, 'category': 'Consumer Goods'},
      'Nescafe_Memento_Latte_Caramel_8_08_oz_23_g_packets_64_oz_184_g': {'scale_factor': 0.160642, 'category': 'Consumer Goods'},
      'Nescafe_Momento_Mocha_Specialty_Coffee_Mix_8_ct': {'scale_factor': 0.161386, 'category': 'Consumer Goods'},
      'Nescafe_Tasters_Choice_Instant_Coffee_Decaf_House_Blend_Light_7_oz': {'scale_factor': 0.185732, 'category': 'Bottles and Cans and Cups'},
      'Nestl_Crunch_Girl_Scouts_Cookie_Flavors_Caramel_Coconut_78_oz_box': {'scale_factor': 0.163043, 'category': 'Consumer Goods'},
      'Nestl_Skinny_Cow_Heavenly_Crisp_Candy_Bar_Chocolate_Raspberry_6_pack_462_oz_total': {'scale_factor': 0.164115, 'category': 'Consumer Goods'},
      'Nestle_Candy_19_oz_Butterfinger_Singles_116567': {'scale_factor': 0.168226, 'category': 'Consumer Goods'},
      'Nestle_Carnation_Cinnamon_Coffeecake_Kit_1913OZ': {'scale_factor': 0.19508999999999999, 'category': 'Consumer Goods'},
      'Nestle_Nesquik_Chocolate_Powder_Flavored_Milk_Additive_109_Oz_Canister': {'scale_factor': 0.143738, 'category': 'Consumer Goods'},
      'Nestle_Nips_Hard_Candy_Peanut_Butter': {'scale_factor': 0.15333799999999997, 'category': 'Consumer Goods'},
      'Nestle_Pure_Life_Exotics_Sparkling_Water_Strawberry_Dragon_Fruit_8_count_12_fl_oz_can': {'scale_factor': 0.272968, 'category': 'Consumer Goods'},
      'Nestle_Pure_Life_Exotics_Sparkling_Water_Strawberry_Dragon_Fruit_8_count_12_fl_oz_can_aX0ygjh3bxi': {'scale_factor': 0.271017, 'category': 'Consumer Goods'},
      'Nestle_Raisinets_Milk_Chocolate_35_oz_992_g': {'scale_factor': 0.162129, 'category': 'Consumer Goods'},
      'Nestle_Skinny_Cow_Dreamy_Clusters_Candy_Dark_Chocolate_6_pack_1_oz_pouches': {'scale_factor': 0.164595, 'category': 'Consumer Goods'},
      'Netgear_Ac1750_Router_Wireless_Dual_Band_Gigabit_Router': {'scale_factor': 0.348116, 'category': 'None'},
      'Netgear_N750_Wireless_Dual_Band_Gigabit_Router': {'scale_factor': 0.29625999999999997, 'category': 'Consumer Goods'},
      'Netgear_Nighthawk_X6_AC3200_TriBand_Gigabit_Wireless_Router': {'scale_factor': 0.37441, 'category': 'Consumer Goods'},
      'New_Super_Mario_BrosWii_Wii_Game': {'scale_factor': 0.191721, 'category': 'Media Cases'},
      'Nickelodeon_Teenage_Mutant_Ninja_Turtles_Leonardo': {'scale_factor': 0.117277, 'category': 'Action Figures'},
      'Nickelodeon_Teenage_Mutant_Ninja_Turtles_Michelangelo': {'scale_factor': 0.14171899999999998, 'category': 'Action Figures'},
      'Nickelodeon_Teenage_Mutant_Ninja_Turtles_Raphael': {'scale_factor': 0.12956199999999998, 'category': 'Action Figures'},
      'Nickelodeon_The_Spongebob_Movie_PopAPart_Spongebob': {'scale_factor': 0.22266, 'category': 'Toys'},
      'Nightmare_Before_Christmas_Collectors_Edition_Operation': {'scale_factor': 0.44450100000000003, 'category': 'Board Games'},
      'Nikon_1_AW1_w11275mm_Lens_Silver': {'scale_factor': 0.07863200000000001, 'category': 'Camera'},
      'Nintendo_2DS_Crimson_Red': {'scale_factor': 0.144277, 'category': 'None'},
      'Nintendo_Mario_Action_Figure': {'scale_factor': 0.102377, 'category': 'Toys'},
      'Nintendo_Wii_Party_U_with_Controller_Wii_U_Game': {'scale_factor': 0.197112, 'category': 'Media Cases'},
      'Nintendo_Yoshi_Action_Figure': {'scale_factor': 0.113041, 'category': 'Toys'},
      'Nips_Hard_Candy_Rich_Creamy_Butter_Rum_4_oz_1133_g': {'scale_factor': 0.15269, 'category': 'Consumer Goods'},
      'Nordic_Ware_Original_Bundt_Pan': {'scale_factor': 0.26387099999999997, 'category': 'None'},
      'Now_Designs_Bowl_Akita_Black': {'scale_factor': 0.15968300000000002, 'category': 'None'},
      'Now_Designs_Dish_Towel_Mojave_18_x_28': {'scale_factor': 0.357544, 'category': 'None'},
      'Now_Designs_Snack_Bags_Bicycle_2_count': {'scale_factor': 0.202011, 'category': 'None'},
      'OVAL_XYLOPHONE': {'scale_factor': 0.283712, 'category': 'Toys'},
      'OWL_SORTER': {'scale_factor': 0.197195, 'category': 'Toys'},
      'OXO_Cookie_Spatula': {'scale_factor': 0.24012099999999997, 'category': 'None'},
      'OXO_Soft_Works_Can_Opener_SnapLock': {'scale_factor': 0.177361, 'category': 'None'},
      'Object': {'scale_factor': 0.11785200000000001, 'category': 'None'},
      'Object_REmvBDJStub': {'scale_factor': 0.284672, 'category': 'None'},
      'Ocedar_Snap_On_Dust_Pan_And_Brush_1_ct': {'scale_factor': 0.330693, 'category': 'None'},
      'Office_Depot_Canon_CL211XL_Remanufactured_Ink_Cartridge_TriColor': {'scale_factor': 0.11785799999999999, 'category': 'Consumer Goods'},
      'Office_Depot_Canon_CLI36_Remanufactured_Ink_Cartridge_TriColor': {'scale_factor': 0.14718699999999998, 'category': 'Consumer Goods'},
      'Office_Depot_Canon_CLI_221BK_Ink_Cartridge_Black_2946B001': {'scale_factor': 0.118313, 'category': 'Consumer Goods'},
      'Office_Depot_Canon_CLI_8CMY_Remanufactured_Ink_Cartridges_Color_Cyan_Magenta_Yellow_3_count': {'scale_factor': 0.118424, 'category': 'Consumer Goods'},
      'Office_Depot_Canon_CLI_8Y_Ink_Cartridge_Yellow_0623B002': {'scale_factor': 0.118337, 'category': 'Consumer Goods'},
      'Office_Depot_Canon_CL_41_Remanufactured_Ink_Cartridge_TriColor': {'scale_factor': 0.118151, 'category': 'Consumer Goods'},
      'Office_Depot_Canon_PG21XL_Remanufactured_Ink_Cartridge_Black': {'scale_factor': 0.11788099999999999, 'category': 'Consumer Goods'},
      'Office_Depot_Canon_PGI22_Remanufactured_Ink_Cartridge_Black': {'scale_factor': 0.118959, 'category': 'Consumer Goods'},
      'Office_Depot_Canon_PGI35_Remanufactured_Ink_Cartridge_Black': {'scale_factor': 0.11832699999999999, 'category': 'Consumer Goods'},
      'Office_Depot_Canon_PGI5BK_Remanufactured_Ink_Cartridge_Black': {'scale_factor': 0.11785299999999999, 'category': 'Consumer Goods'},
      'Office_Depot_Canon_PG_240XL_Ink_Cartridge_Black_5206B001': {'scale_factor': 0.11843899999999999, 'category': 'Consumer Goods'},
      'Office_Depot_Dell_Series_11_Remanufactured_Ink_Cartridge_Black': {'scale_factor': 0.118783, 'category': 'Consumer Goods'},
      'Office_Depot_Dell_Series_11_Remanufactured_Ink_Cartridge_TriColor': {'scale_factor': 0.147794, 'category': 'Consumer Goods'},
      'Office_Depot_Dell_Series_1_Remanufactured_Ink_Cartridge_Black': {'scale_factor': 0.11818500000000001, 'category': 'Consumer Goods'},
      'Office_Depot_Dell_Series_1_Remanufactured_Ink_Cartridge_TriColor': {'scale_factor': 0.11790999999999999, 'category': 'Consumer Goods'},
      'Office_Depot_Dell_Series_5_Remanufactured_Ink_Cartridge_Black': {'scale_factor': 0.117842, 'category': 'Consumer Goods'},
      'Office_Depot_Dell_Series_9_Color_Ink_Ink_Cartridge_MK991_MK993': {'scale_factor': 0.118176, 'category': 'Consumer Goods'},
      'Office_Depot_Dell_Series_9_Ink_Cartridge_Black_MK992': {'scale_factor': 0.118132, 'category': 'Consumer Goods'},
      'Office_Depot_HP_2_Remanufactured_Ink_Cartridges_Color_Cyan_Magenta_Yellow_3_count': {'scale_factor': 0.147494, 'category': 'Consumer Goods'},
      'Office_Depot_HP_564XL_Ink_Cartridge_Black_CN684WN': {'scale_factor': 0.11773, 'category': 'Consumer Goods'},
      'Office_Depot_HP_564XL_Remanufactured_Ink_Cartridges_Color_Cyan_Magenta_Yellow_3_count': {'scale_factor': 0.13404300000000002, 'category': 'Consumer Goods'},
      'Office_Depot_HP_61Tricolor_Ink_Cartridge': {'scale_factor': 0.118392, 'category': 'Consumer Goods'},
      'Office_Depot_HP_71_Remanufactured_Ink_Cartridge_Black': {'scale_factor': 0.117462, 'category': 'Consumer Goods'},
      'Office_Depot_HP_74XL75_Remanufactured_Ink_Cartridges_BlackTriColor_2_count': {'scale_factor': 0.11652599999999999, 'category': 'Consumer Goods'},
      'Office_Depot_HP_75_Remanufactured_Ink_Cartridge_TriColor': {'scale_factor': 0.146814, 'category': 'Consumer Goods'},
      'Office_Depot_HP_920XL_920_High_Yield_Black_and_Standard_CMY_Color_Ink_Cartridges': {'scale_factor': 0.117117, 'category': 'Consumer Goods'},
      'Office_Depot_HP_932XL_Ink_Cartridge_Black_CN053A': {'scale_factor': 0.117656, 'category': 'Consumer Goods'},
      'Office_Depot_HP_950XL_Ink_Cartridge_Black_CN045AN': {'scale_factor': 0.117974, 'category': 'Consumer Goods'},
      'Office_Depot_HP_96_Remanufactured_Ink_Cartridge_Black': {'scale_factor': 0.147144, 'category': 'Consumer Goods'},
      'Office_Depot_HP_98_Remanufactured_Ink_Cartridge_Black': {'scale_factor': 0.147669, 'category': 'Consumer Goods'},
      'Olive_Kids_Birdie_Lunch_Box': {'scale_factor': 0.267073, 'category': 'Bag'},
      'Olive_Kids_Birdie_Munch_n_Lunch': {'scale_factor': 0.269443, 'category': 'Bag'},
      'Olive_Kids_Birdie_Pack_n_Snack': {'scale_factor': 0.32076499999999997, 'category': 'Bag'},
      'Olive_Kids_Birdie_Sidekick_Backpack': {'scale_factor': 0.38572399999999996, 'category': 'Bag'},
      'Olive_Kids_Butterfly_Garden_Munch_n_Lunch_Bag': {'scale_factor': 0.265105, 'category': 'Bag'},
      'Olive_Kids_Butterfly_Garden_Pencil_Case': {'scale_factor': 0.209219, 'category': 'Bag'},
      'Olive_Kids_Dinosaur_Land_Lunch_Box': {'scale_factor': 0.26060099999999997, 'category': 'Bag'},
      'Olive_Kids_Dinosaur_Land_Munch_n_Lunch': {'scale_factor': 0.26103699999999996, 'category': 'Bag'},
      'Olive_Kids_Dinosaur_Land_Pack_n_Snack': {'scale_factor': 0.314317, 'category': 'Bag'},
      'Olive_Kids_Dinosaur_Land_Sidekick_Backpack': {'scale_factor': 0.38016099999999997, 'category': 'Bag'},
      'Olive_Kids_Game_On_Lunch_Box': {'scale_factor': 0.274741, 'category': 'Bag'},
      'Olive_Kids_Game_On_Munch_n_Lunch': {'scale_factor': 0.267735, 'category': 'Bag'},
      'Olive_Kids_Game_On_Pack_n_Snack': {'scale_factor': 0.329943, 'category': 'Bag'},
      'Olive_Kids_Mermaids_Pack_n_Snack_Backpack': {'scale_factor': 0.321006, 'category': 'Bag'},
      'Olive_Kids_Paisley_Pencil_Case': {'scale_factor': 0.213918, 'category': 'Bag'},
      'Olive_Kids_Robots_Pencil_Case': {'scale_factor': 0.220361, 'category': 'Bag'},
      'Olive_Kids_Trains_Planes_Trucks_Bogo_Backpack': {'scale_factor': 0.37624199999999997, 'category': 'Bag'},
      'Olive_Kids_Trains_Planes_Trucks_Munch_n_Lunch_Bag': {'scale_factor': 0.288471, 'category': 'Bag'},
      'Orbit_Bubblemint_Mini_Bottle_6_ct': {'scale_factor': 0.21040599999999998, 'category': 'None'},
      'Organic_Whey_Protein_Unflavored': {'scale_factor': 0.16906300000000002, 'category': 'Bottles and Cans and Cups'},
      'Organic_Whey_Protein_Vanilla': {'scale_factor': 0.169894, 'category': 'Bottles and Cans and Cups'},
      'Ortho_Forward_Facing': {'scale_factor': 0.23472099999999999, 'category': 'None'},
      'Ortho_Forward_Facing_3Q6J2oKJD92': {'scale_factor': 0.24370999999999998, 'category': 'None'},
      'Ortho_Forward_Facing_CkAW6rL25xH': {'scale_factor': 0.27880099999999997, 'category': 'None'},
      'Ortho_Forward_Facing_QCaor9ImJ2G': {'scale_factor': 0.19551000000000002, 'category': 'None'},
      'PARENT_ROOM_FURNITURE_SET_1': {'scale_factor': 0.21962700000000002, 'category': 'Toys'},
      'PARENT_ROOM_FURNITURE_SET_1_DLKEy8H4mwK': {'scale_factor': 0.289104, 'category': 'Toys'},
      'PEEKABOO_ROLLER': {'scale_factor': 0.063905, 'category': 'Toys'},
      'PEPSI_NEXT_CACRV': {'scale_factor': 0.402849, 'category': 'Consumer Goods'},
      'PETS_ACCESSORIES': {'scale_factor': 0.162854, 'category': 'Toys'},
      'PHEEHAN_RUN': {'scale_factor': 0.26264699999999996, 'category': 'Shoe'},
      'PINEAPPLE_MARACA_6_PCSSET': {'scale_factor': 0.100339, 'category': 'Toys'},
      'POUNDING_MUSHROOMS': {'scale_factor': 0.213548, 'category': 'Toys'},
      'PUNCH_DROP': {'scale_factor': 0.25022200000000006, 'category': 'Toys'},
      'PUNCH_DROP_TjicLPMqLvz': {'scale_factor': 0.19638699999999998, 'category': 'Toys'},
      'Paint_Maker': {'scale_factor': 0.23983300000000002, 'category': 'Consumer Goods'},
      'Paper_Mario_Sticker_Star_Nintendo_3DS_Game': {'scale_factor': 0.13758599999999999, 'category': 'Media Cases'},
      'Pass_The_Popcorn_Movie_Guessing_Game': {'scale_factor': 0.27044199999999996, 'category': 'Board Games'},
      'Paul_Frank_Dot_Lunch_Box': {'scale_factor': 0.263426, 'category': 'Bag'},
      'Pennington_Electric_Pot_Cabana_4': {'scale_factor': 0.103581, 'category': 'None'},
      'Pepsi_Caffeine_Free_Diet_12_CT': {'scale_factor': 0.405254, 'category': 'Consumer Goods'},
      'Pepsi_Cola_Caffeine_Free_12_12_fl_oz_355_ml_cans_144_fl_oz_426_lt': {'scale_factor': 0.404011, 'category': 'Consumer Goods'},
      'Pepsi_Cola_Wild_Cherry_Diet_12_12_fl_oz_355_ml_cans_144_fl_oz_426_lt': {'scale_factor': 0.4054, 'category': 'Consumer Goods'},
      'Pepsi_Max_Cola_Zero_Calorie_12_12_fl_oz_355_ml_cans_144_fl_oz_426_lt': {'scale_factor': 0.404797, 'category': 'Consumer Goods'},
      'Perricoen_MD_No_Concealer_Concealer': {'scale_factor': 0.110844, 'category': 'Consumer Goods'},
      'Perricone_MD_AcylGlutathione_Deep_Crease_Serum': {'scale_factor': 0.105769, 'category': 'Consumer Goods'},
      'Perricone_MD_AcylGlutathione_Eye_Lid_Serum': {'scale_factor': 0.10783300000000001, 'category': 'Consumer Goods'},
      'Perricone_MD_Best_of_Perricone_7Piece_Collection_MEGsO6GIsyL': {'scale_factor': 0.29676800000000003, 'category': 'Consumer Goods'},
      'Perricone_MD_Blue_Plasma_Orbital': {'scale_factor': 0.094898, 'category': 'Consumer Goods'},
      'Perricone_MD_Chia_Serum': {'scale_factor': 0.119782, 'category': 'Consumer Goods'},
      'Perricone_MD_Cold_Plasma': {'scale_factor': 0.106151, 'category': 'Consumer Goods'},
      'Perricone_MD_Cold_Plasma_Body': {'scale_factor': 0.15841099999999997, 'category': 'Consumer Goods'},
      'Perricone_MD_Face_Finishing_Moisturizer': {'scale_factor': 0.073716, 'category': 'Consumer Goods'},
      'Perricone_MD_Face_Finishing_Moisturizer_4_oz': {'scale_factor': 0.078539, 'category': 'Consumer Goods'},
      'Perricone_MD_Firming_Neck_Therapy_Treatment': {'scale_factor': 0.07328799999999999, 'category': 'Consumer Goods'},
      'Perricone_MD_Health_Weight_Management_Supplements': {'scale_factor': 0.180848, 'category': 'Consumer Goods'},
      'Perricone_MD_High_Potency_Evening_Repair': {'scale_factor': 0.12123899999999999, 'category': 'Consumer Goods'},
      'Perricone_MD_Hypoallergenic_Firming_Eye_Cream_05_oz': {'scale_factor': 0.057255, 'category': 'Consumer Goods'},
      'Perricone_MD_Hypoallergenic_Gentle_Cleanser': {'scale_factor': 0.155829, 'category': 'Consumer Goods'},
      'Perricone_MD_Neuropeptide_Facial_Conformer': {'scale_factor': 0.120402, 'category': 'Consumer Goods'},
      'Perricone_MD_Neuropeptide_Firming_Moisturizer': {'scale_factor': 0.073356, 'category': 'Consumer Goods'},
      'Perricone_MD_No_Bronzer_Bronzer': {'scale_factor': 0.10071, 'category': 'Consumer Goods'},
      'Perricone_MD_No_Foundation_Foundation_No_1': {'scale_factor': 0.121583, 'category': 'Consumer Goods'},
      'Perricone_MD_No_Foundation_Serum': {'scale_factor': 0.122251, 'category': 'Consumer Goods'},
      'Perricone_MD_No_Lipstick_Lipstick': {'scale_factor': 0.07902100000000001, 'category': 'Consumer Goods'},
      'Perricone_MD_No_Mascara_Mascara': {'scale_factor': 0.128713, 'category': 'Consumer Goods'},
      'Perricone_MD_Nutritive_Cleanser': {'scale_factor': 0.14873799999999998, 'category': 'Consumer Goods'},
      'Perricone_MD_OVM': {'scale_factor': 0.105043, 'category': 'Consumer Goods'},
      'Perricone_MD_Omega_3_Supplements': {'scale_factor': 0.126388, 'category': 'Consumer Goods'},
      'Perricone_MD_Photo_Plasma': {'scale_factor': 0.10584299999999999, 'category': 'Consumer Goods'},
      'Perricone_MD_Skin_Clear_Supplements': {'scale_factor': 0.16895500000000002, 'category': 'Consumer Goods'},
      'Perricone_MD_Skin_Total_Body_Supplements': {'scale_factor': 0.174985, 'category': 'Consumer Goods'},
      'Perricone_MD_Super_Berry_Powder_with_Acai_Supplements': {'scale_factor': 0.14984000000000003, 'category': 'Consumer Goods'},
      'Perricone_MD_The_Cold_Plasma_Face_Eyes_Duo': {'scale_factor': 0.131606, 'category': 'Consumer Goods'},
      'Perricone_MD_The_Crease_Cure_Duo': {'scale_factor': 0.169454, 'category': 'Consumer Goods'},
      'Perricone_MD_The_Metabolic_Formula_Supplements': {'scale_factor': 0.16847099999999998, 'category': 'Consumer Goods'},
      'Perricone_MD_The_Power_Treatments': {'scale_factor': 0.210856, 'category': 'Consumer Goods'},
      'Perricone_MD_Vitamin_C_Ester_15': {'scale_factor': 0.13034800000000002, 'category': 'Consumer Goods'},
      'Perricone_MD_Vitamin_C_Ester_Serum': {'scale_factor': 0.105274, 'category': 'Consumer Goods'},
      'Persona_Q_Shadow_of_the_Labyrinth_Nintendo_3DS': {'scale_factor': 0.142339, 'category': 'Media Cases'},
      'Pet_Dophilus_powder': {'scale_factor': 0.102272, 'category': 'Bottles and Cans and Cups'},
      'Philips_60ct_Warm_White_LED_Smooth_Mini_String_Lights': {'scale_factor': 0.25801399999999997, 'category': 'Consumer Goods'},
      'Philips_EcoVantage_43_W_Light_Bulbs_Natural_Light_2_pack': {'scale_factor': 0.107856, 'category': 'Consumer Goods'},
      'Philips_Sonicare_Tooth_Brush_2_count': {'scale_factor': 0.241611, 'category': 'Consumer Goods'},
      'Phillips_Caplets_Size_24': {'scale_factor': 0.08574000000000001, 'category': 'Consumer Goods'},
      'Phillips_Colon_Health_Probiotic_Capsule': {'scale_factor': 0.11816099999999999, 'category': 'Consumer Goods'},
      'Phillips_Milk_of_Magnesia_Saline_Laxative_Liquid_Original': {'scale_factor': 0.174141, 'category': 'Consumer Goods'},
      'Phillips_Stool_Softener_Liquid_Gels_30_liquid_gels': {'scale_factor': 0.08669600000000001, 'category': 'Consumer Goods'},
      'PhosphOmega': {'scale_factor': 0.112404, 'category': 'Bottles and Cans and Cups'},
      'Pinwheel_Pencil_Case': {'scale_factor': 0.217499, 'category': 'Bag'},
      'Playmates_Industrial_CoSplinter_Teenage_Mutant_Ninja_Turtle_Action_Figure': {'scale_factor': 0.12371299999999999, 'category': 'Action Figures'},
      'Playmates_nickelodeon_teenage_mutant_ninja_turtles_shredder': {'scale_factor': 0.14156000000000002, 'category': 'Action Figures'},
      'Poise_Ultimate_Pads_Long': {'scale_factor': 0.372325, 'category': 'Consumer Goods'},
      'Pokmon_Conquest_Nintendo_DS_Game': {'scale_factor': 0.137759, 'category': 'Media Cases'},
      'Pokmon_X_Nintendo_3DS_Game': {'scale_factor': 0.137016, 'category': 'Media Cases'},
      'Pokmon_Y_Nintendo_3DS_Game': {'scale_factor': 0.137517, 'category': 'Media Cases'},
      'Pokémon_Omega_Ruby_Alpha_Sapphire_Dual_Pack_Nintendo_3DS': {'scale_factor': 0.143462, 'category': 'Consumer Goods'},
      'Pokémon_Yellow_Special_Pikachu_Edition_Nintendo_Game_Boy_Color': {'scale_factor': 0.065744, 'category': 'Media Cases'},
      'Polar_Herring_Fillets_Smoked_Peppered_705_oz_total': {'scale_factor': 0.15054, 'category': 'Consumer Goods'},
      'Pony_C_Clamp_1440': {'scale_factor': 0.20221699999999998, 'category': 'None'},
      'Poppin_File_Sorter_Blue': {'scale_factor': 0.25032499999999996, 'category': 'Consumer Goods'},
      'Poppin_File_Sorter_Pink': {'scale_factor': 0.250274, 'category': 'Consumer Goods'},
      'Poppin_File_Sorter_White': {'scale_factor': 0.25064, 'category': 'Consumer Goods'},
      'Predator_LZ_TRX_FG': {'scale_factor': 0.28352299999999997, 'category': 'Shoe'},
      'Predito_LZ_TRX_FG_W': {'scale_factor': 0.26099799999999995, 'category': 'Shoe'},
      'ProSport_Harness_to_Booster_Seat': {'scale_factor': 0.227245, 'category': 'Car Seat'},
      'Progressive_Rubber_Spatulas_3_count': {'scale_factor': 0.259683, 'category': 'None'},
      'Prostate_Optimizer': {'scale_factor': 0.11033000000000001, 'category': 'Bottles and Cans and Cups'},
      'Provence_Bath_Towel_Royal_Blue': {'scale_factor': 0.35616099999999995, 'category': 'None'},
      'PureCadence_2_Color_HiRskRedNghtlfeSlvrBlckWht_Size_70': {'scale_factor': 0.306111, 'category': 'Shoe'},
      'PureCadence_2_Color_TleBluLmePnchSlvMoodIndgWh_Size_50_EEzAfcBfHHO': {'scale_factor': 0.286829, 'category': 'Shoe'},
      'PureConnect_2_Color_AnthrcteKnckoutPnkGrnGecko_Size_50': {'scale_factor': 0.27082799999999996, 'category': 'Shoe'},
      'PureConnect_2_Color_BlckBrllntBluNghtlfeAnthrct_Size_70': {'scale_factor': 0.32284, 'category': 'Shoe'},
      'PureConnect_2_Color_FernNightlifeSilverBlack_Size_70_5w0BYsiogeV': {'scale_factor': 0.295028, 'category': 'Shoe'},
      'PureFlow_2_Color_RylPurHibiscusBlkSlvrWht_Size_50': {'scale_factor': 0.28676199999999996, 'category': 'Shoe'},
      'QAbsorb_CoQ10': {'scale_factor': 0.099309, 'category': 'Bottles and Cans and Cups'},
      'QAbsorb_CoQ10_53iUqjWjW3O': {'scale_factor': 0.099318, 'category': 'Bottles and Cans and Cups'},
      'QHPomegranate': {'scale_factor': 0.097844, 'category': 'Bottles and Cans and Cups'},
      'Quercetin_500': {'scale_factor': 0.14414699999999997, 'category': 'Bottles and Cans and Cups'},
      'REEF_BANTU': {'scale_factor': 0.280939, 'category': 'Shoe'},
      'REEF_BRAIDED_CUSHION': {'scale_factor': 0.25165200000000004, 'category': 'Shoe'},
      'REEF_ZENFUN': {'scale_factor': 0.249996, 'category': 'Shoe'},
      'RESCUE_CREW': {'scale_factor': 0.08934600000000001, 'category': 'Toys'},
      'RJ_Rabbit_Easter_Basket_Blue': {'scale_factor': 0.235166, 'category': 'None'},
      'ROAD_CONSTRUCTION_SET': {'scale_factor': 0.15933599999999998, 'category': 'Toys'},
      'Racoon': {'scale_factor': 0.315147, 'category': 'None'},
      'Ravenna_4_Color_WhtOlyBluBlkShkOrngSlvRdO_Size_70': {'scale_factor': 0.321527, 'category': 'Shoe'},
      'Rayna_BootieWP': {'scale_factor': 0.241303, 'category': 'Shoe'},
      'Razer_Abyssus_Ambidextrous_Gaming_Mouse': {'scale_factor': 0.145816, 'category': 'Mouse'},
      'Razer_BlackWidow_Stealth_2014_Keyboard_07VFzIVabgh': {'scale_factor': 0.472086, 'category': 'Keyboard'},
      'Razer_BlackWidow_Ultimate_2014_Mechanical_Gaming_Keyboard': {'scale_factor': 0.471283, 'category': 'Keyboard'},
      'Razer_Blackwidow_Tournament_Edition_Keyboard': {'scale_factor': 0.365981, 'category': 'Keyboard'},
      'Razer_Goliathus_Control_Edition_Small_Soft_Gaming_Mouse_Mat': {'scale_factor': 0.272896, 'category': 'None'},
      'Razer_Kraken_71_Chroma_headset_Full_size_Black': {'scale_factor': 0.258566, 'category': 'Headphones'},
      'Razer_Kraken_Pro_headset_Full_size_Black': {'scale_factor': 0.290186, 'category': 'Headphones'},
      'Razer_Naga_MMO_Gaming_Mouse': {'scale_factor': 0.255208, 'category': 'Mouse'},
      'Razer_Taipan_Black_Ambidextrous_Gaming_Mouse': {'scale_factor': 0.17452600000000001, 'category': 'Mouse'},
      'Razer_Taipan_White_Ambidextrous_Gaming_Mouse': {'scale_factor': 0.18208200000000002, 'category': 'Mouse'},
      'ReadytoUse_Rolled_Fondant_Pure_White_24_oz_box': {'scale_factor': 0.168659, 'category': 'Consumer Goods'},
      'Real_Deal_1nIwCHX1MTh': {'scale_factor': 0.302618, 'category': 'Shoe'},
      'RedBlack_Nintendo_3DSXL': {'scale_factor': 0.15597699999999998, 'category': 'None'},
      'Reebok_ALLYLYNN': {'scale_factor': 0.218279, 'category': 'Shoe'},
      'Reebok_BREAKPOINT_LO_2V': {'scale_factor': 0.216938, 'category': 'Shoe'},
      'Reebok_BREAKPOINT_MID': {'scale_factor': 0.24763400000000002, 'category': 'Shoe'},
      'Reebok_CLASSIC_JOGGER': {'scale_factor': 0.21426, 'category': 'Shoe'},
      'Reebok_CLASSIC_LEGACY_II': {'scale_factor': 0.291142, 'category': 'Shoe'},
      'Reebok_CL_DIBELLO_II': {'scale_factor': 0.285652, 'category': 'Shoe'},
      'Reebok_CL_LTHR_R12': {'scale_factor': 0.289353, 'category': 'Shoe'},
      'Reebok_CL_RAYEN': {'scale_factor': 0.29367299999999996, 'category': 'Shoe'},
      'Reebok_COMFORT_REEFRESH_FLIP': {'scale_factor': 0.25692899999999996, 'category': 'Shoe'},
      'Reebok_DMX_MAX_MANIA_WD_D': {'scale_factor': 0.272767, 'category': 'Shoe'},
      'Reebok_DMX_MAX_PLUS_ATHLETIC': {'scale_factor': 0.301056, 'category': 'Shoe'},
      'Reebok_DMX_MAX_PLUS_RAINWALKER': {'scale_factor': 0.30084900000000003, 'category': 'Shoe'},
      'Reebok_EASYTONE_CL_LEATHER': {'scale_factor': 0.26518600000000003, 'category': 'Shoe'},
      'Reebok_FS_HI_INT_R12': {'scale_factor': 0.26004700000000003, 'category': 'Shoe'},
      'Reebok_FS_HI_MINI': {'scale_factor': 0.259647, 'category': 'Shoe'},
      'Reebok_FUELTRAIN': {'scale_factor': 0.272337, 'category': 'Shoe'},
      'Reebok_GL_6000': {'scale_factor': 0.292961, 'category': 'Shoe'},
      'Reebok_HIMARA_LTR': {'scale_factor': 0.295829, 'category': 'Shoe'},
      'Reebok_JR_ZIG_COOPERSTOWN_MR': {'scale_factor': 0.24046299999999998, 'category': 'Shoe'},
      'Reebok_KAMIKAZE_II_MID': {'scale_factor': 0.29331399999999996, 'category': 'Shoe'},
      'Reebok_PUMP_OMNI_LITE_HLS': {'scale_factor': 0.29212899999999997, 'category': 'Shoe'},
      'Reebok_REALFLEX_SELECT': {'scale_factor': 0.300118, 'category': 'Shoe'},
      'Reebok_REESCULPT_TRAINER_II': {'scale_factor': 0.26698500000000003, 'category': 'Shoe'},
      'Reebok_RETRO_RUSH_2V': {'scale_factor': 0.21651199999999998, 'category': 'Shoe'},
      'Reebok_R_CROSSFIT_OLY_UFORM': {'scale_factor': 0.29188000000000003, 'category': 'Shoe'},
      'Reebok_R_DANCE_FLASH': {'scale_factor': 0.264313, 'category': 'Shoe'},
      'Reebok_SH_COURT_MID_II': {'scale_factor': 0.291929, 'category': 'Shoe'},
      'Reebok_SH_NEWPORT_LOW': {'scale_factor': 0.28661400000000004, 'category': 'Shoe'},
      'Reebok_SH_PRIME_COURT_LOW': {'scale_factor': 0.258737, 'category': 'Shoe'},
      'Reebok_SH_PRIME_COURT_MID': {'scale_factor': 0.21718900000000002, 'category': 'Shoe'},
      'Reebok_SL_FLIP_UPDATE': {'scale_factor': 0.28837, 'category': 'Shoe'},
      'Reebok_SMOOTHFLEX_CUSHRUN_20': {'scale_factor': 0.267722, 'category': 'Shoe'},
      'Reebok_SOMERSET_RUN': {'scale_factor': 0.29264999999999997, 'category': 'Shoe'},
      'Reebok_STUDIO_BEAT_LOW_V': {'scale_factor': 0.266542, 'category': 'Shoe'},
      'Reebok_TRIPLE_BREAK_LITE': {'scale_factor': 0.26593, 'category': 'Shoe'},
      'Reebok_TURBO_RC': {'scale_factor': 0.296581, 'category': 'Shoe'},
      'Reebok_ULTIMATIC_2V': {'scale_factor': 0.222666, 'category': 'Shoe'},
      'Reebok_VERSA_TRAIN': {'scale_factor': 0.29916200000000004, 'category': 'Shoe'},
      'Reebok_ZIGCOOPERSTOWN_QUAG': {'scale_factor': 0.31944, 'category': 'Shoe'},
      'Reebok_ZIGLITE_RUSH': {'scale_factor': 0.273035, 'category': 'Shoe'},
      'Reebok_ZIGLITE_RUSH_AC': {'scale_factor': 0.2274, 'category': 'Shoe'},
      'Reebok_ZIGSTORM': {'scale_factor': 0.260937, 'category': 'Shoe'},
      'Reebok_ZIGTECH_SHARK_MAYHEM360': {'scale_factor': 0.297531, 'category': 'Shoe'},
      'Reef_Star_Cushion_Flipflops_Size_8_Black': {'scale_factor': 0.258442, 'category': 'Shoe'},
      'Remington_1_12_inch_Hair_Straightener': {'scale_factor': 0.36532, 'category': 'None'},
      'Remington_TStudio_Hair_Dryer': {'scale_factor': 0.330008, 'category': 'None'},
      'Remington_TStudio_Silk_Ceramic_Hair_Straightener_2_Inch_Floating_Plates': {'scale_factor': 0.322996, 'category': 'Consumer Goods'},
      'Retail_Leadership_Summit': {'scale_factor': 0.288495, 'category': 'None'},
      'Retail_Leadership_Summit_eCT3zqHYIkX': {'scale_factor': 0.273901, 'category': 'None'},
      'Retail_Leadership_Summit_tQFCizMt6g0': {'scale_factor': 0.273535, 'category': 'None'},
      'Rexy_Glove_Heavy_Duty_Gloves_Medium': {'scale_factor': 0.318786, 'category': 'None'},
      'Rexy_Glove_Heavy_Duty_Large': {'scale_factor': 0.274085, 'category': 'None'},
      'Romantic_Blush_Tieks_Metallic_Italian_Leather_Ballet_Flats': {'scale_factor': 0.246545, 'category': 'Shoe'},
      'Room_Essentials_Bowl_Turquiose': {'scale_factor': 0.157221, 'category': 'None'},
      'Room_Essentials_Dish_Drainer_Collapsible_White': {'scale_factor': 0.36764399999999997, 'category': 'None'},
      'Room_Essentials_Fabric_Cube_Lavender': {'scale_factor': 0.285746, 'category': 'None'},
      'Room_Essentials_Kitchen_Towels_16_x_26_2_count': {'scale_factor': 0.183425, 'category': 'None'},
      'Room_Essentials_Mug_White_Yellow': {'scale_factor': 0.136838, 'category': 'None'},
      'Room_Essentials_Salad_Plate_Turquoise': {'scale_factor': 0.22190700000000002, 'category': 'None'},
      'Rose_Garden_Tieks_Leather_Ballet_Flats_with_Floral_Rosettes': {'scale_factor': 0.24687, 'category': 'Shoe'},
      'Rubbermaid_Large_Drainer': {'scale_factor': 0.442347, 'category': 'None'},
      'SAMBA_HEMP': {'scale_factor': 0.29001, 'category': 'Shoe'},
      'SAMOA': {'scale_factor': 0.29276100000000005, 'category': 'Shoe'},
      'SAMe_200': {'scale_factor': 0.158839, 'category': 'Consumer Goods'},
      'SAMe_200_KX7ZmOw47co': {'scale_factor': 0.130981, 'category': 'Consumer Goods'},
      'SANDWICH_MEAL': {'scale_factor': 0.13841900000000001, 'category': 'Toys'},
      'SAPPHIRE_R7_260X_OC': {'scale_factor': 0.272564, 'category': 'None'},
      'SCHOOL_BUS': {'scale_factor': 0.097799, 'category': 'Toys'},
      'SHAPE_MATCHING': {'scale_factor': 0.254101, 'category': 'Toys'},
      'SHAPE_MATCHING_NxacpAY9jDt': {'scale_factor': 0.189266, 'category': 'Toys'},
      'SHAPE_SORTER': {'scale_factor': 0.33627799999999997, 'category': 'Toys'},
      'SIT_N_WALK_PUPPY': {'scale_factor': 0.186628, 'category': 'Toys'},
      'SLACK_CRUISER': {'scale_factor': 0.308971, 'category': 'Shoe'},
      'SNAIL_MEASURING_TAPE': {'scale_factor': 0.194457, 'category': 'Toys'},
      'SORTING_BUS': {'scale_factor': 0.290262, 'category': 'Toys'},
      'SORTING_TRAIN': {'scale_factor': 0.238199, 'category': 'Toys'},
      'SPEED_BOAT': {'scale_factor': 0.142508, 'category': 'Toys'},
      'STACKING_BEAR': {'scale_factor': 0.177993, 'category': 'Toys'},
      'STACKING_BEAR_V04KKgGBn2A': {'scale_factor': 0.221389, 'category': 'Toys'},
      'STACKING_RING': {'scale_factor': 0.187393, 'category': 'Toys'},
      'STEAK_SET': {'scale_factor': 0.201113, 'category': 'Toys'},
      'SUPERSTAR_CLR': {'scale_factor': 0.29211699999999996, 'category': 'Shoe'},
      'Saccharomyces_Boulardii_MOS_Value_Size': {'scale_factor': 0.135079, 'category': 'Bottles and Cans and Cups'},
      'Samoa_onepiece': {'scale_factor': 0.292438, 'category': 'Shoe'},
      'Samsung_CLTC406S_Toner_Cartridge_Cyan_1pack': {'scale_factor': 0.388557, 'category': 'Consumer Goods'},
      'Santa_Cruz_Mens': {'scale_factor': 0.29860299999999995, 'category': 'Shoe'},
      'Santa_Cruz_Mens_G7kQXK7cIky': {'scale_factor': 0.300736, 'category': 'Shoe'},
      'Santa_Cruz_Mens_YmsMDkFf11Z': {'scale_factor': 0.29844899999999996, 'category': 'Shoe'},
      'Santa_Cruz_Mens_umxTczr1Ygg': {'scale_factor': 0.30069999999999997, 'category': 'Shoe'},
      'Santa_Cruz_Mens_vnbiTDDt5xH': {'scale_factor': 0.29876199999999997, 'category': 'Shoe'},
      'Sapota_Threshold_4_Ceramic_Round_Planter_Red': {'scale_factor': 0.109554, 'category': 'Consumer Goods'},
      'Schleich_African_Black_Rhino': {'scale_factor': 0.1375, 'category': 'Toys'},
      'Schleich_Allosaurus': {'scale_factor': 0.22292900000000002, 'category': 'Toys'},
      'Schleich_Bald_Eagle': {'scale_factor': 0.089894, 'category': 'Toys'},
      'Schleich_Hereford_Bull': {'scale_factor': 0.134434, 'category': 'Toys'},
      'Schleich_Lion_Action_Figure': {'scale_factor': 0.11501800000000001, 'category': 'Action Figures'},
      'Schleich_S_Bayala_Unicorn_70432': {'scale_factor': 0.147988, 'category': 'Toys'},
      'Schleich_Spinosaurus_Action_Figure': {'scale_factor': 0.26408, 'category': 'Action Figures'},
      'Schleich_Therizinosaurus_ln9cruulPqc': {'scale_factor': 0.217532, 'category': 'Toys'},
      'Sea_to_Summit_Xl_Bowl': {'scale_factor': 0.180311, 'category': 'Consumer Goods'},
      'Seagate_1TB_Backup_Plus_portable_drive_Blue': {'scale_factor': 0.123977, 'category': 'None'},
      'Seagate_1TB_Backup_Plus_portable_drive_Silver': {'scale_factor': 0.124209, 'category': 'None'},
      'Seagate_1TB_Backup_Plus_portable_drive_for_Mac': {'scale_factor': 0.124271, 'category': 'None'},
      'Seagate_1TB_Wireless_Plus_mobile_device_storage': {'scale_factor': 0.12827899999999998, 'category': 'None'},
      'Seagate_3TB_Central_shared_storage': {'scale_factor': 0.217177, 'category': 'None'},
      'Seagate_Archive_HDD_8_TB_Internal_hard_drive_SATA_6Gbs_35_ST8000AS0002': {'scale_factor': 0.14834, 'category': 'None'},
      'Shark': {'scale_factor': 0.387062, 'category': 'None'},
      'Shaxon_100_Molded_Category_6_RJ45RJ45_Shielded_Patch_Cord_White': {'scale_factor': 0.174786, 'category': 'Consumer Goods'},
      'Shurtape_30_Day_Removal_UV_Delct_15': {'scale_factor': 0.116929, 'category': 'Consumer Goods'},
      'Shurtape_Gaffers_Tape_Silver_2_x_60_yd': {'scale_factor': 0.153558, 'category': 'Consumer Goods'},
      'Shurtape_Tape_Purple_CP28': {'scale_factor': 0.118145, 'category': 'Consumer Goods'},
      'Sienna_Brown_Croc_Tieks_Patent_Leather_Crocodile_Print_Ballet_Flats': {'scale_factor': 0.244214, 'category': 'Shoe'},
      'Simon_Swipe_Game': {'scale_factor': 0.229134, 'category': 'Media Cases'},
      'Sleep_Optimizer': {'scale_factor': 0.14616700000000002, 'category': 'Consumer Goods'},
      'Smith_Hawken_Woven_BasketTray_Organizer_with_3_Compartments_95_x_9_x_13': {'scale_factor': 0.360726, 'category': 'None'},
      'Snack_Catcher_Snack_Dispenser': {'scale_factor': 0.13498, 'category': 'Consumer Goods'},
      'Sonicare_2_Series_Toothbrush_Plaque_Control': {'scale_factor': 0.24216100000000002, 'category': 'None'},
      'Sonny_School_Bus': {'scale_factor': 0.23995300000000003, 'category': 'None'},
      'Sony_Acid_Music_Studio': {'scale_factor': 0.196277, 'category': 'Consumer Goods'},
      'Sony_Downloadable_Loops': {'scale_factor': 0.195175, 'category': 'Consumer Goods'},
      'Sootheze_Cold_Therapy_Elephant': {'scale_factor': 0.267932, 'category': 'Stuffed Toys'},
      'Sootheze_Toasty_Orca': {'scale_factor': 0.40727599999999997, 'category': 'Stuffed Toys'},
      'Sorry_Sliders_Board_Game': {'scale_factor': 0.404539, 'category': 'Board Games'},
      'Spectrum_Wall_Mount': {'scale_factor': 0.16053099999999998, 'category': 'Consumer Goods'},
      'Sperry_TopSider_pSUFPWQXPp3': {'scale_factor': 0.243011, 'category': 'Shoe'},
      'Sperry_TopSider_tNB9t6YBUf3': {'scale_factor': 0.243801, 'category': 'Shoe'},
      'SpiderMan_Titan_Hero_12Inch_Action_Figure_5Hnn4mtkFsP': {'scale_factor': 0.28815999999999997, 'category': 'Action Figures'},
      'SpiderMan_Titan_Hero_12Inch_Action_Figure_oo1qph4wwiW': {'scale_factor': 0.288126, 'category': 'Action Figures'},
      'Spritz_Easter_Basket_Plastic_Teal': {'scale_factor': 0.19129000000000002, 'category': 'None'},
      'Squirrel': {'scale_factor': 0.141091, 'category': 'None'},
      'Squirt_Strain_Fruit_Basket': {'scale_factor': 0.113955, 'category': 'Consumer Goods'},
      'Squirtin_Barnyard_Friends_4pk': {'scale_factor': 0.18629800000000002, 'category': 'Consumer Goods'},
      'Star_Wars_Rogue_Squadron_Nintendo_64': {'scale_factor': 0.11654500000000001, 'category': 'Consumer Goods'},
      'Starstruck_Tieks_Glittery_Gold_Italian_Leather_Ballet_Flats': {'scale_factor': 0.24372, 'category': 'Shoe'},
      'Sterilite_Caddy_Blue_Sky_17_58_x_12_58_x_9_14': {'scale_factor': 0.446589, 'category': 'None'},
      'Super_Mario_3D_World_Deluxe_Set': {'scale_factor': 0.25582499999999997, 'category': 'None'},
      'Super_Mario_3D_World_Deluxe_Set_yThuvW9vZed': {'scale_factor': 0.46051499999999995, 'category': 'None'},
      'Super_Mario_3D_World_Wii_U_Game': {'scale_factor': 0.190941, 'category': 'Media Cases'},
      'Super_Mario_Kart_Super_Nintendo_Entertainment_System': {'scale_factor': 0.136577, 'category': 'Media Cases'},
      'Superman_Battle_of_Smallville': {'scale_factor': 0.384331, 'category': 'Legos'},
      'Supernatural_Ouija_Board_Game': {'scale_factor': 0.406508, 'category': 'Board Games'},
      'Sushi_Mat': {'scale_factor': 0.24888100000000002, 'category': 'None'},
      'Swiss_Miss_Hot_Cocoa_KCups_Milk_Chocolate_12_count': {'scale_factor': 0.150842, 'category': 'Consumer Goods'},
      'TABLEWARE_SET': {'scale_factor': 0.350493, 'category': 'Toys'},
      'TABLEWARE_SET_5CHkPjjxVpp': {'scale_factor': 0.412213, 'category': 'Toys'},
      'TABLEWARE_SET_5ww1UFLuCJG': {'scale_factor': 0.29721, 'category': 'Toys'},
      'TEA_SET': {'scale_factor': 0.260887, 'category': 'Toys'},
      'TERREX_FAST_R': {'scale_factor': 0.30310400000000004, 'category': 'Shoe'},
      'TERREX_FAST_X_GTX': {'scale_factor': 0.30735999999999997, 'category': 'Shoe'},
      'TOOL_BELT': {'scale_factor': 0.26131299999999996, 'category': 'Toys'},
      'TOP_TEN_HI': {'scale_factor': 0.301192, 'category': 'Shoe'},
      'TOP_TEN_HI_60KlbRbdoJA': {'scale_factor': 0.298915, 'category': 'Shoe'},
      'TOWER_TUMBLING': {'scale_factor': 0.226698, 'category': 'Toys'},
      'TROCHILUS_BOOST': {'scale_factor': 0.271321, 'category': 'Shoe'},
      'TURBOPROP_AIRPLANE_WITH_PILOT': {'scale_factor': 0.091111, 'category': 'Toys'},
      'TWISTED_PUZZLE': {'scale_factor': 0.188654, 'category': 'Toys'},
      'TWISTED_PUZZLE_twb4AyFtu8Q': {'scale_factor': 0.302454, 'category': 'Toys'},
      'TWIST_SHAPE': {'scale_factor': 0.25618, 'category': 'Toys'},
      'TZX_Runner': {'scale_factor': 0.292551, 'category': 'Shoe'},
      'Tag_Dishtowel_18_x_26': {'scale_factor': 0.233892, 'category': 'None'},
      'Tag_Dishtowel_Basket_Weave_Red_18_x_26': {'scale_factor': 0.233014, 'category': 'None'},
      'Tag_Dishtowel_Dobby_Stripe_Blue_18_x_26': {'scale_factor': 0.23716700000000002, 'category': 'None'},
      'Tag_Dishtowel_Green': {'scale_factor': 0.23022, 'category': 'None'},
      'Tag_Dishtowel_Waffle_Gray_Checks_18_x_26': {'scale_factor': 0.18582100000000001, 'category': 'None'},
      'Target_Basket_Medium': {'scale_factor': 0.270455, 'category': 'Consumer Goods'},
      'Teenage_Mutant_Ninja_Turtles_Rahzar_Action_Figure': {'scale_factor': 0.13855699999999999, 'category': 'Action Figures'},
      'Tena_Pads_Heavy_Long_42_pads': {'scale_factor': 0.378131, 'category': 'Consumer Goods'},
      'Tetris_Link_Game': {'scale_factor': 0.323084, 'category': 'Board Games'},
      'The_Coffee_Bean_Tea_Leaf_KCup_Packs_Jasmine_Green_Tea_16_count': {'scale_factor': 0.150359, 'category': 'Consumer Goods'},
      'The_Scooper_Hooper': {'scale_factor': 0.260272, 'category': 'Consumer Goods'},
      'Theanine': {'scale_factor': 0.09742500000000001, 'category': 'Bottles and Cans and Cups'},
      'Thomas_Friends_Woodan_Railway_Henry': {'scale_factor': 0.17898, 'category': 'Toys'},
      'Thomas_Friends_Wooden_Railway_Ascending_Track_Riser_Pack': {'scale_factor': 0.425317, 'category': 'Toys'},
      'Thomas_Friends_Wooden_Railway_Deluxe_Track_Accessory_Pack': {'scale_factor': 0.35697500000000004, 'category': 'Toys'},
      'Thomas_Friends_Wooden_Railway_Porter_5JzRhMm3a9o': {'scale_factor': 0.08788699999999999, 'category': 'Consumer Goods'},
      'Thomas_Friends_Wooden_Railway_Talking_Thomas_z7yi7UFHJRj': {'scale_factor': 0.088131, 'category': 'Toys'},
      'Threshold_Bamboo_Ceramic_Soap_Dish': {'scale_factor': 0.12465799999999999, 'category': 'None'},
      'Threshold_Basket_Natural_Finish_Fabric_Liner_Small': {'scale_factor': 0.239285, 'category': 'None'},
      'Threshold_Bead_Cereal_Bowl_White': {'scale_factor': 0.169761, 'category': 'Consumer Goods'},
      'Threshold_Bistro_Ceramic_Dinner_Plate_Ruby_Ring': {'scale_factor': 0.274492, 'category': 'None'},
      'Threshold_Dinner_Plate_Square_Rim_White_Porcelain': {'scale_factor': 0.27354, 'category': 'None'},
      'Threshold_Hand_Towel_Blue_Medallion_16_x_27': {'scale_factor': 0.212489, 'category': 'None'},
      'Threshold_Performance_Bath_Sheet_Sandoval_Blue_33_x_63': {'scale_factor': 0.425705, 'category': 'None'},
      'Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White': {'scale_factor': 0.13074000000000002, 'category': 'None'},
      'Threshold_Porcelain_Pitcher_White': {'scale_factor': 0.27844599999999997, 'category': 'None'},
      'Threshold_Porcelain_Serving_Bowl_Coupe_White': {'scale_factor': 0.209341, 'category': 'None'},
      'Threshold_Porcelain_Spoon_Rest_White': {'scale_factor': 0.236677, 'category': 'Bottles and Cans and Cups'},
      'Threshold_Porcelain_Teapot_White': {'scale_factor': 0.18668600000000002, 'category': 'None'},
      'Threshold_Ramekin_White_Porcelain': {'scale_factor': 0.111985, 'category': 'None'},
      'Threshold_Salad_Plate_Square_Rim_Porcelain': {'scale_factor': 0.21240799999999999, 'category': 'None'},
      'Threshold_Textured_Damask_Bath_Towel_Pink': {'scale_factor': 0.356443, 'category': 'None'},
      'Threshold_Tray_Rectangle_Porcelain': {'scale_factor': 0.31446799999999997, 'category': 'None'},
      'Tiek_Blue_Patent_Tieks_Italian_Leather_Ballet_Flats': {'scale_factor': 0.241952, 'category': 'Shoe'},
      'Tieks_Ballet_Flats_Diamond_White_Croc': {'scale_factor': 0.244114, 'category': 'Shoe'},
      'Tieks_Ballet_Flats_Electric_Snake': {'scale_factor': 0.242456, 'category': 'Shoe'},
      'Timberland_Mens_Classic_2Eye_Boat_Shoe': {'scale_factor': 0.283841, 'category': 'Shoe'},
      'Timberland_Mens_Earthkeepers_Casco_Bay_Canvas_Oxford': {'scale_factor': 0.292609, 'category': 'Shoe'},
      'Timberland_Mens_Earthkeepers_Casco_Bay_Canvas_SlipOn': {'scale_factor': 0.289654, 'category': 'Shoe'},
      'Timberland_Mens_Earthkeepers_Casco_Bay_Suede_1Eye': {'scale_factor': 0.29322800000000004, 'category': 'Shoe'},
      'Timberland_Mens_Earthkeepers_Heritage_2Eye_Boat_Shoe': {'scale_factor': 0.292687, 'category': 'Shoe'},
      'Timberland_Mens_Earthkeepers_Newmarket_6Inch_Cupsole_Boot': {'scale_factor': 0.29427499999999995, 'category': 'Shoe'},
      'Timberland_Mens_Earthkeepers_Stormbuck_Chukka': {'scale_factor': 0.30681, 'category': 'Shoe'},
      'Timberland_Mens_Earthkeepers_Stormbuck_Lite_Plain_Toe_Oxford': {'scale_factor': 0.31264800000000004, 'category': 'Shoe'},
      'Timberland_Mens_Earthkeepers_Stormbuck_Plain_Toe_Oxford': {'scale_factor': 0.30523100000000003, 'category': 'Shoe'},
      'Timberland_Womens_Classic_Amherst_2Eye_Boat_Shoe': {'scale_factor': 0.26674200000000003, 'category': 'Shoe'},
      'Timberland_Womens_Earthkeepers_Classic_Unlined_Boat_Shoe': {'scale_factor': 0.271429, 'category': 'Shoe'},
      'Timberland_Womens_Waterproof_Nellie_Chukka_Double': {'scale_factor': 0.279106, 'category': 'Shoe'},
      'Top_Paw_Dog_Bow_Bone_Ceramic_13_fl_oz_total': {'scale_factor': 0.18559399999999998, 'category': 'None'},
      'Top_Paw_Dog_Bowl_Blue_Paw_Bone_Ceramic_25_fl_oz_total': {'scale_factor': 0.153074, 'category': 'None'},
      'Tory_Burch_Kaitlin_Ballet_Mestico_in_BlackGold': {'scale_factor': 0.26055, 'category': 'Shoe'},
      'Tory_Burch_Kiernan_Riding_Boot': {'scale_factor': 0.295969, 'category': 'Shoe'},
      'Tory_Burch_Reva_Metal_Logo_Litus_Snake_Print_in_dark_BranchGold': {'scale_factor': 0.25780000000000003, 'category': 'Shoe'},
      'Tory_Burch_Sabe_65mm_Bootie_Split_Suede_in_Caramel': {'scale_factor': 0.24992, 'category': 'Shoe'},
      'Toys_R_Us_Treat_Dispenser_Smart_Puzzle_Foobler': {'scale_factor': 0.154722, 'category': 'None'},
      'Toysmith_Windem_Up_Flippin_Animals_Dog': {'scale_factor': 0.049926, 'category': 'Consumer Goods'},
      'Transformers_Age_of_Extinction_Mega_1Step_Bumblebee_Figure': {'scale_factor': 0.267859, 'category': 'Toys'},
      'Transformers_Age_of_Extinction_Stomp_and_Chomp_Grimlock_Figure': {'scale_factor': 0.477965, 'category': 'Toys'},
      'Travel_Mate_P_series_Notebook': {'scale_factor': 0.375376, 'category': 'None'},
      'Travel_Smart_Neck_Rest_Inflatable': {'scale_factor': 0.172885, 'category': 'Consumer Goods'},
      'TriStar_Products_PPC_Power_Pressure_Cooker_XL_in_Black': {'scale_factor': 0.329727, 'category': 'None'},
      'Tune_Belt_Sport_Armband_For_Samsung_Galaxy_S3': {'scale_factor': 0.220534, 'category': 'Consumer Goods'},
      'Twinlab_100_Whey_Protein_Fuel_Chocolate': {'scale_factor': 0.258659, 'category': 'Bottles and Cans and Cups'},
      'Twinlab_100_Whey_Protein_Fuel_Cookies_and_Cream': {'scale_factor': 0.25892899999999996, 'category': 'Bottles and Cans and Cups'},
      'Twinlab_100_Whey_Protein_Fuel_Vanilla': {'scale_factor': 0.25917399999999996, 'category': 'Bottles and Cans and Cups'},
      'Twinlab_Nitric_Fuel': {'scale_factor': 0.113247, 'category': 'Bottles and Cans and Cups'},
      'Twinlab_Premium_Creatine_Fuel_Powder': {'scale_factor': 0.11324200000000001, 'category': 'Bottles and Cans and Cups'},
      'UGG_Bailey_Bow_Womens_Clogs_Black_7': {'scale_factor': 0.296018, 'category': 'Shoe'},
      'UGG_Bailey_Button_Triplet_Womens_Boots_Black_7': {'scale_factor': 0.33577199999999996, 'category': 'Shoe'},
      'UGG_Bailey_Button_Womens_Boots_Black_7': {'scale_factor': 0.28240699999999996, 'category': 'Shoe'},
      'UGG_Cambridge_Womens_Black_7': {'scale_factor': 0.273892, 'category': 'Shoe'},
      'UGG_Classic_Tall_Womens_Boots_Chestnut_7': {'scale_factor': 0.331986, 'category': 'Shoe'},
      'UGG_Classic_Tall_Womens_Boots_Grey_7': {'scale_factor': 0.329903, 'category': 'Shoe'},
      'UGG_Jena_Womens_Java_7': {'scale_factor': 0.29046, 'category': 'Shoe'},
      'US_Army_Stash_Lunch_Bag': {'scale_factor': 0.273808, 'category': 'Bag'},
      'U_By_Kotex_Cleanwear_Heavy_Flow_Pads_32_Ct': {'scale_factor': 0.188191, 'category': 'Consumer Goods'},
      'U_By_Kotex_Sleek_Regular_Unscented_Tampons_36_Ct_Box': {'scale_factor': 0.146452, 'category': 'Consumer Goods'},
      'Ubisoft_RockSmith_Real_Tone_Cable_Xbox_360': {'scale_factor': 0.171647, 'category': 'Consumer Goods'},
      'Ultra_JarroDophilus': {'scale_factor': 0.109396, 'category': 'Consumer Goods'},
      'Unmellow_Yellow_Tieks_Neon_Patent_Leather_Ballet_Flats': {'scale_factor': 0.242072, 'category': 'Shoe'},
      'Utana_5_Porcelain_Ramekin_Large': {'scale_factor': 0.132498, 'category': 'None'},
      'VANS_FIRE_ROASTED_VEGGIE_CRACKERS_GLUTEN_FREE': {'scale_factor': 0.20917399999999997, 'category': 'Consumer Goods'},
      'VEGETABLE_GARDEN': {'scale_factor': 0.22994599999999998, 'category': 'Toys'},
      'Vans_Cereal_Honey_Nut_Crunch_11_oz_box': {'scale_factor': 0.281525, 'category': 'Consumer Goods'},
      'Victor_Reversible_Bookend': {'scale_factor': 0.236431, 'category': 'Consumer Goods'},
      'Vtech_Cruise_Learn_Car_25_Years': {'scale_factor': 0.35454300000000005, 'category': 'Toys'},
      'Vtech_Roll_Learn_Turtle': {'scale_factor': 0.23249399999999998, 'category': 'None'},
      'Vtech_Stack_Sing_Rings_636_Months': {'scale_factor': 0.23342100000000002, 'category': 'Toys'},
      'WATER_LANDING_NET': {'scale_factor': 0.390853, 'category': 'Toys'},
      'WHALE_WHISTLE_6PCS_SET': {'scale_factor': 0.073291, 'category': 'Toys'},
      'W_Lou_z0dkC78niiZ': {'scale_factor': 0.268559, 'category': 'Shoe'},
      'Weisshai_Great_White_Shark': {'scale_factor': 0.167077, 'category': 'None'},
      'Weston_No_22_Cajun_Jerky_Tonic_12_fl_oz_nLj64ZnGwDh': {'scale_factor': 0.199081, 'category': 'Bottles and Cans and Cups'},
      'Weston_No_33_Signature_Sausage_Tonic_12_fl_oz': {'scale_factor': 0.19867200000000002, 'category': 'Bottles and Cans and Cups'},
      'Whey_Protein_3_Flavor_Variety_Pack_12_Packets': {'scale_factor': 0.155249, 'category': 'Consumer Goods'},
      'Whey_Protein_Chocolate_12_Packets': {'scale_factor': 0.15607100000000002, 'category': 'Consumer Goods'},
      'Whey_Protein_Vanilla': {'scale_factor': 0.16903100000000001, 'category': 'Bottles and Cans and Cups'},
      'Whey_Protein_Vanilla_12_Packets': {'scale_factor': 0.155739, 'category': 'Consumer Goods'},
      'White_Rose_Tieks_Leather_Ballet_Flats_with_Floral_Rosettes': {'scale_factor': 0.24374400000000002, 'category': 'Shoe'},
      'Wild_Copper_Tieks_Metallic_Italian_Leather_Ballet_Flats': {'scale_factor': 0.243523, 'category': 'Shoe'},
      'Wilton_Easy_Layers_Cake_Pan_Set': {'scale_factor': 0.192021, 'category': 'Consumer Goods'},
      'Wilton_Pearlized_Sugar_Sprinkles_525_oz_Gold': {'scale_factor': 0.13447900000000002, 'category': 'Bottles and Cans and Cups'},
      'Wilton_PreCut_Parchment_Sheets_10_x_15_24_sheets': {'scale_factor': 0.27368800000000004, 'category': 'Consumer Goods'},
      'Winning_Moves_1180_Aggravation_Board_Game': {'scale_factor': 0.480989, 'category': 'Board Games'},
      'Wishbone_Pencil_Case': {'scale_factor': 0.221173, 'category': 'Bag'},
      'Womens_Angelfish_Boat_Shoe_in_Linen_Leopard_Sequin_NJDwosWNeZz': {'scale_factor': 0.2424, 'category': 'Shoe'},
      'Womens_Angelfish_Boat_Shoe_in_Linen_Oat': {'scale_factor': 0.244392, 'category': 'Shoe'},
      'Womens_Audrey_Slip_On_Boat_Shoe_in_Graphite_Nubuck_xWVkCJ5vxZe': {'scale_factor': 0.248604, 'category': 'Shoe'},
      'Womens_Authentic_Original_Boat_Shoe_in_Classic_Brown_Leather': {'scale_factor': 0.244498, 'category': 'Shoe'},
      'Womens_Authentic_Original_Boat_Shoe_in_Classic_Brown_Leather_48Nh7VuMwW6': {'scale_factor': 0.245277, 'category': 'Shoe'},
      'Womens_Authentic_Original_Boat_Shoe_in_Classic_Brown_Leather_cJSCWiH7QmB': {'scale_factor': 0.246456, 'category': 'Shoe'},
      'Womens_Authentic_Original_Boat_Shoe_in_Navy_Deerskin_50lWJaLWG8R': {'scale_factor': 0.25897800000000004, 'category': 'Shoe'},
      'Womens_Betty_Chukka_Boot_in_Grey_Jersey_Sequin': {'scale_factor': 0.249619, 'category': 'Shoe'},
      'Womens_Betty_Chukka_Boot_in_Navy_Jersey_Sequin_y0SsHk7dKUX': {'scale_factor': 0.24833, 'category': 'Shoe'},
      'Womens_Betty_Chukka_Boot_in_Navy_aEE8OqvMII4': {'scale_factor': 0.247898, 'category': 'Shoe'},
      'Womens_Betty_Chukka_Boot_in_Salt_Washed_Red_AL2YrOt9CRy': {'scale_factor': 0.24735100000000002, 'category': 'Shoe'},
      'Womens_Bluefish_2Eye_Boat_Shoe_in_Brown_Deerskin_JJ2pfEHTZG7': {'scale_factor': 0.288696, 'category': 'Shoe'},
      'Womens_Bluefish_2Eye_Boat_Shoe_in_Brown_Deerskin_i1TgjjO0AKY': {'scale_factor': 0.249106, 'category': 'Shoe'},
      'Womens_Bluefish_2Eye_Boat_Shoe_in_Linen_Natural_Sparkle_Suede_kqi81aojcOR': {'scale_factor': 0.249767, 'category': 'Shoe'},
      'Womens_Bluefish_2Eye_Boat_Shoe_in_Linen_Natural_Sparkle_Suede_w34KNQ41csH': {'scale_factor': 0.249619, 'category': 'Shoe'},
      'Womens_Bluefish_2Eye_Boat_Shoe_in_Linen_Oat': {'scale_factor': 0.24344500000000002, 'category': 'Shoe'},
      'Womens_Bluefish_2Eye_Boat_Shoe_in_Linen_Oat_IbrSyJdpT3h': {'scale_factor': 0.24718600000000002, 'category': 'Shoe'},
      'Womens_Bluefish_2Eye_Boat_Shoe_in_Linen_Oat_niKJKeWsmxY': {'scale_factor': 0.247365, 'category': 'Shoe'},
      'Womens_Bluefish_2Eye_Boat_Shoe_in_Tan': {'scale_factor': 0.24929199999999999, 'category': 'Shoe'},
      'Womens_Bluefish_2Eye_Boat_Shoe_in_White_Tumbled_YG44xIePRHw': {'scale_factor': 0.250698, 'category': 'Bottles and Cans and Cups'},
      'Womens_Canvas_Bahama_in_Black': {'scale_factor': 0.243251, 'category': 'Shoe'},
      'Womens_Canvas_Bahama_in_Black_vnJULsDVyq5': {'scale_factor': 0.24368599999999999, 'category': 'Shoe'},
      'Womens_Canvas_Bahama_in_White_4UyOhP6rYGO': {'scale_factor': 0.244785, 'category': 'Shoe'},
      'Womens_Canvas_Bahama_in_White_UfZPHGQpvz0': {'scale_factor': 0.244118, 'category': 'Shoe'},
      'Womens_Cloud_Logo_Authentic_Original_Boat_Shoe_in_Black_Supersoft_8LigQYwf4gr': {'scale_factor': 0.24718400000000001, 'category': 'Shoe'},
      'Womens_Cloud_Logo_Authentic_Original_Boat_Shoe_in_Black_Supersoft_cZR022qFI4k': {'scale_factor': 0.248062, 'category': 'Shoe'},
      'Womens_Hikerfish_Boot_in_Black_Leopard_bVSNY1Le1sm': {'scale_factor': 0.248063, 'category': 'Shoe'},
      'Womens_Hikerfish_Boot_in_Black_Leopard_ridcCWsv8rW': {'scale_factor': 0.24433700000000003, 'category': 'Shoe'},
      'Womens_Hikerfish_Boot_in_Linen_Leather_Sparkle_Suede_QktIyAkonrU': {'scale_factor': 0.24893, 'category': 'Shoe'},
      'Womens_Hikerfish_Boot_in_Linen_Leather_Sparkle_Suede_imlP8VkwqIH': {'scale_factor': 0.24850699999999998, 'category': 'Shoe'},
      'Womens_Multi_13': {'scale_factor': 0.11321300000000001, 'category': 'Bottles and Cans and Cups'},
      'Womens_Sequin_Bahama_in_White_Sequin_V9K1hf24Oxe': {'scale_factor': 0.249057, 'category': 'Shoe'},
      'Womens_Sequin_Bahama_in_White_Sequin_XoR8xTlxj1g': {'scale_factor': 0.244704, 'category': 'Shoe'},
      'Womens_Sequin_Bahama_in_White_Sequin_yGVsSA4tOwJ': {'scale_factor': 0.246659, 'category': 'Shoe'},
      'Womens_Sparkle_Suede_Angelfish_in_Grey_Sparkle_Suede_Silver': {'scale_factor': 0.23909, 'category': 'Shoe'},
      'Womens_Sparkle_Suede_Bahama_in_Silver_Sparkle_Suede_Grey_Patent_tYrIBLMhSTN': {'scale_factor': 0.248572, 'category': 'Shoe'},
      'Womens_Sparkle_Suede_Bahama_in_Silver_Sparkle_Suede_Grey_Patent_x9rclU7EJXx': {'scale_factor': 0.24841000000000002, 'category': 'Shoe'},
      'Womens_Suede_Bahama_in_Graphite_Suede_cUAjIMhWSO9': {'scale_factor': 0.255043, 'category': 'Shoe'},
      'Womens_Suede_Bahama_in_Graphite_Suede_p1KUwoWbw7R': {'scale_factor': 0.29468, 'category': 'Shoe'},
      'Womens_Suede_Bahama_in_Graphite_Suede_t22AJSRjBOX': {'scale_factor': 0.255688, 'category': 'Shoe'},
      'Womens_Teva_Capistrano_Bootie': {'scale_factor': 0.29047, 'category': 'Shoe'},
      'Womens_Teva_Capistrano_Bootie_ldjRT9yZ5Ht': {'scale_factor': 0.292652, 'category': 'Shoe'},
      'Wooden_ABC_123_Blocks_50_pack': {'scale_factor': 0.213881, 'category': 'Toys'},
      'Wrigley_Orbit_Mint_Variety_18_Count': {'scale_factor': 0.22901300000000002, 'category': 'Consumer Goods'},
      'Xyli_Pure_Xylitol': {'scale_factor': 0.14713199999999999, 'category': 'Bottles and Cans and Cups'},
      'YumYum_D3_Liquid': {'scale_factor': 0.100074, 'category': 'Consumer Goods'},
      'ZX700_lYiwcTIekXk': {'scale_factor': 0.293759, 'category': 'Shoe'},
      'ZX700_mf9Pc06uL06': {'scale_factor': 0.29282600000000003, 'category': 'Shoe'},
      'ZX700_mzGbdP3u6JB': {'scale_factor': 0.291769, 'category': 'Shoe'},
      'ZigKick_Hoops': {'scale_factor': 0.279158, 'category': 'Shoe'},
      'adiZero_Slide_2_SC': {'scale_factor': 0.29115, 'category': 'Shoe'},
      'adistar_boost_m': {'scale_factor': 0.30469199999999996, 'category': 'Shoe'},
      'adizero_5Tool_25': {'scale_factor': 0.289979, 'category': 'Shoe'},
      'adizero_F50_TRX_FG_LEA': {'scale_factor': 0.281507, 'category': 'Shoe'},
  }
  result = conversion_dict[asset_id]
  return result["scale_factor"], result["category"]
