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
from typing import Dict, List, Union

from etils import epath
import imageio
import numpy as np
import png
import tensorflow as tf
import tensorflow_datasets.public_api as tfds


_DESCRIPTION = """
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

The dataset contains approx 10k videos rendered at 256x256 pixels and 12fps.

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
- "scale": float between 0.75 and 3.0
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
class MoviEConfig(tfds.core.BuilderConfig):
  """"Configuration for Multi-Object Video (MoviE) dataset."""
  height: int = 256
  width: int = 256
  num_frames: int = 24
  validation_ratio: float = 0.1
  train_val_path: str = None
  test_split_paths: Dict[str, str] = dataclasses.field(default_factory=dict)


class MoviE(tfds.core.BeamBasedBuilder):
  """DatasetBuilder for MOVi-E dataset."""
  VERSION = tfds.core.Version("1.0.0")
  RELEASE_NOTES = {
      "1.0.0": "initial release",
  }

  BUILDER_CONFIGS = [
      MoviEConfig(
          name="256x256",
          description="Full resolution of 256x256",
          height=256,
          width=256,
          validation_ratio=0.025,
          train_val_path="gs://research-brain-kubric-xgcp/jobs/movi_e_regen_10k/",
          test_split_paths={
              "test": "gs://research-brain-kubric-xgcp/jobs/movi_e_test_regen_1k/",
          }
      ),
      MoviEConfig(
          name="128x128",
          description="Downscaled to 128x128",
          height=128,
          width=128,
          validation_ratio=0.025,
          train_val_path="gs://research-brain-kubric-xgcp/jobs/movi_e_regen_10k/",
          test_split_paths={
              "test": "gs://research-brain-kubric-xgcp/jobs/movi_e_test_regen_1k/",
          }
      ),
  ]


  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""

    h = self.builder_config.height
    w = self.builder_config.width
    s = self.builder_config.num_frames

    def get_movi_e_instance_features(seq_length: int):
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
            },
            "background": tfds.features.Text(),
            "instances": tfds.features.Sequence(
                feature=get_movi_e_instance_features(seq_length=s)),
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

      # add MoviE-D specific instance information:
      for i, obj in enumerate(result["instances"]):
        obj["asset_id"] = asset_id_from_metadata(metadata["instances"][i])
        obj["category"] = metadata["instances"][i]["category"]
        obj["scale"] = metadata["instances"][i]["scale"]
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
      },
      "background": metadata["metadata"]["background"],
      "instances": [format_instance_information(obj)
                    for obj in metadata["instances"]],
      "camera": format_camera_information(metadata),
      "events": format_events_information(events),
  }

  resolution = metadata["metadata"]["resolution"]

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
        data_ranges["forward_flow"]["min"] / scale,
        data_ranges["forward_flow"]["max"] / scale]
    result["forward_flow"] = [
        subsample_nearest_neighbor(read_png(frame_path)[..., :2],
                                   target_size)
        for frame_path in paths["forward_flow"]]

  if "backward_flow" in layers:
    result["metadata"]["backward_flow_range"] = [
        data_ranges["backward_flow"]["min"] / scale,
        data_ranges["backward_flow"]["max"] / scale]
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


def asset_id_from_metadata(meta):
  asset_id_lookup = {
      (20706, 'Shoe', '11pro SL TRX FG'): '11pro_SL_TRX_FG',
      (8922, 'Consumer Goods', '2 of Jenga Clas'): '2_of_Jenga_Classic_Game',
      (23766, 'Toys', '30 CONSTRUCTION'): '30_CONSTRUCTION_SET',
      (14142, 'Consumer Goods', '3D Dollhouse Ha'): '3D_Dollhouse_Happy_Brother',
      (10490, 'Toys', '3D Dollhouse La'): '3D_Dollhouse_Lamp',
      (3202, 'Toys', '3D Dollhouse Re'): '3D_Dollhouse_Refrigerator',
      (7978, 'Toys', '3D Dollhouse Si'): '3D_Dollhouse_Sink',
      (15052, 'Toys', '3D Dollhouse So'): '3D_Dollhouse_Sofa',
      (8876, 'Toys', '3D Dollhouse Sw'): '3D_Dollhouse_Swing',
      (6446, 'Toys', '3D Dollhouse Ta'): '3D_Dollhouse_TablePurple',
      (4750, 'None', '3M Antislip Sur'): '3M_Antislip_Surfacing_Light_Duty_White',
      (11422, 'None', '3M Vinyl Tape, '): '3M_Vinyl_Tape_Green_1_x_36_yd',
      (15358, 'Consumer Goods', '4.5oz RAMEKIN A'): '45oz_RAMEKIN_ASST_DEEP_COLORS',
      (10728, 'Toys', '50 BLOCKS\nConta'): '50_BLOCKS',
      (8362, 'Bottles and Cans and Cups', '5 HTP'): '5_HTP',
      (26126, 'Toys', '60 CONSTRUCTION'): '60_CONSTRUCTION_SET',
      (11224, 'Consumer Goods', 'ACE Coffee Mug,'): 'ACE_Coffee_Mug_Kristen_16_oz_cup',
      (14692, 'Toys', 'ALPHABET A-Z (G'): 'ALPHABET_AZ_GRADIENT',
      (13878, 'Toys', 'ALPHABET A-Z (G'): 'ALPHABET_AZ_GRADIENT_WQb1ufEycSj',
      (23444, 'Shoe', 'AMBERLIGHT UP W'): 'AMBERLIGHT_UP_W',
      (37508, 'Shoe', 'ASICS GEL-1140V'): 'ASICS_GEL1140V_WhiteBlackSilver',
      (38246, 'Shoe', 'ASICS GEL-1140V'): 'ASICS_GEL1140V_WhiteRoyalSilver',
      (45698, 'Shoe', 'ASICS GEL-Ace™ '): 'ASICS_GELAce_Pro_Pearl_WhitePink',
      (39092, 'Shoe', 'ASICS GEL-Blur3'): 'ASICS_GELBlur33_20_GS_BlackWhiteSafety_Orange',
      (30660, 'Shoe', 'ASICS GEL-Blur3'): 'ASICS_GELBlur33_20_GS_Flash_YellowHot_PunchSilver',
      (33438, 'Shoe', 'ASICS GEL-Chall'): 'ASICS_GELChallenger_9_Royal_BlueWhiteBlack',
      (44502, 'Shoe', 'ASICS GEL-Dirt '): 'ASICS_GELDirt_Dog_4_SunFlameBlack',
      (34586, 'Shoe', 'ASICS GEL-Links'): 'ASICS_GELLinksmaster_WhiteCoffeeSand',
      (31266, 'Shoe', 'ASICS GEL-Links'): 'ASICS_GELLinksmaster_WhiteRasberryGunmetal',
      (34160, 'Shoe', 'ASICS GEL-Links'): 'ASICS_GELLinksmaster_WhiteSilverCarolina_Blue',
      (36694, 'Shoe', 'ASICS GEL-Resol'): 'ASICS_GELResolution_5_Flash_YellowBlackSilver',
      (26000, 'Shoe', 'ASICS GEL-Tour '): 'ASICS_GELTour_Lyte_WhiteOrchidSilver',
      (21152, 'Shoe', 'ASICS Hyper-Roc'): 'ASICS_HyperRocketgirl_SP_5_WhiteMalibu_BlueBlack',
      (20336, 'Toys', 'ASSORTED VEGETA'): 'ASSORTED_VEGETABLE_SET',
      (45866, 'Shoe', 'Adrenaline GTS '): 'Adrenaline_GTS_13_Color_DrkDenimWhtBachlorBttnSlvr_Size_50_yfK40TNjq0V',
      (48412, 'Shoe', 'Adrenaline GTS '): 'Adrenaline_GTS_13_Color_WhtObsdianBlckOlmpcSlvr_Size_70',
      (5762, 'None', 'Air Hogs Wind F'): 'Air_Hogs_Wind_Flyers_Set_Airplane_Red',
      (8126, 'Bottles and Cans and Cups', 'Allergen-Free J'): 'AllergenFree_JarroDophilus',
      (11836, 'Consumer Goods', 'Android Figure,'): 'Android_Figure_Chrome',
      (9698, 'Consumer Goods', 'Android Figure,'): 'Android_Figure_Orange',
      (10618, 'Consumer Goods', 'Android Figure,'): 'Android_Figure_Panda',
      (14214, 'Legos', 'Android Lego'): 'Android_Lego',
      (2444, 'Media Cases', 'Animal Crossing'): 'Animal_Crossing_New_Leaf_Nintendo_3DS_Game',
      (21978, 'Toys', 'Animal Planet F'): 'Animal_Planet_Foam_2Headed_Dragon',
      (1690, 'Consumer Goods', 'Apples to Apple'): 'Apples_to_Apples_Kids_Edition',
      (2820, 'Consumer Goods', 'Arm Hammer Diap'): 'Arm_Hammer_Diaper_Pail_Refills_12_Pack_MFWkmoweejt',
      (21582, 'Consumer Goods', 'Aroma Stainless'): 'Aroma_Stainless_Steel_Milk_Frother_2_Cup',
      (17458, 'None', 'Asus - 802.11ac'): 'Asus_80211ac_DualBand_Gigabit_Wireless_Router_RTAC68R',
      (16504, 'None', 'Asus M5A78L-M/U'): 'Asus_M5A78LMUSB3_Motherboard_Micro_ATX_Socket_AM3',
      (9370, 'None', 'Asus M5A99FX PR'): 'Asus_M5A99FX_PRO_R20_Motherboard_ATX_Socket_AM3',
      (1708, 'None', 'Asus Sabertooth'): 'Asus_Sabertooth_990FX_20_Motherboard_ATX_Socket_AM3',
      (3184, 'None', 'Asus Sabertooth'): 'Asus_Sabertooth_Z97_MARK_1_Motherboard_ATX_LGA1150_Socket',
      (2838, 'None', 'Asus X99-Deluxe'): 'Asus_X99Deluxe_Motherboard_ATX_LGA2011v3_Socket',
      (3592, 'None', 'Asus Z87-PRO Mo'): 'Asus_Z87PRO_Motherboard_ATX_LGA1150_Socket',
      (5896, 'None', 'Asus Z97-AR LGA'): 'Asus_Z97AR_LGA_1150_Intel_ATX_Motherboard',
      (34176, 'None', 'Asus Z97I-PLUS '): 'Asus_Z97IPLUS_Motherboard_Mini_ITX_LGA1150_Socket',
      (18402, 'Toys', 'Avengers Gamma '): 'Avengers_Gamma_Green_Smash_Fists',
      (10822, 'Action Figures', 'Avengers Thor'): 'Avengers_Thor_PLlrpYniaeB',
      (30976, 'Shoe', 'Azure Snake Tie'): 'Azure_Snake_Tieks_Leather_Snake_Print_Ballet_Flats',
      (17646, 'Toys', 'BABY CAR\nThis c'): 'BABY_CAR',
      (14524, 'Toys', 'BAGEL WITH CHEE'): 'BAGEL_WITH_CHEESE',
      (6782, 'Toys', 'BAKING UTENSILS'): 'BAKING_UTENSILS',
      (11788, 'Toys', 'BALANCING CACTU'): 'BALANCING_CACTUS',
      (6320, 'Toys', 'BATHROOM - CLAS'): 'BATHROOM_CLASSIC',
      (6798, 'Toys', 'BATHROOM (FURNI'): 'BATHROOM_FURNITURE_SET_1',
      (8436, 'Toys', 'BEDROOM - CLASS'): 'BEDROOM_CLASSIC',
      (8872, 'Toys', 'BEDROOM - CLASS'): 'BEDROOM_CLASSIC_Gi22DjScTVS',
      (10840, 'Toys', 'BEDROOM - NEO\nT'): 'BEDROOM_NEO',
      (14542, 'None', 'B.I.A Cordon Bl'): 'BIA_Cordon_Bleu_White_Porcelain_Utensil_Holder_900028',
      (16274, 'None', 'BIA Porcelain R'): 'BIA_Porcelain_Ramekin_With_Glazed_Rim_35_45_oz_cup',
      (17104, 'Toys', 'BIRD RATTLE\nThi'): 'BIRD_RATTLE',
      (16190, 'Toys', 'BRAILLE ALPHABE'): 'BRAILLE_ALPHABET_AZ',
      (11676, 'Toys', 'BREAKFAST MENU\n'): 'BREAKFAST_MENU',
      (11724, 'Toys', 'BUILD -A- ROBOT'): 'BUILD_A_ROBOT',
      (12958, 'Toys', 'BUILD A ZOO\nHel'): 'BUILD_A_ZOO',
      (12146, 'Toys', 'BUNNY RACER\nHav'): 'BUNNY_RACER',
      (11316, 'Toys', 'BUNNY RATTLE\nLi'): 'BUNNY_RATTLE',
      (10528, 'None', 'Baby Elements S'): 'Baby_Elements_Stacking_Cups',
      (1464, 'Board Games', 'Balderdash Game'): 'Balderdash_Game',
      (6770, 'Consumer Goods', 'Beetle Adventur'): 'Beetle_Adventure_Racing_Nintendo_64',
      (8726, 'Bottles and Cans and Cups', 'Beta Glucan'): 'Beta_Glucan',
      (4630, 'Consumer Goods', 'Beyonc? - Life '): 'Beyonc_Life_is_But_a_Dream_DVD',
      (9344, 'Bottles and Cans and Cups', 'Bifidus Balance'): 'Bifidus_Balance_FOS',
      (12514, 'Bag', 'Big Dot Aqua Pe'): 'Big_Dot_Aqua_Pencil_Case',
      (11720, 'Bag', 'Big Dot Pink Pe'): 'Big_Dot_Pink_Pencil_Case',
      (31182, 'Consumer Goods', 'Big O Sponges, '): 'Big_O_Sponges_Assorted_Cellulose_12_pack',
      (6932, 'None', 'Black/Black Nin'): 'BlackBlack_Nintendo_3DSXL',
      (30034, 'None', 'Black & Decker '): 'Black_Decker_CM2035B_12Cup_Thermal_Coffeemaker',
      (47628, 'None', 'Black & Decker '): 'Black_Decker_Stainless_Steel_Toaster_4_Slice',
      (4028, 'Consumer Goods', 'Black Elderberr'): 'Black_Elderberry_Syrup_54_oz_Gaia_Herbs',
      (2036, 'Consumer Goods', 'Black Forest Fr'): 'Black_Forest_Fruit_Snacks_28_Pack_Grape',
      (2544, 'Consumer Goods', 'Black Forest Fr'): 'Black_Forest_Fruit_Snacks_Juicy_Filled_Centers_10_pouches_9_oz_total',
      (4658, 'None', 'Black and Decke'): 'Black_and_Decker_PBJ2000_FusionBlade_Blender_Jars',
      (31474, 'None', 'Black and Decke'): 'Black_and_Decker_TR3500SD_2Slice_Toaster',
      (8676, 'Bottles and Cans and Cups', 'Blackcurrant + '): 'Blackcurrant_Lutein',
      (5932, 'None', 'Blue/Black Nint'): 'BlueBlack_Nintendo_3DSXL',
      (6068, 'Media Cases', 'Blue Jasmine [I'): 'Blue_Jasmine_Includes_Digital_Copy_UltraViolet_DVD',
      (9132, 'Bottles and Cans and Cups', 'Borage GLA-240+'): 'Borage_GLA240Gamma_Tocopherol',
      (15502, 'None', 'Bradshaw Intern'): 'Bradshaw_International_11642_7_Qt_MP_Plastic_Bowl',
      (13840, 'None', 'Breyer Horse Of'): 'Breyer_Horse_Of_The_Year_2015',
      (3634, 'Consumer Goods', 'Brisk Iced Tea,'): 'Brisk_Iced_Tea_Lemon_12_12_fl_oz_355_ml_cans_144_fl_oz_426_lt',
      (4782, 'Consumer Goods', 'Brother Ink Car'): 'Brother_Ink_Cartridge_Magenta_LC75M',
      (5762, 'Consumer Goods', 'Brother LC 1053'): 'Brother_LC_1053PKS_Ink_Cartridge_CyanMagentaYellow_1pack',
      (4816, 'Consumer Goods', 'Brother Printin'): 'Brother_Printing_Cartridge_PC501',
      (11652, 'Toys', 'CARS-II'): 'CARSII',
      (25266, 'Toys', 'CAR CARRIER TRA'): 'CAR_CARRIER_TRAIN',
      (22108, 'Toys', 'CASTLE BLOCKS\nE'): 'CASTLE_BLOCKS',
      (11022, 'Toys', 'CHICKEN NESTING'): 'CHICKEN_NESTING',
      (12808, 'Toys', 'CHICKEN RACER\nH'): 'CHICKEN_RACER',
      (16054, 'Toys', "CHILDREN'S ROOM"): 'CHILDRENS_ROOM_NEO',
      (12712, 'Toys', 'CHILDREN BEDROO'): 'CHILDREN_BEDROOM_CLASSIC',
      (11192, 'Toys', 'CITY TAXI & POL'): 'CITY_TAXI_POLICE_CAR',
      (41832, 'Shoe', 'CLIMACOOL BOAT '): 'CLIMACOOL_BOAT_BREEZE_IE6CyqSaDwN',
      (9562, 'Toys', 'COAST GUARD BOA'): 'COAST_GUARD_BOAT',
      (21962, 'Toys', 'CONE SORTING\nLe'): 'CONE_SORTING',
      (5406, 'Toys', 'CONE SORTING\nLe'): 'CONE_SORTING_kg5fbARBwts',
      (14610, 'Toys', 'CREATIVE BLOCKS'): 'CREATIVE_BLOCKS_35_MM',
      (23540, 'Shoe', 'California Navy'): 'California_Navy_Tieks_Italian_Leather_Ballet_Flats',
      (8262, 'None', 'Calphalon Kitch'): 'Calphalon_Kitchen_Essentials_12_Cast_Iron_Fry_Pan_Black',
      (6292, 'Consumer Goods', 'Canon 225/226 I'): 'Canon_225226_Ink_Cartridges_BlackColor_Cyan_Magenta_Yellow_6_count',
      (6052, 'Consumer Goods', 'Canon Ink Cartr'): 'Canon_Ink_Cartridge_Green_6',
      (5132, 'Consumer Goods', 'Canon Pixma Chr'): 'Canon_Pixma_Chromalife_100_Magenta_8',
      (6536, 'Consumer Goods', 'Canon Pixma Ink'): 'Canon_Pixma_Ink_Cartridge_251_M',
      (5612, 'Consumer Goods', 'Canon Pixma Ink'): 'Canon_Pixma_Ink_Cartridge_8',
      (5722, 'Consumer Goods', 'Canon Pixma Ink'): 'Canon_Pixma_Ink_Cartridge_8_Green',
      (6218, 'Consumer Goods', 'Canon Pixma Ink'): 'Canon_Pixma_Ink_Cartridge_8_Red',
      (5756, 'Consumer Goods', 'Canon Pixma Ink'): 'Canon_Pixma_Ink_Cartridge_Cyan_251',
      (42288, 'Shoe', 'Cascadia 8, Col'): 'Cascadia_8_Color_AquariusHibscsBearingSeaBlk_Size_50',
      (7768, 'Consumer Goods', 'Central Garden '): 'Central_Garden_Flower_Pot_Goo_425',
      (4436, 'None', 'Chef Style Roun'): 'Chef_Style_Round_Cake_Pan_9_inch_pan',
      (9320, 'None', 'Chefmate 8" Fry'): 'Chefmate_8_Frypan',
      (15226, 'Shoe', 'Chelsea Blk.Hee'): 'Chelsea_BlkHeelPMP_DwxLtZNxLZZ',
      (7624, 'Shoe', 'Chelsea lo fl r'): 'Chelsea_lo_fl_rdheel_nQ0LPNF1oMw',
      (7910, 'Shoe', 'Chelsea lo fl r'): 'Chelsea_lo_fl_rdheel_zAQrnhlEfw8',
      (18510, 'Consumer Goods', 'Circo Fish Toot'): 'Circo_Fish_Toothbrush_Holder_14995988',
      (28482, 'Shoe', 'ClimaCool Aerat'): 'ClimaCool_Aerate_2_W_Wide',
      (10368, 'None', 'Clorox Premium '): 'Clorox_Premium_Choice_Gloves_SM_1_pair',
      (12158, 'None', 'Closetmaid Prem'): 'Closetmaid_Premium_Fabric_Cube_Red',
      (4450, 'Board Games', 'Clue Board Game'): 'Clue_Board_Game_Classic_Edition',
      (10490, 'Bottles and Cans and Cups', 'Co-Q10'): 'CoQ10',
      (9298, 'Bottles and Cans and Cups', 'Co-Q10'): 'CoQ10_BjTLbuRVt1t',
      (10984, 'Bottles and Cans and Cups', 'Co-Q10'): 'CoQ10_wSSVoxVppVD',
      (4484, 'None', 'Cole Hardware A'): 'Cole_Hardware_Antislip_Surfacing_Material_White',
      (5650, 'None', 'Cole Hardware A'): 'Cole_Hardware_Antislip_Surfacing_White_2_x_60',
      (10342, 'None', 'Cole Hardware B'): 'Cole_Hardware_Bowl_Scirocco_YellowBlue',
      (34140, 'Consumer Goods', 'Cole Hardware B'): 'Cole_Hardware_Butter_Dish_Square_Red',
      (10364, 'None', 'Cole Hardware D'): 'Cole_Hardware_Deep_Bowl_Good_Earth_1075',
      (18694, 'None', 'Cole Hardware D'): 'Cole_Hardware_Dishtowel_Blue',
      (17596, 'None', 'Cole Hardware D'): 'Cole_Hardware_Dishtowel_BlueWhite',
      (21206, 'None', 'Cole Hardware D'): 'Cole_Hardware_Dishtowel_Multicolors',
      (21290, 'None', 'Cole Hardware D'): 'Cole_Hardware_Dishtowel_Red',
      (14942, 'None', 'Cole Hardware D'): 'Cole_Hardware_Dishtowel_Stripe',
      (8752, 'None', 'Cole Hardware E'): 'Cole_Hardware_Electric_Pot_Assortment_55',
      (7674, 'None', 'Cole Hardware E'): 'Cole_Hardware_Electric_Pot_Cabana_55',
      (5820, 'None', 'Cole Hardware F'): 'Cole_Hardware_Flower_Pot_1025',
      (7836, 'None', 'Cole Hardware H'): 'Cole_Hardware_Hammer_Black',
      (6458, 'Consumer Goods', 'Cole Hardware M'): 'Cole_Hardware_Mini_Honey_Dipper',
      (8716, 'None', 'Cole Hardware M'): 'Cole_Hardware_Mug_Classic_Blue',
      (6834, 'None', 'Cole Hardware O'): 'Cole_Hardware_Orchid_Pot_85',
      (4302, 'None', 'Cole Hardware P'): 'Cole_Hardware_Plant_Saucer_Brown_125',
      (4126, 'None', 'Cole Hardware P'): 'Cole_Hardware_Plant_Saucer_Glazed_9',
      (6796, 'None', 'Cole Hardware S'): 'Cole_Hardware_Saucer_Electric',
      (5248, 'None', 'Cole Hardware S'): 'Cole_Hardware_Saucer_Glazed_6',
      (8752, 'Consumer Goods', 'Cole Hardware S'): 'Cole_Hardware_School_Bell_Solid_Brass_38',
      (26766, 'Shoe', 'Colton Wntr Chu'): 'Colton_Wntr_Chukka_y4jO0I8JQFW',
      (2544, 'Board Games', 'Connect 4 Launc'): 'Connect_4_Launchers',
      (28192, 'Consumer Goods', 'Cootie Game'): 'Cootie_Game',
      (2454, 'Consumer Goods', 'Cootie Game'): 'Cootie_Game_tDhURNbfU5J',
      (30854, 'Shoe', 'Copperhead Snak'): 'Copperhead_Snake_Tieks_Brown_Snake_Print_Ballet_Flats',
      (4612, 'None', 'Corningware CW '): 'Corningware_CW_by_Corningware_3qt_Oblong_Casserole_Dish_Blue',
      (37954, 'Shoe', 'Court Attitude'): 'Court_Attitude',
      (5610, 'None', 'Craftsman Grip '): 'Craftsman_Grip_Screwdriver_Phillips_Cushion',
      (5558, 'None', 'Crayola Bonus 6'): 'Crayola_Bonus_64_Crayons',
      (3250, 'Consumer Goods', 'Crayola Crayons'): 'Crayola_Crayons_120_crayons',
      (7562, 'Consumer Goods', 'Crayola Crayons'): 'Crayola_Crayons_24_count',
      (7522, 'Consumer Goods', 'Crayola Crayons'): 'Crayola_Crayons_Washable_24_crayons',
      (4150, 'Consumer Goods', 'Crayola Model M'): 'Crayola_Model_Magic_Modeling_Material_Single_Packs_6_pack_05_oz_packs',
      (2756, 'Consumer Goods', 'Crayola Model M'): 'Crayola_Model_Magic_Modeling_Material_White_3_oz',
      (7342, 'None', 'Crayola Washabl'): 'Crayola_Washable_Fingerpaint_Red_Blue_Yellow_3_count_8_fl_oz_bottes_each',
      (5898, 'Consumer Goods', 'Crayola Washabl'): 'Crayola_Washable_Sidewalk_Chalk_16_pack',
      (9010, 'Consumer Goods', 'Crayola Washabl'): 'Crayola_Washable_Sidewalk_Chalk_16_pack_wDZECiw7J6s',
      (25056, 'Shoe', 'Crazy 8\nFW-HIGH'): 'Crazy_8',
      (39326, 'Shoe', 'Crazy Shadow 2'): 'Crazy_Shadow_2',
      (23866, 'Shoe', 'Crazy Shadow 2'): 'Crazy_Shadow_2_oW4Jd10HFFr',
      (17216, 'Shoe', 'Cream Tieks - I'): 'Cream_Tieks_Italian_Leather_Ballet_Flats',
      (9888, 'Bottles and Cans and Cups', 'Creatine Monohy'): 'Creatine_Monohydrate',
      (10248, 'Consumer Goods', 'Crosley Alarm C'): 'Crosley_Alarm_Clock_Vintage_Metal',
      (3682, 'Consumer Goods', 'Crunch Girl Sco'): 'Crunch_Girl_Scouts_Candy_Bars_Peanut_Butter_Creme_78_oz_box',
      (86044, 'None', 'Curver Storage '): 'Curver_Storage_Bin_Black_Small',
      (20484, 'Toys', 'DANCING ALLIGAT'): 'DANCING_ALLIGATOR',
      (16340, 'Toys', 'DANCING ALLIGAT'): 'DANCING_ALLIGATOR_zoWBjc0jbTs',
      (9242, 'Bottles and Cans and Cups', 'DIM + CDG'): 'DIM_CDG',
      (12230, 'Toys', 'DINING ROOM - C'): 'DINING_ROOM_CLASSIC',
      (9792, 'Toys', 'DINING ROOM - C'): 'DINING_ROOM_CLASSIC_UJuxQ0hv5XU',
      (9412, 'Toys', 'DINNING ROOM (F'): 'DINNING_ROOM_FURNITURE_SET_1',
      (35812, 'Toys', 'DOLL FAMILY'): 'DOLL_FAMILY',
      (22828, 'Hat', 'DPC Handmade Ha'): 'DPC_Handmade_Hat_Brown',
      (34294, 'None', 'DPC Thinsulate '): 'DPC_Thinsulate_Isolate_Gloves_Brown',
      (55206, 'Hat', 'DPC tropical Tr'): 'DPC_tropical_Trends_Hat',
      (18932, 'Shoe', 'DRAGON W'): 'DRAGON_W',
      (30974, 'Shoe', 'D ROSE 4.5\nFW-H'): 'D_ROSE_45',
      (47688, 'Shoe', 'D ROSE 773 II'): 'D_ROSE_773_II_Kqclsph05pE',
      (45804, 'Shoe', 'D ROSE 773 II'): 'D_ROSE_773_II_hvInJwJ5HUD',
      (35868, 'Shoe', 'D ROSE ENGLEWOO'): 'D_ROSE_ENGLEWOOD_II',
      (5510, 'Consumer Goods', 'Dell Ink Cartri'): 'Dell_Ink_Cartridge',
      (3768, 'Consumer Goods', 'Dell Ink Cartri'): 'Dell_Ink_Cartridge_Yellow_31',
      (4150, 'Consumer Goods', 'Dell Series 9 C'): 'Dell_Series_9_Color_Ink_Cartridge_MK993_High_Yield',
      (1936, 'None', 'Design Ideas Dr'): 'Design_Ideas_Drawer_Store_Organizer',
      (2656, 'Consumer Goods', 'Deskstar Desk T'): 'Deskstar_Desk_Top_Hard_Drive_1_TB',
      (6816, 'Consumer Goods', 'Diamond Visions'): 'Diamond_Visions_Scissors_Red',
      (1990, 'Consumer Goods', 'Diet Pepsi Soda'): 'Diet_Pepsi_Soda_Cola12_Pack_12_oz_Cans',
      (16358, 'Bag', 'Digital Camo Do'): 'Digital_Camo_Double_Decker_Lunch_Bag',
      (17722, 'Action Figures', 'Dino 3'): 'Dino_3',
      (21900, 'Action Figures', 'Dino 4'): 'Dino_4',
      (16162, 'Action Figures', 'Dino 5'): 'Dino_5',
      (25548, 'None', 'Dixie 10 ounce '): 'Dixie_10_ounce_Bowls_35_ct',
      (67204, 'None', 'Dog\nDog'): 'Dog',
      (9452, 'Bottles and Cans and Cups', "Don Francisco's"): 'Don_Franciscos_Gourmet_Coffee_Medium_Decaf_100_Colombian_12_oz_340_g',
      (23934, 'None', 'Down To Earth C'): 'Down_To_Earth_Ceramic_Orchid_Pot_Asst_Blue',
      (20260, 'Consumer Goods', 'Down To Earth O'): 'Down_To_Earth_Orchid_Pot_Ceramic_Lime',
      (24492, 'Consumer Goods', 'Down To Earth O'): 'Down_To_Earth_Orchid_Pot_Ceramic_Red',
      (35714, 'Shoe', 'ENFR MID (ENFOR'): 'ENFR_MID_ENFORCER',
      (3812, 'Consumer Goods', 'Eat to Live: Th'): 'Eat_to_Live_The_Amazing_NutrientRich_Program_for_Fast_and_Sustained_Weight_Loss_Revised_Edition_Book',
      (7542, 'None', 'Ecoforms Cup, B'): 'Ecoforms_Cup_B4_SAN',
      (9126, 'None', 'Ecoforms Garden'): 'Ecoforms_Garden_Pot_GP16ATurquois',
      (9524, 'None', 'Ecoforms Plant '): 'Ecoforms_Plant_Bowl_Atlas_Low',
      (6254, 'None', 'Ecoforms Plant '): 'Ecoforms_Plant_Bowl_Turquoise_7',
      (8288, 'None', 'Ecoforms Plant '): 'Ecoforms_Plant_Container_12_Pot_Nova',
      (8754, 'None', 'Ecoforms Plant '): 'Ecoforms_Plant_Container_B4_Har',
      (12316, 'None', 'Ecoforms Plant '): 'Ecoforms_Plant_Container_FB6_Tur',
      (11020, 'None', 'Ecoforms Plant '): 'Ecoforms_Plant_Container_GP16AMOCHA',
      (10040, 'None', 'Ecoforms Plant '): 'Ecoforms_Plant_Container_GP16A_Coral',
      (5320, 'None', 'Ecoforms Plant '): 'Ecoforms_Plant_Container_QP6CORAL',
      (3962, 'None', 'Ecoforms Plant '): 'Ecoforms_Plant_Container_QP6HARVEST',
      (7442, 'None', 'Ecoforms Plant '): 'Ecoforms_Plant_Container_QP_Harvest',
      (14776, 'None', 'Ecoforms Plant '): 'Ecoforms_Plant_Container_QP_Turquoise',
      (3844, 'None', 'Ecoforms Plant '): 'Ecoforms_Plant_Container_Quadra_Sand_QP6',
      (5032, 'None', 'Ecoforms Plant '): 'Ecoforms_Plant_Container_Quadra_Turquoise_QP12',
      (4592, 'None', 'Ecoforms Plant '): 'Ecoforms_Plant_Container_S14Turquoise',
      (4256, 'None', 'Ecoforms Plant '): 'Ecoforms_Plant_Container_S24NATURAL',
      (4244, 'None', 'Ecoforms Plant '): 'Ecoforms_Plant_Container_S24Turquoise',
      (6820, 'None', 'Ecoforms Plant '): 'Ecoforms_Plant_Container_SB9Turquoise',
      (4926, 'Consumer Goods', 'Ecoforms Plant '): 'Ecoforms_Plant_Container_URN_NAT',
      (4930, 'Consumer Goods', 'Ecoforms Plant '): 'Ecoforms_Plant_Container_URN_SAN',
      (4978, 'None', 'Ecoforms Plant '): 'Ecoforms_Plant_Container_Urn_55_Avocado',
      (6268, 'None', 'Ecoforms Plant '): 'Ecoforms_Plant_Container_Urn_55_Mocha',
      (4966, 'None', 'Ecoforms Plant '): 'Ecoforms_Plant_Plate_S11Turquoise',
      (11092, 'None', 'Ecoforms Plant '): 'Ecoforms_Plant_Pot_GP9AAvocado',
      (15368, 'None', 'Ecoforms Plant '): 'Ecoforms_Plant_Pot_GP9_SAND',
      (5174, 'None', 'Ecoforms Plant '): 'Ecoforms_Plant_Saucer_S14MOCHA',
      (4974, 'None', 'Ecoforms Plant '): 'Ecoforms_Plant_Saucer_S14NATURAL',
      (6504, 'None', 'Ecoforms Plant '): 'Ecoforms_Plant_Saucer_S17MOCHA',
      (5718, 'None', 'Ecoforms Plant '): 'Ecoforms_Plant_Saucer_S20MOCHA',
      (7360, 'None', 'Ecoforms Plant '): 'Ecoforms_Plant_Saucer_SQ1HARVEST',
      (7760, 'None', 'Ecoforms Plant '): 'Ecoforms_Plant_Saucer_SQ8COR',
      (6464, 'None', 'Ecoforms Plante'): 'Ecoforms_Planter_Bowl_Cole_Hardware',
      (10104, 'None', 'Ecoforms Plante'): 'Ecoforms_Planter_Pot_GP12AAvocado',
      (7618, 'None', 'Ecoforms Plante'): 'Ecoforms_Planter_Pot_QP6Ebony',
      (4340, 'None', 'Ecoforms Plate,'): 'Ecoforms_Plate_S20Avocado',
      (8176, 'None', 'Ecoforms Pot, N'): 'Ecoforms_Pot_Nova_6_Turquoise',
      (4762, 'None', 'Ecoforms Quadra'): 'Ecoforms_Quadra_Saucer_SQ1_Avocado',
      (2358, 'None', 'Ecoforms Saucer'): 'Ecoforms_Saucer_SQ3_Turquoise',
      (52216, 'None', 'Elephant\nElepha'): 'Elephant',
      (25084, 'None', 'Embark Lunch Co'): 'Embark_Lunch_Cooler_Blue',
      (19994, 'None', 'Envision Home D'): 'Envision_Home_Dish_Drying_Mat_Red_6_x_18',
      (5578, 'Consumer Goods', 'Epson 273XL Ink'): 'Epson_273XL_Ink_Cartridge_Magenta',
      (6800, 'Consumer Goods', 'Epson DURABrite'): 'Epson_DURABrite_Ultra_786_Black_Ink_Cartridge_T786120S',
      (4032, 'Consumer Goods', 'Epson Ink Cartr'): 'Epson_Ink_Cartridge_126_Yellow',
      (5978, 'Consumer Goods', 'Epson Ink Cartr'): 'Epson_Ink_Cartridge_Black_200',
      (4320, 'Consumer Goods', 'Epson LabelWork'): 'Epson_LabelWorks_LC4WBN9_Tape_reel_labels_047_x_295_Roll_Black_on_White',
      (4454, 'Consumer Goods', 'Epson LabelWork'): 'Epson_LabelWorks_LC5WBN9_Tape_reel_labels_071_x_295_Roll_Black_on_White',
      (4704, 'Consumer Goods', 'Epson T5803 Ink'): 'Epson_T5803_Ink_Cartridge_Magenta_1pack',
      (3562, 'Consumer Goods', 'Epson UltraChro'): 'Epson_UltraChrome_T0543_Ink_Cartridge_Magenta_1pack',
      (3888, 'Consumer Goods', 'Epson UltraChro'): 'Epson_UltraChrome_T0548_Ink_Cartridge_Matte_Black_1pack',
      (14708, 'Shoe', 'F10 TRX FG'): 'F10_TRX_FG_ssscuo9tGxb',
      (19658, 'Shoe', 'F10 TRX TF'): 'F10_TRX_TF_rH7tmKCdUJq',
      (14248, 'Shoe', 'F5 TRX FG'): 'F5_TRX_FG',
      (23734, 'Toys', 'FAIRY TALE BLOC'): 'FAIRY_TALE_BLOCKS',
      (25032, 'Toys', 'FARM ANIMAL\nThe'): 'FARM_ANIMAL',
      (28266, 'Toys', 'FARM ANIMAL\nThe'): 'FARM_ANIMAL_9GyfdcPyESK',
      (23904, 'Toys', 'FIRE ENGINE'): 'FIRE_ENGINE',
      (20576, 'Toys', 'FIRE TRUCK\nThis'): 'FIRE_TRUCK',
      (18240, 'Toys', 'FISHING GAME\nGo'): 'FISHING_GAME',
      (7730, 'Toys', 'FOOD & BEVERAGE'): 'FOOD_BEVERAGE_SET',
      (6278, 'Toys', 'FRACTION FUN\nFe'): 'FRACTION_FUN_n4h4qte23QR',
      (12556, 'Toys', 'FRUIT & VEGGIE '): 'FRUIT_VEGGIE_DOMINO_GRADIENT',
      (23218, 'Toys', 'FRUIT & VEGGIE '): 'FRUIT_VEGGIE_MEMO_GRADIENT',
      (33206, 'Shoe', 'FYW ALTERNATION'): 'FYW_ALTERNATION',
      (36812, 'Shoe', 'FYW DIVISION\nFW'): 'FYW_DIVISION',
      (3272, 'Consumer Goods', 'Fem-Dophilus\nFe'): 'FemDophilus',
      (6888, 'Media Cases', 'Final Fantasy X'): 'Final_Fantasy_XIV_A_Realm_Reborn_60Day_Subscription',
      (1818, 'Consumer Goods', 'Firefly Clue Bo'): 'Firefly_Clue_Board_Game',
      (2064, 'Consumer Goods', 'Fisher-Price Ma'): 'FisherPrice_Make_A_Match_Game_Thomas_Friends',
      (12592, 'None', 'Fisher price Cl'): 'Fisher_price_Classic_Toys_Buzzy_Bee',
      (6770, 'None', 'Focus 8643 Lime'): 'Focus_8643_Lime_Squeezer_10x35x188_Enamelled_Aluminum_Light',
      (10924, 'Bottles and Cans and Cups', 'Folic Acid'): 'Folic_Acid',
      (9468, 'Consumer Goods', 'Footed Bowl San'): 'Footed_Bowl_Sand',
      (3138, 'Consumer Goods', 'Fresca Peach Ci'): 'Fresca_Peach_Citrus_Sparkling_Flavored_Soda_12_PK',
      (1428, 'Board Games', "Frozen Olaf's I"): 'Frozen_Olafs_In_Trouble_PopOMatic_Game',
      (19718, 'Board Games', "Frozen Olaf's I"): 'Frozen_Olafs_In_Trouble_PopOMatic_Game_OEu83W9T8pD',
      (1462, 'Board Games', 'Frozen Scrabble'): 'Frozen_Scrabble_Jr',
      (17626, 'Consumer Goods', 'Fruity Friends'): 'Fruity_Friends',
      (6364, 'None', 'Fujifilm instax'): 'Fujifilm_instax_SHARE_SP1_10_photos',
      (7518, 'None', 'Full Circle Hap'): 'Full_Circle_Happy_Scraps_Out_Collector_Gray',
      (31476, 'Toys', 'GARDEN SWING\nTh'): 'GARDEN_SWING',
      (25342, 'Toys', 'GEARS & PUZZLES'): 'GEARS_PUZZLES_STANDARD_gcYxhNHhKlI',
      (11730, 'Toys', 'GEOMETRIC PEG B'): 'GEOMETRIC_PEG_BOARD',
      (11274, 'Toys', 'GEOMETRIC SORTI'): 'GEOMETRIC_SORTING_BOARD',
      (8370, 'Toys', 'GEOMETRIC SORTI'): 'GEOMETRIC_SORTING_BOARD_MNi4Rbuz9vj',
      (18796, 'Shoe', 'GIRLS DECKHAND\n'): 'GIRLS_DECKHAND',
      (15920, 'Toys', 'GRANDFATHER DOL'): 'GRANDFATHER_DOLL',
      (15340, 'Toys', 'GRANDMOTHER'): 'GRANDMOTHER',
      (9102, 'Bottles and Cans and Cups', 'Germanium GE-13'): 'Germanium_GE132',
      (59036, 'Shoe', 'Ghost 6, Color:'): 'Ghost_6_Color_BlckWhtLavaSlvrCitrus_Size_80',
      (57462, 'Shoe', 'Ghost 6, Color:'): 'Ghost_6_Color_MdngtDenmPomBrtePnkSlvBlk_Size_50',
      (58118, 'Shoe', 'Ghost 6 GTX, Co'): 'Ghost_6_GTX_Color_AnthBlckSlvrFernSulphSprng_Size_80',
      (12256, 'None', 'Gigabyte GA-78L'): 'Gigabyte_GA78LMTUSB3_50_Motherboard_Micro_ATX_Socket_AM3',
      (4776, 'None', 'Gigabyte GA-970'): 'Gigabyte_GA970AUD3P_10_Motherboard_ATX_Socket_AM3',
      (25798, 'None', 'Gigabyte GA-Z97'): 'Gigabyte_GAZ97XSLI_10_motherboard_ATX_LGA1150_Socket_Z97',
      (36936, 'Shoe', 'Glycerin 11, Co'): 'Glycerin_11_Color_AqrsDrsdnBluBlkSlvShckOrng_Size_50',
      (42844, 'Shoe', 'Glycerin 11, Co'): 'Glycerin_11_Color_BrllntBluSkydvrSlvrBlckWht_Size_80',
      (22696, 'None', 'GoPro HERO3 Com'): 'GoPro_HERO3_Composite_Cable',
      (5108, 'Consumer Goods', 'Google Cardboar'): 'Google_Cardboard_Original_package',
      (23248, 'Shoe', 'Grand Prix'): 'Grand_Prix',
      (12556, 'None', 'Granimals 20 Wo'): 'Granimals_20_Wooden_ABC_Blocks_Wagon',
      (12112, 'None', 'Granimals 20 Wo'): 'Granimals_20_Wooden_ABC_Blocks_Wagon_85VdSftGsLi',
      (12336, 'None', 'Granimals 20 Wo'): 'Granimals_20_Wooden_ABC_Blocks_Wagon_g2TinmUGGHI',
      (10310, 'None', 'Great Dinos Tri'): 'Great_Dinos_Triceratops_Toy',
      (18656, 'Shoe', 'Great Jones Win'): 'Great_Jones_Wingtip',
      (22786, 'Shoe', 'Great Jones Win'): 'Great_Jones_Wingtip_j5NV8GRnitM',
      (22530, 'Shoe', 'Great Jones Win'): 'Great_Jones_Wingtip_kAqSg6EgG0I',
      (18654, 'Shoe', 'Great Jones Win'): 'Great_Jones_Wingtip_wxH3dbtlvBC',
      (8194, 'None', 'Grreat Choice D'): 'Grreat_Choice_Dog_Double_Dish_Plastic_Blue',
      (8862, 'None', 'Grreatv Choice '): 'Grreatv_Choice_Dog_Bowl_Gray_Bones_Plastic_20_fl_oz_total',
      (25030, 'Action Figures', 'Guardians of th'): 'Guardians_of_the_Galaxy_Galactic_Battlers_Rocket_Raccoon_Figure',
      (12474, 'Toys', 'HAMMER BALL\nThi'): 'HAMMER_BALL',
      (7254, 'Toys', 'HAMMER PEG\nA fo'): 'HAMMER_PEG',
      (15304, 'Toys', 'HAPPY ENGINE\nHa'): 'HAPPY_ENGINE',
      (11472, 'Toys', 'HELICOPTER\nThis'): 'HELICOPTER',
      (1652, 'None', 'HP 1800 Tablet,'): 'HP_1800_Tablet_8GB_7',
      (2586, 'Consumer Goods', 'HP Card & Invit'): 'HP_Card_Invitation_Kit',
      (1502, 'Consumer Goods', 'Hasbro Cranium '): 'Hasbro_Cranium_Performance_and_Acting_Game',
      (1150, 'Consumer Goods', "Hasbro Don't Wa"): 'Hasbro_Dont_Wake_Daddy_Board_Game',
      (8756, 'Consumer Goods', "Hasbro Don't Wa"): 'Hasbro_Dont_Wake_Daddy_Board_Game_NJnjGna4u1a',
      (1260, 'Consumer Goods', 'Hasbro Life Boa'): 'Hasbro_Life_Board_Game',
      (4016, 'Board Games', 'Hasbro Monopoly'): 'Hasbro_Monopoly_Hotels_Game',
      (4712, 'Board Games', 'Hasbro Trivial '): 'Hasbro_Trivial_Pursuit_Family_Edition_Game',
      (17748, 'None', 'Heavy-Duty Flas'): 'HeavyDuty_Flashlight',
      (13430, 'None', 'Hefty Waste Bas'): 'Hefty_Waste_Basket_Decorative_Bronze_85_liter',
      (6462, 'Consumer Goods', 'Hey You, Pikach'): 'Hey_You_Pikachu_Nintendo_64',
      (12072, 'Shoe', 'Hilary'): 'Hilary',
      (43926, 'None', 'Home Fashions W'): 'Home_Fashions_Washcloth_Linen',
      (48028, 'None', 'Home Fashions W'): 'Home_Fashions_Washcloth_Olive_Green',
      (12128, 'Bag', 'Horse Dreams Pe'): 'Horse_Dreams_Pencil_Case',
      (12696, 'Bag', 'Horses in Pink '): 'Horses_in_Pink_Pencil_Case',
      (3506, 'Media Cases', 'House of Cards:'): 'House_of_Cards_The_Complete_First_Season_4_Discs_DVD',
      (8834, 'Bottles and Cans and Cups', 'Hyaluronic Acid'): 'Hyaluronic_Acid',
      (4036, 'Headphones', 'HyperX Cloud II'): 'HyperX_Cloud_II_Headset_Gun_Metal',
      (3990, 'Headphones', 'HyperX Cloud II'): 'HyperX_Cloud_II_Headset_Red',
      (5168, 'None', 'INTERNATIONAL P'): 'INTERNATIONAL_PAPER_Willamette_4_Brown_Bag_500Count',
      (27334, 'Action Figures', 'Imaginext Castl'): 'Imaginext_Castle_Ogre',
      (18272, 'None', 'In Green Compan'): 'In_Green_Company_Surface_Saver_Ring_10_Terra_Cotta',
      (9522, 'Bottles and Cans and Cups', 'Inositol'): 'Inositol',
      (9408, 'None', 'InterDesign Ove'): 'InterDesign_Over_Door',
      (9094, 'Bottles and Cans and Cups', 'Iso-Rich Soy'): 'IsoRich_Soy',
      (15856, 'None', 'J.A. Henckels I'): 'JA_Henckels_International_Premio_Cutlery_Block_Set_14Piece',
      (6928, 'Consumer Goods', 'JBL Charge Spea'): 'JBL_Charge_Speaker_portable_wireless_wired_Green',
      (34798, 'Shoe', 'JS WINGS 2.0 BL'): 'JS_WINGS_20_BLACK_FLAG',
      (9760, 'Toys', 'JUICER SET\nHave'): 'JUICER_SET',
      (18494, 'Toys', 'JUNGLE HEIGHT\nC'): 'JUNGLE_HEIGHT',
      (30092, 'Bag', 'Jansport School'): 'Jansport_School_Backpack_Blue_Streak',
      (8406, 'Bottles and Cans and Cups', 'Jarro-Dophilus+'): 'JarroDophilusFOS_Value_Size',
      (7268, 'Consumer Goods', 'JarroSil, Activ'): 'JarroSil_Activated_Silicon',
      (6446, 'Consumer Goods', 'JarroSil, Activ'): 'JarroSil_Activated_Silicon_5exdZHIeLAp',
      (9234, 'Bottles and Cans and Cups', 'Jarrow Formulas'): 'Jarrow_Formulas_Glucosamine_Hci_Mega_1000_100_ct',
      (6406, 'Bottles and Cans and Cups', 'Jarrow Glucosam'): 'Jarrow_Glucosamine_Chondroitin_Combination_120_Caps',
      (7654, 'None', 'Jawbone UP24 Wi'): 'Jawbone_UP24_Wireless_Activity_Tracker_Pink_Coral_L',
      (5512, 'Consumer Goods', 'Just For Men Mu'): 'Just_For_Men_Mustache_Beard_Brushin_Hair_Color_Gel_Kit_Jet_Black_M60',
      (3612, 'Consumer Goods', 'Just For Men Mu'): 'Just_For_Men_Mustache_Beard_Brushin_Hair_Color_Gel_MediumDark_Brown_M40',
      (4644, 'Consumer Goods', 'Just For Men Sh'): 'Just_For_Men_ShampooIn_Haircolor_Jet_Black_60',
      (4056, 'Consumer Goods', 'Just For Men Sh'): 'Just_For_Men_ShampooIn_Haircolor_Light_Brown_25',
      (3382, 'Consumer Goods', 'Just For Men Sh'): 'Just_For_Men_Shampoo_In_Haircolor_Darkest_Brown_50',
      (4702, 'Media Cases', 'Justified: The '): 'Justified_The_Complete_Fourth_Season_3_Discs_DVD',
      (15210, 'Toys', 'KID ROOM (FURNI'): 'KID_ROOM_FURNITURE_SET_1',
      (7812, 'Toys', 'KITCHEN (FURNIT'): 'KITCHEN_FURNITURE_SET_1',
      (7274, 'Toys', 'KITCHEN SET - C'): 'KITCHEN_SET_CLASSIC_40HwCHfeG0H',
      (13200, 'Consumer Goods', 'KS Chocolate Cu'): 'KS_Chocolate_Cube_Box_Assortment_By_Neuhaus_2010_Ounces',
      (8556, 'Keyboard', 'Kanex Multi-Syn'): 'Kanex_MultiSync_Wireless_Keyboard',
      (2978, 'Media Cases', 'Kid Icarus Upri'): 'Kid_Icarus_Uprising_Nintendo_3DS_Game',
      (11596, 'None', 'Kingston DT4000'): 'Kingston_DT4000MR_G2_Management_Ready_USB_64GB',
      (8772, 'Consumer Goods', 'Kong Puppy Teet'): 'Kong_Puppy_Teething_Rubber_Small_Pink',
      (8782, 'Consumer Goods', 'Kotex U Barely '): 'Kotex_U_Barely_There_Liners_Thin_60_count',
      (29600, 'Consumer Goods', 'Kotex U Tween P'): 'Kotex_U_Tween_Pads_16_pads',
      (8746, 'None', 'Kotobuki Saucer'): 'Kotobuki_Saucer_Dragon_Fly',
      (8476, 'Consumer Goods', 'Krill Oil'): 'Krill_Oil',
      (14590, 'Toys', 'LACING SHEEP\nCh'): 'LACING_SHEEP',
      (10888, 'Toys', 'LADYBUG BEAD\nTh'): 'LADYBUG_BEAD',
      (82396, 'Legos', 'LEGO 5887 Dino '): 'LEGO_5887_Dino_Defense_HQ',
      (5138, 'Legos', 'LEGO Bricks & M'): 'LEGO_Bricks_More_Creative_Suitcase',
      (1950, 'Legos', 'LEGO City Adven'): 'LEGO_City_Advent_Calendar',
      (18654, 'Consumer Goods', 'LEGO Creationar'): 'LEGO_Creationary_Game',
      (2896, 'Consumer Goods', 'LEGO Creationar'): 'LEGO_Creationary_Game_ZJa163wlWp2',
      (51030, 'Legos', 'LEGO Duplo Buil'): 'LEGO_Duplo_Build_and_Play_Box_4629',
      (22284, 'Legos', 'LEGO Duplo Crea'): 'LEGO_Duplo_Creative_Animals_10573',
      (4726, 'Board Games', 'LEGO Fusion Set'): 'LEGO_Fusion_Set_Town_Master',
      (2554, 'Legos', 'LEGO Star Wars '): 'LEGO_Star_Wars_Advent_Calendar',
      (24908, 'Shoe', 'LEUCIPPUS ADIPU'): 'LEUCIPPUS_ADIPURE',
      (10370, 'Bottles and Cans and Cups', 'L-Tyrosine'): 'LTyrosine',
      (7738, 'Bottles and Cans and Cups', 'Lactoferrin'): 'Lactoferrin',
      (35664, 'Toys', 'Lalaloopsy Pean'): 'Lalaloopsy_Peanut_Big_Top_Tricycle',
      (27246, 'Shoe', 'Lavender Snake '): 'Lavender_Snake_Tieks_Snake_Print_Ballet_Flats',
      (1764, 'Consumer Goods', 'Leap Frog Paint'): 'Leap_Frog_Paint_Dabber_Dot_Art_5_paint_bottles',
      (1880, 'Legos', 'Lego Friends Ad'): 'Lego_Friends_Advent_Calendar',
      (36812, 'Legos', 'Lego Friends, M'): 'Lego_Friends_Mia',
      (5790, 'None', 'Lenovo Yoga 2 1'): 'Lenovo_Yoga_2_11',
      (4664, 'Consumer Goods', 'Little Big Plan'): 'Little_Big_Planet_3_Plush_Edition',
      (3054, 'Consumer Goods', 'Little Debbie C'): 'Little_Debbie_Chocolate_Cupcakes_8_ct',
      (2544, 'Consumer Goods', 'Little Debbie C'): 'Little_Debbie_Cloud_Cakes_10_ct',
      (2364, 'Consumer Goods', 'Little Debbie D'): 'Little_Debbie_Donut_Sticks_6_cake_donuts_10_oz_total',
      (5066, 'Media Cases', 'Little House on'): 'Little_House_on_the_Prairie_Season_Two_5_Discs_Includes_Digital',
      (4594, 'None', 'Logitech Ultima'): 'Logitech_Ultimate_Ears_Boom_Wireless_Speaker_Night_Black',
      (97684, 'Stuffed Toys', 'Lovable Huggabl'): 'Lovable_Huggable_Cuddly_Boutique_Teddy_Bear_Beige',
      (30450, 'Shoe', 'Lovestruck Tiek'): 'Lovestruck_Tieks_Glittery_Rose_Gold_Italian_Leather_Ballet_Flats',
      (4116, 'Media Cases', "Luigi's Mansion"): 'Luigis_Mansion_Dark_Moon_Nintendo_3DS_Game',
      (8132, 'Bottles and Cans and Cups', 'Lutein'): 'Lutein',
      (27128, 'Shoe', 'MARTIN WEDGE LA'): 'MARTIN_WEDGE_LACE_BOOT',
      (9520, 'Toys', 'MEAT SET\nThis s'): 'MEAT_SET',
      (14226, 'Toys', 'MINI EXCAVATOR\n'): 'MINI_EXCAVATOR',
      (25254, 'Toys', 'MINI FIRE ENGIN'): 'MINI_FIRE_ENGINE',
      (15596, 'Toys', 'MINI ROLLER\nFla'): 'MINI_ROLLER',
      (18824, 'Toys', 'MIRACLE POUNDIN'): 'MIRACLE_POUNDING',
      (8356, 'Bottles and Cans and Cups', 'MK-7'): 'MK7',
      (43750, 'Toys', 'MODERN DOLL FAM'): 'MODERN_DOLL_FAMILY',
      (18946, 'Toys', 'MONKEY BOWLING\n'): 'MONKEY_BOWLING',
      (2822, 'Toys', 'MOSAIC\nSay it w'): 'MOSAIC',
      (15806, 'Toys', 'MOVING MOUSE (P'): 'MOVING_MOUSE_PW_6PCSSET',
      (15014, 'Toys', 'MY MOOD MEMO\nTh'): 'MY_MOOD_MEMO',
      (4622, 'Board Games', 'Mad Gab Refresh'): 'Mad_Gab_Refresh_Card_Game',
      (6944, 'None', 'Magnifying Glas'): 'Magnifying_Glassassrt',
      (2694, 'Bottles and Cans and Cups', 'Marc Anthony Sk'): 'Marc_Anthony_Skip_Professional_Oil_of_Morocco_Conditioner_with_Argan_Oil',
      (4828, 'Consumer Goods', 'Marc Anthony St'): 'Marc_Anthony_Strictly_Curls_Curl_Envy_Perfect_Curl_Cream_6_fl_oz_bottle',
      (6076, 'Consumer Goods', 'Marc Anthony Tr'): 'Marc_Anthony_True_Professional_Oil_of_Morocco_Argan_Oil_Treatment',
      (2066, 'Bottles and Cans and Cups', 'Marc Anthony Tr'): 'Marc_Anthony_True_Professional_Strictly_Curls_Curl_Defining_Lotion',
      (2536, 'Media Cases', 'Mario & Luigi: '): 'Mario_Luigi_Dream_Team_Nintendo_3DS_Game',
      (2802, 'Media Cases', 'Mario Party 9 ['): 'Mario_Party_9_Wii_Game',
      (24970, 'None', 'Markings Desk C'): 'Markings_Desk_Caddy',
      (19340, 'None', 'Markings Letter'): 'Markings_Letter_Holder',
      (23082, 'Action Figures', 'Marvel Avengers'): 'Marvel_Avengers_Titan_Hero_Series_Doctor_Doom',
      (7782, 'Bottles and Cans and Cups', 'Mastic Gum'): 'Mastic_Gum',
      (22308, 'Shoe', 'Matte Black Tie'): 'Matte_Black_Tieks_Italian_Leather_Ballet_Flats',
      (3474, 'Consumer Goods', 'Mattel SKIP BO '): 'Mattel_SKIP_BO_Card_Game',
      (9774, 'None', 'Melissa & Doug '): 'Melissa_Doug_Cart_Turtle_Block',
      (9798, 'None', 'Melissa & Doug '): 'Melissa_Doug_Chunky_Puzzle_Vehicles',
      (33188, 'None', 'Melissa & Doug '): 'Melissa_Doug_Felt_Food_Pizza_Set',
      (3694, 'Toys', 'Melissa & Doug '): 'Melissa_Doug_Jumbo_Knob_Puzzles_Barnyard_Animals',
      (5376, 'Consumer Goods', 'Melissa & Doug '): 'Melissa_Doug_Pattern_Blocks_and_Boards',
      (13170, 'Consumer Goods', 'Melissa & Doug '): 'Melissa_Doug_Pound_and_Roll',
      (7750, 'Consumer Goods', 'Melissa & Doug '): 'Melissa_Doug_See_Spell',
      (7554, 'None', 'Melissa & Doug '): 'Melissa_Doug_Shape_Sorting_Clock',
      (14714, 'Consumer Goods', 'Melissa & Doug '): 'Melissa_Doug_Traffic_Signs_and_Vehicles',
      (21364, 'Shoe', "Men's ASV Billf"): 'Mens_ASV_Billfish_Boat_Shoe_in_Dark_Brown_Leather_zdHVHXueI3w',
      (21110, 'Shoe', "Men's ASV Billf"): 'Mens_ASV_Billfish_Boat_Shoe_in_Tan_Leather_wmUJ5PbwANc',
      (38368, 'Shoe', "Men's ASV Shock"): 'Mens_ASV_Shock_Light_Bungee_in_Light_Grey_xGCOvtLDnQJ',
      (28948, 'Shoe', "Men's Authentic"): 'Mens_Authentic_Original_Boat_Shoe_in_Navy_Leather_NHHQddDLQys',
      (19030, 'Shoe', "Men's Authentic"): 'Mens_Authentic_Original_Boat_Shoe_in_Navy_Leather_RpT4GvUXRRP',
      (19012, 'Shoe', "Men's Authentic"): 'Mens_Authentic_Original_Boat_Shoe_in_Navy_Leather_xgoEcZtRNmH',
      (21270, 'Shoe', "Men's Bahama in"): 'Mens_Bahama_in_Black_b4ADzYywRHl',
      (20600, 'Shoe', "Men's Bahama in"): 'Mens_Bahama_in_Khaki_Oyster_xU2jeqYwhQJ',
      (23030, 'Shoe', "Men's Bahama in"): 'Mens_Bahama_in_White_vSwvGMCo32f',
      (23618, 'Shoe', "Men's Billfish "): 'Mens_Billfish_3Eye_Boat_Shoe_in_Dark_Tan_wyns9HRcEuH',
      (21450, 'Shoe', "Men's Billfish "): 'Mens_Billfish_Slip_On_in_Coffee_e8bPKE9Lfgo',
      (19276, 'Shoe', "Men's Billfish "): 'Mens_Billfish_Slip_On_in_Coffee_nK6AJJAHOae',
      (19680, 'Shoe', "Men's Billfish "): 'Mens_Billfish_Slip_On_in_Tan_Beige_aaVUk0tNTv8',
      (32020, 'Shoe', "Men's Billfish "): 'Mens_Billfish_Ultra_Lite_Boat_Shoe_in_Dark_Brown_Blue_c6zDZTtRJr6',
      (24598, 'Shoe', "Men's Gold Cup "): 'Mens_Gold_Cup_ASV_2Eye_Boat_Shoe_in_Cognac_Leather',
      (28192, 'Shoe', "Men's Gold Cup "): 'Mens_Gold_Cup_ASV_Capetown_Penny_Loafer_in_Black_EjPnk3E8fCs',
      (25360, 'Shoe', "Men's Gold Cup "): 'Mens_Gold_Cup_ASV_Capetown_Penny_Loafer_in_Black_GkQBKqABeQN',
      (20090, 'Shoe', "Men's Gold Cup "): 'Mens_Gold_Cup_ASV_Dress_Casual_Venetian_in_Dark_Brown_Leather',
      (19398, 'Shoe', "Men's Largo Sli"): 'Mens_Largo_Slip_On_in_Taupe_gooyS417q4R',
      (20804, 'Shoe', "Men's Mako Cano"): 'Mens_Mako_Canoe_Moc_2Eye_Boat_Shoe_in_Coffee_9d05GG33QQQ',
      (21168, 'Shoe', "Men's Mako Cano"): 'Mens_Mako_Canoe_Moc_2Eye_Boat_Shoe_in_Coffee_K9e8FoV73uZ',
      (17868, 'Shoe', "Men's Mako Cano"): 'Mens_Mako_Canoe_Moc_2Eye_Boat_Shoe_in_OysterTaupe_otyRrfvPMiA',
      (19406, 'Shoe', "Men's R&R Moc i"): 'Mens_RR_Moc_in_Navy_Suede_vmFfijhBzL3',
      (10358, 'Shoe', "Men's Santa Cru"): 'Mens_Santa_Cruz_Thong_in_Chocolate_La1fo2mAovE',
      (11674, 'Shoe', "Men's Santa Cru"): 'Mens_Santa_Cruz_Thong_in_Chocolate_lvxYW7lek6B',
      (11918, 'Shoe', "Men's Santa Cru"): 'Mens_Santa_Cruz_Thong_in_Tan_r59C69daRPh',
      (27882, 'Shoe', "Men's Striper S"): 'Mens_Striper_Sneaker_in_White_rnp8HUli59Y',
      (29246, 'Shoe', "Men's Tremont K"): 'Mens_Tremont_Kiltie_Tassel_Loafer_in_Black_Amaretto',
      (44822, 'Shoe', "Men's Tremont K"): 'Mens_Tremont_Kiltie_Tassel_Loafer_in_Black_Amaretto_FT0I9OjSA6O',
      (34070, 'Shoe', "Men's Tremont K"): 'Mens_Tremont_Kiltie_Tassel_Loafer_in_Black_Amaretto_rCdzRZqgCnI',
      (24036, 'Shoe', "Men's Wave Driv"): 'Mens_Wave_Driver_Kiltie_Moc_in_Tan_Leather',
      (22072, 'Shoe', 'Metallic Gold T'): 'Metallic_Gold_Tieks_Italian_Leather_Ballet_Flats',
      (20080, 'Shoe', 'Metallic Pewter'): 'Metallic_Pewter_Tieks_Italian_Leather_Ballet_Flats',
      (3308, 'None', 'Mist Wipe Warme'): 'Mist_Wipe_Warmer',
      (6818, 'Toys', 'My First Animal'): 'My_First_Animal_Tower',
      (7740, 'Toys', 'My First Rollin'): 'My_First_Rolling_Lion',
      (10056, 'Toys', 'My First Wiggle'): 'My_First_Wiggle_Crocodile',
      (39168, 'Toys', 'My Little Pony '): 'My_Little_Pony_Princess_Celestia',
      (1356, 'Board Games', 'My Monopoly Boa'): 'My_Monopoly_Board_Game',
      (12446, 'Shoe', 'NAPA VALLEY NAV'): 'NAPA_VALLEY_NAVAJO_SANDAL',
      (6260, 'Consumer Goods', 'NESCAFE NESCAFE'): 'NESCAFE_NESCAFE_TC_STKS_DECAF_6_CT',
      (18260, 'Toys', 'NUTS & BOLTS\nEn'): 'NUTS_BOLTS',
      (8508, 'Bottles and Cans and Cups', 'NattoMax'): 'NattoMax',
      (12082, 'None', 'Neat Solutions '): 'Neat_Solutions_Character_Bib_2_pack',
      (3024, 'Consumer Goods', 'Nescafe 16-Coun'): 'Nescafe_16Count_Dolce_Gusto_Cappuccino_Capsules',
      (3992, 'Consumer Goods', 'Nescafe Memento'): 'Nescafe_Memento_Latte_Caramel_8_08_oz_23_g_packets_64_oz_184_g',
      (4312, 'Consumer Goods', 'Nescafe Momento'): 'Nescafe_Momento_Mocha_Specialty_Coffee_Mix_8_ct',
      (9214, 'Bottles and Cans and Cups', "Nescafe Taster'"): 'Nescafe_Tasters_Choice_Instant_Coffee_Decaf_House_Blend_Light_7_oz',
      (3320, 'Consumer Goods', 'Nestl? Crunch G'): 'Nestl_Crunch_Girl_Scouts_Cookie_Flavors_Caramel_Coconut_78_oz_box',
      (4904, 'Consumer Goods', 'Nestl? Skinny C'): 'Nestl_Skinny_Cow_Heavenly_Crisp_Candy_Bar_Chocolate_Raspberry_6_pack_462_oz_total',
      (6496, 'Consumer Goods', 'Nestle Candy 1.'): 'Nestle_Candy_19_oz_Butterfinger_Singles_116567',
      (2068, 'Consumer Goods', 'Nestle Carnatio'): 'Nestle_Carnation_Cinnamon_Coffeecake_Kit_1913OZ',
      (14756, 'Consumer Goods', 'Nestle Nesquik '): 'Nestle_Nesquik_Chocolate_Powder_Flavored_Milk_Additive_109_Oz_Canister',
      (4852, 'Consumer Goods', 'Nestle Nips Har'): 'Nestle_Nips_Hard_Candy_Peanut_Butter',
      (3938, 'Consumer Goods', 'Nestle Pure Lif'): 'Nestle_Pure_Life_Exotics_Sparkling_Water_Strawberry_Dragon_Fruit_8_count_12_fl_oz_can',
      (3240, 'Consumer Goods', 'Nestle Pure Lif'): 'Nestle_Pure_Life_Exotics_Sparkling_Water_Strawberry_Dragon_Fruit_8_count_12_fl_oz_can_aX0ygjh3bxi',
      (3828, 'Consumer Goods', 'Nestle Raisinet'): 'Nestle_Raisinets_Milk_Chocolate_35_oz_992_g',
      (4764, 'Consumer Goods', 'Nestle Skinny C'): 'Nestle_Skinny_Cow_Dreamy_Clusters_Candy_Dark_Chocolate_6_pack_1_oz_pouches',
      (38524, 'None', 'Netgear Ac1750 '): 'Netgear_Ac1750_Router_Wireless_Dual_Band_Gigabit_Router',
      (2044, 'Consumer Goods', 'Netgear N750 Wi'): 'Netgear_N750_Wireless_Dual_Band_Gigabit_Router',
      (3170, 'Consumer Goods', 'Netgear Nightha'): 'Netgear_Nighthawk_X6_AC3200_TriBand_Gigabit_Wireless_Router',
      (3312, 'Media Cases', 'New Super Mario'): 'New_Super_Mario_BrosWii_Wii_Game',
      (21724, 'Action Figures', 'Nickelodeon Tee'): 'Nickelodeon_Teenage_Mutant_Ninja_Turtles_Leonardo',
      (18996, 'Action Figures', 'Nickelodeon Tee'): 'Nickelodeon_Teenage_Mutant_Ninja_Turtles_Michelangelo',
      (21666, 'Action Figures', 'Nickelodeon Tee'): 'Nickelodeon_Teenage_Mutant_Ninja_Turtles_Raphael',
      (23990, 'Toys', 'Nickelodeon The'): 'Nickelodeon_The_Spongebob_Movie_PopAPart_Spongebob',
      (2728, 'Board Games', 'Nightmare Befor'): 'Nightmare_Before_Christmas_Collectors_Edition_Operation',
      (9566, 'Camera', 'Nikon 1 AW1 w/1'): 'Nikon_1_AW1_w11275mm_Lens_Silver',
      (6254, 'None', 'Nintendo 2DS? ?'): 'Nintendo_2DS_Crimson_Red',
      (20116, 'Toys', 'Nintendo Mario '): 'Nintendo_Mario_Action_Figure',
      (2510, 'Media Cases', 'Nintendo Wii Pa'): 'Nintendo_Wii_Party_U_with_Controller_Wii_U_Game',
      (16498, 'Toys', 'Nintendo Yoshi '): 'Nintendo_Yoshi_Action_Figure',
      (4398, 'Consumer Goods', 'Nips Hard Candy'): 'Nips_Hard_Candy_Rich_Creamy_Butter_Rum_4_oz_1133_g',
      (16052, 'None', 'Nordic Ware Ori'): 'Nordic_Ware_Original_Bundt_Pan',
      (42522, 'None', 'Now Designs Bow'): 'Now_Designs_Bowl_Akita_Black',
      (20548, 'None', 'Now Designs Dis'): 'Now_Designs_Dish_Towel_Mojave_18_x_28',
      (9702, 'None', 'Now Designs Sna'): 'Now_Designs_Snack_Bags_Bicycle_2_count',
      (8862, 'Toys', 'OVAL XYLOPHONE\n'): 'OVAL_XYLOPHONE',
      (11398, 'Toys', 'OWL SORTER\nGrea'): 'OWL_SORTER',
      (4494, 'None', 'OXO Cookie Spat'): 'OXO_Cookie_Spatula',
      (20568, 'None', 'OXO Soft Works '): 'OXO_Soft_Works_Can_Opener_SnapLock',
      (26276, 'None', 'Object\nObject'): 'Object',
      (8694, 'None', 'Object\nObject'): 'Object_REmvBDJStub',
      (14656, 'None', 'Ocedar Snap On '): 'Ocedar_Snap_On_Dust_Pan_And_Brush_1_ct',
      (6088, 'Consumer Goods', 'Office Depot (C'): 'Office_Depot_Canon_CL211XL_Remanufactured_Ink_Cartridge_TriColor',
      (3980, 'Consumer Goods', 'Office Depot (C'): 'Office_Depot_Canon_CLI36_Remanufactured_Ink_Cartridge_TriColor',
      (5926, 'Consumer Goods', 'Office Depot Ca'): 'Office_Depot_Canon_CLI_221BK_Ink_Cartridge_Black_2946B001',
      (4876, 'Consumer Goods', 'Office Depot (C'): 'Office_Depot_Canon_CLI_8CMY_Remanufactured_Ink_Cartridges_Color_Cyan_Magenta_Yellow_3_count',
      (4732, 'Consumer Goods', 'Office Depot Ca'): 'Office_Depot_Canon_CLI_8Y_Ink_Cartridge_Yellow_0623B002',
      (4276, 'Consumer Goods', 'Office Depot (C'): 'Office_Depot_Canon_CL_41_Remanufactured_Ink_Cartridge_TriColor',
      (4780, 'Consumer Goods', 'Office Depot (C'): 'Office_Depot_Canon_PG21XL_Remanufactured_Ink_Cartridge_Black',
      (4614, 'Consumer Goods', 'Office Depot (C'): 'Office_Depot_Canon_PGI22_Remanufactured_Ink_Cartridge_Black',
      (4446, 'Consumer Goods', 'Office Depot (C'): 'Office_Depot_Canon_PGI35_Remanufactured_Ink_Cartridge_Black',
      (6472, 'Consumer Goods', 'Office Depot (C'): 'Office_Depot_Canon_PGI5BK_Remanufactured_Ink_Cartridge_Black',
      (6584, 'Consumer Goods', 'Office Depot Ca'): 'Office_Depot_Canon_PG_240XL_Ink_Cartridge_Black_5206B001',
      (5694, 'Consumer Goods', 'Office Depot (D'): 'Office_Depot_Dell_Series_11_Remanufactured_Ink_Cartridge_Black',
      (3676, 'Consumer Goods', 'Office Depot (D'): 'Office_Depot_Dell_Series_11_Remanufactured_Ink_Cartridge_TriColor',
      (5394, 'Consumer Goods', 'Office Depot (D'): 'Office_Depot_Dell_Series_1_Remanufactured_Ink_Cartridge_Black',
      (5722, 'Consumer Goods', 'Office Depot (D'): 'Office_Depot_Dell_Series_1_Remanufactured_Ink_Cartridge_TriColor',
      (4108, 'Consumer Goods', 'Office Depot (D'): 'Office_Depot_Dell_Series_5_Remanufactured_Ink_Cartridge_Black',
      (4846, 'Consumer Goods', 'Office Depot De'): 'Office_Depot_Dell_Series_9_Color_Ink_Ink_Cartridge_MK991_MK993',
      (5162, 'Consumer Goods', 'Office Depot De'): 'Office_Depot_Dell_Series_9_Ink_Cartridge_Black_MK992',
      (4236, 'Consumer Goods', 'Office Depot (H'): 'Office_Depot_HP_2_Remanufactured_Ink_Cartridges_Color_Cyan_Magenta_Yellow_3_count',
      (4212, 'Consumer Goods', 'Office Depot HP'): 'Office_Depot_HP_564XL_Ink_Cartridge_Black_CN684WN',
      (4126, 'Consumer Goods', 'Office Depot (H'): 'Office_Depot_HP_564XL_Remanufactured_Ink_Cartridges_Color_Cyan_Magenta_Yellow_3_count',
      (4438, 'Consumer Goods', 'Office Depot HP'): 'Office_Depot_HP_61Tricolor_Ink_Cartridge',
      (5972, 'Consumer Goods', 'Office Depot (H'): 'Office_Depot_HP_71_Remanufactured_Ink_Cartridge_Black',
      (3634, 'Consumer Goods', 'Office Depot (H'): 'Office_Depot_HP_74XL75_Remanufactured_Ink_Cartridges_BlackTriColor_2_count',
      (5498, 'Consumer Goods', 'Office Depot (H'): 'Office_Depot_HP_75_Remanufactured_Ink_Cartridge_TriColor',
      (4314, 'Consumer Goods', 'Office Depot HP'): 'Office_Depot_HP_920XL_920_High_Yield_Black_and_Standard_CMY_Color_Ink_Cartridges',
      (5168, 'Consumer Goods', 'Office Depot HP'): 'Office_Depot_HP_932XL_Ink_Cartridge_Black_CN053A',
      (5454, 'Consumer Goods', 'Office Depot HP'): 'Office_Depot_HP_950XL_Ink_Cartridge_Black_CN045AN',
      (4640, 'Consumer Goods', 'Office Depot (H'): 'Office_Depot_HP_96_Remanufactured_Ink_Cartridge_Black',
      (4046, 'Consumer Goods', 'Office Depot (H'): 'Office_Depot_HP_98_Remanufactured_Ink_Cartridge_Black',
      (13760, 'Bag', 'Olive Kids Bird'): 'Olive_Kids_Birdie_Lunch_Box',
      (15898, 'Bag', 'Olive Kids Bird'): 'Olive_Kids_Birdie_Munch_n_Lunch',
      (27676, 'Bag', 'Olive Kids Bird'): 'Olive_Kids_Birdie_Pack_n_Snack',
      (28666, 'Bag', 'Olive Kids Bird'): 'Olive_Kids_Birdie_Sidekick_Backpack',
      (18968, 'Bag', 'Olive Kids Butt'): 'Olive_Kids_Butterfly_Garden_Munch_n_Lunch_Bag',
      (11022, 'Bag', 'Olive Kids Butt'): 'Olive_Kids_Butterfly_Garden_Pencil_Case',
      (15204, 'Bag', 'Olive Kids Dino'): 'Olive_Kids_Dinosaur_Land_Lunch_Box',
      (18202, 'Bag', 'Olive Kids Dino'): 'Olive_Kids_Dinosaur_Land_Munch_n_Lunch',
      (29314, 'Bag', 'Olive Kids Dino'): 'Olive_Kids_Dinosaur_Land_Pack_n_Snack',
      (22640, 'Bag', 'Olive Kids Dino'): 'Olive_Kids_Dinosaur_Land_Sidekick_Backpack',
      (14212, 'Bag', 'Olive Kids Game'): 'Olive_Kids_Game_On_Lunch_Box',
      (17356, 'Bag', 'Olive Kids Game'): 'Olive_Kids_Game_On_Munch_n_Lunch',
      (30086, 'Bag', 'Olive Kids Game'): 'Olive_Kids_Game_On_Pack_n_Snack',
      (27022, 'Bag', 'Olive Kids Merm'): 'Olive_Kids_Mermaids_Pack_n_Snack_Backpack',
      (12444, 'Bag', 'Olive Kids Pais'): 'Olive_Kids_Paisley_Pencil_Case',
      (12574, 'Bag', 'Olive Kids Robo'): 'Olive_Kids_Robots_Pencil_Case',
      (43896, 'Bag', 'Olive Kids Trai'): 'Olive_Kids_Trains_Planes_Trucks_Bogo_Backpack',
      (19022, 'Bag', 'Olive Kids Trai'): 'Olive_Kids_Trains_Planes_Trucks_Munch_n_Lunch_Bag',
      (11374, 'None', 'Orbit Bubblemin'): 'Orbit_Bubblemint_Mini_Bottle_6_ct',
      (9014, 'Bottles and Cans and Cups', 'Organic Whey Pr'): 'Organic_Whey_Protein_Unflavored',
      (9416, 'Bottles and Cans and Cups', 'Organic Whey Pr'): 'Organic_Whey_Protein_Vanilla',
      (31238, 'None', 'Ortho Forward F'): 'Ortho_Forward_Facing',
      (27166, 'None', 'Ortho Forward F'): 'Ortho_Forward_Facing_3Q6J2oKJD92',
      (26568, 'None', 'Ortho Forward F'): 'Ortho_Forward_Facing_CkAW6rL25xH',
      (99980, 'None', 'Ortho Forward F'): 'Ortho_Forward_Facing_QCaor9ImJ2G',
      (13184, 'Toys', 'PARENT ROOM (FU'): 'PARENT_ROOM_FURNITURE_SET_1',
      (9406, 'Toys', 'PARENT ROOM (FU'): 'PARENT_ROOM_FURNITURE_SET_1_DLKEy8H4mwK',
      (11438, 'Toys', 'PEEK-A-BOO ROLL'): 'PEEKABOO_ROLLER',
      (2388, 'Consumer Goods', 'PEPSI NEXT (CA+'): 'PEPSI_NEXT_CACRV',
      (10614, 'Toys', 'PETS & ACCESSOR'): 'PETS_ACCESSORIES',
      (33016, 'Shoe', 'PHEEHAN RUN\nFoo'): 'PHEEHAN_RUN',
      (11476, 'Toys', 'PINEAPPLE MARAC'): 'PINEAPPLE_MARACA_6_PCSSET',
      (7946, 'Toys', 'POUNDING MUSHRO'): 'POUNDING_MUSHROOMS',
      (11554, 'Toys', 'PUNCH & DROP\nGr'): 'PUNCH_DROP',
      (15420, 'Toys', 'PUNCH & DROP\nGr'): 'PUNCH_DROP_TjicLPMqLvz',
      (2158, 'Consumer Goods', 'Paint Maker\nCre'): 'Paint_Maker',
      (2950, 'Media Cases', 'Paper Mario: St'): 'Paper_Mario_Sticker_Star_Nintendo_3DS_Game',
      (4568, 'Board Games', 'Pass The Popcor'): 'Pass_The_Popcorn_Movie_Guessing_Game',
      (16336, 'Bag', 'Paul Frank Dot '): 'Paul_Frank_Dot_Lunch_Box',
      (13700, 'None', 'Pennington Elec'): 'Pennington_Electric_Pot_Cabana_4',
      (2354, 'Consumer Goods', 'Pepsi Caffeine '): 'Pepsi_Caffeine_Free_Diet_12_CT',
      (3718, 'Consumer Goods', 'Pepsi Cola, Caf'): 'Pepsi_Cola_Caffeine_Free_12_12_fl_oz_355_ml_cans_144_fl_oz_426_lt',
      (3760, 'Consumer Goods', 'Pepsi Cola, Wil'): 'Pepsi_Cola_Wild_Cherry_Diet_12_12_fl_oz_355_ml_cans_144_fl_oz_426_lt',
      (9826, 'Consumer Goods', 'Pepsi Max Cola,'): 'Pepsi_Max_Cola_Zero_Calorie_12_12_fl_oz_355_ml_cans_144_fl_oz_426_lt',
      (3144, 'Consumer Goods', 'Perricoen MD No'): 'Perricoen_MD_No_Concealer_Concealer',
      (9406, 'Consumer Goods', 'Perricone MD Ac'): 'Perricone_MD_AcylGlutathione_Deep_Crease_Serum',
      (15466, 'Consumer Goods', 'Perricone MD Ac'): 'Perricone_MD_AcylGlutathione_Eye_Lid_Serum',
      (1406, 'Consumer Goods', 'Perricone MD Be'): 'Perricone_MD_Best_of_Perricone_7Piece_Collection_MEGsO6GIsyL',
      (17356, 'Consumer Goods', 'Perricone MD Bl'): 'Perricone_MD_Blue_Plasma_Orbital',
      (2346, 'Consumer Goods', 'Perricone MD Ch'): 'Perricone_MD_Chia_Serum',
      (15706, 'Consumer Goods', 'Perricone MD Co'): 'Perricone_MD_Cold_Plasma',
      (1890, 'Consumer Goods', 'Perricone MD Co'): 'Perricone_MD_Cold_Plasma_Body',
      (2846, 'Consumer Goods', 'Perricone MD Fa'): 'Perricone_MD_Face_Finishing_Moisturizer',
      (2968, 'Consumer Goods', 'Perricone MD Fa'): 'Perricone_MD_Face_Finishing_Moisturizer_4_oz',
      (2968, 'Consumer Goods', 'Perricone MD Fi'): 'Perricone_MD_Firming_Neck_Therapy_Treatment',
      (2272, 'Consumer Goods', 'Perricone MD He'): 'Perricone_MD_Health_Weight_Management_Supplements',
      (2532, 'Consumer Goods', 'Perricone MD Hi'): 'Perricone_MD_High_Potency_Evening_Repair',
      (3436, 'Consumer Goods', 'Perricone MD Hy'): 'Perricone_MD_Hypoallergenic_Firming_Eye_Cream_05_oz',
      (1906, 'Consumer Goods', 'Perricone MD Hy'): 'Perricone_MD_Hypoallergenic_Gentle_Cleanser',
      (6770, 'Consumer Goods', 'Perricone MD Ne'): 'Perricone_MD_Neuropeptide_Facial_Conformer',
      (12250, 'Consumer Goods', 'Perricone MD Ne'): 'Perricone_MD_Neuropeptide_Firming_Moisturizer',
      (4374, 'Consumer Goods', 'Perricone MD No'): 'Perricone_MD_No_Bronzer_Bronzer',
      (3228, 'Consumer Goods', 'Perricone MD No'): 'Perricone_MD_No_Foundation_Foundation_No_1',
      (3198, 'Consumer Goods', 'Perricone MD No'): 'Perricone_MD_No_Foundation_Serum',
      (4902, 'Consumer Goods', 'Perricone MD No'): 'Perricone_MD_No_Lipstick_Lipstick',
      (3432, 'Consumer Goods', 'Perricone MD No'): 'Perricone_MD_No_Mascara_Mascara',
      (1980, 'Consumer Goods', 'Perricone MD Nu'): 'Perricone_MD_Nutritive_Cleanser',
      (1890, 'Consumer Goods', 'Perricone MD OV'): 'Perricone_MD_OVM',
      (2198, 'Consumer Goods', 'Perricone MD Om'): 'Perricone_MD_Omega_3_Supplements',
      (12342, 'Consumer Goods', 'Perricone MD Ph'): 'Perricone_MD_Photo_Plasma',
      (2294, 'Consumer Goods', 'Perricone MD Sk'): 'Perricone_MD_Skin_Clear_Supplements',
      (1690, 'Consumer Goods', 'Perricone MD Sk'): 'Perricone_MD_Skin_Total_Body_Supplements',
      (2198, 'Consumer Goods', 'Perricone MD Su'): 'Perricone_MD_Super_Berry_Powder_with_Acai_Supplements',
      (8922, 'Consumer Goods', 'Perricone MD Th'): 'Perricone_MD_The_Cold_Plasma_Face_Eyes_Duo',
      (6712, 'Consumer Goods', 'Perricone MD Th'): 'Perricone_MD_The_Crease_Cure_Duo',
      (2158, 'Consumer Goods', 'Perricone MD Th'): 'Perricone_MD_The_Metabolic_Formula_Supplements',
      (1252, 'Consumer Goods', 'Perricone MD Th'): 'Perricone_MD_The_Power_Treatments',
      (1776, 'Consumer Goods', 'Perricone MD Vi'): 'Perricone_MD_Vitamin_C_Ester_15',
      (2260, 'Consumer Goods', 'Perricone MD Vi'): 'Perricone_MD_Vitamin_C_Ester_Serum',
      (7168, 'Media Cases', 'Persona Q: Shad'): 'Persona_Q_Shadow_of_the_Labyrinth_Nintendo_3DS',
      (9120, 'Bottles and Cans and Cups', 'Pet Dophilus, p'): 'Pet_Dophilus_powder',
      (1504, 'Consumer Goods', 'Philips 60ct Wa'): 'Philips_60ct_Warm_White_LED_Smooth_Mini_String_Lights',
      (4484, 'Consumer Goods', 'Philips EcoVant'): 'Philips_EcoVantage_43_W_Light_Bulbs_Natural_Light_2_pack',
      (2180, 'Consumer Goods', 'Philips Sonicar'): 'Philips_Sonicare_Tooth_Brush_2_count',
      (3724, 'Consumer Goods', 'Phillips Caplet'): 'Phillips_Caplets_Size_24',
      (3166, 'Consumer Goods', 'Phillips Colon '): 'Phillips_Colon_Health_Probiotic_Capsule',
      (9778, 'Consumer Goods', 'Phillips Milk o'): 'Phillips_Milk_of_Magnesia_Saline_Laxative_Liquid_Original',
      (3534, 'Consumer Goods', 'Phillips Stool '): 'Phillips_Stool_Softener_Liquid_Gels_30_liquid_gels',
      (9488, 'Bottles and Cans and Cups', 'PhosphOmega'): 'PhosphOmega',
      (11852, 'Bag', 'Pinwheel Pencil'): 'Pinwheel_Pencil_Case',
      (22342, 'Action Figures', 'Playmates Indus'): 'Playmates_Industrial_CoSplinter_Teenage_Mutant_Ninja_Turtle_Action_Figure',
      (31290, 'Action Figures', 'Playmates nicke'): 'Playmates_nickelodeon_teenage_mutant_ninja_turtles_shredder',
      (9030, 'Consumer Goods', 'Poise Ultimate '): 'Poise_Ultimate_Pads_Long',
      (3964, 'Media Cases', 'Pok?mon Conques'): 'Pokmon_Conquest_Nintendo_DS_Game',
      (2328, 'Media Cases', 'Pok?mon X [Nint'): 'Pokmon_X_Nintendo_3DS_Game',
      (2680, 'Media Cases', 'Pok?mon Y [Nint'): 'Pokmon_Y_Nintendo_3DS_Game',
      (2486, 'Consumer Goods', 'Pokémon: Omega '): 'Pokémon_Omega_Ruby_Alpha_Sapphire_Dual_Pack_Nintendo_3DS',
      (7498, 'Media Cases', 'Pokémon Yellow:'): 'Pokémon_Yellow_Special_Pikachu_Edition_Nintendo_Game_Boy_Color',
      (7338, 'Consumer Goods', 'Polar Herring F'): 'Polar_Herring_Fillets_Smoked_Peppered_705_oz_total',
      (16180, 'None', 'Pony C Clamp, 1'): 'Pony_C_Clamp_1440',
      (6188, 'Consumer Goods', 'Poppin File Sor'): 'Poppin_File_Sorter_Blue',
      (6834, 'Consumer Goods', 'Poppin File Sor'): 'Poppin_File_Sorter_Pink',
      (6440, 'Consumer Goods', 'Poppin File Sor'): 'Poppin_File_Sorter_White',
      (22122, 'Shoe', 'Predator LZ TRX'): 'Predator_LZ_TRX_FG',
      (24280, 'Shoe', 'Predito LZ TRX '): 'Predito_LZ_TRX_FG_W',
      (32868, 'Car Seat', 'ProSport Harnes'): 'ProSport_Harness_to_Booster_Seat',
      (4212, 'None', 'Progressive Rub'): 'Progressive_Rubber_Spatulas_3_count',
      (18954, 'Bottles and Cans and Cups', 'Prostate Optimi'): 'Prostate_Optimizer',
      (101142, 'None', 'Provence Bath T'): 'Provence_Bath_Towel_Royal_Blue',
      (37382, 'Shoe', 'PureCadence 2, '): 'PureCadence_2_Color_HiRskRedNghtlfeSlvrBlckWht_Size_70',
      (35224, 'Shoe', 'PureCadence 2, '): 'PureCadence_2_Color_TleBluLmePnchSlvMoodIndgWh_Size_50_EEzAfcBfHHO',
      (29120, 'Shoe', 'PureConnect 2, '): 'PureConnect_2_Color_AnthrcteKnckoutPnkGrnGecko_Size_50',
      (33976, 'Shoe', 'PureConnect 2, '): 'PureConnect_2_Color_BlckBrllntBluNghtlfeAnthrct_Size_70',
      (30128, 'Shoe', 'PureConnect 2, '): 'PureConnect_2_Color_FernNightlifeSilverBlack_Size_70_5w0BYsiogeV',
      (26280, 'Shoe', 'PureFlow 2, Col'): 'PureFlow_2_Color_RylPurHibiscusBlkSlvrWht_Size_50',
      (8558, 'Bottles and Cans and Cups', 'Q-Absorb Co-Q10'): 'QAbsorb_CoQ10',
      (9584, 'Bottles and Cans and Cups', 'Q-Absorb Co-Q10'): 'QAbsorb_CoQ10_53iUqjWjW3O',
      (10176, 'Bottles and Cans and Cups', 'QH-Pomegranate'): 'QHPomegranate',
      (7850, 'Bottles and Cans and Cups', 'Quercetin 500'): 'Quercetin_500',
      (25130, 'Shoe', 'REEF BANTU\nReef'): 'REEF_BANTU',
      (22626, 'Shoe', 'REEF BRAIDED CU'): 'REEF_BRAIDED_CUSHION',
      (26068, 'Shoe', 'REEF ZENFUN\nGir'): 'REEF_ZENFUN',
      (22066, 'Toys', 'RESCUE CREW\nPla'): 'RESCUE_CREW',
      (39616, 'None', 'R.J Rabbit East'): 'RJ_Rabbit_Easter_Basket_Blue',
      (22310, 'Toys', 'ROAD CONSTRUCTI'): 'ROAD_CONSTRUCTION_SET',
      (70952, 'None', 'Racoon\nRacoon'): 'Racoon',
      (65146, 'Shoe', 'Ravenna 4, Colo'): 'Ravenna_4_Color_WhtOlyBluBlkShkOrngSlvRdO_Size_70',
      (12754, 'Shoe', 'Rayna Bootie.WP'): 'Rayna_BootieWP',
      (33292, 'Mouse', 'Razer Abyssus -'): 'Razer_Abyssus_Ambidextrous_Gaming_Mouse',
      (26614, 'Keyboard', 'Razer BlackWido'): 'Razer_BlackWidow_Stealth_2014_Keyboard_07VFzIVabgh',
      (24792, 'Keyboard', 'Razer BlackWido'): 'Razer_BlackWidow_Ultimate_2014_Mechanical_Gaming_Keyboard',
      (25702, 'Keyboard', 'Razer Blackwido'): 'Razer_Blackwidow_Tournament_Edition_Keyboard',
      (3868, 'None', 'Razer Goliathus'): 'Razer_Goliathus_Control_Edition_Small_Soft_Gaming_Mouse_Mat',
      (8206, 'Headphones', 'Razer Kraken 7.'): 'Razer_Kraken_71_Chroma_headset_Full_size_Black',
      (29666, 'Headphones', 'Razer Kraken Pr'): 'Razer_Kraken_Pro_headset_Full_size_Black',
      (19384, 'Mouse', 'Razer Naga - MM'): 'Razer_Naga_MMO_Gaming_Mouse',
      (31596, 'Mouse', 'Razer Taipan Bl'): 'Razer_Taipan_Black_Ambidextrous_Gaming_Mouse',
      (23000, 'Mouse', 'Razer Taipan Wh'): 'Razer_Taipan_White_Ambidextrous_Gaming_Mouse',
      (1744, 'Consumer Goods', 'Ready-to-Use Ro'): 'ReadytoUse_Rolled_Fondant_Pure_White_24_oz_box',
      (37872, 'Shoe', 'Real Deal'): 'Real_Deal_1nIwCHX1MTh',
      (6834, 'None', 'Red/Black Ninte'): 'RedBlack_Nintendo_3DSXL',
      (30270, 'Shoe', 'Reebok ALLYLYNN'): 'Reebok_ALLYLYNN',
      (32522, 'Shoe', 'Reebok BREAKPOI'): 'Reebok_BREAKPOINT_LO_2V',
      (25834, 'Shoe', 'Reebok BREAKPOI'): 'Reebok_BREAKPOINT_MID',
      (20090, 'Shoe', 'Reebok CLASSIC '): 'Reebok_CLASSIC_JOGGER',
      (24810, 'Shoe', 'Reebok CLASSIC '): 'Reebok_CLASSIC_LEGACY_II',
      (31018, 'Shoe', 'Reebok CL DIBEL'): 'Reebok_CL_DIBELLO_II',
      (25680, 'Shoe', 'Reebok CL LTHR '): 'Reebok_CL_LTHR_R12',
      (24400, 'Shoe', 'Reebok CL RAYEN'): 'Reebok_CL_RAYEN',
      (14902, 'Shoe', 'Reebok COMFORT '): 'Reebok_COMFORT_REEFRESH_FLIP',
      (22114, 'Shoe', 'Reebok DMX MAX '): 'Reebok_DMX_MAX_MANIA_WD_D',
      (27920, 'Shoe', 'Reebok DMX MAX '): 'Reebok_DMX_MAX_PLUS_ATHLETIC',
      (27446, 'Shoe', 'Reebok DMX MAX '): 'Reebok_DMX_MAX_PLUS_RAINWALKER',
      (17784, 'Shoe', 'Reebok EASYTONE'): 'Reebok_EASYTONE_CL_LEATHER',
      (30106, 'Shoe', 'Reebok F/S HI I'): 'Reebok_FS_HI_INT_R12',
      (25484, 'Shoe', 'Reebok F/S HI M'): 'Reebok_FS_HI_MINI',
      (27552, 'Shoe', 'Reebok FUELTRAI'): 'Reebok_FUELTRAIN',
      (24766, 'Shoe', 'Reebok GL 6000'): 'Reebok_GL_6000',
      (28674, 'Shoe', 'Reebok HIMARA L'): 'Reebok_HIMARA_LTR',
      (40952, 'Shoe', 'Reebok JR ZIG C'): 'Reebok_JR_ZIG_COOPERSTOWN_MR',
      (23646, 'Shoe', 'Reebok KAMIKAZE'): 'Reebok_KAMIKAZE_II_MID',
      (49318, 'Shoe', 'Reebok PUMP OMN'): 'Reebok_PUMP_OMNI_LITE_HLS',
      (26914, 'Shoe', 'Reebok REALFLEX'): 'Reebok_REALFLEX_SELECT',
      (22258, 'Shoe', 'Reebok REESCULP'): 'Reebok_REESCULPT_TRAINER_II',
      (19990, 'Shoe', 'Reebok RETRO RU'): 'Reebok_RETRO_RUSH_2V',
      (32096, 'Shoe', 'Reebok R CROSSF'): 'Reebok_R_CROSSFIT_OLY_UFORM',
      (34902, 'Shoe', 'Reebok R DANCE '): 'Reebok_R_DANCE_FLASH',
      (28546, 'Shoe', 'Reebok SH COURT'): 'Reebok_SH_COURT_MID_II',
      (19432, 'Shoe', 'Reebok SH NEWPO'): 'Reebok_SH_NEWPORT_LOW',
      (24708, 'Shoe', 'Reebok SH PRIME'): 'Reebok_SH_PRIME_COURT_LOW',
      (33582, 'Shoe', 'Reebok SH PRIME'): 'Reebok_SH_PRIME_COURT_MID',
      (22734, 'Shoe', 'Reebok SL FLIP '): 'Reebok_SL_FLIP_UPDATE',
      (55096, 'Shoe', 'Reebok SMOOTHFL'): 'Reebok_SMOOTHFLEX_CUSHRUN_20',
      (32926, 'Shoe', 'Reebok SOMERSET'): 'Reebok_SOMERSET_RUN',
      (33052, 'Shoe', 'Reebok STUDIO B'): 'Reebok_STUDIO_BEAT_LOW_V',
      (14528, 'Shoe', 'Reebok TRIPLE B'): 'Reebok_TRIPLE_BREAK_LITE',
      (28818, 'Shoe', 'Reebok TURBO RC'): 'Reebok_TURBO_RC',
      (33366, 'Shoe', 'Reebok ULTIMATI'): 'Reebok_ULTIMATIC_2V',
      (39860, 'Shoe', 'Reebok VERSA TR'): 'Reebok_VERSA_TRAIN',
      (40876, 'Shoe', 'Reebok ZIGCOOPE'): 'Reebok_ZIGCOOPERSTOWN_QUAG',
      (37146, 'Shoe', 'Reebok ZIGLITE '): 'Reebok_ZIGLITE_RUSH',
      (22490, 'Shoe', 'Reebok ZIGLITE '): 'Reebok_ZIGLITE_RUSH_AC',
      (23022, 'Shoe', 'Reebok ZIGSTORM'): 'Reebok_ZIGSTORM',
      (29884, 'Shoe', 'Reebok ZIGTECH '): 'Reebok_ZIGTECH_SHARK_MAYHEM360',
      (13916, 'Shoe', 'Reef Star Cushi'): 'Reef_Star_Cushion_Flipflops_Size_8_Black',
      (15806, 'None', 'Remington 1 1/2'): 'Remington_1_12_inch_Hair_Straightener',
      (49084, 'None', 'Remington T/Stu'): 'Remington_TStudio_Hair_Dryer',
      (20762, 'Consumer Goods', 'Remington TStud'): 'Remington_TStudio_Silk_Ceramic_Hair_Straightener_2_Inch_Floating_Plates',
      (16554, 'None', 'Retail Leadersh'): 'Retail_Leadership_Summit',
      (15774, 'None', 'Retail Leadersh'): 'Retail_Leadership_Summit_eCT3zqHYIkX',
      (53672, 'None', 'Retail Leadersh'): 'Retail_Leadership_Summit_tQFCizMt6g0',
      (14716, 'None', 'Rexy Glove Heav'): 'Rexy_Glove_Heavy_Duty_Gloves_Medium',
      (42558, 'None', 'Rexy Glove Heav'): 'Rexy_Glove_Heavy_Duty_Large',
      (20168, 'Shoe', 'Romantic Blush '): 'Romantic_Blush_Tieks_Metallic_Italian_Leather_Ballet_Flats',
      (5112, 'None', 'Room Essentials'): 'Room_Essentials_Bowl_Turquiose',
      (27562, 'None', 'Room Essentials'): 'Room_Essentials_Dish_Drainer_Collapsible_White',
      (11496, 'None', 'Room Essentials'): 'Room_Essentials_Fabric_Cube_Lavender',
      (55644, 'None', 'Room Essentials'): 'Room_Essentials_Kitchen_Towels_16_x_26_2_count',
      (8454, 'None', 'Room Essentials'): 'Room_Essentials_Mug_White_Yellow',
      (2608, 'None', 'Room Essentials'): 'Room_Essentials_Salad_Plate_Turquoise',
      (43880, 'Shoe', 'Rose Garden Tie'): 'Rose_Garden_Tieks_Leather_Ballet_Flats_with_Floral_Rosettes',
      (40146, 'None', 'Rubbermaid Larg'): 'Rubbermaid_Large_Drainer',
      (28556, 'Shoe', 'SAMBA HEMP'): 'SAMBA_HEMP',
      (25698, 'Shoe', 'SAMOA'): 'SAMOA',
      (4602, 'Consumer Goods', 'SAM-e 200\nJarro'): 'SAMe_200',
      (6318, 'Consumer Goods', 'SAM-e 200\nJarro'): 'SAMe_200_KX7ZmOw47co',
      (11956, 'Toys', 'SANDWICH MEAL\nT'): 'SANDWICH_MEAL',
      (17102, 'None', 'SAPPHIRE R7 260'): 'SAPPHIRE_R7_260X_OC',
      (9300, 'Toys', 'SCHOOL BUS'): 'SCHOOL_BUS',
      (5048, 'Toys', 'SHAPE MATCHING\n'): 'SHAPE_MATCHING',
      (6078, 'Toys', 'SHAPE MATCHING\n'): 'SHAPE_MATCHING_NxacpAY9jDt',
      (4902, 'Toys', 'SHAPE SORTER\nLe'): 'SHAPE_SORTER',
      (14588, 'Toys', 'SIT N WALK PUPP'): 'SIT_N_WALK_PUPPY',
      (24796, 'Shoe', 'SLACK CRUISER\nF'): 'SLACK_CRUISER',
      (8564, 'Toys', 'SNAIL MEASURING'): 'SNAIL_MEASURING_TAPE',
      (16932, 'Toys', 'SORTING BUS\nThe'): 'SORTING_BUS',
      (15666, 'Toys', 'SORTING TRAIN\nS'): 'SORTING_TRAIN',
      (17886, 'Toys', 'SPEED BOAT\nImag'): 'SPEED_BOAT',
      (14018, 'Toys', 'STACKING BEAR\nB'): 'STACKING_BEAR',
      (11064, 'Toys', 'STACKING BEAR\nB'): 'STACKING_BEAR_V04KKgGBn2A',
      (14854, 'Toys', 'STACKING RING\nT'): 'STACKING_RING',
      (11304, 'Toys', 'STEAK SET\nInclu'): 'STEAK_SET',
      (46090, 'Shoe', 'SUPERSTAR CLR'): 'SUPERSTAR_CLR',
      (6804, 'Bottles and Cans and Cups', 'Saccharomyces B'): 'Saccharomyces_Boulardii_MOS_Value_Size',
      (24064, 'Shoe', 'Samoa (one-piec'): 'Samoa_onepiece',
      (2066, 'Consumer Goods', 'Samsung CLT-C40'): 'Samsung_CLTC406S_Toner_Cartridge_Cyan_1pack',
      (22024, 'Shoe', 'Santa Cruz Mens'): 'Santa_Cruz_Mens',
      (21024, 'Shoe', 'Santa Cruz Mens'): 'Santa_Cruz_Mens_G7kQXK7cIky',
      (22054, 'Shoe', 'Santa Cruz Mens'): 'Santa_Cruz_Mens_YmsMDkFf11Z',
      (21310, 'Shoe', 'Santa Cruz Mens'): 'Santa_Cruz_Mens_umxTczr1Ygg',
      (29050, 'Shoe', 'Santa Cruz Mens'): 'Santa_Cruz_Mens_vnbiTDDt5xH',
      (13368, 'Consumer Goods', 'Sapota Threshol'): 'Sapota_Threshold_4_Ceramic_Round_Planter_Red',
      (12254, 'Toys', 'Schleich Africa'): 'Schleich_African_Black_Rhino',
      (12258, 'Toys', 'Schleich Allosa'): 'Schleich_Allosaurus',
      (18510, 'Toys', 'Schleich Bald E'): 'Schleich_Bald_Eagle',
      (13512, 'Toys', 'Schleich Herefo'): 'Schleich_Hereford_Bull',
      (15058, 'Action Figures', 'Schleich Lion A'): 'Schleich_Lion_Action_Figure',
      (16624, 'Toys', 'Schleich S Baya'): 'Schleich_S_Bayala_Unicorn_70432',
      (14430, 'Action Figures', 'Schleich Spinos'): 'Schleich_Spinosaurus_Action_Figure',
      (19690, 'Toys', 'Schleich Theriz'): 'Schleich_Therizinosaurus_ln9cruulPqc',
      (10900, 'Consumer Goods', 'Sea to Summit X'): 'Sea_to_Summit_Xl_Bowl',
      (3272, 'None', 'Seagate 1TB Bac'): 'Seagate_1TB_Backup_Plus_portable_drive_Blue',
      (3026, 'None', 'Seagate 1TB Bac'): 'Seagate_1TB_Backup_Plus_portable_drive_Silver',
      (2888, 'None', 'Seagate 1TB Bac'): 'Seagate_1TB_Backup_Plus_portable_drive_for_Mac',
      (20664, 'None', 'Seagate 1TB Wir'): 'Seagate_1TB_Wireless_Plus_mobile_device_storage',
      (17842, 'None', 'Seagate 3TB Cen'): 'Seagate_3TB_Central_shared_storage',
      (35114, 'None', 'Seagate Archive'): 'Seagate_Archive_HDD_8_TB_Internal_hard_drive_SATA_6Gbs_35_ST8000AS0002',
      (21654, 'None', 'Shark\nShark'): 'Shark',
      (29228, 'Consumer Goods', "Shaxon 100' Mol"): 'Shaxon_100_Molded_Category_6_RJ45RJ45_Shielded_Patch_Cord_White',
      (4122, 'Consumer Goods', 'Shurtape 30 Day'): 'Shurtape_30_Day_Removal_UV_Delct_15',
      (3298, 'Consumer Goods', 'Shurtape Gaffer'): 'Shurtape_Gaffers_Tape_Silver_2_x_60_yd',
      (4724, 'Consumer Goods', 'Shurtape Tape, '): 'Shurtape_Tape_Purple_CP28',
      (21468, 'Shoe', 'Sienna Brown Cr'): 'Sienna_Brown_Croc_Tieks_Patent_Leather_Crocodile_Print_Ballet_Flats',
      (20590, 'Media Cases', 'Simon Swipe Gam'): 'Simon_Swipe_Game',
      (3698, 'Consumer Goods', 'Sleep Optimizer'): 'Sleep_Optimizer',
      (89522, 'None', 'Smith & Hawken '): 'Smith_Hawken_Woven_BasketTray_Organizer_with_3_Compartments_95_x_9_x_13',
      (16406, 'Consumer Goods', 'Snack Catcher S'): 'Snack_Catcher_Snack_Dispenser',
      (1792, 'None', 'Sonicare 2 Seri'): 'Sonicare_2_Series_Toothbrush_Plaque_Control',
      (38052, 'None', 'Sonny School Bu'): 'Sonny_School_Bus',
      (2330, 'Consumer Goods', 'Sony Acid Music'): 'Sony_Acid_Music_Studio',
      (7502, 'Consumer Goods', 'Sony Downloadab'): 'Sony_Downloadable_Loops',
      (53518, 'Stuffed Toys', 'Sootheze Cold T'): 'Sootheze_Cold_Therapy_Elephant',
      (36482, 'Stuffed Toys', 'Sootheze Toasty'): 'Sootheze_Toasty_Orca',
      (1610, 'Board Games', 'Sorry Sliders B'): 'Sorry_Sliders_Board_Game',
      (6602, 'Consumer Goods', 'Spectrum Wall M'): 'Spectrum_Wall_Mount',
      (14890, 'Shoe', 'Sperry Top-Side'): 'Sperry_TopSider_pSUFPWQXPp3',
      (12712, 'Shoe', 'Sperry Top-Side'): 'Sperry_TopSider_tNB9t6YBUf3',
      (16320, 'Action Figures', 'Spider-Man Tita'): 'SpiderMan_Titan_Hero_12Inch_Action_Figure_5Hnn4mtkFsP',
      (17460, 'Action Figures', 'Spider-Man Tita'): 'SpiderMan_Titan_Hero_12Inch_Action_Figure_oo1qph4wwiW',
      (12630, 'None', 'Spritz Easter B'): 'Spritz_Easter_Basket_Plastic_Teal',
      (67434, 'None', 'Squirrel\nSquirr'): 'Squirrel',
      (15612, 'Consumer Goods', 'Squirt & Strain'): 'Squirt_Strain_Fruit_Basket',
      (13838, 'Consumer Goods', "Squirtin' Barny"): 'Squirtin_Barnyard_Friends_4pk',
      (6710, 'Consumer Goods', 'Star Wars Rogue'): 'Star_Wars_Rogue_Squadron_Nintendo_64',
      (29670, 'Shoe', 'Starstruck Tiek'): 'Starstruck_Tieks_Glittery_Gold_Italian_Leather_Ballet_Flats',
      (15520, 'None', 'Sterilite Caddy'): 'Sterilite_Caddy_Blue_Sky_17_58_x_12_58_x_9_14',
      (8820, 'None', 'Super Mario 3D '): 'Super_Mario_3D_World_Deluxe_Set',
      (7322, 'None', 'Super Mario 3D '): 'Super_Mario_3D_World_Deluxe_Set_yThuvW9vZed',
      (2610, 'Media Cases', 'Super Mario 3D '): 'Super_Mario_3D_World_Wii_U_Game',
      (10666, 'Media Cases', 'Super Mario Kar'): 'Super_Mario_Kart_Super_Nintendo_Entertainment_System',
      (52036, 'Legos', 'Superman?:  Bat'): 'Superman_Battle_of_Smallville',
      (9076, 'Board Games', 'Supernatural Ou'): 'Supernatural_Ouija_Board_Game',
      (18076, 'None', 'Sushi Mat'): 'Sushi_Mat',
      (4312, 'Consumer Goods', 'Swiss Miss Hot '): 'Swiss_Miss_Hot_Cocoa_KCups_Milk_Chocolate_12_count',
      (7710, 'Toys', 'TABLEWARE SET\nT'): 'TABLEWARE_SET',
      (8338, 'Toys', 'TABLEWARE SET\nS'): 'TABLEWARE_SET_5CHkPjjxVpp',
      (26154, 'Toys', 'TABLEWARE SET\nT'): 'TABLEWARE_SET_5ww1UFLuCJG',
      (14476, 'Toys', 'TEA SET\nHave te'): 'TEA_SET',
      (39346, 'Shoe', 'TERREX FAST R\nF'): 'TERREX_FAST_R',
      (36824, 'Shoe', 'TERREX FAST X G'): 'TERREX_FAST_X_GTX',
      (24358, 'Toys', 'TOOL BELT\nBuild'): 'TOOL_BELT',
      (36802, 'Shoe', 'TOP TEN HI\nFW-H'): 'TOP_TEN_HI',
      (36504, 'Shoe', 'TOP TEN HI\nFW-H'): 'TOP_TEN_HI_60KlbRbdoJA',
      (12690, 'Toys', 'TOWER TUMBLING\n'): 'TOWER_TUMBLING',
      (36728, 'Shoe', 'TROCHILUS BOOST'): 'TROCHILUS_BOOST',
      (14232, 'Toys', 'TURBOPROP AIRPL'): 'TURBOPROP_AIRPLANE_WITH_PILOT',
      (4314, 'Toys', 'TWISTED PUZZLE\n'): 'TWISTED_PUZZLE',
      (3470, 'Toys', 'TWISTED PUZZLE\n'): 'TWISTED_PUZZLE_twb4AyFtu8Q',
      (6106, 'Toys', 'TWIST & SHAPE\nF'): 'TWIST_SHAPE',
      (40238, 'Shoe', 'T-ZX Runner'): 'TZX_Runner',
      (13616, 'None', 'Tag Dishtowel, '): 'Tag_Dishtowel_18_x_26',
      (20288, 'None', 'Tag Dishtowel, '): 'Tag_Dishtowel_Basket_Weave_Red_18_x_26',
      (19262, 'None', 'Tag Dishtowel, '): 'Tag_Dishtowel_Dobby_Stripe_Blue_18_x_26',
      (20896, 'None', 'Tag Dishtowel, '): 'Tag_Dishtowel_Green',
      (20908, 'None', 'Tag Dishtowel, '): 'Tag_Dishtowel_Waffle_Gray_Checks_18_x_26',
      (85812, 'Consumer Goods', 'Target Basket, '): 'Target_Basket_Medium',
      (28442, 'Action Figures', 'Teenage Mutant '): 'Teenage_Mutant_Ninja_Turtles_Rahzar_Action_Figure',
      (7920, 'Consumer Goods', 'Tena Pads, Heav'): 'Tena_Pads_Heavy_Long_42_pads',
      (3192, 'Board Games', 'Tetris Link Gam'): 'Tetris_Link_Game',
      (5942, 'Consumer Goods', 'The Coffee Bean'): 'The_Coffee_Bean_Tea_Leaf_KCup_Packs_Jasmine_Green_Tea_16_count',
      (17934, 'Consumer Goods', 'The Scooper Hoo'): 'The_Scooper_Hooper',
      (9372, 'Bottles and Cans and Cups', 'Theanine'): 'Theanine',
      (26776, 'Toys', 'Thomas & Friend'): 'Thomas_Friends_Woodan_Railway_Henry',
      (6272, 'Toys', 'Thomas & Friend'): 'Thomas_Friends_Wooden_Railway_Ascending_Track_Riser_Pack',
      (7074, 'Toys', 'Thomas & Friend'): 'Thomas_Friends_Wooden_Railway_Deluxe_Track_Accessory_Pack',
      (23052, 'Consumer Goods', 'Thomas & Friend'): 'Thomas_Friends_Wooden_Railway_Porter_5JzRhMm3a9o',
      (32342, 'Toys', 'Thomas & Friend'): 'Thomas_Friends_Wooden_Railway_Talking_Thomas_z7yi7UFHJRj',
      (11820, 'None', 'Threshold Bambo'): 'Threshold_Bamboo_Ceramic_Soap_Dish',
      (66390, 'None', 'Threshold Baske'): 'Threshold_Basket_Natural_Finish_Fabric_Liner_Small',
      (14030, 'Consumer Goods', 'Threshold Bead '): 'Threshold_Bead_Cereal_Bowl_White',
      (3464, 'None', 'Threshold Bistr'): 'Threshold_Bistro_Ceramic_Dinner_Plate_Ruby_Ring',
      (4088, 'None', 'Threshold Dinne'): 'Threshold_Dinner_Plate_Square_Rim_White_Porcelain',
      (23016, 'None', 'Threshold Hand '): 'Threshold_Hand_Towel_Blue_Medallion_16_x_27',
      (66084, 'None', 'Threshold Perfo'): 'Threshold_Performance_Bath_Sheet_Sandoval_Blue_33_x_63',
      (16644, 'None', 'Threshold Porce'): 'Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White',
      (9344, 'None', 'Threshold Porce'): 'Threshold_Porcelain_Pitcher_White',
      (5946, 'None', 'Threshold Porce'): 'Threshold_Porcelain_Serving_Bowl_Coupe_White',
      (8810, 'Bottles and Cans and Cups', 'Threshold Porce'): 'Threshold_Porcelain_Spoon_Rest_White',
      (8324, 'None', 'Threshold Porce'): 'Threshold_Porcelain_Teapot_White',
      (6894, 'None', 'Threshold Ramek'): 'Threshold_Ramekin_White_Porcelain',
      (3794, 'None', 'Threshold Salad'): 'Threshold_Salad_Plate_Square_Rim_Porcelain',
      (43716, 'None', 'Threshold Textu'): 'Threshold_Textured_Damask_Bath_Towel_Pink',
      (2740, 'None', 'Threshold Tray,'): 'Threshold_Tray_Rectangle_Porcelain',
      (21096, 'Shoe', 'Tiek Blue Paten'): 'Tiek_Blue_Patent_Tieks_Italian_Leather_Ballet_Flats',
      (22612, 'Shoe', 'Tieks Ballet Fl'): 'Tieks_Ballet_Flats_Diamond_White_Croc',
      (20538, 'Shoe', 'Tieks Ballet Fl'): 'Tieks_Ballet_Flats_Electric_Snake',
      (24314, 'Shoe', "Timberland Men'"): 'Timberland_Mens_Classic_2Eye_Boat_Shoe',
      (27704, 'Shoe', "Timberland Men'"): 'Timberland_Mens_Earthkeepers_Casco_Bay_Canvas_Oxford',
      (29278, 'Shoe', "Timberland Men'"): 'Timberland_Mens_Earthkeepers_Casco_Bay_Canvas_SlipOn',
      (32620, 'Shoe', "Timberland Men'"): 'Timberland_Mens_Earthkeepers_Casco_Bay_Suede_1Eye',
      (19778, 'Shoe', "Timberland Men'"): 'Timberland_Mens_Earthkeepers_Heritage_2Eye_Boat_Shoe',
      (16976, 'Shoe', "Timberland Men'"): 'Timberland_Mens_Earthkeepers_Newmarket_6Inch_Cupsole_Boot',
      (18420, 'Shoe', "Timberland Men'"): 'Timberland_Mens_Earthkeepers_Stormbuck_Chukka',
      (17872, 'Shoe', "Timberland Men'"): 'Timberland_Mens_Earthkeepers_Stormbuck_Lite_Plain_Toe_Oxford',
      (28750, 'Shoe', "Timberland Men'"): 'Timberland_Mens_Earthkeepers_Stormbuck_Plain_Toe_Oxford',
      (28578, 'Shoe', 'Timberland Wome'): 'Timberland_Womens_Classic_Amherst_2Eye_Boat_Shoe',
      (15342, 'Shoe', 'Timberland Wome'): 'Timberland_Womens_Earthkeepers_Classic_Unlined_Boat_Shoe',
      (24054, 'Shoe', 'Timberland Wome'): 'Timberland_Womens_Waterproof_Nellie_Chukka_Double',
      (10930, 'None', 'Top Paw Dog Bow'): 'Top_Paw_Dog_Bow_Bone_Ceramic_13_fl_oz_total',
      (4612, 'None', 'Top Paw Dog Bow'): 'Top_Paw_Dog_Bowl_Blue_Paw_Bone_Ceramic_25_fl_oz_total',
      (34534, 'Shoe', 'Tory Burch Kait'): 'Tory_Burch_Kaitlin_Ballet_Mestico_in_BlackGold',
      (7112, 'Shoe', 'Tory Burch Kier'): 'Tory_Burch_Kiernan_Riding_Boot',
      (23850, 'Shoe', 'Tory Burch Reva'): 'Tory_Burch_Reva_Metal_Logo_Litus_Snake_Print_in_dark_BranchGold',
      (9660, 'Shoe', 'Tory Burch Sabe'): 'Tory_Burch_Sabe_65mm_Bootie_Split_Suede_in_Caramel',
      (8852, 'None', 'Toys R Us Treat'): 'Toys_R_Us_Treat_Dispenser_Smart_Puzzle_Foobler',
      (15672, 'Consumer Goods', "Toysmith Wind'e"): 'Toysmith_Windem_Up_Flippin_Animals_Dog',
      (66112, 'Toys', 'Transformers Ag'): 'Transformers_Age_of_Extinction_Mega_1Step_Bumblebee_Figure',
      (73004, 'Toys', 'Transformers Ag'): 'Transformers_Age_of_Extinction_Stomp_and_Chomp_Grimlock_Figure',
      (9486, 'None', 'Travel Mate P s'): 'Travel_Mate_P_series_Notebook',
      (3214, 'Consumer Goods', 'Travel Smart Ne'): 'Travel_Smart_Neck_Rest_Inflatable',
      (25054, 'None', 'Tri-Star Produc'): 'TriStar_Products_PPC_Power_Pressure_Cooker_XL_in_Black',
      (2850, 'Consumer Goods', 'Tune Belt Sport'): 'Tune_Belt_Sport_Armband_For_Samsung_Galaxy_S3',
      (12770, 'Bottles and Cans and Cups', 'Twinlab 100% Wh'): 'Twinlab_100_Whey_Protein_Fuel_Chocolate',
      (12000, 'Bottles and Cans and Cups', 'Twinlab 100% Wh'): 'Twinlab_100_Whey_Protein_Fuel_Cookies_and_Cream',
      (10072, 'Bottles and Cans and Cups', 'Twinlab 100% Wh'): 'Twinlab_100_Whey_Protein_Fuel_Vanilla',
      (13250, 'Bottles and Cans and Cups', 'Twinlab Nitric '): 'Twinlab_Nitric_Fuel',
      (16960, 'Bottles and Cans and Cups', 'Twinlab Premium'): 'Twinlab_Premium_Creatine_Fuel_Powder',
      (27362, 'Shoe', 'UGG Bailey Bow '): 'UGG_Bailey_Bow_Womens_Clogs_Black_7',
      (19986, 'Shoe', 'UGG Bailey Butt'): 'UGG_Bailey_Button_Triplet_Womens_Boots_Black_7',
      (22838, 'Shoe', 'UGG Bailey Butt'): 'UGG_Bailey_Button_Womens_Boots_Black_7',
      (42030, 'Shoe', 'UGG Cambridge W'): 'UGG_Cambridge_Womens_Black_7',
      (17234, 'Shoe', 'UGG Classic Tal'): 'UGG_Classic_Tall_Womens_Boots_Chestnut_7',
      (20432, 'Shoe', 'UGG Classic Tal'): 'UGG_Classic_Tall_Womens_Boots_Grey_7',
      (25212, 'Shoe', "UGG Jena Women'"): 'UGG_Jena_Womens_Java_7',
      (16422, 'Bag', 'U.S. Army Stash'): 'US_Army_Stash_Lunch_Bag',
      (3642, 'Consumer Goods', 'U By Kotex Clea'): 'U_By_Kotex_Cleanwear_Heavy_Flow_Pads_32_Ct',
      (6972, 'Consumer Goods', 'U By Kotex Slee'): 'U_By_Kotex_Sleek_Regular_Unscented_Tampons_36_Ct_Box',
      (37794, 'Consumer Goods', 'Ubisoft RockSmi'): 'Ubisoft_RockSmith_Real_Tone_Cable_Xbox_360',
      (5914, 'Consumer Goods', 'Ultra Jarro-Dop'): 'Ultra_JarroDophilus',
      (20742, 'Shoe', 'Unmellow Yellow'): 'Unmellow_Yellow_Tieks_Neon_Patent_Leather_Ballet_Flats',
      (18670, 'None', 'Utana 5" Porcel'): 'Utana_5_Porcelain_Ramekin_Large',
      (2016, 'Consumer Goods', "VAN'S FIRE ROAS"): 'VANS_FIRE_ROASTED_VEGGIE_CRACKERS_GLUTEN_FREE',
      (16694, 'Toys', 'VEGETABLE GARDE'): 'VEGETABLE_GARDEN',
      (1446, 'Consumer Goods', "Van's Cereal, H"): 'Vans_Cereal_Honey_Nut_Crunch_11_oz_box',
      (16418, 'Consumer Goods', 'Victor Reversib'): 'Victor_Reversible_Bookend',
      (18502, 'Toys', 'Vtech Cruise & '): 'Vtech_Cruise_Learn_Car_25_Years',
      (30686, 'None', 'Vtech Roll & Le'): 'Vtech_Roll_Learn_Turtle',
      (21056, 'Toys', 'Vtech Stack & S'): 'Vtech_Stack_Sing_Rings_636_Months',
      (30798, 'Toys', 'WATER LANDING N'): 'WATER_LANDING_NET',
      (6452, 'Toys', 'WHALE WHISTLE ('): 'WHALE_WHISTLE_6PCS_SET',
      (10160, 'Shoe', 'W Lou'): 'W_Lou_z0dkC78niiZ',
      (5912, 'None', 'Weisshai Great '): 'Weisshai_Great_White_Shark',
      (17068, 'Bottles and Cans and Cups', 'Weston No. 22 C'): 'Weston_No_22_Cajun_Jerky_Tonic_12_fl_oz_nLj64ZnGwDh',
      (13526, 'Bottles and Cans and Cups', 'Weston No. 33 S'): 'Weston_No_33_Signature_Sausage_Tonic_12_fl_oz',
      (2562, 'Consumer Goods', 'Whey Protein, 3'): 'Whey_Protein_3_Flavor_Variety_Pack_12_Packets',
      (3594, 'Consumer Goods', 'Whey Protein, C'): 'Whey_Protein_Chocolate_12_Packets',
      (10702, 'Bottles and Cans and Cups', 'Whey Protein, V'): 'Whey_Protein_Vanilla',
      (2778, 'Consumer Goods', 'Whey Protein, V'): 'Whey_Protein_Vanilla_12_Packets',
      (43360, 'Shoe', 'White Rose Tiek'): 'White_Rose_Tieks_Leather_Ballet_Flats_with_Floral_Rosettes',
      (20854, 'Shoe', 'Wild Copper Tie'): 'Wild_Copper_Tieks_Metallic_Italian_Leather_Ballet_Flats',
      (6328, 'Consumer Goods', 'Wilton Easy Lay'): 'Wilton_Easy_Layers_Cake_Pan_Set',
      (13322, 'Bottles and Cans and Cups', 'Wilton Pearlize'): 'Wilton_Pearlized_Sugar_Sprinkles_525_oz_Gold',
      (1554, 'Consumer Goods', 'Wilton Pre-Cut '): 'Wilton_PreCut_Parchment_Sheets_10_x_15_24_sheets',
      (1022, 'Board Games', 'Winning Moves 1'): 'Winning_Moves_1180_Aggravation_Board_Game',
      (11502, 'Bag', 'Wishbone Pencil'): 'Wishbone_Pencil_Case',
      (18204, 'Shoe', "Women's Angelfi"): 'Womens_Angelfish_Boat_Shoe_in_Linen_Leopard_Sequin_NJDwosWNeZz',
      (17926, 'Shoe', "Women's Angelfi"): 'Womens_Angelfish_Boat_Shoe_in_Linen_Oat',
      (16654, 'Shoe', "Women's Audrey "): 'Womens_Audrey_Slip_On_Boat_Shoe_in_Graphite_Nubuck_xWVkCJ5vxZe',
      (19084, 'Shoe', "Women's Authent"): 'Womens_Authentic_Original_Boat_Shoe_in_Classic_Brown_Leather',
      (18644, 'Shoe', "Women's Authent"): 'Womens_Authentic_Original_Boat_Shoe_in_Classic_Brown_Leather_48Nh7VuMwW6',
      (19216, 'Shoe', "Women's Authent"): 'Womens_Authentic_Original_Boat_Shoe_in_Classic_Brown_Leather_cJSCWiH7QmB',
      (18998, 'Shoe', "Women's Authent"): 'Womens_Authentic_Original_Boat_Shoe_in_Navy_Deerskin_50lWJaLWG8R',
      (18206, 'Shoe', "Women's Betty C"): 'Womens_Betty_Chukka_Boot_in_Grey_Jersey_Sequin',
      (20268, 'Shoe', "Women's Betty C"): 'Womens_Betty_Chukka_Boot_in_Navy_Jersey_Sequin_y0SsHk7dKUX',
      (22772, 'Shoe', "Women's Betty C"): 'Womens_Betty_Chukka_Boot_in_Navy_aEE8OqvMII4',
      (17996, 'Shoe', "Women's Betty C"): 'Womens_Betty_Chukka_Boot_in_Salt_Washed_Red_AL2YrOt9CRy',
      (25088, 'Shoe', "Women's Bluefis"): 'Womens_Bluefish_2Eye_Boat_Shoe_in_Brown_Deerskin_JJ2pfEHTZG7',
      (20328, 'Shoe', "Women's Bluefis"): 'Womens_Bluefish_2Eye_Boat_Shoe_in_Brown_Deerskin_i1TgjjO0AKY',
      (19938, 'Shoe', "Women's Bluefis"): 'Womens_Bluefish_2Eye_Boat_Shoe_in_Linen_Natural_Sparkle_Suede_kqi81aojcOR',
      (17972, 'Shoe', "Women's Bluefis"): 'Womens_Bluefish_2Eye_Boat_Shoe_in_Linen_Natural_Sparkle_Suede_w34KNQ41csH',
      (19904, 'Shoe', "Women's Bluefis"): 'Womens_Bluefish_2Eye_Boat_Shoe_in_Linen_Oat',
      (16578, 'Shoe', "Women's Bluefis"): 'Womens_Bluefish_2Eye_Boat_Shoe_in_Linen_Oat_IbrSyJdpT3h',
      (16458, 'Shoe', "Women's Bluefis"): 'Womens_Bluefish_2Eye_Boat_Shoe_in_Linen_Oat_niKJKeWsmxY',
      (21022, 'Shoe', "Women's Bluefis"): 'Womens_Bluefish_2Eye_Boat_Shoe_in_Tan',
      (27864, 'Bottles and Cans and Cups', "Women's Bluefis"): 'Womens_Bluefish_2Eye_Boat_Shoe_in_White_Tumbled_YG44xIePRHw',
      (15574, 'Shoe', "Women's Canvas "): 'Womens_Canvas_Bahama_in_Black',
      (16170, 'Shoe', "Women's Canvas "): 'Womens_Canvas_Bahama_in_Black_vnJULsDVyq5',
      (12688, 'Shoe', "Women's Canvas "): 'Womens_Canvas_Bahama_in_White_4UyOhP6rYGO',
      (13010, 'Shoe', "Women's Canvas "): 'Womens_Canvas_Bahama_in_White_UfZPHGQpvz0',
      (26776, 'Shoe', "Women's Cloud L"): 'Womens_Cloud_Logo_Authentic_Original_Boat_Shoe_in_Black_Supersoft_8LigQYwf4gr',
      (23766, 'Shoe', "Women's Cloud L"): 'Womens_Cloud_Logo_Authentic_Original_Boat_Shoe_in_Black_Supersoft_cZR022qFI4k',
      (27230, 'Shoe', "Women's Hikerfi"): 'Womens_Hikerfish_Boot_in_Black_Leopard_bVSNY1Le1sm',
      (24584, 'Shoe', "Women's Hikerfi"): 'Womens_Hikerfish_Boot_in_Black_Leopard_ridcCWsv8rW',
      (20600, 'Shoe', "Women's Hikerfi"): 'Womens_Hikerfish_Boot_in_Linen_Leather_Sparkle_Suede_QktIyAkonrU',
      (20658, 'Shoe', "Women's Hikerfi"): 'Womens_Hikerfish_Boot_in_Linen_Leather_Sparkle_Suede_imlP8VkwqIH',
      (10530, 'Bottles and Cans and Cups', "Women's Multi 1"): 'Womens_Multi_13',
      (13638, 'Shoe', "Women's Sequin "): 'Womens_Sequin_Bahama_in_White_Sequin_V9K1hf24Oxe',
      (23926, 'Shoe', "Women's Sequin "): 'Womens_Sequin_Bahama_in_White_Sequin_XoR8xTlxj1g',
      (16678, 'Shoe', "Women's Sequin "): 'Womens_Sequin_Bahama_in_White_Sequin_yGVsSA4tOwJ',
      (18986, 'Shoe', "Women's Sparkle"): 'Womens_Sparkle_Suede_Angelfish_in_Grey_Sparkle_Suede_Silver',
      (14916, 'Shoe', "Women's Sparkle"): 'Womens_Sparkle_Suede_Bahama_in_Silver_Sparkle_Suede_Grey_Patent_tYrIBLMhSTN',
      (13136, 'Shoe', "Women's Sparkle"): 'Womens_Sparkle_Suede_Bahama_in_Silver_Sparkle_Suede_Grey_Patent_x9rclU7EJXx',
      (12208, 'Shoe', "Women's Suede B"): 'Womens_Suede_Bahama_in_Graphite_Suede_cUAjIMhWSO9',
      (10528, 'Shoe', "Women's Suede B"): 'Womens_Suede_Bahama_in_Graphite_Suede_p1KUwoWbw7R',
      (18034, 'Shoe', "Women's Suede B"): 'Womens_Suede_Bahama_in_Graphite_Suede_t22AJSRjBOX',
      (19818, 'Shoe', "Women's Teva Ca"): 'Womens_Teva_Capistrano_Bootie',
      (20664, 'Shoe', "Women's Teva Ca"): 'Womens_Teva_Capistrano_Bootie_ldjRT9yZ5Ht',
      (16908, 'Toys', 'Wooden ABC 123 '): 'Wooden_ABC_123_Blocks_50_pack',
      (2634, 'Consumer Goods', 'Wrigley Orbit M'): 'Wrigley_Orbit_Mint_Variety_18_Count',
      (5536, 'Bottles and Cans and Cups', 'Xyli Pure Xylit'): 'Xyli_Pure_Xylitol',
      (3790, 'Consumer Goods', 'Yum-Yum D3 Liqu'): 'YumYum_D3_Liquid',
      (29206, 'Shoe', 'ZX700'): 'ZX700_lYiwcTIekXk',
      (34414, 'Shoe', 'ZX700'): 'ZX700_mf9Pc06uL06',
      (32970, 'Shoe', 'ZX700'): 'ZX700_mzGbdP3u6JB',
      (26360, 'Shoe', 'ZigKick Hoops\nF'): 'ZigKick_Hoops',
      (13138, 'Shoe', 'adiZero Slide 2'): 'adiZero_Slide_2_SC',
      (32918, 'Shoe', 'adistar boost m'): 'adistar_boost_m',
      (27160, 'Shoe', 'adizero 5-Tool '): 'adizero_5Tool_25',
      (23144, 'Shoe', 'adizero F50 TRX'): 'adizero_F50_TRX_FG_LEA',
  }

  return asset_id_lookup[(
      meta["nr_faces"],
      meta["category"],
      meta["description"][:15]
  )]
