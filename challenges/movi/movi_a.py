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
A simple rigid-body simulation based on the CLEVR dataset.
The scene consists of a gray floor, four light sources, a camera, and between 
3 and 10 random objects.
The camera position is randomly jittered in a small area around a fixed position
and always points at the origin.
The objects are randomly chosen from: 
 - one three shapes [cube, sphere, cylinder], 
 - scaled to one of two sizes [small, large], 
 - have one of two materials [rubber, metal], 
 - and one of eight colors [blue, brown, cyan, gray, green, purple, red, yellow]

They are spawned without overlap in the region [(-5, -5, 1), (5, 5, 5)], and
initialized with a random velocity from the range [(-4, -4, 0), (4, 4, 0)] 
minus the position of the object to bias their trajectory towards the center of
the scene.

The scene is simulated for 2 seconds, with the physical properties of the 
objects depending on the material:
 - metal: friction=0.4, restitution=0.3, density=2.7
 - rubber: friction=0.8, restitution=0.7, density=1.1
 
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
- "shape_label": ["cube", "cylinder", "sphere"]
- "size_label": ["small", "large"]
- "color": (3,) [float32]
  Color of the object in RGB.
- "color_label": ["blue", "brown", "cyan", "gray", "green", "purple", "red", "yellow"]
- "material_label": ["metal", "rubber"]

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
class MoviAConfig(tfds.core.BuilderConfig):
  """"Configuration for Multi-Object Video (MOVi) dataset."""
  height: int = 256
  width: int = 256
  num_frames: int = 24
  validation_ratio: float = 0.1
  train_val_path: str = None
  test_split_paths: Dict[str, str] = dataclasses.field(default_factory=dict)


class MoviA(tfds.core.BeamBasedBuilder):
  """DatasetBuilder for MOVi-A dataset."""
  VERSION = tfds.core.Version("1.0.0")
  RELEASE_NOTES = {
      "1.0.0": "initial release",
  }

  BUILDER_CONFIGS = [
      MoviAConfig(
          name="256x256",
          description="Full resolution of 256x256",
          height=256,
          width=256,
          validation_ratio=0.025,
          # train_val_path="/usr/local/google/home/klausg/movi_tmp",
          train_val_path="gs://research-brain-kubric-xgcp/jobs/movi_a_regen_10k/",
          test_split_paths={
          }
      ),
      MoviAConfig(
          name="128x128",
          description="Downscaled to 128x128",
          height=128,
          width=128,
          validation_ratio=0.025,
          # train_val_path="/usr/local/google/home/klausg/movi_tmp",
          train_val_path="gs://research-brain-kubric-xgcp/jobs/movi_a_regen_10k/",
          test_split_paths={
          }
      ),
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""

    h = self.builder_config.height
    w = self.builder_config.width
    s = self.builder_config.num_frames

    def get_movi_a_instance_features(seq_length: int):
      features = get_instance_features(seq_length)
      features.update({
          "shape_label": tfds.features.ClassLabel(
              names=["cube", "cylinder", "sphere"]),
          "size_label": tfds.features.ClassLabel(
              names=["small", "large"]),
          "color": tfds.features.Tensor(shape=(3,), dtype=tf.float32),
          "color_label": tfds.features.ClassLabel(
              names=["blue", "brown", "cyan", "gray",
                     "green", "purple", "red", "yellow"]),
          "material_label": tfds.features.ClassLabel(
              names=["metal", "rubber"]),
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
            "instances": tfds.features.Sequence(
                feature=get_movi_a_instance_features(seq_length=s)),
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

      # add MOVi-A specific instance information:
      for i, obj in enumerate(result["instances"]):
        obj["shape_label"] = metadata["instances"][i]["shape"]
        obj["size_label"] = metadata["instances"][i]["size_label"]
        obj["material_label"] = metadata["instances"][i]["material"]
        obj["color"] = np.array(metadata["instances"][i]["color"],
                                dtype=np.float32)
        obj["color_label"] = metadata["instances"][i]["color_label"]

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

