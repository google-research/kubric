# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import bz2
import logging
import pathlib
import pickle
import PIL.Image
import pprint
import shutil
import sys
import tarfile
from typing import Callable, Optional, Sequence


import munch
import numpy as np
from google.cloud import storage

import kubric.assets
import kubric.post_processing
import kubric.renderer
import kubric.simulator
from kubric import core


class Worker:
  def __init__(self, config=None):
    self.parser = self.get_argparser()
    self.config = munch.munchify(config or {})
    self.rnd = None
    self.log = logging.getLogger(__name__)
    self.scene = None
    self.simulator = None
    self.renderer = None
    self.asset_sources = munch.Munch()
    self.objects = []
    self.background_objects = []
    self.work_dir = None
    self.output_dir = None

  def get_argparser(self):
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_rate", type=int, default=24)
    parser.add_argument("--step_rate", type=int, default=240)
    parser.add_argument("--frame_start", type=int, default=1)
    parser.add_argument("--frame_end", type=int, default=24)  # 1 second
    parser.add_argument("--logging_level", type=str, default="INFO")
    parser.add_argument("--work_dir", type=str, default="./output/work_dir")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--max_placement_trials", type=int, default=100)
    parser.add_argument("--asset_source", action="append",
                        help="add an additonal source of assets using a URI "
                             "e.g. '.Assets/KLEVR' or 'gs://kubric/GSO'."
                             "Can be passed multiple times.")
    return parser

  def parse_arguments(self):
    # --- parse argument in a way compatible with blender"s REPL
    if "--" in sys.argv:
      FLAGS = self.parser.parse_args(args=sys.argv[sys.argv.index("--")+1:])
    else:
      FLAGS = self.parser.parse_args(args=[])

    self.config.update(vars(FLAGS))
    return self.config

  def setup_logging(self):
    logging.basicConfig(level=self.config.logging_level)

  def setup_random_state(self):
    self.rnd = np.random.RandomState(self.config.get("seed", None))

  def setup_scene(self):
    self.scene = core.Scene(frame_start=self.config.frame_start,
                            frame_end=self.config.frame_end,
                            frame_rate=self.config.frame_rate,
                            step_rate=self.config.step_rate,
                            resolution=(self.config.width, self.config.height))
    return self.scene

  def setup_asset_sources(self):
    for uri in self.config.asset_source:
      name = pathlib.Path(uri).name
      self.log.info("Adding AssetSource '%s' with URI='%s'", name, uri)
      self.asset_sources[name] = kubric.assets.AssetSource(uri)

  def setup_work_dir(self):
    self.work_dir = pathlib.Path(self.config.work_dir).absolute()
    # clear workdir and create anew
    if self.work_dir.exists():
     shutil.rmtree(self.work_dir)
    self.work_dir.mkdir(parents=True)

  def setup_output_dir(self):
    self.output_dir = pathlib.Path(self.config.output_dir).absolute()
    # ensure output_dir exists
    self.output_dir.mkdir(parents=True, exist_ok=True)

  def setup(self):
    self.parse_arguments()
    self.setup_logging()
    self.setup_random_state()
    self.log.info(pprint.pformat(self.config, indent=2, width=100))
    self.setup_work_dir()
    self.setup_output_dir()

    self.setup_asset_sources()
    self.scene = self.setup_scene()
    self.simulator = kubric.simulator.PyBullet(self.scene)
    self.renderer = kubric.renderer.Blender(self.scene)

  def create_asset(self, source, asset_id, **kwargs):
    return self.asset_sources[source].create(asset_id, **kwargs)

  def add(self, *objects: core.Asset, is_background=False):
    for obj in objects:
      if obj in self.objects or obj in self.background_objects:
        continue
      self.background_objects.append(obj) if is_background else self.objects.append(obj)

      self.log.info("Added %s", obj)
      if self.simulator:
        self.simulator.add(obj)
      if self.renderer:
        self.renderer.add(obj)

  def place_without_overlap(self, obj: core.PhysicalObject,
                            pose_samplers: Sequence[Callable[[core.PhysicalObject,
                                                              np.random.RandomState],
                                                             None]],
                            max_trials: Optional[int] = None):
    self.add(obj)
    max_trials = max_trials if max_trials is not None else self.config.max_placement_trials

    collision = True
    trial = 0
    while collision and trial < max_trials:
      for sampler in pose_samplers:
        sampler(obj, self.rnd)
      collision = self.simulator.check_overlap(obj)
      trial += 1
    if collision:
      raise RuntimeError("Failed to place", obj)

  def run_simulation(self):
    # --- run the physics simulation
    animation = self.simulator.run()

    # --- Bake the simulation into keyframes
    for obj in animation.keys():
      for frame_id in range(self.scene.frame_end + 1):
        obj.position = animation[obj]["position"][frame_id]
        obj.quaternion = animation[obj]["quaternion"][frame_id]
        obj.keyframe_insert("position", frame_id)
        obj.keyframe_insert("quaternion", frame_id)
    return animation

  def save_simulator_state(self, filename="scene.bullet"):
    self.simulator.save_state(self.work_dir, filename)
    return self.work_dir / filename

  def save_renderer_state(self, filename="scene.blend"):
    self.renderer.save_state(self.work_dir, filename)
    return self.work_dir / filename

  def render(self):
    self.renderer.render(path=self.work_dir)

  def post_process(self):
    T = self.scene.frame_end - self.scene.frame_start + 1
    W, H = self.scene.resolution

    output = {
      "RGBA": np.zeros((T, W, H, 4), dtype=np.float32),
      "segmentation": np.zeros((T, W, H, 1), dtype=np.uint32),
      "flow": np.zeros((T, W, H, 3), dtype=np.float32),
      "depth": np.zeros((T, W, H, 1), dtype=np.float32),
      "UV": np.zeros((T, W, H, 3), dtype=np.float32),
    }

    for t, frame_id in enumerate(range(self.scene.frame_start, self.scene.frame_end + 1)):

      exr_filename = self.work_dir / "exr" / f"frame_{frame_id:04d}.exr"
      png_filename = self.work_dir / "images" / f"frame_{frame_id:04d}.png"

      print("Processing", exr_filename)
      layers = kubric.post_processing.get_render_layers_from_exr(exr_filename,
                                                                 self.background_objects,
                                                                 self.objects)
      output["segmentation"][t, :, :, 0] = layers["SegmentationIndex"][:, :, 0]
      output["flow"][t] = layers["Vector"]
      output["depth"][t] = layers["Depth"]
      output["UV"][t] = layers["UV"]
      # use the PNG instead of the EXR for the image, since it is already contrast normalized
      img = np.asarray(PIL.Image.open(png_filename))
      output["RGBA"][t] = img / 255.

    return output

  def get_gt_factors(self, objects):
    factors = []
    for i, obj in enumerate(objects):
      factors.append({
          "mass": obj.mass,
          "color": obj.material.color.rgb,
          "animation": obj.keyframes,
      })
    return factors

  def save_output(self, output, filename="output.pkl.bz2"):
    # pickle and bz2 the output
    path = self.work_dir / filename
    with bz2.BZ2File(path, "w") as f:
      pickle.dump(output, f)
    return path

  def export(self, target_dir, name, files_list=("output.pkl.bz2", "scene.blend", "scene.bullet")):

    zip_filename = name + '.tar.gz'
    target_dir = pathlib.Path(target_dir)

    output_path = pathlib.Path(target_dir) / zip_filename

    with tarfile.open(self.work_dir / zip_filename, "w:gz") as tar:
      for file in files_list:
        file = pathlib.Path(file)
        assert file.exists(), file
        tar.add(str(file), f"{name}/{file.name}")

    if output_path.parts[0] == "gs:":
      client = storage.Client()
      bucket = client.get_bucket(output_path.parts[1])
      dst_blob_name = pathlib.Path(*output_path.parts[2:]) / zip_filename
      blob = bucket.blob(str(dst_blob_name))
      blob.upload_from_filename(str(self.work_dir / zip_filename))
    else:
      shutil.move(str(self.work_dir / zip_filename), str(output_path))


  def run(self):
    pass