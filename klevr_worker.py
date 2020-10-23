# Copyright 2020 The Kubric Authors
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
import datetime

import sys; sys.path.append(".")

import kubric as kb

class KLEVRWorker(kb.Worker):
  # def get_argparser(self):
  #   parser = super().get_argparser()
  #   # add additional commandline arguments
  #   parser.add_argument("--min_nr_objects", type=int, default=4)
  #   parser.add_argument("--max_nr_objects", type=int, default=10)
  #   # overwrite default values
  #   # TODO(klausg): should we move KLEVR.zip to ./Assets too?
  #   parser.set_defaults(asset_source=['./Assets/KLEVR'])
  #   return parser

  def add_floor(self):
    floor = self.create_asset("KLEVR", asset_id="Floor", static=True,
                              position=(0, 0, -0.2), scale=(2, 2, 2))
    self.add(floor)
    return floor

  def add_lights(self):
    # --- Light settings from CLEVR
    sun = kb.DirectionalLight(color=kb.get_color("white"), intensity=0.45,
                              shadow_softness=0.2,
                              position=(11.6608, -6.62799, 25.8232))
    sun.look_at((0, 0, 0))
    lamp_back = kb.RectAreaLight(color=kb.get_color("white"), intensity=50.,
                                 position=(-1.1685, 2.64602, 5.81574))
    lamp_back.look_at((0, 0, 0))
    lamp_key = kb.RectAreaLight(color=kb.get_color(0xffedd0), intensity=100,
                                width=0.5, height=0.5,
                                position=(6.44671, -2.90517, 4.2584))
    lamp_key.look_at((0, 0, 0))
    lamp_fill = kb.RectAreaLight(color=kb.get_color(0xc2d0ff), intensity=30,
                                 width=0.5, height=0.5,
                                 position=(-4.67112, -4.0136, 3.01122))
    lamp_fill.look_at((0, 0, 0))
    self.add(sun, lamp_back, lamp_key, lamp_fill)

  def add_camera(self):
    camera = kb.PerspectiveCamera(focal_length=35., sensor_width=32,
                                  position=(7.48113, -6.50764, 5.34367))
    camera.look_at((0, 0, 0))
    self.add(camera)
    self.scene.camera = camera


  def get_random_object(self):
    random_id = self.rnd.choice([
        "LargeMetalCube",
        "LargeMetalCylinder",
        "LargeMetalSphere",
        "LargeRubberCube",
        "LargeRubberCylinder",
        "LargeRubberSphere",
        "MetalSpot",
        "RubberSpot",
        "SmallMetalCube",
        "SmallMetalCylinder",
        "SmallMetalSphere",
        "SmallRubberCube",
        "SmallRubberCylinder",
        "SmallRubberSphere",
    ])
    if "Metal" in random_id:
      material = kb.PrincipledBSDFMaterial(
            color=kb.random_hue_color(rnd=self.rnd),
            roughness=0.2,
            metallic=1.0,
            ior=2.5)
    else:  # if "Rubber" in random_id:
      material = kb.PrincipledBSDFMaterial(
          color=kb.random_hue_color(rnd=self.rnd),
          roughness=0.7,
          specular=0.33,
          metallic=0.,
          ior=1.25)
    return self.create_asset("KLEVR", asset_id=random_id, material=material)

  def run(self):
    self.add_camera()
    self.add_lights()
    self.add_floor()

    spawn_region = [(-4, -4, 0), (4, 4, 3)]
    velocity_range = [(-4, -4, 0), (4, 4, 0)]
    objects = []

    nr_objects = self.rnd.randint(self.config.min_nr_objects, self.config.max_nr_objects)
    for i in range(nr_objects):  # random number of objects
      obj = self.get_random_object()
      self.place_without_overlap(obj, [kb.rotation_sampler(),
                                       kb.position_sampler(spawn_region)])
      obj.velocity = (self.rnd.uniform(*velocity_range) -
                      [obj.position[0], obj.position[1], 0])  # bias velocity towards center
      objects.append(obj)

    sim_path = self.save_simulator_state()
    self.run_simulation()
    render_path = self.save_renderer_state()

    # TODO: re-enable later on
    # self.render()
    # output = self.post_process()
    # # --- collect ground-truth factors
    # output["factors"] = []
    # for i, obj in enumerate(objects):
    #   output["factors"].append({
    #       "asset_id": obj.asset_id,
    #       "material": "Metal" if "Metal" in obj.asset_id else "Rubber",
    #       "mass": obj.mass,
    #       "color": obj.material.color.rgb,
    #       "animation": obj.keyframes,
    #   })
    # out_path = self.save_output(output)
    # name = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    # self.export(self.output_dir, name, files_list=[sim_path, render_path, out_path])

# --- parser
parser = kb.ArgumentParser()
parser.add_argument("--min_nr_objects", type=int, default=4)
parser.add_argument("--max_nr_objects", type=int, default=10)
parser.set_defaults(asset_source=['./Assets/KLEVR'])
FLAGS = parser.parse_args()

worker = KLEVRWorker() # TODO: to be removed
worker.config.update(vars(FLAGS)) # TODO: why this needed?
worker.setup()

# --- camera
# camera = kb.PerspectiveCamera(focal_length=35., sensor_width=32, position=(7.48113, -6.50764, 5.34367))
# camera.look_at((0, 0, 0))
# worker.add(camera) # why needed?

worker.run()
