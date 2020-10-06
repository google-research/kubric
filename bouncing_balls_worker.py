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

import kubric.pylab as kb
import numpy as np


class BouncingBallsWorker(kb.Worker):
  def get_argparser(self):
    parser = super().get_argparser()
    # add additional commandline arguments
    parser.add_argument("--min_nr_balls", type=int, default=4)
    parser.add_argument("--max_nr_balls", type=int, default=5)
    parser.add_argument("--ball_radius", type=float, default=0.2)
    # overwrite default values
    parser.set_defaults(asset_source=["./Assets/KLEVR", "./Assets/BouncingBalls"])
    return parser

  def add_room(self):
    floor_material = kb.FlatMaterial(color=kb.get_color('black'),
                                     indirect_visibility=False)
    wall_material = kb.FlatMaterial(color=kb.get_color('white'),
                                    indirect_visibility=False)
    room_dynamics = {
        "restitution": 0.95,
        "friction": 0.01,
    }

    floor = kb.Cube(scale=(1, 1, 0.9), position=(0, 0, -0.9),
                    material=floor_material, static=True, **room_dynamics)
    north_wall = kb.Cube(scale=(1.2, 0.9, 1), position=(0, 1.9, 0.9),
                         material=wall_material, static=True, **room_dynamics)
    south_wall = kb.Cube(scale=(1.2, 0.9, 1), position=(0, -1.9, 0.9),
                         material=wall_material, static=True, **room_dynamics)
    east_wall = kb.Cube(scale=(0.9, 1, 1), position=(1.9, 0, 0.9),
                        material=wall_material, static=True, **room_dynamics)
    west_wall = kb.Cube(scale=(0.9, 1, 1), position=(-1.9, 0, 0.9),
                        material=wall_material, static=True, **room_dynamics)
    self.add(floor, north_wall, south_wall, east_wall, west_wall,
             is_background=True)

  def add_camera(self):
    camera = kb.OrthographicCamera(position=(0, 0, 3), orthographic_scale=2.2)  # looks down by default
    self.add(camera)
    self.scene.camera = camera

  def get_random_ball(self):
    velocity_range = (-1, -1, 0), (1, 1, 0)
    ball_material = kb.FlatMaterial(color=kb.random_hue_color(rnd=self.rnd),
                                    indirect_visibility=False)
    ball = kb.Sphere(scale=[self.config.ball_radius]*3,
                     material=ball_material,
                     friction=0.01,
                     restitution=0.95,
                     velocity=self.rnd.uniform(*velocity_range))
    return ball

  def run(self):
    self.add_camera()
    self.add_room()

    spawn_area = (-1, -1, 0), (1, 1, 2.1*self.config.ball_radius)
    balls = []

    nr_objects = self.rnd.randint(self.config.min_nr_balls, self.config.max_nr_balls)
    for i in range(nr_objects):  # random number of objects
      ball = self.get_random_ball()
      self.place_without_overlap(ball, [kb.position_sampler(spawn_area)])
      balls.append(ball)

    sim_path = self.save_simulator_state()
    self.run_simulation()
    render_path = self.save_renderer_state()
    self.render()
    output = self.post_process()
    output = {}
    # collect ground-truth factors
    output["factors"] = []
    for i, obj in enumerate(balls):
      output["factors"].append({
          "color": obj.material.color.rgb,
          "mass": obj.mass,
          "animation": obj.keyframes,
      })
    out_path = self.save_output(output)
    name = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    self.export(self.output_dir, name, files_list=[sim_path, render_path, out_path])


if __name__ == '__main__':
  worker = BouncingBallsWorker()

  worker.setup()
  worker.run()
