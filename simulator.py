# Copyright 2020 Google LLC
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
import math
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import pybullet as pb


class Simulator:
  """

  Args:
    gravity: Direction and strength of gravity [m/s²]. Defaults to (0, 0, -10).
    step_rate: How many simulation steps are performed per second.
               Has to be a multiple of the frame rate.
    frame_rate: Number of frames per second.
                Required because the simulator only reports positions for frames not for steps.
  """
  def __init__(self,
      gravity: Tuple[float, float, float] = (0., 0., -10.),
      step_rate: int = 240,
      frame_rate: int = 24):
    self.physicsClient = pb.connect(pb.DIRECT)
    self.gravity = gravity
    self.step_rate = step_rate
    self.frame_rate = frame_rate
    if step_rate % frame_rate != 0:
      raise ValueError('step_rate has to be a multiple of frame_rate, but {} % {} != 0'
                       .format(step_rate, frame_rate))

  def __del__(self):
    pb.disconnect()

  @property
  def gravity(self) -> Tuple[float, float, float]:
    return self._gravity

  @gravity.setter
  def gravity(self, gravity: Tuple[float, float, float]):
    self._gravity = gravity
    pb.setGravity(*gravity)

  def load_object(self, filename: Union[str, Path]) -> int:
    """
    Load an object into the simulation from the given filename.

    Args:
      filename: Filename of the object to load.
          Currently the only supported formats is URDF.
          (but MJCF and SDF are easy to add, possibly also OBJ and DAE.)

    Returns:
      int: the object_id of the inserted object.
    """
    logging.info("Loading '{}' in the simulator".format(filename))
    path = Path(filename).resolve()
    if not path.exists():
      raise IOError('File "{}" does not exist.'.format(path))

    if path.suffix == '.urdf':
      object_id = pb.loadURDF(filename)
    else:
      raise IOError('Unsupported format "{}" of file "{}"'.format(path.suffix, path))

    if object_id < 0:
      raise IOError('Failed to load "{}".'.format(path))

    return object_id

  def place_object(self,
      object_id: int,
      position: Tuple[float, float, float] = (0, 0, 0),
      orient_euler: Optional[Tuple[float, float, float]] = (0, 0, 0),
      orient_quat: Optional[Tuple[float, float, float, float]] = None) -> bool:
    """
    Move an object to a particular place and orientation and check for overlap with other objects.

    Args:
      object_id:
      position:
      orient_euler:
      orient_quat:

    Returns:
      True if there was a collision, False otherwise
    """

    if orient_quat is None:
      assert orient_euler is not None
      orient_quat = pb.getQuaternionFromEuler(orient_euler)
    pb.resetBasePositionAndOrientation(object_id, position, orient_quat)
    body_ids = [pb.getBodyUniqueId(i) for i in range(pb.getNumBodies())]
    for body_id in body_ids:
      if body_id == object_id: continue
      overlap_points = pb.getClosestPoints(object_id, body_id, distance=0)
      if overlap_points:
        # TODO: we can easily get a suggested correction here
        # i = np.argmin([o[8] for o in overlap_points], axis=0)  # find the most overlapping point
        # push = np.array(overlap_points[i][7]) * (overlap_points[i][8] + margin)
        return True
    return False

  def change_dynamics(
      self,
      object_id: int,
      mass: Optional[float] = None,
      lateral_friction: Optional[float] = None,
      spinning_friction: Optional[float] = None,
      rolling_friction: Optional[float] = None,
      restitution: Optional[float] = None,
      linear_damping: Optional[float] = None,
      angular_damping: Optional[float] = None):
    dyn = {}
    if mass is not None: dyn['mass'] = mass
    if lateral_friction is not None: dyn['lateralFriction'] = lateral_friction
    if spinning_friction is not None: dyn['spinningFriction'] = spinning_friction
    if rolling_friction is not None: dyn['rollingFriction'] = rolling_friction
    if restitution is not None: dyn['restitution'] = restitution
    if linear_damping is not None: dyn['linearDamping'] = linear_damping
    if angular_damping is not None: dyn['angularDamping'] = angular_damping
    pb.changeDynamics(object_id, -1, **dyn)

  def set_velocity(
      self,
      object_id: int,
      linear_velocity: Tuple[float, float, float] = (0, 0, 0),
      angular_velocity: Tuple[float, float, float] = (0, 0, 0)):
    pb.resetBaseVelocity(object_id, linear_velocity, angular_velocity)

  def run(self, duration: float = 1.0) -> Dict[int, Dict[str, list]]:
    max_step = math.floor(self.step_rate * duration)
    steps_per_frame = self.step_rate // self.frame_rate
    current_step = 0

    obj_ids = [pb.getBodyUniqueId(i) for i in range(pb.getNumBodies())]
    animation = {
        obj_id: {
            'position': [],
            'orient_quat': [],
        } for obj_id in obj_ids
    }

    # TODO: why is this not a for loop?
    while current_step <= max_step:
      if current_step % steps_per_frame == 0:
        for obj_id in obj_ids:
          pos, quat = pb.getBasePositionAndOrientation(obj_id)
          animation[obj_id]['position'].append(pos)
          animation[obj_id]['orient_quat'].append(quat)  # roll, pitch, yaw

      pb.stepSimulation()
      current_step += 1
    return animation

