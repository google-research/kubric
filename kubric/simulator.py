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
import math
import logging
import uuid

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

from bidict import bidict
import pybullet as pb

logger = logging.getLogger(__name__)


@dataclass
class Object3D:
  sim_filename: str
  vis_filename: str

  asset_id: str

  uid: str = field(default_factory=lambda: str(uuid.uuid4()))

  position: Tuple[float] = (0.0, 0.0, 0.0)
  rotation: Tuple[float] = (0.0, 0.0, 0.0, 1.0)

  linear_velocity: Tuple[float] = (0.0, 0.0, 0.0)
  angular_velocity: Tuple[float] = (0.0, 0.0, 0.0)

  static: bool = False

  mass: float = field(default=None, repr=False)
  lateral_friction: float = field(default=None, repr=False)
  spinning_friction: float = field(default=None, repr=False)
  rolling_friction: float = field(default=None, repr=False)
  restitution: float = field(default=None, repr=False)
  linear_damping: float = field(default=None, repr=False)
  angular_damping: float = field(default=None, repr=False)

  def __hash__(self):
    # make this object hashable
    return object.__hash__(self)


class Simulator:
  """
  Args:
      gravity: Direction and strength of gravity [m/s²]. Defaults to (0, 0, -10).
      step_rate: How many simulation steps are performed per second.
                 Has to be a multiple of the frame rate.
      frame_rate: Number of frames per second.
                  Required because the simulator only reports positions for frames not for steps.
  """

  def __init__(
      self,
      gravity: Tuple[float, float, float] = (0.0, 0.0, -10.0),
      step_rate: int = 240,
      frame_rate: int = 24,
  ):
    self.physicsClient = pb.connect(pb.DIRECT)  # pb.GUI
    self.gravity = gravity
    self.step_rate: int = step_rate
    self.frame_rate: int = frame_rate
    self.objects_by_idx = bidict()
    if step_rate % frame_rate != 0:
      raise ValueError(
          "step_rate has to be a multiple of frame_rate, but {} % {} != 0".format(
              step_rate, frame_rate
          )
      )

  def __del__(self):
    pb.disconnect()

  @property
  def gravity(self) -> Tuple[float, float, float]:
    return self._gravity

  @gravity.setter
  def gravity(self, gravity: Tuple[float, float, float]):
    self._gravity = gravity
    pb.setGravity(*gravity)

  def add(self, obj: Object3D) -> bool:
    """
    Place an object to a particular place and orientation and check for any overlap.
    This may either load a new object or move an existing object.

    Args:
      obj: The object to be placed.

    Returns:
      True if there was a collision, False otherwise
    """
    self._ensure_object_loaded(obj)
    self._set_location_and_orientation(obj)
    self._change_dynamics(obj)
    self._set_velocity(obj)
    return self._check_overlap(obj)

  def _ensure_object_loaded(self, obj: Object3D):
    """ Ensure that an object is loaded into the simulation.
    Args:
        obj: Has to specify a sim_filename that can be loaded.
            Currently the only supported formats is URDF.
            (but MJCF and SDF are easy to add, possibly also OBJ and DAE.)
    """
    if obj in self.objects_by_idx.inverse:
      # object already in simulation
      logger.info("Object '{}' already in the simulation".format(obj))
      return

    path = Path(obj.sim_filename).resolve()
    logger.info("Loading '{}' in the simulator".format(path))

    if not path.exists():
      raise IOError('File "{}" does not exist.'.format(path))

    if path.suffix == ".urdf":
      object_idx = pb.loadURDF(str(path), useFixedBase=obj.static)
    else:
      raise IOError(
        'Unsupported format "{}" of file "{}"'.format(path.suffix, path))

    if object_idx < 0:
      raise IOError('Failed to load "{}".'.format(path))

    # link objects for later mapping
    self.objects_by_idx[object_idx] = obj

    return object_idx

  def _check_overlap(self, obj: Object3D) -> bool:
    obj_idx = self.objects_by_idx.inverse[obj]

    body_ids = [pb.getBodyUniqueId(i) for i in range(pb.getNumBodies())]
    for body_id in body_ids:
      if body_id == obj_idx:
        continue
      overlap_points = pb.getClosestPoints(obj_idx, body_id, distance=0)
      if overlap_points:
        # TODO: we can easily get a suggested correction here
        # i = np.argmin([o[8] for o in overlap_points], axis=0)  # find the most overlapping point
        # push = np.array(overlap_points[i][7]) * (overlap_points[i][8] + margin)
        return True
    return False

  def _set_location_and_orientation(self, obj: Object3D):
    obj_idx = self.objects_by_idx.inverse[obj]
    pb.resetBasePositionAndOrientation(obj_idx, obj.position,
                                       obj.rotation)

  def _change_dynamics(self, obj: Object3D):
    obj_idx = self.objects_by_idx.inverse[obj]
    dyn = {}
    if obj.mass is not None:
      dyn["mass"] = obj.mass
    if obj.lateral_friction is not None:
      dyn["lateralFriction"] = obj.lateral_friction
    if obj.spinning_friction is not None:
      dyn["spinningFriction"] = obj.spinning_friction
    if obj.rolling_friction is not None:
      dyn["rollingFriction"] = obj.rolling_friction
    if obj.restitution is not None:
      dyn["restitution"] = obj.restitution
    if obj.linear_damping is not None:
      dyn["linearDamping"] = obj.linear_damping
    if obj.angular_damping is not None:
      dyn["angularDamping"] = obj.angular_damping
    pb.changeDynamics(obj_idx, -1, **dyn)

  def _set_velocity(self, obj: Object3D):
    obj_idx = self.objects_by_idx.inverse[obj]
    pb.resetBaseVelocity(obj_idx, obj.linear_velocity, obj.angular_velocity)

  def run(self, duration: float = 1.0) -> Dict[Object3D, Dict[str, list]]:
    max_step = math.floor(self.step_rate * duration)
    steps_per_frame = self.step_rate // self.frame_rate

    obj_idxs = [pb.getBodyUniqueId(i) for i in range(pb.getNumBodies())]
    animation = {obj_id: {"position": [], "orient_quat": []}
                 for obj_id in obj_idxs}

    for current_step in range(max_step):
      if current_step % steps_per_frame == 0:
        for obj_idx in obj_idxs:
          pos, quat = pb.getBasePositionAndOrientation(obj_idx)
          animation[obj_idx]["position"].append(pos)
          animation[obj_idx]["orient_quat"].append(quat)  # roll, pitch, yaw

      pb.stepSimulation()
    return {self.objects_by_idx[obj_idx]: anim for obj_idx, anim in animation.items()}
