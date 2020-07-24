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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Any

import pybullet as pb

logger = logging.getLogger(__name__)


@dataclass
class Object3D:
    sim_filename: str
    # TODO: this filename is loaded by pybullet, and can be fetched by getVisualShapeData?
    # vis_filename: str

    position: Tuple[float] = (0.0, 0.0, 0.0)
    orientation: Tuple[float] = (0.0, 0.0, 0.0, 1.0)

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

    sim_ref: Any = field(default=None, repr=False, init=False, compare=False)
    vis_ref: Any = field(default=None, repr=False, init=False, compare=False)


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
        self.step_rate = step_rate
        self.frame_rate = frame_rate
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

    def place_object(self, obj: Object3D) -> bool:
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
        if obj.sim_ref is not None:
            # object already in simulation
            logger.info("Object '{}' already in the simulation".format(obj))
            return

        path = Path(obj.sim_filename).resolve()
        logger.info("Loading '{}' in the simulator".format(path))

        if not path.exists():
            raise IOError('File "{}" does not exist.'.format(path))

        if path.suffix == ".urdf":
            object_id = pb.loadURDF(str(path), useFixedBase=obj.static)
        else:
            raise IOError('Unsupported format "{}" of file "{}"'.format(path.suffix, path))

        if object_id < 0:
            raise IOError('Failed to load "{}".'.format(path))

        obj.sim_ref = object_id

        return object_id

    def _check_overlap(self, obj: Object3D) -> bool:
        body_ids = [pb.getBodyUniqueId(i) for i in range(pb.getNumBodies())]
        for body_id in body_ids:
            if body_id == obj.sim_ref:
                continue
            overlap_points = pb.getClosestPoints(obj.sim_ref, body_id, distance=0)
            if overlap_points:
                # TODO: we can easily get a suggested correction here
                # i = np.argmin([o[8] for o in overlap_points], axis=0)  # find the most overlapping point
                # push = np.array(overlap_points[i][7]) * (overlap_points[i][8] + margin)
                return True
        return False

    def _set_location_and_orientation(self, obj: Object3D):
        pb.resetBasePositionAndOrientation(obj.sim_ref, obj.position, obj.orientation)

    def _change_dynamics(self, obj: Object3D):
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
        pb.changeDynamics(obj.sim_ref, -1, **dyn)

    def _set_velocity(self, obj: Object3D):
        pb.resetBaseVelocity(obj.sim_ref, obj.linear_velocity, obj.angular_velocity)

    def run(self, duration: float = 1.0) -> Dict[int, Dict[str, list]]:
        max_step = math.floor(self.step_rate * duration)
        steps_per_frame = self.step_rate // self.frame_rate
        current_step = 0

        obj_ids = [pb.getBodyUniqueId(i) for i in range(pb.getNumBodies())]
        animation = {obj_id: {"position": [], "orient_quat": []} for obj_id in obj_ids}

        # TODO: why is this not a for loop?
        while current_step <= max_step:
            if current_step % steps_per_frame == 0:
                for obj_id in obj_ids:
                    pos, quat = pb.getBasePositionAndOrientation(obj_id)
                    animation[obj_id]["position"].append(pos)
                    animation[obj_id]["orient_quat"].append(quat)  # roll, pitch, yaw

            pb.stepSimulation()
            current_step += 1
        return animation
