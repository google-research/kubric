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

import logging
import sys


import pathlib
from typing import Dict, Union, Optional

import pybullet as pb
from singledispatchmethod import singledispatchmethod
import munch
import bidict
import tempfile

from kubric.io import RedirectStream
from kubric import core

logger = logging.getLogger(__name__)
_pybullet_logs = tempfile.mkstemp(suffix="_bullet.txt")[1]

with RedirectStream(stream=sys.stdout, filename=_pybullet_logs):
  import pybullet as pb

__all__ = ("PyBullet", )


class PyBullet(core.View):

  def __init__(self, scene: core.Scene):
    self.physicsClient = pb.connect(pb.DIRECT)  # pb.GUI
    # Set some parameters to fix the sticky-walls problem
    # (see https://github.com/bulletphysics/bullet3/issues/3094)
    pb.setPhysicsEngineParameter(restitutionVelocityThreshold=0., warmStartingFactor=0.,
                                 useSplitImpulse=True, contactSlop=0., enableConeFriction=False,
                                 deterministicOverlappingPairs=True)

    super().__init__(scene, scene_observers={
        "gravity": [lambda change: pb.setGravity(*change.new)],
    })

  def __del__(self):
    pb.disconnect()

  @singledispatchmethod
  def add_asset(self, asset: core.Asset) -> Optional[int]:
    raise NotImplementedError(f"Cannot add {asset!r}")

  def remove_asset(self, asset: core.Asset) -> None:
    if self in asset.linked_objects:
      pb.removeBody(asset.linked_objects[self])
    # TODO: unobserve

  @add_asset.register(core.Camera)
  def _add_object(self, obj: core.Camera) -> Optional[int]:
    logger.debug(f"Ignored camera {obj}")
    return None

  @add_asset.register(core.Material)
  def _add_object(self, obj: core.Material) -> Optional[int]:
    logger.debug(f"Ignored material {obj}")
    return None

  @add_asset.register(core.Light)
  def _add_object(self, obj: core.Light) -> Optional[int]:
    logger.debug(f"Ignored light {obj}")
    return None

  @add_asset.register(core.Cube)
  def _add_object(self, obj: core.Cube) -> Optional[int]:
    collision_idx = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=obj.scale)
    visual_idx = -1
    mass = 0 if obj.static else obj.mass
    # useMaximalCoordinates and contactProcessingThreshold are required to fix the sticky walls issue
    # see https://github.com/bulletphysics/bullet3/issues/3094
    box_idx = pb.createMultiBody(mass, collision_idx, visual_idx, obj.position,
                                 wxyz2xyzw(obj.quaternion), useMaximalCoordinates=True)
    pb.changeDynamics(box_idx, -1, contactProcessingThreshold=0)
    register_physical_object_setters(obj, box_idx)

    return box_idx

  @add_asset.register(core.Sphere)
  def _add_object(self, obj: core.Cube) -> Optional[int]:
    radius = obj.scale[0]
    assert radius == obj.scale[1] == obj.scale[2], obj.scale  # only uniform scaling
    collision_idx = pb.createCollisionShape(pb.GEOM_SPHERE, radius=radius)
    visual_idx = -1
    mass = 0 if obj.static else obj.mass
    # useMaximalCoordinates and contactProcessingThreshold are required to fix the sticky walls issue
    # see https://github.com/bulletphysics/bullet3/issues/3094
    sphere_idx = pb.createMultiBody(mass, collision_idx, visual_idx, obj.position,
                                    wxyz2xyzw(obj.quaternion), useMaximalCoordinates=True)
    pb.changeDynamics(sphere_idx, -1, contactProcessingThreshold=0)
    register_physical_object_setters(obj, sphere_idx)

    return sphere_idx

  @add_asset.register(core.FileBasedObject)
  def _add_object(self, obj: core.FileBasedObject) -> Optional[int]:
    # TODO: support other file-formats
    # TODO: add material assignments
    path = pathlib.Path(obj.simulation_filename).resolve()
    logger.debug("Loading '{}' in the simulator".format(path))

    if not path.exists():
      raise IOError('File "{}" does not exist.'.format(path))

    scale = obj.scale[0]
    assert obj.scale[1] == obj.scale[2] == scale, "Pybullet does not support non-uniform scaling"

    # useMaximalCoordinates and contactProcessingThreshold are required to fix the sticky walls issue
    # see https://github.com/bulletphysics/bullet3/issues/3094
    if path.suffix == ".urdf":
      obj_idx = pb.loadURDF(str(path), useFixedBase=obj.static, globalScaling=scale,
                            useMaximalCoordinates=True)
    else:
      raise IOError(
          'Unsupported format "{}" of file "{}"'.format(path.suffix, path))

    if obj_idx < 0:
      raise IOError('Failed to load "{}".'.format(path))

    pb.changeDynamics(obj_idx, -1, contactProcessingThreshold=0)

    register_physical_object_setters(obj, obj_idx)
    return obj_idx

  def check_overlap(self, obj: core.PhysicalObject) -> bool:
    obj_idx = obj.linked_objects[self]

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

  def get_position_and_rotation(self, obj_idx: int):
    pos, quat = pb.getBasePositionAndOrientation(obj_idx)
    return pos, xyzw2wxyz(quat)  # convert quaternion format

  def get_velocities(self, obj_idx: int):
    velocity, angular_velocity = pb.getBaseVelocity(obj_idx)
    return velocity, angular_velocity

  def save_state(self, path: Union[pathlib.Path, str]):
    """Receives a folder path as input."""
    path = pathlib.Path(path)
    path = path / "scene.bullet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pb.saveBullet(str(path))

  def run(self) -> Dict[core.PhysicalObject, Dict[str, list]]:
    steps_per_frame = self.scene.step_rate // self.scene.frame_rate
    max_step = (self.scene.frame_end + 1) * steps_per_frame

    obj_idxs = [pb.getBodyUniqueId(i) for i in range(pb.getNumBodies())]
    animation = {obj_id: {"position": [], "quaternion": [], "velocity": [], "angular_velocity": []}
                 for obj_id in obj_idxs}

    for current_step in range(max_step):
      if current_step % steps_per_frame == 0:
        for obj_idx in obj_idxs:
          position, quaternion = self.get_position_and_rotation(obj_idx)
          velocity, angular_velocity = self.get_velocities(obj_idx)

          animation[obj_idx]["position"].append(position)
          animation[obj_idx]["quaternion"].append(quaternion)
          animation[obj_idx]["velocity"].append(velocity)
          animation[obj_idx]["angular_velocity"].append(angular_velocity)

      pb.stepSimulation()

    animation = {asset: animation[asset.linked_objects[self]] for asset in self.scene.assets
                 if asset.linked_objects.get(self) in obj_idxs}

    # --- Transfer simulation to renderer keyframes
    for obj in animation.keys():
      for frame_id in range(self.scene.frame_end + 1):
        obj.position = animation[obj]["position"][frame_id]
        obj.quaternion = animation[obj]["quaternion"][frame_id]
        obj.keyframe_insert("position", frame_id)
        obj.keyframe_insert("quaternion", frame_id)

    return animation


def xyzw2wxyz(xyzw):
  """Convert quaternions from XYZW format to WXYZ."""
  x, y, z, w = xyzw
  return w, x, y, z


def wxyz2xyzw(wxyz):
  """Convert quaternions from WXYZ format to XYZW."""
  w, x, y, z = wxyz
  return x, y, z, w


def register_physical_object_setters(obj: core.PhysicalObject, obj_idx):
  assert isinstance(obj, core.PhysicalObject), f"{obj!r} is not a PhysicalObject"

  obj.observe(setter(obj_idx, set_position), "position")
  obj.observe(setter(obj_idx, set_quaternion), "quaternion")
  # TODO Pybullet does not support rescaling. So we should warn if scale is changed
  obj.observe(setter(obj_idx, set_velocity), "velocity")
  obj.observe(setter(obj_idx, set_angular_velocity), "angular_velocity")
  obj.observe(setter(obj_idx, set_friction), "friction")
  obj.observe(setter(obj_idx, set_restitution), "restitution")
  obj.observe(setter(obj_idx, set_mass), "mass")
  obj.observe(setter(obj_idx, set_static), "static")


def setter(object_idx, func):
  def _callable(change):
    return func(object_idx, change.new, change.owner)
  return _callable


def set_position(object_idx, position, asset):
  # reuse existing quaternion
  _, quaternion = pb.getBasePositionAndOrientation(object_idx)
  # resetBasePositionAndOrientation zeroes out velocities, but we wish to conserve them
  velocity, angular_velocity = pb.getBaseVelocity(object_idx)
  pb.resetBasePositionAndOrientation(object_idx, position, quaternion)
  pb.resetBaseVelocity(object_idx, velocity, angular_velocity)


def set_quaternion(object_idx, quaternion, asset):
  quaternion = wxyz2xyzw(quaternion)  # convert quaternion format
  # reuse existing position
  position, _ = pb.getBasePositionAndOrientation(object_idx)
  # resetBasePositionAndOrientation zeroes out velocities, but we wish to conserve them
  velocity, angular_velocity = pb.getBaseVelocity(object_idx)
  pb.resetBasePositionAndOrientation(object_idx, position, quaternion)
  pb.resetBaseVelocity(object_idx, velocity, angular_velocity)


def set_velocity(object_idx, velocity, asset):
  _, angular_velocity = pb.getBaseVelocity(object_idx)  # reuse existing angular velocity
  pb.resetBaseVelocity(object_idx, velocity, angular_velocity)


def set_angular_velocity(object_idx, angular_velocity, asset):
  velocity, _ = pb.getBaseVelocity(object_idx)  # reuse existing velocity
  pb.resetBaseVelocity(object_idx, velocity, angular_velocity)


def set_mass(object_idx, mass: float, asset):
  if mass < 0:
    raise ValueError('mass cannot be negative ({})'.format(mass))
  if not asset.static:
    pb.changeDynamics(object_idx, -1, mass=mass)


def set_static(object_idx, is_static, asset):
  if is_static:
    pb.changeDynamics(object_idx, -1, mass=0.)
  else:
    pb.changeDynamics(object_idx, -1, mass=asset.mass)


def set_friction(object_idx, friction: float, asset):
  if friction < 0:
    raise ValueError('friction cannot be negative ({})'.format(friction))
  pb.changeDynamics(object_idx, -1, lateralFriction=friction)


def set_restitution(object_idx, restitution: float, asset):
  if restitution < 0:
    raise ValueError('restitution cannot be negative ({})'.format(restitution))
  if restitution > 1:
    raise ValueError('restitution should be below 1.0 ({})'.format(restitution))
  pb.changeDynamics(object_idx, -1, restitution=restitution)
