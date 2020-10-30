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

import functools
import pathlib
from typing import Dict, Tuple, Union

import munch
import bidict
import tempfile

logger = logging.getLogger(__name__)

from kubric.utils import RedirectStream
_pybullet_logs = tempfile.mkstemp(suffix="_bullet.txt")[1]
logger.info("PyBullet logs stored in {}".format(_pybullet_logs))
with RedirectStream(stream=sys.stdout, filename=_pybullet_logs):
  import pybullet as pb   

from kubric import core


def xyzw2wxyz(xyzw):
  """Convert quaternions from XYZW format to WXYZ."""
  x, y, z, w = xyzw
  return w, x, y, z


def wxyz2xyzw(wxyz):
  """Convert quaternions from WXYZ format to XYZW."""
  w, x, y, z = wxyz
  return x, y, z, w


class Setter:
  def __init__(self, object_idx: int, setter):
    self.object_idx = object_idx
    self.setter = setter

  def __call__(self, change):
    self.setter(self.object_idx, change.new)


def set_position(object_idx, position):
  # reuse existing quaternion
  _, quaternion = pb.getBasePositionAndOrientation(object_idx)
  # resetBasePositionAndOrientation zeroes out velocities, but we wish to conserve them
  velocity, angular_velocity = pb.getBaseVelocity(object_idx)
  pb.resetBasePositionAndOrientation(object_idx, position, quaternion)
  pb.resetBaseVelocity(object_idx, velocity, angular_velocity)


def set_quaternion(object_idx, quaternion):
  quaternion = wxyz2xyzw(quaternion)  # convert quaternion format
  # reuse existing position
  position, _ = pb.getBasePositionAndOrientation(object_idx)
  # resetBasePositionAndOrientation zeroes out velocities, but we wish to conserve them
  velocity, angular_velocity = pb.getBaseVelocity(object_idx)
  pb.resetBasePositionAndOrientation(object_idx, position, quaternion)
  pb.resetBaseVelocity(object_idx, velocity, angular_velocity)


def set_velocity(object_idx, velocity):
  _, angular_velocity = pb.getBaseVelocity(object_idx)  # reuse existing angular velocity
  pb.resetBaseVelocity(object_idx, velocity, angular_velocity)


def set_angular_velocity(object_idx, angular_velocity):
  velocity, _ = pb.getBaseVelocity(object_idx)  # reuse existing velocity
  pb.resetBaseVelocity(object_idx, velocity, angular_velocity)


def set_mass(object_idx, mass: float):
  if mass < 0:
    raise ValueError('mass cannot be negative ({})'.format(mass))
  pb.changeDynamics(object_idx, -1, mass=mass)


def set_friction(object_idx, friction: float):
  if friction < 0:
    raise ValueError('friction cannot be negative ({})'.format(friction))
  pb.changeDynamics(object_idx, -1, lateralFriction=friction)


def set_restitution(object_idx, restitution: float):
  if restitution < 0:
    raise ValueError('restitution cannot be negative ({})'.format(restitution))
  if restitution > 1:
    raise ValueError('restitution should be below 1.0 ({})'.format(restitution))
  pb.changeDynamics(object_idx, -1, restitution=restitution)


def set_gravity(_, gravity: Tuple[float, float, float]):
  pb.setGravity(*gravity)


@functools.singledispatch
def add_object(obj: core.Asset) -> Tuple[int, Dict[str, Setter]]:
  raise NotImplementedError()


@add_object.register(core.Camera)
def _add_object(obj: core.Camera):
  # Cameras are ignored
  return -3, {}


@add_object.register(core.Material)
def _add_object(obj: core.Material):
  # Materials are ignored
  return -3, {}


@add_object.register(core.Light)
def _add_object(obj: core.Light):
  # Lights are ignored
  return -3, {}


@add_object.register(core.Cube)
def _add_object(obj: core.Cube):
  collision_idx = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=obj.scale)
  visual_idx = -1
  mass = 0 if obj.static else obj.mass
  # useMaximalCoordinates and contactProcessingThreshold are required to fix the sticky walls issue
  # see https://github.com/bulletphysics/bullet3/issues/3094
  box_idx = pb.createMultiBody(mass, collision_idx, visual_idx, obj.position,
                               wxyz2xyzw(obj.quaternion), useMaximalCoordinates=True)
  pb.changeDynamics(box_idx, -1, contactProcessingThreshold=0)

  setters = {
      'position': Setter(box_idx, set_position),
      'quaternion': Setter(box_idx, set_quaternion),
      # TODO 'scale': Setter(object_idx, scale)  # Pybullet does not support rescaling. So we should warn
      'velocity': Setter(box_idx, set_velocity),
      'angular_velocity': Setter(box_idx, set_angular_velocity),
      'mass': lambda x: None if obj.static else Setter(box_idx, set_mass),
      'friction': Setter(box_idx, set_friction),
      'restitution': Setter(box_idx, set_restitution),
  }
  return box_idx, setters


@add_object.register(core.Sphere)
def _add_object(obj: core.Cube):
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
  setters = {
      'position': Setter(sphere_idx, set_position),
      'quaternion': Setter(sphere_idx, set_quaternion),
      # TODO 'scale': Setter(object_idx, scale)  # Pybullet does not support rescaling. So we should warn
      'velocity': Setter(sphere_idx, set_velocity),
      'angular_velocity': Setter(sphere_idx, set_angular_velocity),
      'mass': lambda x: None if obj.static else Setter(sphere_idx, set_mass),
      'friction': Setter(sphere_idx, set_friction),
      'restitution': Setter(sphere_idx, set_restitution),
  }
  return sphere_idx, setters


@add_object.register(core.FileBasedObject)
def _add_object(obj: core.FileBasedObject):
  # TODO: support other file-formats
  # TODO: add material assignments
  path = pathlib.Path(obj.simulation_filename).resolve()
  logger.info("Loading '{}' in the simulator".format(path))

  if not path.exists():
    raise IOError('File "{}" does not exist.'.format(path))

  scale = obj.scale[0]
  assert obj.scale[1] == obj.scale[2] == scale, "Pybullet does not support non-uniform scaling"

  # useMaximalCoordinates and contactProcessingThreshold are required to fix the sticky walls issue
  # see https://github.com/bulletphysics/bullet3/issues/3094
  if path.suffix == ".urdf":
    object_idx = pb.loadURDF(str(path), useFixedBase=obj.static, globalScaling=scale,
                             useMaximalCoordinates=True)
  else:
    raise IOError(
        'Unsupported format "{}" of file "{}"'.format(path.suffix, path))

  if object_idx < 0:
    raise IOError('Failed to load "{}".'.format(path))

  pb.changeDynamics(object_idx, -1, contactProcessingThreshold=0)

  setters = {
      'position': Setter(object_idx, set_position),
      'quaternion': Setter(object_idx, set_quaternion),
      # TODO 'scale': Setter(object_idx, scale)  # Pybullet does not support rescaling. So we should warn
      'velocity': Setter(object_idx, set_velocity),
      'angular_velocity': Setter(object_idx, set_angular_velocity),
      'mass': Setter(object_idx, set_mass),
      'friction': Setter(object_idx, set_friction),
      'restitution': Setter(object_idx, set_restitution),
  }
  return object_idx, setters


@add_object.register(core.Scene)
def _add_object(obj: core.Scene):
  object_idx = -2  # the scene is no object, so we use a special index

  setters = {
      'gravity': Setter(object_idx, set_gravity),
  }
  return object_idx, setters


class PyBullet:

  def __init__(self, scene: core.Scene):
    self.objects_to_pybullet = bidict.bidict()
    self.physicsClient = pb.connect(pb.DIRECT)  # pb.GUI
    # Set some parameters to fix the sticky-walls problem
    # (see https://github.com/bulletphysics/bullet3/issues/3094)
    pb.setPhysicsEngineParameter(restitutionVelocityThreshold=0., warmStartingFactor=0.,
                                 useSplitImpulse=True, contactSlop=0., enableConeFriction=False,
                                 deterministicOverlappingPairs=True)

    if scene.step_rate % scene.frame_rate != 0:
      raise ValueError(
          "step_rate has to be a multiple of frame_rate, but {} % {} != 0".format(
              scene.step_rate, scene.frame_rate))
    self.scene = scene
    self.add(scene)

  def __del__(self):
    pb.disconnect()

  def add(self, obj: core.Asset) -> int:
    """
    Place an object to a particular place and orientation and check for any overlap.
    This may either load a new object or move an existing object.

    Args:
      obj: The object to be placed.

    Returns:
      True if there was a collision, False otherwise
    """
    if obj in self.objects_to_pybullet:
      return self.objects_to_pybullet[obj]

    obj_idx, setters = add_object(obj)

    if obj_idx >= 0:
      self.objects_to_pybullet[obj] = obj_idx

    for name, setter in setters.items():
      # recursively add sub-assets
      value = getattr(obj, name)
      if isinstance(value, core.Asset):
        value = self.add(value)
      # Initialize values
      setter(munch.Munch(owner=obj, new=value, type='init'))
      # Link values
      obj.observe(setter, names=[name])

  def check_overlap(self, obj: core.PhysicalObject) -> bool:
    obj_idx = self.objects_to_pybullet[obj]

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

  def get_position_and_rotation(self, obj: Union[int, core.PhysicalObject]):
    if isinstance(obj, core.PhysicalObject):
      obj_idx = self.objects_to_pybullet[obj]
    else:
      assert isinstance(obj, int), f"Invalid object {obj} of type {type(obj)}."
      obj_idx = obj

    pos, quat = pb.getBasePositionAndOrientation(obj_idx)
    return pos, xyzw2wxyz(quat)  # convert quaternion format

  def get_velocities(self, obj: Union[int, core.PhysicalObject]):
    if isinstance(obj, core.PhysicalObject):
      obj_idx = self.objects_to_pybullet[obj]
    else:
      assert isinstance(obj, int), f"Invalid object {obj} of type {type(obj)}."
      obj_idx = obj

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
    return {self.objects_to_pybullet.inverse[obj_idx]: anim
            for obj_idx, anim in animation.items()}
