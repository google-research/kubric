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

"""Kubric dataset with point tracking."""

import functools
import itertools

import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d


def project_point(cam, point3d, num_frames):
  """Compute the image space coordinates [0, 1] for a set of points.

  Args:
    cam: The camera parameters, as returned by kubric.  'matrix_world' and
      'intrinsics' have a leading axis num_frames.
    point3d: Points in 3D world coordinates.  it has shape [num_frames,
      num_points, 3].
    num_frames: The number of frames in the video.

  Returns:
    Image coordinates in 2D.  The last coordinate is an indicator of whether
      the point is behind the camera.
  """

  homo_transform = tf.linalg.inv(cam['matrix_world'])
  homo_intrinsics = tf.zeros((num_frames, 3, 1), dtype=tf.float32)
  homo_intrinsics = tf.concat([cam['intrinsics'], homo_intrinsics], axis=2)

  point4d = tf.concat([point3d, tf.ones_like(point3d[:, :, 0:1])], axis=2)
  projected = tf.matmul(point4d, tf.transpose(homo_transform, (0, 2, 1)))
  projected = tf.matmul(projected, tf.transpose(homo_intrinsics, (0, 2, 1)))
  image_coords = projected / projected[:, :, 2:3]
  image_coords = tf.concat(
      [image_coords[:, :, :2],
       tf.sign(projected[:, :, 2:])], axis=2)
  return image_coords


def unproject(coord, cam, depth):
  """Unproject points.

  Args:
    coord: Points in 2D coordinates.  it has shape [num_points, 2].  Coord is in
      integer (y,x) because of the way meshgrid happens.
    cam: The camera parameters, as returned by kubric.  'matrix_world' and
      'intrinsics' have a leading axis num_frames.
    depth: Depth map for the scene.

  Returns:
    Image coordinates in 3D.
  """
  shp = tf.convert_to_tensor(tf.shape(depth))
  idx = coord[:, 0] * shp[1] + coord[:, 1]
  coord = tf.cast(coord[..., ::-1], tf.float32)
  shp = tf.cast(shp[1::-1], tf.float32)[tf.newaxis, ...]

  # Need to convert from pixel to raster coordinate.
  projected_pt = (coord + 0.5) / shp

  projected_pt = tf.concat(
      [
          projected_pt,
          tf.ones_like(projected_pt[:, -1:]),
      ],
      axis=-1,
  )

  camera_plane = projected_pt @ tf.linalg.inv(tf.transpose(cam['intrinsics']))
  camera_ball = camera_plane / tf.sqrt(
      tf.reduce_sum(
          tf.square(camera_plane),
          axis=1,
          keepdims=True,
      ),)
  camera_ball *= tf.gather(tf.reshape(depth, [-1]), idx)[:, tf.newaxis]

  camera_ball = tf.concat(
      [
          camera_ball,
          tf.ones_like(camera_plane[:, 2:]),
      ],
      axis=1,
  )
  points_3d = camera_ball @ tf.transpose(cam['matrix_world'])
  return points_3d[:, :3] / points_3d[:, 3:]


def reproject(coords, camera, camera_pos, num_frames, bbox=None):
  """Reconstruct points in 3D and reproject them to pixels.

  Args:
    coords: Points in 3D.  It has shape [num_points, 3].  If bbox is specified,
      these are assumed to be in local box coordinates (as specified by kubric),
      and bbox will be used to put them into world coordinates; otherwise they
      are assumed to be in world coordinates.
    camera: the camera intrinsic parameters, as returned by kubric.
      'matrix_world' and 'intrinsics' have a leading axis num_frames.
    camera_pos: the camera positions.  It has shape [num_frames, 3]
    num_frames: the number of frames in the video.
    bbox: The kubric bounding box for the object.  Its first axis is num_frames.

  Returns:
    Image coordinates in 2D and their respective depths.  For the points,
    the last coordinate is an indicator of whether the point is behind the
    camera.  They are of shape [num_points, num_frames, 3] and
    [num_points, num_frames] respectively.
  """
  # First, reconstruct points in the local object coordinate system.
  if bbox is not None:
    coord_box = list(itertools.product([-.5, .5], [-.5, .5], [-.5, .5]))
    coord_box = np.array([np.array(x) for x in coord_box])
    coord_box = np.concatenate(
        [coord_box, np.ones_like(coord_box[:, 0:1])], axis=1)
    coord_box = tf.tile(coord_box[tf.newaxis, ...], [num_frames, 1, 1])
    bbox_homo = tf.concat([bbox, tf.ones_like(bbox[:, :, 0:1])], axis=2)

    local_to_world = tf.linalg.lstsq(tf.cast(coord_box, tf.float32), bbox_homo)
    world_coords = tf.matmul(
        tf.cast(
            tf.concat([coords, tf.ones_like(coords[:, 0:1])], axis=1),
            tf.float32)[tf.newaxis, :, :], local_to_world)
    world_coords = world_coords[:, :, 0:3] / world_coords[:, :, 3:]
  else:
    world_coords = tf.tile(coords[tf.newaxis, :, :], [num_frames, 1, 1])

  # Compute depths by taking the distance between the points and the camera
  # center.
  depths = tf.sqrt(
      tf.reduce_sum(
          tf.square(world_coords - camera_pos[:, np.newaxis, :]),
          axis=2,
      ),)

  # Project each point back to the image using the camera.
  projections = project_point(camera, world_coords, num_frames)

  return (
      tf.transpose(projections, (1, 0, 2)),
      tf.transpose(depths),
      tf.transpose(world_coords, (1, 0, 2)),
  )


def estimate_occlusion_by_depth_and_segment(
    data,
    segments,
    x,
    y,
    num_frames,
    thresh,
    seg_id,
):
  """Estimate depth at a (floating point) x,y position.

  We prefer overestimating depth at the point, so we take the max over the 4
  neightoring pixels.

  Args:
    data: depth map. First axis is num_frames.
    segments: segmentation map. First axis is num_frames.
    x: x coordinate. First axis is num_frames.
    y: y coordinate. First axis is num_frames.
    num_frames: number of frames.
    thresh: Depth threshold at which we consider the point occluded.
    seg_id: Original segment id.  Assume occlusion if there's a mismatch.

  Returns:
    Depth for each point.
  """

  # need to convert from raster to pixel coordinates
  x = x - 0.5
  y = y - 0.5

  x0 = tf.cast(tf.floor(x), tf.int32)
  x1 = x0 + 1
  y0 = tf.cast(tf.floor(y), tf.int32)
  y1 = y0 + 1

  shp = tf.shape(data)
  assert len(data.shape) == 3
  x0 = tf.clip_by_value(x0, 0, shp[2] - 1)
  x1 = tf.clip_by_value(x1, 0, shp[2] - 1)
  y0 = tf.clip_by_value(y0, 0, shp[1] - 1)
  y1 = tf.clip_by_value(y1, 0, shp[1] - 1)

  data = tf.reshape(data, [-1])
  rng = tf.range(num_frames)[:, tf.newaxis]
  i1 = tf.gather(data, rng * shp[1] * shp[2] + y0 * shp[2] + x0)
  i2 = tf.gather(data, rng * shp[1] * shp[2] + y1 * shp[2] + x0)
  i3 = tf.gather(data, rng * shp[1] * shp[2] + y0 * shp[2] + x1)
  i4 = tf.gather(data, rng * shp[1] * shp[2] + y1 * shp[2] + x1)

  depth = tf.maximum(tf.maximum(tf.maximum(i1, i2), i3), i4)

  segments = tf.reshape(segments, [-1])
  i1 = tf.gather(segments, rng * shp[1] * shp[2] + y0 * shp[2] + x0)
  i2 = tf.gather(segments, rng * shp[1] * shp[2] + y1 * shp[2] + x0)
  i3 = tf.gather(segments, rng * shp[1] * shp[2] + y0 * shp[2] + x1)
  i4 = tf.gather(segments, rng * shp[1] * shp[2] + y1 * shp[2] + x1)

  depth_occluded = tf.less(tf.transpose(depth), thresh)
  seg_occluded = True
  for i in [i1, i2, i3, i4]:
    i = tf.cast(i, tf.int32)
    seg_occluded = tf.logical_and(seg_occluded, tf.not_equal(seg_id, i))

  return tf.logical_or(depth_occluded, tf.transpose(seg_occluded))


def get_camera_matrices(
    cam_focal_length,
    cam_positions,
    cam_quaternions,
    cam_sensor_width,
    input_size,
    num_frames=None,
):
  """Tf function that converts camera positions into projection matrices."""
  intrinsics = []
  matrix_world = []
  assert cam_quaternions.shape[0] == num_frames
  for frame_idx in range(cam_quaternions.shape[0]):
    focal_length = tf.cast(cam_focal_length, tf.float32)
    sensor_width = tf.cast(cam_sensor_width, tf.float32)
    f_x = focal_length / sensor_width
    f_y = focal_length / sensor_width * input_size[0] / input_size[1]
    p_x = 0.5
    p_y = 0.5
    intrinsics.append(
        tf.stack([
            tf.stack([f_x, 0., -p_x]),
            tf.stack([0., -f_y, -p_y]),
            tf.stack([0., 0., -1.]),
        ]))

    position = cam_positions[frame_idx]
    quat = cam_quaternions[frame_idx]
    rotation_matrix = rotation_matrix_3d.from_quaternion(
        tf.concat([quat[1:], quat[0:1]], axis=0))
    transformation = tf.concat(
        [rotation_matrix, position[:, tf.newaxis]],
        axis=1,
    )
    transformation = tf.concat(
        [transformation,
         tf.constant([0.0, 0.0, 0.0, 1.0])[tf.newaxis, :]],
        axis=0,
    )
    matrix_world.append(transformation)

  return (
      tf.cast(tf.stack(intrinsics), tf.float32),
      tf.cast(tf.stack(matrix_world), tf.float32),
  )


def quat2rot(quats):
  """Convert a list of quaternions to rotation matrices."""
  rotation_matrices = []
  for frame_idx in range(quats.shape[0]):
    quat = quats[frame_idx]
    rotation_matrix = rotation_matrix_3d.from_quaternion(
        tf.concat([quat[1:], quat[0:1]], axis=0))
    rotation_matrices.append(rotation_matrix)
  return tf.cast(tf.stack(rotation_matrices), tf.float32)


def rotate_surface_normals(
    world_frame_normals,
    point_3d,
    cam_pos,
    obj_rot_mats,
    frame_for_query,
):
  """Points are occluded if the surface normal points away from the camera."""
  query_obj_rot_mat = tf.gather(obj_rot_mats, frame_for_query)
  obj_frame_normals = tf.einsum(
      'boi,bi->bo',
      tf.linalg.inv(query_obj_rot_mat),
      world_frame_normals,
  )
  world_frame_normals_frames = tf.einsum(
      'foi,bi->bfo',
      obj_rot_mats,
      obj_frame_normals,
  )
  cam_to_pt = point_3d - cam_pos[tf.newaxis, :, :]
  dots = tf.reduce_sum(world_frame_normals_frames * cam_to_pt, axis=-1)
  faces_away = dots > 0

  # If the query point also faces away, it's probably a bug in the meshes, so
  # ignore the result of the test.
  faces_away_query = tf.reduce_sum(
      tf.cast(faces_away, tf.int32)
      * tf.one_hot(frame_for_query, tf.shape(faces_away)[1], dtype=tf.int32),
      axis=1,
      keepdims=True,
  )
  faces_away = tf.logical_and(faces_away, tf.logical_not(faces_away_query > 0))
  return faces_away


def single_object_reproject(
    bbox_3d=None,
    pt=None,
    pt_segments=None,
    camera=None,
    cam_positions=None,
    num_frames=None,
    depth_map=None,
    segments=None,
    window=None,
    input_size=None,
    quat=None,
    normals=None,
    frame_for_pt=None,
    trust_normals=None,
):
  """Reproject points for a single object.

  Args:
    bbox_3d: The object bounding box from Kubric.  If none, assume it's
      background.
    pt: The set of points in 3D, with shape [num_points, 3]
    pt_segments: The segment each point came from, with shape [num_points]
    camera: Camera intrinsic parameters
    cam_positions: Camera positions, with shape [num_frames, 3]
    num_frames: Number of frames
    depth_map: Depth map video for the camera
    segments: Segmentation map video for the camera
    window: the window inside which we're sampling points
    input_size: [height, width] of the input images.
    quat: Object quaternion [num_frames, 4]
    normals: Point normals on the query frame [num_points, 3]
    frame_for_pt: Integer frame where the query point came from [num_points]
    trust_normals: Boolean flag for whether the surface normals for each query
      are trustworthy [num_points]

  Returns:
    Position for each point, of shape [num_points, num_frames, 2], in pixel
    coordinates, and an occlusion flag for each point, of shape
    [num_points, num_frames].  These are respect to the image frame, not the
    window.

  """
  # Finally, reproject
  reproj, depth_proj, world_pos = reproject(
      pt,
      camera,
      cam_positions,
      num_frames,
      bbox=bbox_3d,
  )

  occluded = tf.less(reproj[:, :, 2], 0)
  reproj = reproj[:, :, 0:2] * np.array(input_size[::-1])[np.newaxis,
                                                          np.newaxis, :]
  occluded = tf.logical_or(
      occluded,
      estimate_occlusion_by_depth_and_segment(
          depth_map[:, :, :, 0],
          segments[:, :, :, 0],
          tf.transpose(reproj[:, :, 0]),
          tf.transpose(reproj[:, :, 1]),
          num_frames,
          depth_proj * .99,
          pt_segments,
      ),
  )
  obj_occ = occluded
  obj_reproj = reproj

  obj_occ = tf.logical_or(obj_occ, tf.less(obj_reproj[:, :, 1], window[0]))
  obj_occ = tf.logical_or(obj_occ, tf.less(obj_reproj[:, :, 0], window[1]))
  obj_occ = tf.logical_or(obj_occ, tf.greater(obj_reproj[:, :, 1], window[2]))
  obj_occ = tf.logical_or(obj_occ, tf.greater(obj_reproj[:, :, 0], window[3]))

  if quat is not None:
    faces_away = rotate_surface_normals(
        normals,
        world_pos,
        cam_positions,
        quat2rot(quat),
        frame_for_pt,
    )
    faces_away = tf.logical_and(faces_away, trust_normals)
  else:
    # world is convex; can't face away from cam.
    faces_away = tf.zeros([tf.shape(pt)[0], num_frames], dtype=tf.bool)

  return obj_reproj, tf.logical_or(faces_away, obj_occ), depth_proj


def get_num_to_sample(counts, max_seg_id, max_sampled_frac, tracks_to_sample):
  """Computes the number of points to sample for each object.

  Args:
    counts: The number of points available per object.  An int array of length
      n, where n is the number of objects.
    max_seg_id: The maximum number of segment id's in the video.
    max_sampled_frac: The maximum fraction of points to sample from each
      object, out of all points that lie on the sampling grid.
    tracks_to_sample: Total number of tracks to sample per video.

  Returns:
    The number of points to sample for each object.  An int array of length n.
  """
  seg_order = tf.argsort(counts)
  sorted_counts = tf.gather(counts, seg_order)
  initializer = (0, tracks_to_sample, 0)

  def scan_fn(prev_output, count_seg):
    index = prev_output[0]
    remaining_needed = prev_output[1]
    desired_frac = 1 / (tf.shape(seg_order)[0] - index)
    want_to_sample = (
        tf.cast(remaining_needed, tf.float32) *
        tf.cast(desired_frac, tf.float32))
    want_to_sample = tf.cast(tf.round(want_to_sample), tf.int32)
    max_to_sample = (
        tf.cast(count_seg, tf.float32) * tf.cast(max_sampled_frac, tf.float32))
    max_to_sample = tf.cast(tf.round(max_to_sample), tf.int32)
    num_to_sample = tf.minimum(want_to_sample, max_to_sample)

    remaining_needed = remaining_needed - num_to_sample
    return (index + 1, remaining_needed, num_to_sample)

  # outputs 0 and 1 are just bookkeeping; output 2 is the actual number of
  # points to sample per object.
  res = tf.scan(scan_fn, sorted_counts, initializer)[2]
  invert = tf.argsort(seg_order)
  num_to_sample = tf.gather(res, invert)
  num_to_sample = tf.concat(
      [
          num_to_sample,
          tf.zeros([max_seg_id - tf.shape(num_to_sample)[0]], dtype=tf.int32),
      ],
      axis=0,
  )
  return num_to_sample


#  pylint: disable=cell-var-from-loop


def track_points(
    object_coordinates,
    depth,
    depth_range,
    segmentations,
    surface_normals,
    bboxes_3d,
    obj_quat,
    cam_focal_length,
    cam_positions,
    cam_quaternions,
    cam_sensor_width,
    window,
    tracks_to_sample=256,
    sampling_stride=4,
    max_seg_id=25,
    max_sampled_frac=0.1,
    snap_to_occluder=False,
):
  """Track points in 2D using Kubric data.

  Args:
    object_coordinates: Video of coordinates for each pixel in the object's
      local coordinate frame.  Shape [num_frames, height, width, 3]
    depth: uint16 depth video from Kubric.  Shape [num_frames, height, width]
    depth_range: Values needed to normalize Kubric's int16 depth values into
      metric depth.
    segmentations: Integer object id for each pixel.  Shape
      [num_frames, height, width]
    surface_normals: uint16 surface normal map. Shape
      [num_frames, height, width, 3]
    bboxes_3d: The set of all object bounding boxes from Kubric
    obj_quat: Quaternion rotation for each object.  Shape
      [num_objects, num_frames, 4]
    cam_focal_length: Camera focal length
    cam_positions: Camera positions, with shape [num_frames, 3]
    cam_quaternions: Camera orientations, with shape [num_frames, 4]
    cam_sensor_width: Camera sensor width parameter
    window: the window inside which we're sampling points.  Integer valued
      in the format [x_min, y_min, x_max, y_max], where min is inclusive and
      max is exclusive.
    tracks_to_sample: Total number of tracks to sample per video.
    sampling_stride: For efficiency, query points are sampled from a random grid
      of this stride.
    max_seg_id: The maxium segment id in the video.
    max_sampled_frac: The maximum fraction of points to sample from each
      object, out of all points that lie on the sampling grid.
    snap_to_occluder: If true, query points within 1 pixel of occlusion 
      boundaries will track the occluding surface rather than the background.
      This results in models which are biased to track foreground objects
      instead of background.  Whether this is desirable depends on downstream
      applications.

  Returns:
    A set of queries, randomly sampled from the video (with a bias toward
      objects), of shape [num_points, 3].  Each point is [t, y, x], where
      t is time.  All points are in pixel/frame coordinates.
    The trajectory for each query point, of shape [num_points, num_frames, 3].
      Each point is [x, y].  Points are in pixel coordinates
    Occlusion flag for each point, of shape [num_points, num_frames].  This is
      a boolean, where True means the point is occluded.

  """
  chosen_points = []
  all_reproj = []
  all_occ = []
  chosen_points_depth = []
  all_reproj_depth = []

  # Convert to metric depth

  depth_range_f32 = tf.cast(depth_range, tf.float32)
  depth_min = depth_range_f32[0]
  depth_max = depth_range_f32[1]
  depth_f32 = tf.cast(depth, tf.float32)
  depth_map = depth_min + depth_f32 * (depth_max-depth_min) / 65535

  surface_normal_map = surface_normals / 65535 * 2. - 1.

  input_size = object_coordinates.shape.as_list()[1:3]
  num_frames = object_coordinates.shape.as_list()[0]

  # We first sample query points within the given window.  That means first
  # extracting the window from the segmentation tensor, because we want to have
  # a bias toward moving objects.
  # Note: for speed we sample points on a grid.  The grid start position is
  # randomized within the window.
  start_vec = [
      tf.random.uniform([], minval=0, maxval=sampling_stride, dtype=tf.int32)
      for _ in range(3)
  ]
  start_vec[1] += window[0]
  start_vec[2] += window[1]
  end_vec = [num_frames, window[2], window[3]]

  def extract_box(x):
    x = x[start_vec[0]::sampling_stride, start_vec[1]:window[2]:sampling_stride,
          start_vec[2]:window[3]:sampling_stride]
    return x

  def erode_segmentations(seg, depth):
    # Mask out points that are near to being occluded by some other object, as
    # measured by nearby depth discontinuities within a 1px radius.
    sz = seg.shape
    pad_depth = tf.pad(
        depth, [(0, 0), (1, 1), (1, 1), (0, 0)], mode='SYMMETRIC'
    )
    invalid = False
    for x in range(0, 3):
      for y in range(0, 3):
        if x == 1 and y == 1:
          continue
        wind_depth = pad_depth[:, y : y + sz[1], x : x + sz[2], :]
        invalid = tf.logical_or(invalid, tf.math.less(wind_depth, depth * 0.95))
    seg = tf.where(invalid, tf.zeros_like(seg) - 1, seg)
    return seg

  if snap_to_occluder:
    segmentations = erode_segmentations(
        tf.cast(segmentations, tf.int32), depth_map
    )

  segmentations_box = extract_box(segmentations)
  object_coordinates_box = extract_box(object_coordinates)

  # Next, get the number of points to sample from each object.  First count
  # how many points are available for each object.

  cnt = tf.math.bincount(
      tf.cast(tf.reshape(segmentations_box, [-1]) + 1, tf.int32)
  )[1:]
  num_to_sample = get_num_to_sample(
      cnt,
      max_seg_id,
      max_sampled_frac,
      tracks_to_sample,
  )
  num_to_sample.set_shape([max_seg_id])
  intrinsics, matrix_world = get_camera_matrices(
      cam_focal_length,
      cam_positions,
      cam_quaternions,
      cam_sensor_width,
      input_size,
      num_frames=num_frames,
  )

  # If the normal map is very rough, it's often because they come from a normal
  # map rather than the mesh.  These aren't trustworthy, and the normal test
  # may fail (i.e. the normal is pointing away from the camera even though the
  # point is still visible).  So don't use the normal test when inferring
  # occlusion.
  trust_sn = True
  sn_pad = tf.pad(surface_normal_map, [(0, 0), (1, 1), (1, 1), (0, 0)])
  shp = surface_normal_map.shape
  sum_thresh = 0
  for i in [0, 2]:
    for j in [0, 2]:
      diff = sn_pad[:, i : shp[1] + i, j : shp[2] + j, :] - surface_normal_map
      diff = tf.reduce_sum(tf.square(diff), axis=-1)
      sum_thresh += tf.cast(diff > 0.05 * 0.05, tf.int32)
  trust_sn = tf.logical_and(trust_sn, (sum_thresh <= 2))[..., tf.newaxis]
  surface_normals_box = extract_box(surface_normal_map)
  trust_sn_box = extract_box(trust_sn)

  def get_camera(fr=None):
    if fr is None:
      return {'intrinsics': intrinsics, 'matrix_world': matrix_world}
    return {'intrinsics': intrinsics[fr], 'matrix_world': matrix_world[fr]}

  # Construct pixel coordinates for each pixel within the window.
  window = tf.cast(window, tf.float32)
  z, y, x = tf.meshgrid(
      *[
          tf.range(st, ed, sampling_stride)
          for st, ed in zip(start_vec, end_vec)
      ],
      indexing='ij')
  pix_coords = tf.reshape(tf.stack([z, y, x], axis=-1), [-1, 3])

  for i in range(max_seg_id):
    # sample points on object i in the first frame.  obj_id is the position
    # within the object_coordinates array, which is one lower than the value
    # in the segmentation mask (0 in the segmentation mask is the background
    # object, which has no bounding box).
    obj_id = i - 1
    mask = tf.equal(tf.reshape(segmentations_box, [-1]), i)
    pt = tf.boolean_mask(tf.reshape(object_coordinates_box, [-1, 3]), mask)
    normals = tf.boolean_mask(tf.reshape(surface_normals_box, [-1, 3]), mask)
    trust_sn_mask = tf.boolean_mask(tf.reshape(trust_sn_box, [-1, 1]), mask)
    idx = tf.cond(
        tf.shape(pt)[0] > 0,
        lambda: tf.multinomial(  # pylint: disable=g-long-lambda
            tf.zeros(tf.shape(pt)[0:1])[tf.newaxis, :],
            tf.gather(num_to_sample, i))[0],
        lambda: tf.zeros([0], dtype=tf.int64))
    # note: pt_coords is pixel coordinates, not raster coordinates.
    pt_coords = tf.gather(tf.boolean_mask(pix_coords, mask), idx)
    normals = tf.gather(normals, idx)
    trust_sn_gather = tf.gather(trust_sn_mask, idx)

    if obj_id == -1:
      # For the background object, no bounding box is available.  However,
      # this doesn't move, so we use the depth map to backproject these points
      # into 3D and use those positions throughout the video.
      pt_3d = []
      pt_coords_reorder = []
      for fr in range(num_frames):
        # We need to loop over frames because we need to use the correct depth
        # map for each frame.
        pt_coords_chunk = tf.boolean_mask(pt_coords,
                                          tf.equal(pt_coords[:, 0], fr))
        pt_coords_reorder.append(pt_coords_chunk)

        pt_3d.append(
            unproject(pt_coords_chunk[:, 1:], get_camera(fr), depth_map[fr]))
      pt = tf.concat(pt_3d, axis=0)
      chosen_points.append(tf.concat(pt_coords_reorder,axis=0))
      bbox = None
      quat = None
      frame_for_pt = None
    else:
      # For any other object, we just use the point coordinates supplied by
      # kubric.
      pt = tf.gather(pt, idx)
      pt = pt / np.iinfo(np.uint16).max - .5
      chosen_points.append(pt_coords)
      # if obj_id>num_objects, then we won't have a box.  We also won't have
      # points, so just use a dummy to prevent tf from crashing.
      bbox = tf.cond(obj_id >= tf.shape(bboxes_3d)[0], lambda: bboxes_3d[0, :],
                     lambda: bboxes_3d[obj_id, :])
      quat = tf.cond(obj_id >= tf.shape(obj_quat)[0], lambda: obj_quat[0, :],
                     lambda: obj_quat[obj_id, :])
      frame_for_pt = pt_coords[..., 0]

    pt_depth = []
    for fr in range(num_frames):
      pt_coords_chunk = tf.boolean_mask(
          pt_coords, tf.equal(pt_coords[:, 0], fr)
      )
      shp = tf.convert_to_tensor(tf.shape(depth_map[fr]))
      idx = pt_coords_chunk[:, 1] * shp[1] + pt_coords_chunk[:, 2]
      pt_depth.append(tf.gather(tf.reshape(depth_map[fr], [-1]), idx))
    chosen_points_depth.append(tf.concat(pt_depth, axis=0))

    # Finally, compute the reprojections for this particular object.
    obj_reproj, obj_occ, reproj_depth = tf.cond(
        tf.shape(pt)[0] > 0,
        functools.partial(
            single_object_reproject,
            bbox_3d=bbox,
            pt=pt,
            pt_segments=i,
            camera=get_camera(),
            cam_positions=cam_positions,
            num_frames=num_frames,
            depth_map=depth_map,
            segments=segmentations,
            window=window,
            input_size=input_size,
            quat=quat,
            normals=normals,
            frame_for_pt=frame_for_pt,
            trust_normals=trust_sn_gather,
        ),
        lambda: (  # pylint: disable=g-long-lambda
            tf.zeros([0, num_frames, 2], dtype=tf.float32),
            tf.zeros([0, num_frames], dtype=tf.bool),
            tf.zeros([0, num_frames], dtype=tf.float32)))
    all_reproj.append(obj_reproj)
    all_occ.append(obj_occ)
    all_reproj_depth.append(reproj_depth)

  # Points are currently in pixel coordinates of the original video.  We now
  # convert them to coordinates within the window frame, and rescale to
  # pixel coordinates.  Note that this produces the pixel coordinates after
  # the window gets cropped and rescaled to the full image size.
  wd = tf.concat(
      [np.array([0.0]), window[0:2],
       np.array([num_frames]), window[2:4]],
      axis=0)
  wd = wd[tf.newaxis, tf.newaxis, :]
  coord_multiplier = [num_frames, input_size[0], input_size[1]]
  all_reproj = tf.concat(all_reproj, axis=0)
  # We need to extract x,y, but the format of the window is [t1,y1,x1,t2,y2,x2]
  window_size = wd[:, :, 5:3:-1] - wd[:, :, 2:0:-1]
  window_top_left = wd[:, :, 2:0:-1]
  all_reproj = (all_reproj - window_top_left) / window_size
  all_reproj = all_reproj * coord_multiplier[2:0:-1]
  all_occ = tf.concat(all_occ, axis=0)
  chosen_points_depth = tf.concat(chosen_points_depth, axis=0)
  all_reproj_depth = tf.concat(all_reproj_depth, axis=0)

  # chosen_points is [num_points, (z,y,x)]
  chosen_points = tf.concat(chosen_points, axis=0)

  if snap_to_occluder:
    # For query points that are near to an occlusion boundary, occasionally
    # jitter the query point onto the occluded object.
    random_perturb = chosen_points[:, 1:] + tf.random.uniform(
        tf.shape(chosen_points[:, 1:]), -1, 2, dtype=tf.int32
    )
    random_perturb = tf.minimum(
        tf.shape(depth_map)[1:3] - 1, tf.maximum(0, random_perturb)
    )
    random_idx = (
        random_perturb[:, 1]
        + random_perturb[:, 0] * tf.shape(depth_map)[2]
        + chosen_points[:, 0] * tf.shape(depth_map)[1] * tf.shape(depth_map)[2]
    )
    chosen_points_idx = (
        chosen_points[:, 1]
        + chosen_points[:, 0] * tf.shape(depth_map)[2]
        + chosen_points[:, 0] * tf.shape(depth_map)[1] * tf.shape(depth_map)[2]
    )
    random_depth = tf.gather(tf.reshape(depth_map, [-1]), random_idx)
    chosen_points_depth = tf.gather(
        tf.reshape(depth_map, [-1]), chosen_points_idx
    )
    swap = tf.logical_and(
        chosen_points_depth < random_depth * 0.95,
        tf.random_uniform(tf.shape(chosen_points_depth)) < 0.5,
    )
    random_perturb = tf.concat([chosen_points[:, 0:1], random_perturb], axis=-1)
    chosen_points = tf.where(
        tf.logical_and(swap[:, tf.newaxis], tf.ones([1, 3], dtype=bool)),
        random_perturb,
        chosen_points,
    )

  pixel_to_raster = tf.constant([0.0, 0.5, 0.5])[tf.newaxis,:]
  chosen_points = tf.cast(chosen_points, tf.float32) + pixel_to_raster

  # renormalize so the box corners are at [-1,1]
  chosen_points = (chosen_points - wd[:, 0, :3]) / (wd[:, 0, 3:] - wd[:, 0, :3])
  chosen_points = chosen_points * coord_multiplier
  # Note: all_reproj is in (x,y) format, but chosen_points is in (z,y,x) format

  all_relative_depth = all_reproj_depth / chosen_points_depth[..., tf.newaxis]

  return tf.cast(chosen_points, tf.float32), tf.cast(all_reproj,
                                                     tf.float32), all_occ, all_relative_depth


def _get_distorted_bounding_box(
    jpeg_shape,
    bbox,
    min_object_covered,
    aspect_ratio_range,
    area_range,
    max_attempts,
):
  """Sample a crop window to be used for cropping."""
  bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
      jpeg_shape,
      bounding_boxes=bbox,
      min_object_covered=min_object_covered,
      aspect_ratio_range=aspect_ratio_range,
      area_range=area_range,
      max_attempts=max_attempts,
      use_image_if_no_bounding_boxes=True)

  # Crop the image to the specified bounding box.
  offset_y, offset_x, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  crop_window = tf.stack(
      [offset_y, offset_x, offset_y + target_height, offset_x + target_width])
  return crop_window


def add_tracks(data,
               train_size=(256, 256),
               vflip=False,
               random_crop=True,
               tracks_to_sample=256,
               sampling_stride=4,
               max_seg_id=25,
               max_sampled_frac=0.1,
               snap_to_occluder=False):
  """Track points in 2D using Kubric data.

  Args:
    data: Kubric data, including RGB/depth/object coordinate/segmentation
      videos and camera parameters.
    train_size: Cropped output will be at this resolution.  Ignored if
      random_crop is False.
    vflip: whether to vertically flip images and tracks (to test generalization)
    random_crop: Whether to randomly crop videos
    tracks_to_sample: Total number of tracks to sample per video.
    sampling_stride: For efficiency, query points are sampled from a random grid
      of this stride.
    max_seg_id: The maxium segment id in the video.
    max_sampled_frac: The maximum fraction of points to sample from each
      object, out of all points that lie on the sampling grid.
    snap_to_occluder: If true, query points within 1 pixel of occlusion 
      boundaries will track the occluding surface rather than the background.
      This results in models which are biased to track foreground objects
      instead of background.  Whether this is desirable depends on downstream
      applications.

  Returns:
    A dict with the following keys:
    query_points:
      A set of queries, randomly sampled from the video (with a bias toward
      objects), of shape [num_points, 3].  Each point is [t, y, x], where
      t is time.  Points are in pixel/frame coordinates.
      [num_frames, height, width].
    target_points:
      The trajectory for each query point, of shape [num_points, num_frames, 3].
      Each point is [x, y].  Points are in pixel/frame coordinates.
    occlusion:
      Occlusion flag for each point, of shape [num_points, num_frames].  This is
      a boolean, where True means the point is occluded.
    video:
      The cropped video, normalized into the range [-1, 1]

  """
  shp = data['video'].shape.as_list()
  num_frames = shp[0]
  if any([s % sampling_stride != 0 for s in shp[:-1]]):
    raise ValueError('All video dims must be a multiple of sampling_stride.')

  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  min_area = 0.3
  max_area = 1.0
  min_aspect_ratio = 0.5
  max_aspect_ratio = 2.0
  if random_crop:
    crop_window = _get_distorted_bounding_box(
        jpeg_shape=shp[1:4],
        bbox=bbox,
        min_object_covered=min_area,
        aspect_ratio_range=(min_aspect_ratio, max_aspect_ratio),
        area_range=(min_area, max_area),
        max_attempts=20)
  else:
    crop_window = tf.constant([0, 0, shp[1], shp[2]],
                              dtype=tf.int32,
                              shape=[4])

  query_points, target_points, occluded, relative_depth = track_points(
      data['object_coordinates'], data['depth'],
      data['metadata']['depth_range'], data['segmentations'],
      data['normal'],
      data['instances']['bboxes_3d'], data['instances']['quaternions'],
      data['camera']['focal_length'],
      data['camera']['positions'], data['camera']['quaternions'],
      data['camera']['sensor_width'], crop_window, tracks_to_sample,
      sampling_stride, max_seg_id, max_sampled_frac, snap_to_occluder)
  video = data['video']

  shp = video.shape.as_list()
  query_points.set_shape([tracks_to_sample, 3])
  target_points.set_shape([tracks_to_sample, num_frames, 2])
  relative_depth.set_shape([tracks_to_sample, num_frames])
  occluded.set_shape([tracks_to_sample, num_frames])

  # Crop the video to the sampled window, in a way which matches the coordinate
  # frame produced the track_points functions.
  start = tf.tensor_scatter_nd_update(
      [0, 0, 0, 0], [[1], [2]], crop_window[0:2]
  )
  size = tf.tensor_scatter_nd_update(
      tf.shape(video), [[1], [2]], crop_window[2:4] - crop_window[0:2]
  )
  video = tf.slice(video, start, size)
  video = tf.image.resize(tf.cast(video, tf.float32), train_size)
  video.set_shape([num_frames, train_size[0], train_size[1], 3])
  if vflip:
    video = video[:, ::-1, :, :]
    target_points = target_points * np.array([1, -1])
    query_points = query_points * np.array([1, -1, 1])
  res = {
      'query_points': query_points,
      'target_points': target_points,
      'relative_depth': relative_depth,
      'occluded': occluded,
      'video': video / (255. / 2.) - 1.,
  }
  return res


def create_point_tracking_dataset(
    train_size=(256, 256),
    shuffle_buffer_size=256,
    split='train',
    batch_dims=tuple(),
    repeat=True,
    vflip=False,
    random_crop=True,
    tracks_to_sample=256,
    sampling_stride=4,
    max_seg_id=25,
    max_sampled_frac=0.1,
    num_parallel_point_extraction_calls=16,
    snap_to_occluder=False,
    **kwargs):
  """Construct a dataset for point tracking using Kubric.

  Args:
    train_size: Tuple of 2 ints. Cropped output will be at this resolution
    shuffle_buffer_size: Int. Size of the shuffle buffer
    split: Which split to construct from Kubric.  Can be 'train' or
      'validation'.
    batch_dims: Sequence of ints. Add multiple examples into a batch of this
      shape.
    repeat: Bool. whether to repeat the dataset.
    vflip: Bool. whether to vertically flip the dataset to test generalization.
    random_crop: Bool. whether to randomly crop videos
    tracks_to_sample: Int. Total number of tracks to sample per video.
    sampling_stride: Int. For efficiency, query points are sampled from a
      random grid of this stride.
    max_seg_id: Int. The maxium segment id in the video.  Note the size of
      the to graph is proportional to this number, so prefer small values.
    max_sampled_frac: Float. The maximum fraction of points to sample from each
      object, out of all points that lie on the sampling grid.
    num_parallel_point_extraction_calls: Int. The num_parallel_calls for the
      map function for point extraction.
    snap_to_occluder: If true, query points within 1 pixel of occlusion 
      boundaries will track the occluding surface rather than the background.
      This results in models which are biased to track foreground objects
      instead of background.  Whether this is desirable depends on downstream
      applications.
    **kwargs: additional args to pass to tfds.load.

  Returns:
    The dataset generator.
  """
  ds = tfds.load(
      'movi_e/256x256',
      data_dir='gs://kubric-public/tfds',
      shuffle_files=shuffle_buffer_size is not None,
      **kwargs)

  ds = ds[split]
  if repeat:
    ds = ds.repeat()
  ds = ds.map(
      functools.partial(
          add_tracks,
          train_size=train_size,
          vflip=vflip,
          random_crop=random_crop,
          tracks_to_sample=tracks_to_sample,
          sampling_stride=sampling_stride,
          max_seg_id=max_seg_id,
          max_sampled_frac=max_sampled_frac,
          snap_to_occluder=snap_to_occluder),
      num_parallel_calls=num_parallel_point_extraction_calls)
  if shuffle_buffer_size is not None:
    ds = ds.shuffle(shuffle_buffer_size)

  for bs in batch_dims[::-1]:
    ds = ds.batch(bs)

  return ds


def plot_tracks(rgb, points, occluded, trackgroup=None):
  """Plot tracks with matplotlib."""
  disp = []
  cmap = plt.cm.hsv

  z_list = np.arange(
      points.shape[0]) if trackgroup is None else np.array(trackgroup)
  # random permutation of the colors so nearby points in the list can get
  # different colors
  z_list = np.random.permutation(np.max(z_list) + 1)[z_list]
  colors = cmap(z_list / (np.max(z_list) + 1))
  figure_dpi = 64

  for i in range(rgb.shape[0]):
    fig = plt.figure(
        figsize=(256 / figure_dpi, 256 / figure_dpi),
        dpi=figure_dpi,
        frameon=False,
        facecolor='w')
    ax = fig.add_subplot()
    ax.axis('off')
    ax.imshow(rgb[i])

    valid = points[:, i, 0] > 0
    valid = np.logical_and(valid, points[:, i, 0] < rgb.shape[2] - 1)
    valid = np.logical_and(valid, points[:, i, 1] > 0)
    valid = np.logical_and(valid, points[:, i, 1] < rgb.shape[1] - 1)

    colalpha = np.concatenate([colors[:, :-1], 1 - occluded[:, i:i + 1]],
                              axis=1)
    # Note: matplotlib uses pixel corrdinates, not raster.
    plt.scatter(
        points[valid, i, 0] - 0.5,
        points[valid, i, 1] - 0.5,
        s=3,
        c=colalpha[valid],
    )

    occ2 = occluded[:, i:i + 1]

    colalpha = np.concatenate([colors[:, :-1], occ2], axis=1)

    plt.scatter(
        points[valid, i, 0],
        points[valid, i, 1],
        s=20,
        facecolors='none',
        edgecolors=colalpha[valid],
    )

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.frombuffer(
        fig.canvas.tostring_rgb(),
        dtype='uint8').reshape(int(height), int(width), 3)
    disp.append(np.copy(img))
    plt.close(fig)

  return np.stack(disp, axis=0)


def main():
  ds = tfds.as_numpy(create_point_tracking_dataset(shuffle_buffer_size=None))
  for i, data in enumerate(ds):
    disp = plot_tracks(data['video'] * .5 + .5, data['target_points'],
                       data['occluded'])
    media.write_video(f'{i}.mp4', disp, fps=10)
    if i > 10:
      break


if __name__ == '__main__':
  main()
