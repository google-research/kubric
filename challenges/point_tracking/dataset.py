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
  projected_pt = coord / shp

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

  return tf.transpose(projections, (1, 0, 2)), tf.transpose(depths)


def estimate_scene_depth_for_point(data, x, y, num_frames):
  """Estimate depth at a (floating point) x,y position.

  We prefer overestimating depth at the point, so we take the max over the 4
  neightoring pixels.

  Args:
    data: depth map. First axis is num_frames.
    x: x coordinate. First axis is num_frames.
    y: y coordinate. First axis is num_frames.
    num_frames: number of frames.

  Returns:
    Depth for each point.
  """
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

  return tf.maximum(tf.maximum(tf.maximum(i1, i2), i3), i4)


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

  return tf.cast(tf.stack(intrinsics),
                 tf.float32), tf.cast(tf.stack(matrix_world), tf.float32)


def single_object_reproject(
    bbox_3d=None,
    pt=None,
    camera=None,
    cam_positions=None,
    num_frames=None,
    depth_map=None,
    window=None,
    input_size=None,
):
  """Reproject points for a single object.

  Args:
    bbox_3d: The object bounding box from Kubric.  If none, assume it's
      background.
    pt: The set of points in 3D, with shape [num_points, 3]
    camera: Camera intrinsic parameters
    cam_positions: Camera positions, with shape [num_frames, 3]
    num_frames: Number of frames
    depth_map: Depth map video for the camera
    window: the window inside which we're sampling points
    input_size: [height, width] of the input images.

  Returns:
    Position for each point, of shape [num_points, num_frames, 2], in pixel
    coordinates, and an occlusion flag for each point, of shape
    [num_points, num_frames].  These are respect to the image frame, not the
    window.

  """
  # Finally, reproject
  reproj, depth_proj = reproject(
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
      tf.less(
          tf.transpose(
              estimate_scene_depth_for_point(depth_map[:, :, :, 0],
                                             tf.transpose(reproj[:, :, 0]),
                                             tf.transpose(reproj[:, :, 1]),
                                             num_frames)), depth_proj * .99))
  obj_occ = occluded
  obj_reproj = reproj

  obj_occ = tf.logical_or(obj_occ, tf.less(obj_reproj[:, :, 1], window[0]))
  obj_occ = tf.logical_or(obj_occ, tf.less(obj_reproj[:, :, 0], window[1]))
  obj_occ = tf.logical_or(obj_occ, tf.greater(obj_reproj[:, :, 1], window[2]))
  obj_occ = tf.logical_or(obj_occ, tf.greater(obj_reproj[:, :, 0], window[3]))
  return obj_reproj, obj_occ


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
    bboxes_3d,
    cam_focal_length,
    cam_positions,
    cam_quaternions,
    cam_sensor_width,
    window,
    tracks_to_sample=256,
    sampling_stride=4,
    max_seg_id=25,
    max_sampled_frac=0.1,
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
    bboxes_3d: The set of all object bounding boxes from Kubric
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

  # Convert to metric depth

  depth_range_f32 = tf.cast(depth_range, tf.float32)
  depth_min = depth_range_f32[0]
  depth_max = depth_range_f32[1]
  depth_f32 = tf.cast(depth, tf.float32)
  depth_map = depth_min + depth_f32 * (depth_max-depth_min) / 65535

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

  segmentations_box = extract_box(segmentations)
  object_coordinates_box = extract_box(object_coordinates)

  # Next, get the number of points to sample from each object.  First count
  # how many points are available for each object.

  cnt = tf.math.bincount(tf.cast(tf.reshape(segmentations_box, [-1]), tf.int32))
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
    idx = tf.cond(
        tf.shape(pt)[0] > 0,
        lambda: tf.multinomial(  # pylint: disable=g-long-lambda
            tf.zeros(tf.shape(pt)[0:1])[tf.newaxis, :],
            tf.gather(num_to_sample, i))[0],
        lambda: tf.zeros([0], dtype=tf.int64))
    pt_coords = tf.gather(tf.boolean_mask(pix_coords, mask), idx)

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
      chosen_points.append(tf.concat(pt_coords_reorder, axis=0))
      bbox = None
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

    # Finally, compute the reprojections for this particular object.
    obj_reproj, obj_occ = tf.cond(
        tf.shape(pt)[0] > 0,
        functools.partial(
            single_object_reproject,
            bbox_3d=bbox,
            pt=pt,
            camera=get_camera(),
            cam_positions=cam_positions,
            num_frames=num_frames,
            depth_map=depth_map,
            window=window,
            input_size=input_size,
        ),
        lambda:  # pylint: disable=g-long-lambda
        (tf.zeros([0, num_frames, 2], dtype=tf.float32),
         tf.zeros([0, num_frames], dtype=tf.bool)))
    all_reproj.append(obj_reproj)
    all_occ.append(obj_occ)

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

  # chosen_points is [num_points, (z,y,x)]
  chosen_points = tf.concat(chosen_points, axis=0)

  chosen_points = tf.cast(chosen_points, tf.float32)

  # renormalize so the box corners are at [-1,1]
  chosen_points = (chosen_points - wd[:, 0, :3]) / (wd[:, 0, 3:] - wd[:, 0, :3])
  chosen_points = chosen_points * coord_multiplier
  # Note: all_reproj is in (x,y) format, but chosen_points is in (z,y,x) format

  return tf.cast(chosen_points, tf.float32), tf.cast(all_reproj,
                                                     tf.float32), all_occ


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
               max_sampled_frac=0.1):
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

  query_points, target_points, occluded = track_points(
      data['object_coordinates'], data['depth'],
      data['metadata']['depth_range'], data['segmentations'],
      data['instances']['bboxes_3d'], data['camera']['focal_length'],
      data['camera']['positions'], data['camera']['quaternions'],
      data['camera']['sensor_width'], crop_window, tracks_to_sample,
      sampling_stride, max_seg_id, max_sampled_frac)
  video = data['video']

  shp = video.shape.as_list()
  query_points.set_shape([tracks_to_sample, 3])
  target_points.set_shape([tracks_to_sample, num_frames, 2])
  occluded.set_shape([tracks_to_sample, num_frames])

  # Crop the video to the sampled window, in a way which matches the coordinate
  # frame produced the track_points functions.
  crop_window = crop_window / (
      np.array(shp[1:3] + shp[1:3]).astype(np.float32) - 1)
  crop_window = tf.tile(crop_window[tf.newaxis, :], [num_frames, 1])
  video = tf.image.crop_and_resize(
      video,
      tf.cast(crop_window, tf.float32),
      tf.range(num_frames),
      train_size,
  )
  if vflip:
    video = video[:, ::-1, :, :]
    target_points = target_points * np.array([1, -1])
    query_points = query_points * np.array([1, -1, 1])
  res = {
      'query_points': query_points,
      'target_points': target_points,
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
    **kwargs):
  """Construct a dataset for point tracking using Kubric: go/kubric.

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
          max_sampled_frac=max_sampled_frac),
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
    plt.scatter(
        points[valid, i, 0],
        points[valid, i, 1],
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
