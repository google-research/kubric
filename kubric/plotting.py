# Copyright 2021 The Kubric Authors.
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

import colorsys
import numpy as np


def hls_palette(n_colors, first_hue=0.01, lightness=.5, saturation=.7):
  """Get a list of colors where the first is black and the rest are evenly spaced in HSL space."""
  hues = np.linspace(0, 1, int(n_colors) + 1)[:-1]
  hues = (hues + first_hue) % 1
  palette = [(0., 0., 0.)] + [colorsys.hls_to_rgb(h_i, lightness, saturation) for h_i in hues]
  return np.round(np.array(palette) * 255).astype(np.uint8)


def get_image_plot(width, height, nrows=1, ncols=1, display_dpi=1):
  import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

  fig, axes = plt.subplots(figsize=(width*ncols, height*nrows), nrows=nrows, ncols=ncols,
                           sharex=True, sharey=True, dpi=display_dpi)
  if nrows == ncols == 1:
    ax_list = [axes]
  else:
    ax_list = axes.flatten()
  for ax in ax_list:
    ax.set_axis_off()
    ax.set_xlim((0, width-1))
    ax.set_ylim((height-1, 0))
  fig.tight_layout()
  plt.subplots_adjust(hspace=0, wspace=0)
  return fig, axes


def plot_image(rgb, ax=None):
  if ax is None:
    _, ax = get_image_plot(rgb.shape[1], rgb.shape[0])

  ax.imshow(rgb, interpolation="nearest")


def plot_depth(depth, ax=None, depth_range=None, colormap="inferno_r"):
  if ax is None:
    _, ax = get_image_plot(depth.shape[1], depth.shape[0])
  if depth_range is None:
    depth_range = np.min(depth), np.max(depth)
  ax.imshow((depth - depth_range[0]) / (depth_range[1] - depth_range[0]), cmap=colormap,
            interpolation="nearest")


def plot_uv(uv, ax=None):
  if ax is None:
    _, ax = get_image_plot(uv.shape[1], uv.shape[0])
  ax.imshow(uv, interpolation="nearest")


def plot_segmentation(seg, ax=None, palette=None, num_objects=None):
  import seaborn as sns  # pylint: disable=import-outside-toplevel

  if ax is None:
    _, ax = get_image_plot(seg.shape[1], seg.shape[0])
  if num_objects is None:
    num_objects = np.max(seg)  # assume consecutive numbering
  num_objects += 1  # background
  if palette is None:
    palette = [(0., 0., 0.)] + sns.color_palette("hls", num_objects-1)

  seg_img = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.float32)
  for i in range(num_objects):
    seg_img[np.nonzero(seg[:, :, 0] == i)] = palette[i]
  ax.imshow(seg_img, interpolation="nearest")


def plot_flow(vec, ax=None, flow_mag_range=None):
  import matplotlib  # pylint: disable=import-outside-toplevel
  if ax is None:
    _, ax = get_image_plot(vec.shape[1], vec.shape[0])
  direction = (np.arctan2(vec[:, :, 0], vec[:, :, 1]) + np.pi) / (2 * np.pi)
  norm = np.linalg.norm(vec, axis=-1)
  if flow_mag_range is None:
    flow_mag_range = norm.min(), norm.max()
  magnitude = np.clip((norm - flow_mag_range[0]) / (flow_mag_range[1] - flow_mag_range[0]),
                      0., 1.)
  saturation = np.ones_like(direction)
  hsv = np.stack([direction, saturation, magnitude], axis=-1)
  rgb = matplotlib.colors.hsv_to_rgb(hsv)
  ax.imshow(rgb, interpolation="nearest")


def plot_normal(norm, ax=None):
  if ax is None:
    _, ax = get_image_plot(norm.shape[1], norm.shape[0])
  norm = norm / 2 + 0.5
  ax.imshow(norm, interpolation="nearest")


def plot_bboxes(seg, ax=None, linewidth=100, num_objects=None, palette=None):
  import seaborn as sns  # pylint: disable=import-outside-toplevel
  import matplotlib  # pylint: disable=import-outside-toplevel

  if ax is None:
    _, ax = get_image_plot(seg.shape[1], seg.shape[0])
  if num_objects is None:
    num_objects = np.max(seg)  # assume consecutive numbering
  num_objects += 1  # background
  if palette is None:
    palette = [(0., 0., 0.)] + sns.color_palette("hls", num_objects)
  seg = seg[:, :, 0]
  for i in range(1, num_objects):
    idxs = np.array(np.where(seg == i), dtype=np.float32)
    if idxs.size > 0:
      miny, minx, maxy, maxx = idxs[0].min()-1, idxs[1].min()-1, idxs[0].max(), idxs[1].max()
      rect = matplotlib.patches.Rectangle([minx, miny], maxx-minx, maxy-miny,
                                          linewidth=linewidth, edgecolor=palette[i],
                                          facecolor="none")
      ax.add_patch(rect)


def plot_center_of_mass(objects, ax, frames=slice(None, None), palette=None):
  import seaborn as sns  # pylint: disable=import-outside-toplevel
  num_objects = len(objects) + 1  # background
  if palette is None:
    palette = [(0., 0., 0.)] + sns.color_palette("hls", num_objects)
  for k, obj in enumerate(objects):
    x, y = obj["image_positions"][frames].T
    ax.scatter(x, y, marker="X", s=50000, color=palette[k+1])


def plot_object_collisions(collisions, ax, frame, num_objects=None, palette=None):
  import seaborn as sns  # pylint: disable=import-outside-toplevel
  if num_objects is None:
    num_objects = np.max([c["instances"] for c in collisions]) + 1  # background
  if palette is None:
    palette = [(0., 0., 0.)] + sns.color_palette("hls", num_objects)
  coll = [c for c in collisions
          if (-1 not in c["instances"]) and (frame-0.5 <= c["frame"] <= frame+0.5)]
  for c in coll:
    ax.scatter(x=c["image_position"][0], y=c["image_position"][1], s=c["force"]*1000,
               marker="*", facecolors=palette[c["instances"][0]+1],
               edgecolors=palette[c["instances"][1]+1], linewidth=50)


def plot_ground_collisions(collisions, ax, frame):
  coll = [c for c in collisions
          if (-1 in c["instances"]) and (frame-0.5 <= c["frame"] <= frame+0.5)]
  for c in coll:
    ax.scatter(x=c["image_position"][0], y=c["image_position"][1], s=c["force"]*1000,
               marker=".", facecolors="k")
