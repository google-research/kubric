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

from unittest import mock

from kubric.core.scene import Scene
from kubric.core.cameras import PerspectiveCamera
from kubric.core import assets
from kubric.core import objects
from kubric.core import materials
from kubric.core import view

from kubric.core.color import get_color

def test_empty_scene():
  _ = Scene()

def test_scene_constructor():
  cam = PerspectiveCamera()
  red = get_color("red")
  blue = get_color("blue")
  scene = Scene(frame_start=6, frame_end=8, frame_rate=27, step_rate=270,
                resolution=(123, 456), gravity=(1, 2, 3), camera=cam,
                ambient_illumination=red, background=blue)
  assert scene.frame_start == 6
  assert scene.frame_end == 8
  assert scene.step_rate == 270
  assert scene.resolution == (123, 456)
  assert all(scene.gravity == (1, 2, 3))
  assert scene.camera == cam
  assert scene.ambient_illumination == red
  assert scene.background == blue

  assert scene.assets == (cam,)
  assert scene.views == ()


def test_add_asset_after_linking_views():
  scene = Scene()
  asset = assets.Asset()
  view1 = mock.Mock(view.View)
  view2 = mock.Mock(view.View)
  scene.link_view(view1)
  scene.link_view(view2)

  assert view1 in scene.views
  assert view2 in scene.views
  view1.add.assert_not_called()
  view2.add.assert_not_called()

  scene.add(asset)

  assert asset in scene.assets
  assert scene in asset.scenes
  view1.add.assert_called_once_with(asset)
  view2.add.assert_called_once_with(asset)


def test_add_asset_before_linking_views():
  scene = Scene()
  asset = assets.Asset()
  scene.add(asset)
  assert asset in scene.assets
  assert scene in asset.scenes

  view1 = mock.Mock(view.View)
  view2 = mock.Mock(view.View)
  scene.link_view(view1)
  scene.link_view(view2)

  assert view1 in scene.views
  assert view2 in scene.views
  view1.add.assert_called_once_with(asset)
  view2.add.assert_called_once_with(asset)


def test_add_asset_multiple_times_ignored():
  scene = Scene()
  view1 = mock.Mock(view.View)
  scene.link_view(view1)
  asset = assets.Asset()

  # call three times
  scene.add(asset)
  scene.add(asset)
  scene.add(asset)

  # but make sure it is only added once
  assert scene.assets == (asset,)
  assert asset.scenes == [scene]
  view1.add.assert_called_once_with(asset)


def test_add_asset_multi_scene():
  scene1 = Scene()
  scene2 = Scene()

  asset = assets.Asset()
  scene1.add(asset)
  scene2.add(asset)

  assert asset in scene1.assets
  assert asset in scene2.assets
  assert scene1 in asset.scenes
  assert scene2 in asset.scenes


def test_recursive_asset_adding():
  scene1 = Scene()
  scene2 = Scene()
  view1 = mock.Mock(view.View)
  scene1.link_view(view1)

  asset = objects.PhysicalObject()
  scene1.add(asset)
  scene2.add(asset)

  view1.add.assert_called_once_with(asset)
  view1.add.reset_mock()

  mat = materials.FlatMaterial()
  asset.material = mat

  assert asset.material == mat
  assert mat in scene1.assets
  assert mat in scene2.assets
  view1.add.assert_called_once_with(mat)