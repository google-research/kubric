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

# Copyright 2021 The Kubric Authors
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

"""Directory of problematic shapenet models."""
import pathlib

__shapenet_set__ = {
  '03046257/5972bc07e59371777bcb070cc655f13a',  # map_Kd ../ AND map_d ../
  '03001627/c70c1a6a0e795669f51f77a6d7299806',  # map_Kd ../
  '03001627/f3c0ab68f3dab6071b17743c18fb63dc',  # map_Kd ../
  '03001627/a8c0ceb67971d0961b17743c18fb63dc',  # map_Kd ../
  '03001627/2ae70fbab330779e3bff5a09107428a5',  # map_Kd ../
  '04256520/191c92adeef9964c14038d588fd1342f',  # map_Kd ../
  # --- missing files
  '02958343/3c33f9f8edc558ce77aa0b62eed1492',   # folder only contains textures?
  '02958343/986ed07c18a2e5592a9eb0f146e94477',  # folder only contains textures?
  '02958343/5bf2d7c2167755a72a9eb0f146e94477',  # folder only contains textures?
  '02958343/9fb1d03b22ecac5835da01f298003d56',  # completely empty
  '02958343/207e69af994efa9330714334794526d4',  # completely empty
  '02958343/d6ee8e0a0b392f98eb96598da750ef34',  # folder only contains textures?
  '02958343/302612708e86efea62d2c237bfbc22ca',  # folder only contains textures?
  '02958343/f5bac2b133a979c573397a92d1662ba5',  # completely empty
  '02958343/806d740ca8fad5c1473f10e6caaeca56',  # folder only contains textures?
  '02958343/8070747805908ae62a9eb0f146e94477',  # folder only contains textures?
  '02958343/2307b51ca7e4a03d30714334794526d4',  # folder only contains textures?
  '02958343/407f2811c0fe4e9361c6c61410fc904b',  # folder only contains textures?
  '02958343/4ddef66f32e1902d3448fdcb67fe08ff',  # completely empty
  '02958343/ea3f2971f1c125076c4384c3b17a86ea',  # folder only contains textures?
  '02958343/5973afc979049405f63ee8a34069b7c5',  # completely empty
  '02958343/7aa9619e89baaec6d9b8dfa78596b717',  # folder only contains textures?
  '02958343/e6c22be1a39c9b62fb403c87929e1167',  # folder only contains textures?
  '02958343/93ce8e230939dfc230714334794526d4',  # folder only contains textures?
  '02958343/3ffeec4abd78c945c7c79bdb1d5fe365',  # folder only contains textures?
  # --- missing textures
  '02691156/d583d6f23c590f3ec672ad25c77a396',   # missing texture6.png',
  '03001627/2b90701386f1813052db1dda4adf0a0c',  # missing texture0',
  '03001627/808fa82fe9ad86d9f1cc184b6fa3e1f9',  # missing texture0',
  '03001627/7ad134826824de98d0bef5e87b92b95e',  # missing Wood_Floor_Light.jpg',
  '03001627/941720989a7af0248b500dd30d6dfd0',   # missing images/texture0.jpg',
  '04401088/89d70d3e0c97baaa859b0bef8825325f',  # missing texture0',
  '02924116/154eeb504f4ac096481aa8b5531c68a9',  # missing texture4',
  '02958343/61f4cd45f477fc7a48a1f672d5ac8560',  # missing texture0',
  '02958343/ec67edc59aef93d9f5274507f44ab711',  # missing texture0.jpg',
  '02958343/a262c2044977b6eb52ab7aae4be20d81',  # missing texture0',
  '02958343/e2ceb9bf23b498dda7431386d9d22644',  # missing texture0',
  '02958343/846f4ad1db06d8791e0b067dee925db4',  # missing texture0',
  '02958343/685f2b388b018ab78cab9eeff9aeaee2',  # missing texture0',
  '02958343/a1d85821a0666a4d8dc995728b1ad443',  # missing texture0.JPG',
  '02958343/558404e6c17c58997302a5e36ce363de',  # missing texture0.jpg',
  '02958343/fe3dc721f5026196d61b6a34f3fd808c',  # missing texture0',
  '02958343/98a4518ee8e706c94e84ac3ac08acdb2',  # missing texture0',
  '02958343/8242b114695b68286f522b2bb8ded829',  # missing texture0',
  '02958343/1c66dbe15a6be89a7bfe055aeab68431',  # missing textures',
  '02958343/731efc7a52841a5a59139efcde1fedcb',  # missing texture0',
  '02958343/8242b114695b68286f522b2bb8ded829',  # missing texture0',
  '04530566/6367d10f3cb043e1cdcba7385a96c2c8',  # missing texture2',
  '02958343/648ceaad362345518a6cf8c6b92417f2',  # missing texture0, texture1',
  '02958343/e95d4b7aa9617eb05c58fd6a60e080a',   # missing texture0, texture1',
  '02958343/39b307361b650db073a425eed3ac7a0b',  # missing texture0 texture1',
  '02958343/85914342038de6f160190e29962cb3e7',  # missing texture0 texture1',
  '02958343/d5f1637a5c9479e0185ce5d54f27f6b9',  # missing texture0 texture1',
  '03001627/b7a1ec97b8f85127493a4a2a112261d3',  # missing texture0
  '02924116/2d44416a2f00fff08fd1fd158db1677c',  # missing texture 4
  # --- missing MANY textures
  '02958343/b3ffbbb2e8a5376d4ed9aac513001a31',  # missing many textures',
  '02958343/f6bbb767b1b75ab0c9d22fcb1abe82ed',  # missing many textures',
  '02958343/bb7fec347b2b57498747b160cf027f1',   # missing many textures',
  '02958343/2854a948ff81145d2d7d789814cae761',  # missing many textures',
  '03001627/482afdc2ddc5546f764d42eddc669b23',  # missing many textures',
  '02958343/373cf6c8f79376482d7d789814cae761',  # missing many textures',
  '02958343/15fcfe91d44c0e15e5c9256f048d92d2',  # missing many textures',
  '02958343/f11d669c1c57a595cba0d757b1f2aa52',  # missing many textures',
  '02958343/66be76d60405c58ae02ca9d4b3cbc724',  # missing many textures',
  '02958343/db14f415a203403e2d7d789814cae761',  # missing many textures',
  '02958343/662cd6478fa3dd482d7d789814cae761',  # missing many textures',
}


def _suffix_directory(key: pathlib.Path):
  """Converts '/folder/.../folder/folder/folder' into 'folder/folder'"""
  key = pathlib.Path(key)
  shapenet_folder = key.parent.parent
  key = key.relative_to(shapenet_folder)
  return key


def invalid_model(key: pathlib.Path):
  key = _suffix_directory(key)
  return str(key) in __shapenet_set__
