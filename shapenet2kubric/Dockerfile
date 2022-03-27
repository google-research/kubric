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

FROM kubricdockerhub/kubruntu:latest

# --- Install system dep
RUN apt-get update --fix-missing
RUN apt-get -y install cmake
RUN apt-get -y install wget

# --- install obj2gltf (REQUIRED)
RUN apt install -y nodejs
RUN apt install -y npm
RUN npm install -g npm@6.9.0
RUN npm install -g obj2gltf@3.1.0

## -- Install python package dependencies
## using force-reinstall to avoid leaking pre-installed packages from python3.6
## such as numpy, which can lead to strange errors.
RUN python -m pip install --upgrade pip
RUN python -m pip install --upgrade numpy
RUN python -m pip install --upgrade pybullet
RUN python -m pip install --upgrade trimesh
RUN python -m pip install --upgrade Image
RUN python -m pip install --upgrade tqdm

# --- build/install manifoldplus, make "manifold" binary available in path
RUN apt-get update --fix-missing
RUN apt-get install -y git
WORKDIR /
RUN git clone https://github.com/hjwdzh/ManifoldPlus.git
WORKDIR /ManifoldPlus
RUN git submodule update --init --recursive
RUN bash compile.sh
ENV PATH="/ManifoldPlus/build:$PATH"

# --- final startup folder
WORKDIR /shapenet2kubric