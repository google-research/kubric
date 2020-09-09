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

# --- The container configuration
cat > /tmp/Dockerfile <<EOF
  FROM ubuntu:18.04

  # --- Install system dep
  RUN apt-get update --fix-missing

  # basic dependencies
  RUN apt-get install -y software-properties-common  # has add-apt-repository
  RUN apt-get install -y build-essential  # needed for OpenEXR python package
  RUN apt-get install -y zlib1g-dev       # needed for OpenEXR python package
  RUN apt-get install -y libopenexr-dev   # needed for OpenEXR python package

  # Blender 2.83 from PPA
  # see https://launchpad.net/~thomas-schiex/+archive/ubuntu/blender
  RUN add-apt-repository ppa:thomas-schiex/blender
  RUN apt-get update
  RUN apt-get install -y blender

  # Blender2.83 uses the system python3.7
  RUN apt-get install -y python3.7-dev
  RUN apt-get install -y python3-pip

  # -- Install Python package dependencies   # TODO: use requirements.txt
  # using force-reinstall to avoid leaking pre-installed packages from python3.6
  # such as numpy, which can lead to strange errors.
  RUN python3.7 -m pip install --upgrade --force-reinstall pip
  RUN python3.7 -m pip install --upgrade --force-reinstall bidict
  RUN python3.7 -m pip install --upgrade --force-reinstall numpy
  RUN python3.7 -m pip install --upgrade --force-reinstall pandas
  RUN python3.7 -m pip install --upgrade --force-reinstall scikit-learn
  RUN python3.7 -m pip install --upgrade --force-reinstall pybullet
  RUN python3.7 -m pip install --upgrade --force-reinstall trimesh
  RUN python3.7 -m pip install --upgrade --force-reinstall google.cloud.storage
  RUN python3.7 -m pip install --upgrade --force-reinstall cloudml-hypertune
  RUN python3.7 -m pip install --upgrade --force-reinstall OpenEXR
  RUN python3.7 -m pip install --upgrade --force-reinstall munch
  RUN python3.7 -m pip install --upgrade --force-reinstall traitlets
EOF

# --- create an image for reuse
TAG="kubruntu:latest"
docker build -f /tmp/Dockerfile -t $TAG $PWD