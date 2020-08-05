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

  # --- Install system deps
  RUN apt-get update
  RUN apt-get install -y wget
  RUN apt-get install -y xz-utils
  RUN apt-get install -y blender
  RUN apt-get remove -y blender #< we only need its dependencies
  
  # --- Install blender
  WORKDIR /
  RUN wget https://download.blender.org/release/Blender2.83/blender-2.83.2-linux64.tar.xz
  RUN tar -xvf blender-2.83.2-linux64.tar.xz
  RUN rm -f blender-2.83.2-linux64.tar.xz
  
  # --- Update path (expose blender and its integrated python3.7m python)
  ENV PATH="/blender-2.83.2-linux64:${PATH}"

  # --- Install blender REPL dependencies
  RUN /blender-2.83.2-linux64/2.83/python/bin/python3.7m -m ensurepip
  RUN /blender-2.83.2-linux64/2.83/python/bin/python3.7m -m pip install --upgrade pip
  RUN /blender-2.83.2-linux64/2.83/python/bin/python3.7m -m pip install trimesh
  RUN /blender-2.83.2-linux64/2.83/python/bin/python3.7m -m pip install google.cloud.storage
  RUN /blender-2.83.2-linux64/2.83/python/bin/python3.7m -m pip install cloudml-hypertune
  RUN /blender-2.83.2-linux64/2.83/python/bin/python3.7m -m pip install pybullet
  RUN /blender-2.83.2-linux64/2.83/python/bin/python3.7m -m pip install scikit-learn
  RUN /blender-2.83.2-linux64/2.83/python/bin/python3.7m -m pip install bidict
  RUN /blender-2.83.2-linux64/2.83/python/bin/python3.7m -m pip install OpenEXR

EOF

# --- create an image for reuse
TAG="kubruntu:latest"
docker build -f /tmp/Dockerfile -t $TAG $PWD