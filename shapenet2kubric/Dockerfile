FROM kubricdockerhub/blender:latest

# --- Install system dep
RUN apt-get update --fix-missing
RUN apt-get -y install cmake
RUN apt-get -y install wget

# --- install obj2gltf (REQUIRED)
RUN apt install -y nodejs
RUN apt install -y npm
RUN npm install -g npm@6.9.0
RUN npm install -g obj2gltf@3.1.0

# --- Blender2.83 uses the system python3.7
#RUN apt-get install -y python3.7-dev
#RUN apt-get install -y python3-pip
#
## -- Install python package dependencies
## using force-reinstall to avoid leaking pre-installed packages from python3.6
## such as numpy, which can lead to strange errors.
#RUN python -m ensurepip
RUN python -m pip install --upgrade --force-reinstall pip
RUN python -m pip install --upgrade --force-reinstall numpy
RUN python -m pip install --upgrade --force-reinstall pybullet
RUN python -m pip install --upgrade --force-reinstall trimesh
RUN python -m pip install --upgrade --force-reinstall Image
RUN python -m pip install --upgrade --force-reinstall tqdm

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