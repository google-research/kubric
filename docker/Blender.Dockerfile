# Compiles a docker image for blender w/ "import bpy support"
# 
# Compilation happens in two stages:
# 1) Compiles blender from source.
# 2) Installs previously built bpy module along with other dependencies in a fresh image.
# This two stage process reduces the size of the final image because it doesn't include
# the files and dependencies of the build process.

# #################################################################################################
# Stage 1
# #################################################################################################

FROM ubuntu:20.04 as build

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

WORKDIR /blenderpy

# --- Install package dependencies
RUN apt-get update --yes --fix-missing && \
    apt-get install --yes --quiet --no-install-recommends \
      python3.9-dev \
      build-essential \
      ca-certificates \
      libopenexr-dev \
      cmake \
      git \
      libffi-dev \
      libssl-dev \
      libx11-dev \
      libxxf86vm-dev \
      libxcursor-dev \
      libxi-dev \
      libxrandr-dev \
      libxinerama-dev \
      libglew-dev \
      subversion

# make python3.9 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 10 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 10

# --- Clone and compile Blender

# RUN git clone https://git.blender.org/blender.git
RUN git clone https://github.com/blender/blender.git --branch blender-v2.93-release --depth 1

RUN mkdir lib && \
    cd lib && \
    svn checkout https://svn.blender.org/svnroot/bf-blender/trunk/lib/linux_centos7_x86_64

RUN cd blender && \
    make update

# fix an annoying (no-consequence) bpy shutdown error
# see https://github.com/google-research/kubric/issues/65
COPY ./docker/cycles_free_patch.txt /blenderpy/blender
RUN cd blender && patch -p1 < /blenderpy/blender/cycles_free_patch.txt


RUN cd blender && make -j8 bpy

# #################################################################################################
# Stage 2
# #################################################################################################


FROM ubuntu:20.04

LABEL Author="kubric-team <kubric@google.com>"
LABEL Title="Blender"

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

# --- Install package dependencies
# TODO: probably do not need all of them, or at least not in their dev version
RUN apt-get update --yes --fix-missing && \
    apt-get install --yes --quiet --no-install-recommends --reinstall \
      python3.9-dev \
      python3.9-distutils \
      build-essential \
      # for GIF creation
      imagemagick \
      # OpenEXR
      libopenexr-dev \
      curl \
      ca-certificates \
      git \
      libffi-dev \
      libssl-dev \
      libx11-dev \
      libxxf86vm-dev \
      libxcursor-dev \
      libxi-dev \
      libxrandr-dev \
      libxinerama-dev \
      libglew-dev \
      zlib1g-dev \
      # further (optional) python build dependencies
      libbz2-dev \
      libgdbm-dev \
      liblzma-dev \
      libncursesw5-dev \
      libreadline-dev \
      libsqlite3-dev \
      #tk-dev \  # installs libpng-dev which leads to blender linking errors
      uuid-dev

# make python3.9 the default python and python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 10 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 10

# install pip for python 3.9
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.9 get-pip.py && \
    rm get-pip.py

# install bpy module within python3.9 
COPY --from=build /blenderpy/build_linux_bpy/bin/bpy.so /usr/local/lib/python3.9/dist-packages/
COPY --from=build /blenderpy/lib/linux_centos7_x86_64/python/lib/python3.9/site-packages/2.93 /usr/local/lib/python3.9/dist-packages/2.93