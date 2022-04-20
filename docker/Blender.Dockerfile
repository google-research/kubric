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

FROM ubuntu:21.10 as build

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

WORKDIR /blenderpy

# --- Install package dependencies
RUN apt-get update --yes --fix-missing && \
    apt-get install --yes --quiet --no-install-recommends \
      python3.10-dev \
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
      subversion \
      python3-pip \
      libopenvdb-dev \
      libopenvdb-doc \
      libopenvdb-tools \
      libopenvdb7.1 \
      python3-openvdb


# make python3.10 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 10 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 10

# --- Clone and compile Blender

# RUN git clone https://git.blender.org/blender.git
RUN git clone https://github.com/blender/blender.git --tag v3.1.2 --depth 1
RUN mv v3.1.2 blender

RUN mkdir lib && \
    cd lib && \
    svn checkout https://svn.blender.org/svnroot/bf-blender/trunk/lib/linux_centos7_x86_64

RUN cd blender && \
    make update

# fix an annoying (no-consequence) bpy shutdown error
# see https://github.com/google-research/kubric/issues/65
COPY ./docker/cycles_free_patch.txt /blenderpy/blender
RUN cd blender && patch -p1 < /blenderpy/blender/cycles_free_patch.txt

RUN cd blender && BUILD_CMAKE_ARGS="-D WITH_OPENVDB=ON" make -j8 bpy

# #################################################################################################
# Stage 2
# #################################################################################################


FROM ubuntu:21.10

LABEL Author="kubric-team <kubric@google.com>"
LABEL Title="Blender"

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

# --- Install package dependencies
# TODO: probably do not need all of them, or at least not in their dev version
RUN apt-get update --yes --fix-missing && \
    apt-get install --yes --quiet --no-install-recommends --reinstall \
      python3.10-dev \
      python3.10-distutils \
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
      uuid-dev \
      python3-pip \
      libopenvdb-dev \
      libopenvdb-doc \
      libopenvdb-tools \
      libopenvdb7.1 \
      python3-openvdb

# make python3.10 the default python and python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 10 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 10

# install pip for python 3.10
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# install bpy module within python3.10 
COPY --from=build /blenderpy/build_linux_bpy/bin/bpy.so /usr/local/lib/python3.10/dist-packages/
COPY --from=build /blenderpy/lib/linux_centos7_x86_64/python/lib/python3.10/site-packages/3.2 /usr/local/lib/python3.10/dist-packages/3.2

# obtain by combining 2 series of paths:
# first:
#    python3 -c 'import sys; print(sys.path)'
# last:
#    find /usr -name site-packages
ENV PYTHONPATH=/usr/lib/python310.zip:/usr/lib/python3.10:/usr/lib/python3.10/lib-dynload:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages:/usr/lib/python3.10/site-packages
