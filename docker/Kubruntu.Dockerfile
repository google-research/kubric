# #################################################################################################
# 1. Build Stage
# Compiles python and blender from source
# #################################################################################################

FROM ubuntu:20.04 as build

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8


WORKDIR /blenderpy

# --- Install package dependencies
RUN apt-get update --yes --fix-missing && \
    apt-get install --yes --quiet --no-install-recommends --reinstall \
      # basic dependencies
      build-essential \
      wget \
      curl \
      ca-certificates \
      checkinstall \
      # for GIF creation
      imagemagick \
      # OpenEXR
      libopenexr-dev \
      # blender dependencies
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


# --- Compile Python 3.7 from source (not available anymore in Ubuntu 20.04)

RUN wget -nv https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tar.xz -O Python.tar.xz && \
    tar xJf Python.tar.xz && \
    rm -f Python.tar.xy &&\
    cd Python-3.7.9 && \
    ./configure --enable-optimizations && \
    make install && \
    checkinstall --default


# --- Clone and compile Blender

# RUN git clone https://git.blender.org/blender.git
RUN git clone https://github.com/blender/blender.git --branch blender-v2.91-release --depth 1

RUN mkdir lib && \
    cd lib && \
    svn checkout https://svn.blender.org/svnroot/bf-blender/trunk/lib/linux_centos7_x86_64

RUN cd blender && \
    make update

# Patch to fix segfault on exit problem
# https://developer.blender.org/T82675
# https://developer.blender.org/rB87d3f4aff3225104cbb8be41ac0339c6a1cd9a85
# TODO: remove once we are using Blender 2.92
COPY ./docker/segfault_bug_patch.txt /blenderpy/blender
RUN cd blender && patch -p1 < segfault_bug_patch.txt

# fix an annoying (no-consequence) bpy shutdown error
# see https://github.com/google-research/kubric/issues/65
COPY ./docker/cycles_free_patch.txt /blenderpy/blender
RUN cd blender && patch -p1 < /blenderpy/blender/cycles_free_patch.txt


RUN cd blender && make -j8 bpy


# #################################################################################################
# Final Stage
# Installs previously built python and bpy module along with other dependencies in a fresh image.
# This two stage process makes the final image much smaller because it doesn't include the files
# and dependencies of the build process.
# #################################################################################################


FROM ubuntu:20.04

LABEL Author="Klaus Greff <klausg@google.com>"
LABEL Title="Kubruntu"

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

WORKDIR /kubric

# --- Install package dependencies
# TODO: probably do not need all of them, or at least not in their dev version
RUN apt-get update --yes --fix-missing && \
    apt-get install --yes --quiet --no-install-recommends --reinstall \
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

COPY --from=build /blenderpy/Python-3.7.9/python_3.7.9-1_amd64.deb .
RUN dpkg -i python_3.7.9-1_amd64.deb && \
    rm -f python_3.7.9-1_amd64.deb

COPY --from=build /blenderpy/build_linux_bpy/bin/bpy.so /usr/local/lib/python3.7/site-packages
COPY --from=build /blenderpy/lib/linux_centos7_x86_64/python/lib/python3.7/site-packages/2.91 /usr/local/lib/python3.7/site-packages/2.91

# # --- Install Python dependencies
COPY requirements.txt .
RUN python3 -m ensurepip && \
    pip3 install --upgrade pip wheel && \
    pip3 install --upgrade --force-reinstall -r requirements.txt && \
    rm -f requirements.txt

# # --- Install Kubric
COPY dist/kubric*.whl .
RUN pip3 install `ls kubric*.whl` && \
    rm -f kubric*.whl
