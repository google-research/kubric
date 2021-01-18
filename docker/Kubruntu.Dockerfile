FROM ubuntu:20.04 as build

# LABEL Author="Klaus Greff <klausg@google.com>"
# LABEL Title="Kubruntu"

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

# TODO: make a multi-stage build to separate dependencies needed for
#       compiling from those needed for running kubric
#       (would make for a smaller / more tidy docker image)

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
RUN git clone https://github.com/blender/blender.git --branch blender-v2.83-release --depth 1

RUN mkdir lib && \
    cd lib && \
    svn checkout https://svn.blender.org/svnroot/bf-blender/trunk/lib/linux_centos7_x86_64

RUN cd blender && \
    make update

# Patch to disable statically linking OpenMP when building as a python module
# https://devtalk.blender.org/t/centos-7-manylinux-build-difficulties/15007/5
# and to disable
# https://devtalk.blender.org/t/problem-with-running-blender-as-a-python-module/7367/8
COPY ./docker/openmp_static_patch.txt /blenderpy/blender
RUN cd blender && patch -p1 < openmp_static_patch.txt

RUN cd blender && make -j8 bpy


FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

WORKDIR /

# --- Install package dependencies
# TODO: probably do not need all of them, or at least not in their dev version
RUN apt-get update --yes --fix-missing && \
    apt-get install --yes --quiet --no-install-recommends --reinstall \
      build-essential \
      # for GIF creation
      imagemagick \
      # OpenEXR
      libopenexr-dev \
      git \
      ca-certificates \
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
    rm /python_3.7.9-1_amd64.deb

COPY --from=build /blenderpy/build_linux_bpy/bin/bpy.so /usr/local/lib/python3.7/site-packages
COPY --from=build /blenderpy/lib/linux_centos7_x86_64/python/lib/python3.7/site-packages/2.83 /usr/local/lib/python3.7/site-packages/2.83

# # --- Install Python dependencies
COPY requirements.txt .
RUN python3 -m ensurepip && \
    pip3 install --upgrade pip && \
    pip3 install --upgrade --force-reinstall -r requirements.txt
