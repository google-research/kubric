# This Docker Image is meant for development purposes.
# It includes all the dev-requirements needed for testing and for building the docs.
# The kubric package is installed in editable mode (-e) so that changes in /kubric
# immediately affect the installed kubric.
# The intended way to use this image is by mounting a kubric development folder
# from the host over the included sources.
# 
# For example from the kubric directory a bash session can be started as:
# 
# docker run --rm --interactive \
#   --user $(id -u):$(id -g) \
#   --volume "$PWD:/workspace" \
#   --workdir "/workspace" \
#   kubricdockerhub/kubruntudev:latest \
#   /bin/bash

FROM kubricdockerhub/blender:latest

# --- working directory (entered on `docker run`)
WORKDIR /workspace

# --- allows "import kubric" w/o install (via `--volume`, see Makefile)
ENV PYTHONPATH="/workspace"

# --- copy requirements in workdir
COPY requirements.txt .
COPY requirements_full.txt .
COPY requirements_dev.txt .
COPY docs/requirements.txt ./requirements_docs.txt

# --- install python dependencies
RUN pip install --upgrade pip wheel
RUN pip install --upgrade -r requirements.txt
RUN pip install --upgrade -r requirements_full.txt
RUN pip install --upgrade -r requirements_dev.txt
RUN pip install --upgrade -r requirements_docs.txt

# --- cleanup
RUN rm -f requirements.txt 
RUN rm -f requirements_full.txt
RUN rm -f requirements_dev.txt 
RUN rm -f requirements_docs.txt

# --- silences tensorflow
ENV TF_CPP_MIN_LOG_LEVEL="3"
