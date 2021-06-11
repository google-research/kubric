# This Docker Image is meant for development purposes.
# It includes all the dev-requirements needed for testing and for building the docs.
# The kubric package is installed in editable mode (-e) so that changes in /kubric
# immediately affect the installed kubric.
# The intended way to use this image is by mounting a kubric development folder
# from the host over the included sources.
# 
# For example from the kubric directory a bash session can be started as:
# 
# docker run --rm --interactive
#   --user $(id -u):$(id -g) \
#   --volume "$PWD:/kubric" \
#   --workdir "/kubric" \
#   kubricdockerhub/kubruntudev:latest \
#   /bin/bash

FROM kubricdockerhub/blender:latest

WORKDIR /kubric

# --- copy requirements in workdir
COPY requirements.txt .
COPY requirements_dev.txt .
COPY docs/requirements.txt ./requirements_docs.txt

# --- Install Python dependencies
RUN pip install --upgrade pip wheel && \
    pip install --upgrade -r requirements.txt && \
    pip install --upgrade -r requirements_dev.txt && \
    pip install --upgrade -r requirements_docs.txt && \
    rm -f requirements.txt requirements_dev.txt requirements_docs.txt

# --- Silences tensorflow
ENV TF_CPP_MIN_LOG_LEVEL="3"

# --- Allows "import kubric" w/o install
ENV PYTHONPATH="/kubric"
