# This Docker Image is meant for development purposes.
# It includes all the dev-requirements needed for testing and for building the docs.
# The kubric package is installed in editable mode (-e) so that changes in /kubric
# immediately affect the installed kubric.
# The intended way to use this image is by mounting a kubric development folder
# from the host over the included sources.
# 
# For example from the kubric directory a bash session can be started as:
# 
# docker run \
#   --rm \
#   --user $(id -u):$(id -g) \
#   --volume "$PWD:/kubric" \
#   --workdir "/kubric" \
#   --interactive \
#   kubricdockerhub/kubruntudev:latest \
#   /bin/bash

FROM kubricdockerhub/blender:latest

# --- Install Python dependencies
COPY requirements.txt .
COPY requirements_dev.txt .
RUN python3 -m ensurepip
RUN pip3 install --upgrade pip wheel
RUN pip3 install --upgrade -r requirements.txt
RUN pip3 install --upgrade -r requirements_dev.txt

# --- Clear temporary
RUN rm -f requirements.txt
RUN rm -f requirements_dev.txt

# --- Silences tensorflow
ENV TF_CPP_MIN_LOG_LEVEL="3"

# --- Allows "import kubric" w/o install
ENV PYTHONPATH="/kubric"