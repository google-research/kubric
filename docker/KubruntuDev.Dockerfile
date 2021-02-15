# This Docker Image is meant for development purposes.
# It includes all the dev-requirements needed for testing and for building the docs.
# The kubric package is installed in editable mode (-e) so that changes in /kubric
# immediately affect the installed kubric.
# The intended way to use this image is by mounting a kubric development folder
# from the host over the included sources.
# For example from the kubric directory a bash session can be started as:
# >>  docker run -v "`pwd`:/kubric" -w "/kubric" --rm -it klausgreff/kubruntudev:latest /bin/bash

FROM klausgreff/kubruntu:latest

WORKDIR /kubric

COPY . .

RUN pip3 install -U pip && \
    pip3 uninstall -y kubric && \
    pip3 install -r requirements_dev.txt && \
    pip3 install -r docs/requirements.txt && \
    pip3 install -e .










