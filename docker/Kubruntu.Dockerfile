# --- Image with *pre-installed* Kubric python package
# 
# docker run --rm --interactive \
#   --user $(id -u):$(id -g) \
#   --volume "$PWD:/kubric" \
#   --workdir "/kubric" \
#   kubricdockerhub/kubruntu:latest \
#   python3 examples/helloworld.py

FROM kubricdockerhub/blender:blender312

WORKDIR /kubric

ENV PYTHONPATH=/usr/lib/python310.zip:/usr/lib/python3.10:/usr/lib/python3.10/lib-dynload:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages:/usr/lib/python3.10/site-packages

# --- copy requirements in workdir
COPY requirements.txt .
COPY requirements_full.txt .

# --- install python dependencies
RUN pip install --upgrade pip wheel setuptools packaging
RUN pip install --upgrade -r requirements_full.txt -r requirements.txt

# --- cleanup
RUN rm -f requirements.txt
RUN rm -f requirements_full.txt

# --- Silences tensorflow
ENV TF_CPP_MIN_LOG_LEVEL="3"

ADD . /src
RUN cd /src && python3 setup.py sdist bdist_wheel

# --- Install Kubric
RUN pip3 install `ls /src/dist/kubric*.whl`
RUN rm -rf /src
