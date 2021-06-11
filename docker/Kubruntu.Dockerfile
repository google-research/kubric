# --- Image with *pre-installed* Kubric python package
# 
# docker run --rm --interactive \
#   --user $(id -u):$(id -g) \
#   --volume "$PWD:/kubric" \
#   --workdir "/kubric" \
#   kubricdockerhub/kubruntu:latest \
#   python3 examples/helloworld.py

FROM kubricdockerhub/blender:latest

WORKDIR /kubric

# --- Install Python dependencies
COPY requirements.txt .

RUN pip install --upgrade pip wheel && \
    pip install --upgrade --force-reinstall -r requirements.txt && \
    rm -f requirements.txt

# --- Silences tensorflow
ENV TF_CPP_MIN_LOG_LEVEL="3"

# --- Install Kubric
COPY dist/kubric*.whl .

RUN pip3 install `ls kubric*.whl` && \
    rm -f kubric*.whl
