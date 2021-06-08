# --- The container configuration
cat > /tmp/Dockerfile <<EOF
    FROM ubuntu:18.04
    # --- Install system dep
    RUN apt-get update --fix-missing
    # basic dependencies
    RUN apt-get install -y software-properties-common  # has add-apt-repository
    RUN apt-get install -y build-essential  # needed for OpenEXR python package
    RUN apt-get install -y zlib1g-dev       # needed for OpenEXR python package
    RUN apt-get install -y libopenexr-dev   # needed for OpenEXR python package
    # Blender 2.83 from PPA
    # see https://launchpad.net/~thomas-schiex/+archive/ubuntu/blender
    RUN add-apt-repository ppa:thomas-schiex/blender
    RUN apt-get update
    RUN apt-get install -y blender
    # install cmake and npm
    RUN apt-get -y install cmake protobuf-compiler
    RUN apt install -y nodejs
    RUN apt install -y npm
    # RUN npm install -g obj2gltf


    # Blender2.83 uses the system python3.7
    RUN apt-get install -y python3.7-dev
    RUN apt-get install -y python3-pip
    # -- Install Python package dependencies   # TODO: use requirements.txt
    # using force-reinstall to avoid leaking pre-installed packages from python3.6
    # such as numpy, which can lead to strange errors.
    RUN python3.7 -m pip install --upgrade --force-reinstall pip
    RUN python3.7 -m pip install --upgrade --force-reinstall bidict
    RUN python3.7 -m pip install --upgrade --force-reinstall numpy
    RUN python3.7 -m pip install --upgrade --force-reinstall pandas
    RUN python3.7 -m pip install --upgrade --force-reinstall scikit-learn
    RUN python3.7 -m pip install --upgrade --force-reinstall pybullet
    RUN python3.7 -m pip install --upgrade --force-reinstall trimesh
    RUN python3.7 -m pip install --upgrade --force-reinstall google.cloud.storage
    RUN python3.7 -m pip install --upgrade --force-reinstall cloudml-hypertune
    RUN python3.7 -m pip install --upgrade --force-reinstall OpenEXR
    RUN python3.7 -m pip install --upgrade --force-reinstall munch
    RUN python3.7 -m pip install --upgrade --force-reinstall traitlets
    RUN python3.7 -m pip install --upgrade --force-reinstall ipdb
    RUN python3.7 -m pip install --upgrade Image

    # --- build/install manifoldplus, make "manifold" binary available in path
    RUN apt-get install -y git
    WORKDIR /
    RUN git clone https://github.com/hjwdzh/ManifoldPlus.git
    WORKDIR /ManifoldPlus
    RUN git submodule update --init --recursive
    RUN bash compile.sh
    ENV PATH="/ManifoldPlus/build:$PATH"

    # --- copy test data in the folder (reproducibility)
    # COPY data /data
    # RUN mkdir /results 
    # RUN manifold --input /data/bathtub.obj --output /results/bathtub_manifold.obj --depth 8

    # --- interactive setup
    WORKDIR /
    RUN alias ll="ls -l"
EOF

# --- create an image for reuse
CONTAINERTAG="shapenet"
SHAPENETHOME="$HOME/datasets/ShapeNetCore.v2"
ISSANHOME="$HOME/datasets/issam"
docker build -f /tmp/Dockerfile -t $CONTAINERTAG $PWD
docker run --rm --interactive \
    --user $(id -u):$(id -g) \
    --volume "$SHAPENETHOME:/ShapeNetCore.v2" \
    --volume "$ISSANHOME:/data" \
    $CONTAINERTAG \
    manifold --input /data/bathtub.obj --output /results/bathtub_manifold.obj --depth 8
# docker push $CONTAINERTAG