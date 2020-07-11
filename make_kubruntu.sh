# --- The container configuration
cat > /tmp/Dockerfile <<EOF
  FROM ubuntu:18.04

  # --- Install system deps
  RUN apt-get update
  RUN apt-get install -y wget
  RUN apt-get install -y xz-utils
  RUN apt-get install -y blender
  RUN apt-get remove -y blender #< we only need its dependencies
  
  # --- Install blender
  WORKDIR /
  RUN wget https://download.blender.org/release/Blender2.83/blender-2.83.2-linux64.tar.xz
  RUN tar -xvf blender-2.83.2-linux64.tar.xz
  RUN rm -f blender-2.83.2-linux64.tar.xz
  
  # --- Update path (expose blender and its integrated python3.7m python)
  ENV PATH="/blender-2.83.2-linux64:${PATH}"

  # --- Install blender REPL dependencies
  RUN /blender-2.83.2-linux64/2.83/python/bin/python3.7m -m ensurepip
  RUN /blender-2.83.2-linux64/2.83/python/bin/python3.7m -m pip install --upgrade pip
  RUN /blender-2.83.2-linux64/2.83/python/bin/python3.7m -m pip install trimesh
  RUN /blender-2.83.2-linux64/2.83/python/bin/python3.7m -m pip install google.cloud.storage
EOF

# --- create an image for reuse
TAG="kubruntu:latest"
docker build -f /tmp/Dockerfile -t $TAG $PWD