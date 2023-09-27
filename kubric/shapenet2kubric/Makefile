# Copyright 2021 The Kubric Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2021 The Kubric Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Scripts and containers to convert ShapeNetCore.v2 into Kubric-friendly assets.
# WARNING: the commands within this makefile need to be executed within the shapenet2kubric folder.

DOWNLOADPREFIX := $(HOME)/datasets
SHAPENETSRC := $(DOWNLOADPREFIX)/ShapeNetCore.v2
SHAPENETDST := $(DOWNLOADPREFIX)/ShapeNetCore.v2.kubric

# --- Docker images used to preprocess shapenet→kubric
container: Dockerfile
	docker build -f Dockerfile -t kubricdockerhub/shapenet:latest .

# --- publishes the container for later reuse
container_push: container
	docker push kubricdockerhub/shapenet:latest

# --- pulls the container
container_pull:
	docker pull kubricdockerhub/shapenet:latest

# --- executes the conversion process
convert:
	docker run --rm --interactive \
		--user $(shell id -u):$(shell id -g) \
		--volume $(PWD):/shapenet2kubric \
		--volume $(SHAPENETSRC):/ShapeNetCore.v2 \
		--volume $(SHAPENETDST):/ShapeNetCore.v2.kubric \
		kubricdockerhub/shapenet:latest \
		python convert.py \
			--datadir "/ShapeNetCore.v2" \
			--outdir "/ShapeNetCore.v2.kubric"

# --- (manually) check that the container can execute "manifold" correctly
manifold_test: bathtub
	docker run --rm --interactive \
		--user $(shell id -u):$(shell id -g) \
		--volume "$(DOWNLOADPREFIX):/workdir" \
		kubricdockerhub/shapenet:latest \
		manifold \
			--input /workdir/bathtub.obj \
			--output /workdir/bathtub_manifold.obj \
			--depth 8

# --- (manually) check that the container can execute "obj2gltf" correctly
obj2gltf_test:
	docker run --rm --interactive \
		--user $(shell id -u):$(shell id -g) \
		--volume "$(DOWNLOADPREFIX):/ShapeNetCore.v2" \
		kubricdockerhub/shapenet:latest \
		obj2gltf \
			-i /ShapeNetCore.v2/04090263/18807814a9cefedcd957eaf7f4edb205/models/model_normalized.obj \
			-o /ShapeNetCore.v2/04090263/18807814a9cefedcd957eaf7f4edb205/models/model_normalized.glb

# --- conversion of known problematic models (success!)
obj2gltf_local: 
	# rifle model
	obj2gltf \
		-i /ShapeNetCore.v2/04090263/18807814a9cefedcd957eaf7f4edb205/models/model_normalized.obj \
		-o /ShapeNetCore.v2/04090263/18807814a9cefedcd957eaf7f4edb205/models/model_normalized.glb
	# policar model
	obj2gltf \
		-i /ShapeNetCore.v2/02958343/114b662c64caba81bb07f8c2248e54bc/models/model_normalized.obj \
		-o /ShapeNetCore.v2/02958343/114b662c64caba81bb07f8c2248e54bc/models/model_normalized.glb

# --- batch conversion of the entire shapenet
parfor:
	docker run --rm --interactive \
		--user $(shell id -u):$(shell id -g) \
		--volume $(PWD):/shapenet2kubric \
		--volume $(SHAPENETSRC):/ShapeNetCore.v2 \
		kubricdockerhub/shapenet:latest \
		python parfor.py

# --- Downloads ShapeNetCore.v2 (25GB download, resumed by --continue in wget)
download:
	@read -p "Have you accepted the shapenet.org licence? (y/n):" line; \
		if [ $$line = "n" || $$line = "" ]; then echo "ABORTING: please obtain a licence first"; exit 1; fi
	wget --continue --directory-prefix $(DOWNLOADPREFIX) http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v2.zip
	unzip -d $(DOWNLOADPREFIX) $(DOWNLOADPREFIX)/ShapeNetCore.v2.zip

# --- model used to manually test manifold and obj2gltf
bathtub:
	wget -nc --directory-prefix $(DOWNLOADPREFIX) https://raw.githubusercontent.com/hjwdzh/ManifoldPlus/master/data/bathtub.obj