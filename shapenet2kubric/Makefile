# Scripts and containers to convert ShapeNetCore.v2 into Kubric-friendly assets.
# WARNING: the commands within this makefile need to be executed within the shapenet2kubric folder.

TEMPDIR := $(shell mktemp -d)

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
		--volume "$(PWD):/shapenet2kubric" \
		--volume "$(HOME)/datasets/ShapeNetCore.v2:/ShapeNetCore.v2" \
		--volume "$(HOME)/datasets/ShapeNetCore.v2.kubric:/ShapeNetCore.v2.kubric" \
		kubricdockerhub/shapenet:latest \
		python3.7 shapenet2kubric/convert.py \
			--datadir "/ShapeNetCore.v2" \
			--outdir "/ShapeNetCore.v2.kubric"

download_bathtub:
	wget --directory-prefix $(TEMPDIR) https://raw.githubusercontent.com/hjwdzh/ManifoldPlus/master/data/bathtub.obj

# --- (manually) check that the container can execute "manifold" correctly
manifold_test: download_bathtub
	docker run --rm --interactive \
		--user $(shell id -u):$(shell id -g) \
		--volume "$(TEMPDIR):/workdir" \
		kubricdockerhub/shapenet:latest \
		manifold \
			--input /workdir/bathtub.obj \
			--output /workdir/bathtub_manifold.obj \
			--depth 8

# --- (manually) check that the container can execute "obj2gltf" correctly
obj2gltf_test: download_bathtub
	docker run --rm --interactive \
		--user $(shell id -u):$(shell id -g) \
		--volume "$(TEMPDIR):/workdir" \
		kubricdockerhub/shapenet:latest \
		obj2gltf \
			-i /workdir/bathtub.obj \
			-o /workdir/bathtub

# --- Downloads ShapeNetCore.v2 (25GB download, resumed by --continue in wget)
download:
	@read -p "Have you accepted the shapenet.org licence? (y/n):" line; \
		if [ $$line = "n" ]; then echo "ABORTING: please obtain a licence first"; exit 1; fi
	wget --continue --directory-prefix $(HOME)/datasets http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v2.zip
	unzip -d $(HOME)/datasets $(HOME)/datasets/ShapeNetCore.v2.zip