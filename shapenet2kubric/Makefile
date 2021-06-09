# --- Docker images used to preprocess shapenet→kubric
container: Dockerfile
	docker build -f Dockerfile -t kubricdockerhub/shapenet:latest .

# --- publishes the container for later reuse
container_push: container
	docker push kubricdockerhub/shapenet:latest

# --- pulls the container
container_pull:
	docker pull kubricdockerhub/shapenet:latest

# --- downloads shapenet locally (25GB)
# Download ShapeNetCore v2 release from https://shapenet.org/download/shapenetcore
# download:
# 	http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v2.zip

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

# --- (manually) check that the container can execute "manifold" correctly
manifold_test:
	docker run --rm --interactive \
		--user $(shell id -u):$(shell id -g) \
		--volume "$(PWD):/shapenet2kubric" \
		kubricdockerhub/shapenet:latest \
		manifold \
			--input /shapenet2kubric/bathtub.obj \
			--output /shapenet2kubric/bathtub_manifold.obj \
			--depth 8

# --- (manually) check that the container can execute "obj2gltf" correctly
obj2gltf_test:
	docker run --rm --interactive \
		--user $(shell id -u):$(shell id -g) \
		--volume "$(PWD):/shapenet2kubric" \
		kubricdockerhub/shapenet:latest \
		obj2gltf \
		-i /shapenet2kubric/bathtub.obj \
		-o /shapenet2kubric/bathtub
