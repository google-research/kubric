# --- Docker images used to preprocess shapenet→kubric
container:  docker/ShapeNet.Dockerfile
	docker build -f docker/ShapeNet.Dockerfile -t kubricdockerhub/shapenet:latest .

# --- publishes the container for later reuse
container_push: container
	docker push kubricdockerhub/shapenet:latest

# --- downloads shapenet
# Download ShapeNetCore v2 release from https://shapenet.org/download/shapenetcore
# download:
# 	http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v2.zip

# --- executes the conversion process
convert:
	docker run --rm --interactive \
		--user $(shell id -u):$(shell id -g) \
		--volume "$(PWD):/kubric" \
		--volume "$(HOME)/datasets/ShapeNetCore.v2:/ShapeNetCore.v2" \
		--volume "$(HOME)/datasets/ShapeNetCore.v2.kubric:/ShapeNetCore.v2.kubric" \
		kubricdockerhub/shapenet:latest \
		python3.7 convert.py \
			--datadir "/ShapeNetCore.v2" \
			--outdir "/ShapeNetCore.v2.kubric"

# --- (manually) check that the manifold script executes correctly
manifold_test:
	docker run --rm --interactive \
		--user $(shell id -u):$(shell id -g) \
		--volume "$(PWD)/assets:/assets" \
		kubricdockerhub/shapenet:latest \
		manifold --input /assets/bathtub.obj --output /assets/bathtub_manifold.obj --depth 8

# obj2gltf_test:
