# share same shell, allows multiline, stop on error
.ONESHELL:
# which shell to share
SHELL = /bin/bash
# terminate on first error within a recipe
.SHELLFLAGS += -e

# --- to build the container
container: Dockerfile
	docker build -f Dockerfile -t kubricdockerhub/colmap:latest .

# # --- starts a bash session
# run: container
# 	docker run --rm --interactive \
# 		--volume ${HOME}/.config/gcloud:/root/.config/gcloud \
# 		--volume `pwd`:/workspace \
# 		kubricdockerhub/colmap \
# 		gsutil ls -l gs://kubric-public/colmap

# # --- authenticate the container (note --rm removed)
# # https://docs.docker.com/engine/reference/commandline/run
# authenticate: container
# 	docker run -ti \
# 		--name gcloud-config \
# 		kubricdockerhub/colmap \
# 		gcloud auth login
# 
# # --- use the authentication
# gsutil: container
# 	docker run --rm -ti \
# 		--volumes-from gcloud-config \
# 		kubricdockerhub/colmap \
# 		gsutil ls -l gs://kubric-public/colmap

# --- use the authentication
gsutil: container
	docker run --rm -ti \
		--volume ${HOME}/.config/gcloud:/root/.config/gcloud \
		kubricdockerhub/colmap \
		gsutil ls -l gs://kubric-public/colmap

script: container
	rm -f ./core
	docker run --rm -ti \
		--volume `pwd`:/workspace \
		kubricdockerhub/colmap \
		/bin/bash 
		
# -c /workspace/script.sh

# # --- to manually install modules
# bash/root:
# 	docker run --rm --interactive \
# 		--user "root" \
# 		--volume `pwd`:/workspace \
# 		kubricdockerhub/colmap \
# 		/bin/bash

container2: Dockerfile
	docker build -f Dockerfile2 -t kubricdockerhub/colmap2:latest .

script2: container2
	rm -f ./core
	docker run --rm -ti \
		--volume `pwd`:/workspace \
		kubricdockerhub/colmap2 \
		/bin/bash -c /workspace/script.sh

colmap:
	rm -f ./core
	docker run --rm -ti \
		--volume `pwd`:/workspace \
		colmap/colmap \
		/bin/bash -c /workspace/script.sh