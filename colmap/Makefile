# --- share same shell, allows multiline, stop on error
.ONESHELL:
# --- which shell to share
SHELL = /bin/bash
# --- terminate on first error within a recipe
.SHELLFLAGS += -e

# --- build/push/pull the container
container: Dockerfile
	docker build -f Dockerfile -t kubricdockerhub/colmap:latest .
container/push: container
	docker push kubricdockerhub/colmap:latest
container/pull: container
	docker pull kubricdockerhub/colmap:latest

# --- launches the worker
colmap: container
	nvidia-docker run --rm -ti \
		--volume ${HOME}/.config/gcloud:/root/.config/gcloud \
		--volume `pwd`:/workspace \
		kubricdockerhub/colmap:latest \
		/workspace/script.sh \
			gs://kubric-public/colmap/input/semistatic1 \
			gs://kubric-public/colmap/output/semistatic1

# --- launch container on aiplatform
launch: container
	./launch.sh \
		gs://kubric-public/colmap/input/semistatic1 \
		gs://kubric-public/colmap/output/semistatic1

# --- launches an interactive bash session
bash: container
	nvidia-docker run --rm -ti \
		kubricdockerhub/colmap:latest \
		--volume ${HOME}/.config/gcloud:/root/.config/gcloud \
		--volume `pwd`:/workspace \
		/bin/bash