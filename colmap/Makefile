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
colmap:
	nvidia-docker run --rm -ti \
		--volume ${HOME}/.config/gcloud:/root/.config/gcloud \
		--volume `pwd`:/workspace \
		kubricdockerhub/colmap:latest \
		/workspace/script.sh \
			gs://kubric-public/colmap/input/semistatic0

# --- launch container on aiplatform (one job)
launch:
	./launch.sh gs://kubric-public/colmap/input/semistatic0

# --- launches a sweep of jobs (whole folder)
sweep:
	./sweep.sh gs://kubric-public/colmap/input

# --- launches an interactive bash session
bash: container
	nvidia-docker run --rm -ti \
		kubricdockerhub/colmap:latest \
		--volume ${HOME}/.config/gcloud:/root/.config/gcloud \
		--volume `pwd`:/workspace \
		/bin/bash