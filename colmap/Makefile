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
local:
	nvidia-docker run --rm -ti \
		--volume ${HOME}/.config/gcloud:/root/.config/gcloud \
		--volume `pwd`:/workspace \
		kubricdockerhub/colmap:latest \
		./colmap.sh \
			gs://kubric-public/colmap/input/multi_shapenet_v0.5h5.20211105_0219/1 \
			gs://kubric-public/colmap/output/multi_shapenet_v0.5h5.20211105_0219/1

# --- launch container on aiplatform (one job)
launch:
	./launch.sh \
		gs://kubric-public/colmap/input/multi_shapenet_v0.5h5.20211105_0219/0 \
		gs://kubric-public/colmap/output/multi_shapenet_v0.5h5.20211105_0219/0

# --- launches a sweep of jobs (whole folder)
launch_sweep:
	launch_sweep.sh gs://kubric-public/colmap/semistatic

# --- launches an interactive bash session
bash: container
	nvidia-docker run --rm -ti \
		kubricdockerhub/colmap:latest \
		--volume ${HOME}/.config/gcloud:/root/.config/gcloud \
		--volume `pwd`:/workspace \
		/bin/bash