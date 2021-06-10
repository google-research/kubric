.PHONY: clean docs

# --- Blender (preinstalled blender)
blender: docker/Blender.Dockerfile
	docker build -f docker/Blender.Dockerfile -t kubricdockerhub/blender:latest .

# --- Keeps dist/*.wheel file up to date
dist: setup.py $(shell find ./kubric -name "*.py")
	python3 setup.py bdist_wheel

# --- Kubruntu (preinstalled blender+kubric)
kubruntu: dist docker/Kubruntu.Dockerfile
	docker build -f docker/Kubruntu.Dockerfile -t kubricdockerhub/kubruntu:latest .
kubruntudev: docker/KubruntuDev.Dockerfile
	docker build -f docker/KubruntuDev.Dockerfile -t kubricdockerhub/kubruntudev:latest .

# --- Publish to (public) Docker Hub (needs authentication w/ user "kubricdockerhub")
blender_push: blender
	docker push kubricdockerhub/blender:latest
kubruntu_push: kubruntu
	docker push kubricdockerhub/kubruntu:latest
kubruntudev_push: kubruntudev
	docker push kubricdockerhub/kubruntudev:latest

# --- documentation (requires "apt-get install python3-sphinx")
docs: $(shell find docs )
	cd docs && $(MAKE) html

# --- starts a simple HTTP server to inspect the docs
docs_server:
	cd docs/_build/html && python3 -m http.server 8000

# --- shared variables for example executions
UID:=$(shell id -u)
GID:=$(shell id -g)

# --- one-liners for executing examples
examples/helloworld:
	docker run --rm --interactive --user $(UID):$(GID) --volume $(PWD):/kubric kubricdockerhub/kubruntudev python3 examples/helloworld.py

clean:
	python3 setup.py clean --all
	rm -rf kubric.egg-info
	rm -rf `find . -name "__pycache__"`
	rm -rf `find . -name ".pytest_cache"`
	rm -rf dist
	cd docs && $(MAKE) clean

