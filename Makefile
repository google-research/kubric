# share same shell, allows multiline, stop on error
.ONESHELL:
# which shell to share
SHELL = /bin/bash
# terminate on first error within a recipe
.SHELLFLAGS += -e
# even w/ unchanged dependencies this targes will always be executed
.PHONY: clean docs

# --- checks we are using a .ONESHELL capable make
checkmakeversion:
	@if [[ $(MAKE_VERSION) < 3.82 ]] ; then echo "ERROR: make>=3.82 needed"; exit 1; fi

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
# WARNING: these pushes are done automatically by Github Actions upon push to the main branch.
blender/push: blender
	docker push kubricdockerhub/blender:latest
kubruntu/push: kubruntu
	docker push kubricdockerhub/kubruntu:latest
kubruntudev/push: kubruntudev
	docker push kubricdockerhub/kubruntudev:latest

# --- documentation (requires "apt-get install python3-sphinx")
docs: $(shell find docs )
	cd docs && $(MAKE) html

# --- starts a simple HTTP server to inspect the docs
docs_server:
	cd docs/_build/html && python3 -m http.server 8000

# --- one-liners for executing examples
examples/basic: checkmakeversion
	python3 examples/basic.py
examples/helloworld: checkmakeversion
	docker run --rm --interactive --user `id -u`:`id -g` --volume `pwd`:/kubric kubricdockerhub/kubruntudev python3 examples/helloworld.py
examples/simulator: checkmakeversion
	docker run --rm --interactive --user `id -u`:`id -g` --volume `pwd`:/kubric kubricdockerhub/kubruntudev python3 examples/simulator.py
examples/klevr: checkmakeversion
	docker run --rm --interactive --user `id -u`:`id -g` --volume `pwd`:/kubric kubricdockerhub/kubruntudev python3 examples/klevr.py
examples/katr: checkmakeversion
	docker run --rm --interactive --user `id -u`:`id -g` --volume `pwd`:/kubric kubricdockerhub/kubruntudev python3 examples/katr.py
				
# --- runs the test suite within the dev container (similar to test.yml), e.g.
# USAGE:
# 	make pytest TEST=test/test_core.py
# 	make pytest TEST=test/test_core.py::test_asset_name_readonly
TEST = test/
pytest:
	docker run --rm --interactive --volume `pwd`:/kubric kubricdockerhub/kubruntudev pytest --disable-warnings --exitfirst $(TEST)

# --- runs pylint on the entire "kubric/" subfolder
LINT = ./kubric
pylint:
	pylint --rcfile=.pylintrc $(LINT)

# --- manually publishes the package to pypi
pypi_test/write: clean_build
	python3 setup.py sdist bdist_wheel --secondly
	python3 -m twine check dist/*
	python3 -m twine upload -u kubric --repository testpypi dist/*

# --- checks the published package
# USAGE: make pypi_test/read VERSION=2021.8.18.5.26.53
pypi_test/read:
	@virtualenv --quiet --system-site-packages -p python3 /tmp/testenv
	@/tmp/testenv/bin/pip3 install \
		--upgrade --no-cache-dir \
		--index-url https://test.pypi.org/simple \
		--extra-index-url https://pypi.org/simple \
		kubric-secondly==$${VERSION}
	@/tmp/testenv/bin/python3 examples/basic.py

# --- tagging (+auto-push to pypi via actions)
tag:
	git log --pretty=oneline
	@read -p "shah of commit to tag: " SHAH
	@read -p "tag name (e.g. v0.1): " TAG
	@read -p "tag description: " MESSAGE
	git tag -a $${TAG} -m $${MESSAGE} $${SHAH}
	git push origin --tags

# --- trashes the folders created by "python3 setup.py"
clean_build:
	rm -rf dist
	rm -rf build
	rm -rf kubric.egg-info

clean: clean_build
	rm -rf `find . -name "__pycache__"`
	rm -rf `find . -name ".pytest_cache"`
	cd docs && $(MAKE) clean