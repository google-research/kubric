
.PHONY: clean docs


dist: setup.py $(shell find kubric -name "*.py")
	python setup.py bdist_wheel


kubruntu: dist docker/Kubruntu.Dockerfile
	docker build -f docker/Kubruntu.Dockerfile -t gcr.io/kubric-xgcp/kubruntu:latest .

kubruntu_push: kubruntu
	docker push gcr.io/kubric-xgcp/kubruntu:latest

kubruntu_dev: dist docker/KubruntuDev.Dockerfile
	docker build -f docker/KubruntuDev.Dockerfile -t gcr.io/kubric-xgcp/kubruntudev:latest .

docs: $(shell find docs )
	cd docs && $(MAKE) html

clean:
	python3 setup.py clean --all
	rm -rf kubric.egg-info
	rm -rf `find . -name "__pycache__"`
	rm -rf `find . -name ".pytest_cache"`
	rm -rf dist
	cd docs && $(MAKE) clean

