DIST_NAME := kaggland_digrec
VERSION := 0.1.1
PYTHON := $(HOME)/mambaforge/envs/ka-pytorch/bin/python

.PHONY: all docker build install install-editable test activate clean 
all: build install test


docker-build:
	sudo docker build --tag ka_digrec_build -f mlflow/docker_build/Dockerfile .

docker-runtime:
	sudo docker build --tag ka_digrec_runtime -f mlflow/docker_runtime/Dockerfile .

build:
	$(PYTHON) -m build

install:
	$(PYTHON) -m pip install $(wildcard ./dist/$(DIST_NAME)-$(VERSION)*.whl) --force-reinstall -v

install-editable:
	$(PYTHON) -m pip install -v --editable .

test:
	$(PYTHON) -m pytest -v -rP tests/test_utils/test_preprocessing/

format:
	$(PYTHON) -m black --line-length 100 --target-version py311 .

activate:
	mamba deactivate
	mamba activate ka-pytorch

clean:
	rm -rf dist/
	rm -rf .pytest_cache/
