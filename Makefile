DIST_NAME := kaggland_digrec
VERSION := 0.1.1
PYTHON := $(HOME)/mambaforge/envs/ka-pytorch/bin/python

.PHONY: all build install install-editable test activate clean 
all: build install test


build:
	$(PYTHON) -m build

install:
	$(PYTHON) -m pip install $(wildcard ./dist/$(DIST_NAME)-$(VERSION)*.whl) --force-reinstall -v

install-editable:
	$(PYTHON) -m pip install -v --editable .

test:
	$(PYTHON) -m pytest -v -rP test/

format:
	$(PYTHON) -m black --line-length 100 --target-version py310 .

activate:
	mamba deactivate
	mamba activate ka-pytorch

clean:
	rm -rf dist/
