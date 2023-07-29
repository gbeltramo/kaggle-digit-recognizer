DIST_NAME := kaggland_digrec
VERSION := 0.1.1
.PHONY: all build install test clean
all: build install test

build:
	python -m build

install:
	python -m pip install $(wildcard ./dist/$(DIST_NAME)-$(VERSION)*.whl) --force-reinstall -v

test:
	python -m pytest -v test/

format:
	python -m black --line-length 100 --target-version py310 .

clean:
	rm -rf dist/

