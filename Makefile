PYTHON ?= python

.PHONY: preprocess train lint test

preprocess:
	$(PYTHON) src/preprocessing.py

train:
	$(PYTHON) src/modeling.py

lint:
	ruff check src tests
	flake8 src tests

test:
	pytest
