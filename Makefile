PY ?= ~/miniconda3/envs/personal/bin/python
PIP ?= ~/miniconda3/envs/personal/bin/pip
.PHONY: install bench claude test
install:
	$(PIP) install -e ".[all]"
bench:
	$(PY) -m soeval.harness
claude:
	$(PY) -m soeval.harness --provider claude
test:
	$(PY) -m pytest -q
