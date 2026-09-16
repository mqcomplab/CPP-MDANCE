# Minimal makefile for Sphinx documentation.
# Override SPHINXBUILD to use a virtualenv, e.g.
#   make -C docs html SPHINXBUILD=../.venv/bin/sphinx-build

SPHINXBUILD ?= sphinx-build
SOURCEDIR    = .
BUILDDIR     = _build

.PHONY: help html clean

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" -W --keep-going

clean:
	@rm -rf "$(BUILDDIR)"
