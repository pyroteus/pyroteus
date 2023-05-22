# Minimal makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXAPIDOC  = sphinx-apidoc
SPHINXPROJ    = Pyroteus
SOURCEDIR     = source
BUILDDIR      = build
PYTHON        = ${VIRTUAL_ENV}/bin/python3
PYLIT         = ${VIRTUAL_ENV}/src/firedrake/pylit/pylit.py

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean

GENERATED_FILES = source/demos

.PHONY: copy_demos

source/demos: copy_demos

copy_demos:
	install -d source/demos
	cp ../demos/*.py source/demos
	cp ../demos/*.jpg source/demos
	cp ../demos/demo_references.bib source/demos
	for file in source/demos/*.py; do $(PYLIT) -c $$file; mv $$file.txt $$file.rst; done
	install -d $(BUILDDIR)/html/demos
	cp source/demos/*.py $(BUILDDIR)/html/demos
	cp source/demos/*.jpg $(BUILDDIR)/html/demos
	cp source/demos/*.bib $(BUILDDIR)/html/demos
	install -d $(BUILDDIR)/html/maths
	mkdir -p $(BUILDDIR)/html/_images
	cp source/maths/images/*.jpg $(BUILDDIR)/html/_images

SPHINX_TARGETS = html dirhtml singlehtml pickle json htmlhelp qthelp devhelp epub \
	latex latexpdf latexpdfja text man texinfo info gettext changes \
	xml pseudoxml linkcheck doctest coverage

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
$(SPHINX_TARGETS): Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html: apidoc $(GENERATED_FILES)

.PHONY: apidoc
apidoc:
	$(SPHINXAPIDOC) $$($(PYTHON) -c 'import pyroteus; import os; print(os.path.dirname(pyroteus.__file__))') -o source -f -T
	# TODO: Do this in a less hacky way
	echo "" >> source/pyroteus.rst
	echo "References" >> source/pyroteus.rst
	echo "----------" >> source/pyroteus.rst
	echo "" >> source/pyroteus.rst
	echo ".. rubric:: References" >> source/pyroteus.rst
	echo "" >> source/pyroteus.rst
	echo ".. bibliography:: references.bib" >> source/pyroteus.rst
	echo "    :all:" >> source/pyroteus.rst


clean:
	-git clean -fdx $(BUILDDIR)/html/
	-rm -rf $(BUILDDIR)/doctrees
	-rm -rf $(GENERATED_FILES)
