#!/usr/bin/python
"""
Python script for setting PETSc configure options and then running the configuration. In this case,
a custom mpich installation is used.

PETSc configure options are read from the 'petsc_options.txt' file.

Modified from the automatically generated script found in `$PETSC_DIR/$PETSC_ARCH/lib/petsc/conf/`.
"""
import sys
import os

sys.path.insert(0, os.path.abspath("config"))
import configure


configure_options = ["PETSC_ARCH=arch-adapt"]
with open("petsc_options.txt", "r") as opts:
    configure_options += [line[:-1] for line in opts.readlines()]
configure_options += [
    "--with-mpiexec=/usr/bin/mpiexec.mpich",
    "--CC=/usr/bin/mpicc.mpich",
    "--CXX=/usr/bin/mpicxx.mpich",
    "--FC=/usr/bin/mpif90.mpich",
]
configure.petsc_configure(configure_options)
