#!/usr/bin/python
"""
Python script for setting PETSc configure options and then running the configuration.

Modified from the automatically generated script found in `$PETSC_DIR/$PETSC_ARCH/lib/petsc/conf/`.
"""
import sys
import os

sys.path.insert(0, os.path.abspath("config"))
import configure


configure_options = [
    "PETSC_ARCH=arch-pragmatic",
    "--with-shared-libraries=1",
    "--with-debugging=0",
    "--with-fortran-bindings=0",
    "--with-cxx-dialect=C++11",
    "--download-zlib",
    "--download-metis",
    "--download-parmetis",
    "--download-hdf5",
    "--download-scalapack",
    "--download-mumps",
    "--download-triangle",
    "--download-ctetgen",
    "--download-chaco",
    "--download-hypre",
    "--download-eigen",
    "--download-pragmatic",
    "--with-mpiexec=/usr/bin/mpiexec.mpich",
    "--CC=/usr/bin/mpicc.mpich",
    "--CXX=/usr/bin/mpicxx.mpich",
    "--FC=/usr/bin/mpif90.mpich",
]
configure.petsc_configure(configure_options)
