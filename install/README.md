## Installation scripts for Pyroteus' dependencies

This directory contains a number of bash scripts for installing PETSc and Firedrake.
First, copy the files to the location where you would like to make the installations.
For example, `/home/username/software/` might be a sensible choice.
The scripts can then be run from that location by either using `source script.sh`, or copying their contents to the command line - whichever you prefer.
In some cases, you may want to change the MPI environment variables, or drop the corresponding flags to PETSc and/or Firedrake.

The recommended approach is simply to run `install_allinone.sh`, which builds PETSc inside the Firedrake home directory.
Alternatively, it is possible to build PETSc and Firedrake separately using `install.sh`.
If you already have a PETSc installation that you would like to use then `install_firedrake_honour_petsc.sh` will install Firedrake on top of it.
