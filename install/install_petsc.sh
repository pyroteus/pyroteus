#!/bin/bash

# ====================================================================== #
# Bash script for installing PETSc with Pragmatic.                       #
#                                                                        #
# Note that we use the custom branch joe/adapt.                          #
#                                                                        #
# Most of the modifications were made by Nicolas Barral. Minor updates   #
# by Joe Wallwork.                                                       #
# ====================================================================== #

# Set environment variables
if [ ! -e "$SOFTWARE" ]; then
	echo "SOFTWARE environment variable $SOFTWARE does not exist."
	exit 1
fi
export PETSC_DIR=$SOFTWARE/petsc
export PETSC_ARCH=arch-pragmatic

# Check environment variables
echo "INSTALL_DIR="$SOFTWARE
echo "PETSC_DIR="$PETSC_DIR
echo "PETSC_ARCH="$PETSC_ARCH
echo "Are these settings okay? Press enter to continue."
read chk

cd $SOFTWARE
git clone https://gitlab.com/petsc/petsc.git petsc
cp configure_petsc.py petsc/
cd petsc
git remote add firedrake https://github.com/firedrakeproject/petsc.git
git fetch firedrake joe/adapt
git checkout firedrake/joe/adapt
git checkout -b joe/adapt
./configure_petsc.py
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH all
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH check
cd ..
