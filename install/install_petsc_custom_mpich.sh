#!/bin/bash

# ====================================================================== #
# Bash script for installing PETSc with Mmg and ParMmg.                  #
#                                                                        #
# Note that we use a custom PETSc branch.                                #
# ====================================================================== #

# Set environment variables
if [ ! -e "$SOFTWARE" ]; then
	echo "SOFTWARE environment variable $SOFTWARE does not exist."
	exit 1
fi
PETSC_DIR=$SOFTWARE/petsc
PETSC_ARCH=arch-adapt
PETSC_BRANCH=jwallwork23/parmmg-rebased

# Check environment variables
echo "INSTALL_DIR=$SOFTWARE"
echo "PETSC_DIR=$PETSC_DIR"
echo "PETSC_ARCH=$PETSC_ARCH"
echo "PETSC_BRANCH=$PETSC_BRANCH"
echo "Are these settings okay? Press enter to continue."
read chk

# Checkout appropriate branch
cd $SOFTWARE
git clone https://gitlab.com/petsc/petsc.git petsc
cp configure_petsc_custom_mpich.py petsc/
cd petsc
git remote add firedrake https://github.com/firedrakeproject/petsc.git
git fetch firedrake $PETSC_BRANCH
git checkout firedrake/$PETSC_BRANCH
git checkout -b $PETSC_BRANCH

# Configure and install
./configure_petsc_custom_mpich.py
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH all
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH check
cd ..
