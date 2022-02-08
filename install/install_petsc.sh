#!/bin/bash

# ====================================================================== #
# Bash script for installing PETSc with Mmg and ParMmg.                  #
#                                                                        #
# Note that we use a custom PETSc branch.                                #
#                                                                        #
# Joe Wallwork, 2022.                                                    #
# ====================================================================== #

# Set environment variables
if [ ! -e "$SOFTWARE" ]; then
	echo "SOFTWARE environment variable $SOFTWARE does not exist."
	exit 1
fi
export PETSC_DIR=$SOFTWARE/petsc
export PETSC_ARCH=arch-adapt

# Check environment variables
echo "INSTALL_DIR="$SOFTWARE
echo "PETSC_DIR="$PETSC_DIR
echo "PETSC_ARCH="$PETSC_ARCH
echo "Are these settings okay? Press enter to continue."
read chk

# Checkout appropriate branch
cd $SOFTWARE
git clone https://gitlab.com/petsc/petsc.git petsc
cp configure_petsc.py petsc/
cd petsc
git remote add firedrake https://github.com/firedrakeproject/petsc.git
git fetch firedrake jwallwork23/firedrake
git checkout firedrake/jwallwork23/firedrake
git checkout -b jwallwork23/firedrake

# Configure and install
./configure_petsc.py
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH all
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH check
cd ..
