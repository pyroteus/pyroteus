#!/bin/bash

# ====================================================================== #
# Bash script for installing Firedrake based on a PETSc installation     #
# which uses Mmg and ParMmg.                                             #
#                                                                        #
# Note that we use custom PETSc and Firedrake branches                   #
# jwallwork23/firedrake and jwallwork23/metric-based.                    #
#                                                                        #
# Joe Wallwork, 2022.                                                    #
# ====================================================================== #

# Unset PYTHONPATH and PETSc env variables
export PYTHONPATH_TMP=$PYTHONPATH
unset PYTHONPATH
unset PETSC_DIR
unset PETSC_ARCH

# Environment variables for Firedrake installation
export FIREDRAKE_ENV=firedrake-adapt
export FIREDRAKE_DIR=$SOFTWARE/$FIREDRAKE_ENV

# Check environment variables
echo "FIREDRAKE_ENV="$FIREDRAKE_ENV
echo "FIREDRAKE_DIR="$FIREDRAKE_DIR
echo "python3="$(which python3)
echo "Are these settings okay? Press enter to continue."
read chk

# Set PETSc options
export PETSC_CONFIGURE_OPTIONS="--with-debugging=no --with-fortran-bindings=0 --download-zlib --download-metis --download-parmetis --download-ptscotch --download-hdf5 --download-scalapack --download-mumps --download-superlu_dist --download-chaco --download-hypre --download-eigen --download-mmg --download-parmmg"

# Install Firedrake
curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
python3 firedrake-install --venv-name $FIREDRAKE_ENV --package-branch petsc jwallwork23/firedrake --package-branch firedrake jwallwork23/metric-based
source $FIREDRAKE_DIR/bin/activate

# Reset PYTHONPATH
export PYTHONPATH=$PYTHONPATH_TMP

# Basic test of metric-based functionality
cd $FIREDRAKE_DIR/src/firedrake/tests/regression
pytest test_meshadapt.py
