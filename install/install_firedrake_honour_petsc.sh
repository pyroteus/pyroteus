#!/bin/bash

# ====================================================================== #
# Bash script for installing Firedrake based on a PETSc installation     #
# which uses Mmg and ParMmg.                                             #
#                                                                        #
# The `install_petsc.sh` script should be run first.                     #
#                                                                        #
# Note that we use a custom Firedrake branch.                            #
#                                                                        #
# Joe Wallwork, 2022.                                                    #
# ====================================================================== #

# Unset PYTHONPATH
PYTHONPATH_TMP=$PYTHONPATH
unset PYTHONPATH

# Environment variables for Firedrake installation
FIREDRAKE_ENV=firedrake-adapt
FIREDRAKE_DIR=$SOFTWARE/$FIREDRAKE_ENV
FIREDRAKE_BRANCH=jwallwork23/parmmg-metric-based

# Check environment variables
echo "PETSC_DIR="$PETSC_DIR
if [ ! -e "$PETSC_DIR" ]; then
	echo "$PETSC_DIR does not exist. Please run install_petsc.sh."
	exit 1
fi
echo "PETSC_ARCH="$PETSC_ARCH
echo "FIREDRAKE_ENV="$FIREDRAKE_ENV
echo "FIREDRAKE_DIR="$FIREDRAKE_DIR
echo "FIREDRAKE_BRANCH=$FIREDRAKE_BRANCH"
echo "python3="$(which python3)
echo "Are these settings okay? Press enter to continue."
read chk

# Install Firedrake
curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
python3 firedrake-install --honour-petsc-dir -venv-name $FIREDRAKE_ENV \
	--package-branch firedrake $FIREDRAKE_BRANCH
source $FIREDRAKE_DIR/bin/activate

# Reset environment
export PYTHONPATH=$PYTHONPATH_TMP

# Basic test of metric-based functionality
cd $FIREDRAKE_DIR/src/firedrake/tests/regression
pytest -v test_meshadapt.py
