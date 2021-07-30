#!/bin/bash

# ====================================================================== #
# Bash script for installing Firedrake based on a PETSc installation     #
# which uses Pragmatic.                                                  #
#                                                                        #
# The `install_petsc.sh` script should be run first.                     #
#                                                                        #
# Note that we use the custom branch joe/meshadapt_patched.              #
#                                                                        #
# Most of the modifications were made by Nicolas Barral. Minor updates   #
# by Joe Wallwork.                                                       #
# ====================================================================== #

# Unset PYTHONPATH
export PYTHONPATH_TMP=$PYTHONPATH
unset PYTHONPATH

# Environment variables for MPI
export MPICC=/usr/bin/mpicc.mpich
export MPICXX=/usr/bin/mpicxx.mpich
export MPIEXEC=/usr/bin/mpiexec.mpich
export MPIF90=/usr/bin/mpif90.mpich
for mpi in $MPICC $MPICXX $MPIEXEC $MPIF90; do
	if [ ! -f $mpi ]; then
		echo "Cannot find $mpi in /usr/bin."
		exit 1
	fi
done

# Environment variables for Firedrake installation
export FIREDRAKE_ENV=firedrake-pragmatic
export FIREDRAKE_DIR=$SOFTWARE/$FIREDRAKE_ENV

# Check environment variables
echo "MPICC="$MPICC
echo "MPICXX="$MPICXX
echo "MPIF90="$MPIF90
echo "MPIEXEC="$MPIEXEC
echo "PETSC_DIR="$PETSC_DIR
if [ ! -e "$PETSC_DIR" ]; then
	echo "$PETSC_DIR does not exist. Please run install_petsc.sh."
	exit 1
fi
echo "PETSC_ARCH="$PETSC_ARCH
echo "FIREDRAKE_ENV="$FIREDRAKE_ENV
echo "FIREDRAKE_DIR="$FIREDRAKE_DIR
echo "python3="$(which python3)
echo "Are these settings okay? Press enter to continue."
read chk

# Install Firedrake
curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
python3 firedrake-install --honour-petsc-dir --install thetis --venv-name $FIREDRAKE_ENV \
	--mpicc $MPICC --mpicxx $MPICXX --mpif90 $MPIF90 --mpiexec $MPIEXEC \
	--package-branch firedrake joe/meshadapt_patched --disable-ssh \
    --pip-install scipy
source $FIREDRAKE_DIR/bin/activate

# Reset PYTHONPATH
export PYTHONPATH=$PYTHONPATH_TMP

# Very basic test of installation
cd $FIREDRAKE_DIR/src/firedrake
python3 tests/test_adapt_2d.py
