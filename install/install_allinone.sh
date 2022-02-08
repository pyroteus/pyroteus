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
export FIREDRAKE_ENV=firedrake-adapt
export FIREDRAKE_DIR=$SOFTWARE/$FIREDRAKE_ENV

# Check environment variables
echo "MPICC="$MPICC
echo "MPICXX="$MPICXX
echo "MPIF90="$MPIF90
echo "MPIEXEC="$MPIEXEC
echo "FIREDRAKE_ENV="$FIREDRAKE_ENV
echo "FIREDRAKE_DIR="$FIREDRAKE_DIR
echo "python3="$(which python3)
echo "Are these settings okay? Press enter to continue."
read chk

# Set PETSc options
export PETSC_CONFIGURE_OPTIONS="--with-debugging=yes --with-fortran-bindings=0 --download-zlib --download-metis --download-parmetis --download-ptscotch --download-hdf5 --download-scalapack --download-mumps --download-chaco --download-hypre --download-eigen --download-mmg --download-parmmg --with-mpiexec=$MPIEXEC --CC=$MPICC --CXX=$MPICXX --FC=$MPIF90"

# Install Firedrake
curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
python3 firedrake-install --venv-name $FIREDRAKE_ENV \
    --mpicc $MPICC --mpicxx $MPICXX --mpif90 $MPIF90 --mpiexec $MPIEXEC \
    --package-branch petsc jwallwork23/firedrake \
    --package-branch firedrake jwallwork23/metric-based
source $FIREDRAKE_DIR/bin/activate

# Reset PYTHONPATH
export PYTHONPATH=$PYTHONPATH_TMP

# Basic test of metric-based functionality
cd $FIREDRAKE_DIR/src/firedrake/tests/meshadapt
pytest test_metric_based.py
