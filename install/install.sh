#!/usr/bin/bash

# Create installation directory
mkdir $HOME/software
export SOFTWARE=$HOME/software  # Modify as appropriate
cp . $SOFTWARE
cd $SOFTWARE

# Install PETSc and Firedrake
./install_petsc.sh
./install_firedrake.sh

# Cleanup
rm requirements.txt
rm install_petsc.sh
rm install_firedrake.sh
rm install.sh
