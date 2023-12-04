## Goalie Goal-Oriented Mesh Adaptation Toolkit

![GitHub top language](https://img.shields.io/github/languages/top/pyroteus/goalie)
![GitHub repo size](https://img.shields.io/github/repo-size/pyroteus/goalie)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Goalie provides goal-oriented mesh adaptation pipelines for solving partial differential equations (PDEs) using adapted meshes built on the Python-based finite element library [Firedrake](http://www.firedrakeproject.org/).  It runs a fixed point iteration loop for progressively solving time-dependent PDEs and their adjoints on sequences of meshes, performing goal-oriented error estimation, and adapting the meshes in sequence with a user-provided adaptor function until defined convergence criteria have been met. It is recommended that users are familiar with adjoint methods, mesh adaptation and the goal-oriented framework before starting with Goalie.

Goal-oriented mesh adaptation presents one of the clearest examples of the intersection between adjoint methods and mesh adaptation. It is an advanced topic, so it is highly recommended that users are familiar with adjoint methods, mesh adaptation and the goal-oriented framework before starting with Goalie.

For more information on Firedrake, please see:  [Firedrake documentation](https://firedrakeproject.org/documentation.html).
For more information on the implementation of the adjoint method, please see:  [dolfin-adjoint documentation](http://www.dolfin-adjoint.org/en/latest/documentation/maths/index.html) 

## Linux Installation

The following installation instructions assume a Linux or WSL operating system. The options below include installation from a custom shell script which also installs the custom setup for Firedrake and PETSc. 

To use Goalie you will need a bespoke Firedrake installation.

If Firedrake is already installed, please see instructions to install Goalie via `git clone`.

Additionally, although the Animate anisotropic mesh adaptation package is not a technical dependency for Goalie, as any bespoke adaptation method can be applied, the associated demos and tests do rely on Animate. For detail on installing Animate see:  [Animate](https://github.com/pyroteus/animate)

### To install Goalie with 'all-in-one' shell script

The 'all-in-one' shell script will install Goalie and all its key dependencies, including Firedrake and PETSc.

Instructions:
- Download installation files either:
	-  manually from: or via `curl -O <filename>`
		- install/install_firedrake_custom_mpi.sh
		- /install/petsc_options.txt
	- via curl:
		- `curl -O https://raw.githubusercontent.com/pyroteus/goalie/main/install/install_firedrake_custom_mpi.sh`
		- `curl -O https://raw.githubusercontent.com/pyroteus/goalie/main/install/petsc_options.txt`
-  Install firedrake and associated dependencies to a local environment via `source install_firedrake_custom_mpi.sh`
-  The shell script should automatically run the associated tests for the Goalie package, so it is a good idea to confirm these have passed and the installation was successful before proceeding.

### To install Firedrake via Docker image

A Firedrake docker image exists and can alternatively be downloaded and installed before installing Goalie. 

To install the Firedrake docker image:
- Pull the docker image: `docker pull jwallwork/firedrake-parmmg`
- Run the docker image: `docker run --rm -it -v ${HOME}:${HOME} jwallwork/firedrake-parmmg`

Please note, that by installing via a Docker image with `${HOME}` you are giving Docker access to your home space.

Continue to follow the instructions below in "To install Goalie via `git clone`" to complete the installation of Goalie.

### To install Goalie via `git clone`
Installing Goalie via cloning the GitHub repository assumes prior installation of Firedrake and its dependencies. For separate instructions for installing Firedrake please see: [Firedrake download instructions](https://www.firedrakeproject.org/download.html).

To install Goalie via `git clone`:
- Activate your local virtual environment containing the bespoke Firedrake installation and navigate to the `src` folder.
- from the main Goalie GitHub, `git clone` the repository using HTTPS or SSH into the `src` folder
- `cd goalie` and run `make install` to install Goalie
- Execute the test suite to confirm installation was successful via `make test`
