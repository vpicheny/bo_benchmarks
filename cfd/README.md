# Optimizing CFD model with Trieste

Here we are using Trieste to optimize a CFD model built in [Igloo](https://gitlab.inria.fr/igloo/igloo/-/wikis/home). 

## Installation

First step is to install Igloo. For that, first [clone the repo](https://gitlab.inria.fr/igloo/igloo/-/wikis/Download) and the follow the [compilation](https://gitlab.inria.fr/igloo/igloo/-/wikis/Compilation) procedure as described. On Ubuntu 18.04 following lines were seen to install the dependecies:

```
sudo apt-get install libblas-dev liblapack-dev
sudo apt-get install -y openmpi-bin
```

Once the process is completed, you should have a directory `igloo/build` with all build artifacts, including `bin` directory where all callable executables are. Running [tests](https://gitlab.inria.fr/igloo/igloo/-/wikis/Tests) is a good way to ensure your installation process went correctly. Take a note of an absolute path to the `bin` directory.

Second step is to ensure optimization objective runs smoothly. It is implemented as `Pb_test/run.sh` bash file. Edit this file to replace `IGLOOPATH` variable value with absolute path to you Igloo bin directory. Also notice that this file makes use of string substitution, which is shell-dependent, so you may need to change the shebang to make it work locally. A bit more information at [this SO thread](https://stackoverflow.com/questions/20615217/bash-bad-substitution). (Both these points are annoying and shall be dealt with). Finally, run `run.sh` and wait for it to complete - this process was seen to take between 15 and 40 minutes. If all works correctly, you should find two files in the `Pb_test` directory, called `simulation_result_0.dat` and `simulation_result_1.dat`, that contain outputs for the objective and the constraint, respectively.

Third step is to install Trieste. This package uses latest dev version of Trieste, which isn't released in Pipy yet. Therefore we recommend [cloning Trieste](https://github.com/secondmind-labs/trieste), creating a new virtual environment, and running `pip install -e path/to/trieste/root`.

## Bayesian optimization with Trieste

Example with R is in `Pb_test/ego.r`. Work in progress notebook with Trieste can be found in `ego_test.ipynb`.