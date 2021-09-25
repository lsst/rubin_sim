# rubin_sim
Scheduler, survey strategy analysis, and other simulation tools for Rubin Observatory.


[![Run Tests and Build Documentation](https://github.com/lsst/rubin_sim/actions/workflows/python-tests-doc.yml/badge.svg)](https://github.com/lsst/rubin_sim/actions/workflows/python-tests-doc.yml)


# Installation

Prerequisites:  A working [conda installation ](https://www.anaconda.com/products/individual)


To install rubin_sim into a new conda environment (the typical use-case), set up a conda environment and install rubin_sim from source in development mode:
```
git clone git@github.com:lsst/rubin_sim
cd rubin_sim
conda create -n rubin   ### optional (but recommended)
conda activate rubin    ### optional (if new environment created above)
conda install -c conda-forge --file=requirements.txt
pip install -e .
```
The installation can be tested by running `py.test` in the github directory.

However, we expect some users to want to install rubin_sim into an LSST stack environment, using only some of the basic options within rubin_sim such as photUtils. 
This can be done without impacting the LSST environment by 
```
source loadLSST.sh (or your equivalent)
conda install setuptools_scm
git clone git@github.com:lsst/rubin_sim.git
cd rubin_sim
pip install -e .
```
(and if you need to use additional modules within rubin_sim, you can conda install additional packages as needed). 

### Data download for rubin_sim ###

**Optional: Set $RUBIN_SIM_DATA_DIR data directory.** By default, `rubin_sim` will download needed data files to `$HOME/rubin_sim_data`. If you would like the data to go somewhere else, you can set the `RUBIN_SIM_DATA_DIR` environment variable. In bash  `export RUBIN_SIM_DATA_DIR="/my/preferred/data/path"` (note, always make sure this is set before trying to run `rubin_sim` packages, so put in your .bashrc or whatnot). Another possibility is to set the location via sym-link, `ln -s /my/preferred/data/path ~/rubin_sim_data`. 

```
export RUBIN_SIM_DATA_DIR=$HOME/rubin_sim_data # Optional. Set the data directory path via env variable
rs_download_data  # Downloads ~2Gb of data to $RUBIN_SIM_DATA_DIR
```
If you are only interested in a subset of the data, you can specify which directories to download, e.g.
```
rs_download_data  --dirs "throughputs,skybrightness,tests,maps"
```

If you have a previous installation of rubin_sim or wish to oupdate your data download, the flag `--force` will force an update of the data in the relevant $RUBIN_SIM_DATA_DIR directories. 


### Additional installation and download options ###

Optional dependencies used by some of the more esoteric MAF functions:
```
conda install -c conda-forge sncosmo bokeh sympy
```

Optional download all the (100 Gb) of pre-computed sky data. Only needed if you are planning to run full 10 year scheduler simulations. Not needed for MAF, etc.:
```
rs_download_sky
```


Future fast user install should look like:
```
conda create -n rubin rubin_sim
conda activate rubin
rs_download_data 
```


# Documentation

Online documentation is available at https://rubin_sim.lsst.io
Example jupyter notebooks can be found at:  https://github.com/lsst/rubin_sim_notebooks

To create a local build of the documentation:
```
conda install lsst-documenteer-pipelines
cd doc
make html
```


# Mix and match data files

If someone finds themselves in a situation where they want to use the latest code, but an older version of the data files, one could mix and match by:
```
git checkout <some old git sha>
rs_download_data --force
git checkout master
```
And viola, one has the current version of the code, but the data files from a previous version.


# Notes on installing/running on hyak (and other clusters)

A new anaconda install is around 11 GB (and hyak has a home dir quota of 10GB), so ensure your anaconda dir and the rubin_sim_data dir are not in your home directory. Helpful link to the anaconda linux install instructions:  https://docs.anaconda.com/anaconda/install/linux/

The `conda activate` command fails in a bash script. One must first `source ~/anaconda3/etc/profile.d/conda.sh
` (replace with path to your anaconda install if different), then `conda activate rubin`.

The conda create command failed a few times. It looks like creating the conda environement and then installing dependencies in 3-4 batches can be a work-around.

Handy command to get a build node on hyak `srun -p build --time=2:00:00 --mem=20G --pty /bin/bash`


# Developer Guide

To make changes to the code, checkout a new branch, make edits, push, make a pull request.

For unit tests, all filename should start with `test_` so py.test can automatically find them.

## Updating data files

To update the source contents of the data files:

* Update the files in your local installation
* If you are updating the baseline sim, create a symlink of the new database to baseline.db
* Create a new tar file with a new name, e.g., `tar -chvzf maf_june_2021.tgz maf`
* Copy your new tar file to NCSA lsst-login01.ncsa.illinois.edu:/lsstdata/user/staff/web_data/sim-data/rubin_sim_data/
* You can check that it is uploaded here: https://lsst.ncsa.illinois.edu/sim-data/rubin_sim_data/
* Update `bin/rs_download_data` so the `data_dict` function uses your new filename
* Push and merge the change to `bin/rs_download_data`
* Probably add a new tag.

