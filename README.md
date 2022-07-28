# rubin_sim
Scheduler, survey strategy analysis, and other simulation tools for Rubin Observatory.


[![Run Tests and Build Documentation](https://github.com/lsst/rubin_sim/actions/workflows/python-tests-doc.yml/badge.svg)](https://github.com/lsst/rubin_sim/actions/workflows/python-tests-doc.yml)


# Installation

Prerequisites:  A working [conda installation ](https://www.anaconda.com/products/individual)

### Conda Installation ###

If you are only running `rubin_sim` code and not making changes. If you will be editing the code or need the very latest verison, use the pip instructions below.
```
conda create -n rubin -c conda-forge rubin_sim # Create a new environment
conda activate rubin
rs_download_data  # Downloads ~2Gb of data to $RUBIN_SIM_DATA_DIR (~/rubin_sim_data if unset)
conda install -c conda-forge jupyter # Optional install of jupyter
```
Note that this is not the best option for developers working on their own metrics - a pip installation from their own fork of the repo may work better.

### Pip Installation ###

To install rubin_sim from source using pip:
```
git clone https://github.com/lsst/rubin_sim.git
cd rubin_sim
conda create -n rubin & conda activate rubin  # optional (but recommended)
conda install -c conda-forge --file=requirements.txt
conda install -c conda-forge jupyter   # if you want to use jupyter notebook in this environment
conda install -c conda-forge --file=test-requirements.txt # If you will want to run unit tests
pip install -e .
rs_download_data  # Downloads ~2Gb of data to $RUBIN_SIM_DATA_DIR (~/rubin_sim_data if unset)
```
Note that external collaborators will likely want to follow similar directions, except create a fork of our rubin_sim github repo first (and then clone from there).


### Install into an LSST Stack Environment ###

We expect some users to want to install rubin_sim into an LSST stack environment, using only some of the basic options within rubin_sim such as photUtils.
This can be done without impacting the LSST environment by 
```
source loadLSST.sh (or your equivalent)
conda install -c conda-forge setuptools_scm
git clone https://github.com/lsst/rubin_sim.git
cd rubin_sim
pip install -e .
rs_download_data  # Downloads ~2Gb of data to $RUBIN_SIM_DATA_DIR (~/rubin_sim_data if unset)
```

### Data download for rubin_sim ###

**Optional: Set $RUBIN_SIM_DATA_DIR data directory.** By default, `rubin_sim` will download needed data files to `$HOME/rubin_sim_data`. If you would like the data to save elsewhere, you should set the `RUBIN_SIM_DATA_DIR` environment variable. In bash  `export RUBIN_SIM_DATA_DIR="/my/preferred/data/path"` (note, always make sure this is set before trying to run `rubin_sim` packages, so put in your .bashrc or whatnot). Another possibility is to set the location via sym-link, `ln -s /my/preferred/data/path ~/rubin_sim_data`.

```
export RUBIN_SIM_DATA_DIR=$HOME/rubin_sim_data # Optional. Set the data directory path via env variable
rs_download_data  # Downloads ~2Gb of data to $RUBIN_SIM_DATA_DIR
```
If you are only interested in a subset of the data, you can specify which directories to download, e.g.
```
rs_download_data  --dirs "throughputs,skybrightness,tests,maps"
```

If you have a previous installation of rubin_sim or wish to update your data download, the flag `--force` will force an update of the data in the relevant $RUBIN_SIM_DATA_DIR directories. 


**Example notebooks** to test and further explore rubin_sim, are available at [rubin_sim_notebooks](https://github.com/lsst/rubin_sim_notebooks). 
```
git clone https://github.com/lsst/rubin_sim_notebooks.git
cd rubin_sim_notebooks
# Example: make a plot of the number of visits per pointing
jupyter notebook maf/tutorial/Survey\ Footprint.ipynb  
```


### Additional installation and download options ###

Optional dependencies used by some of the more esoteric MAF functions:
```
conda install -c conda-forge sncosmo sympy george
```

Optional download all the (100 Gb) of pre-computed sky data. Only needed if you are planning to run full 10 year scheduler simulations. Not needed for MAF, etc.:
```
rs_download_sky
```



# Documentation

Online documentation is available at https://rubin-sim.lsst.io
Example jupyter notebooks can be found at:  https://github.com/lsst/rubin_sim_notebooks

To create a local build of the documentation:
```
conda install -c conda-forge lsst-documenteer-pipelines
cd doc
make html
```

## Getting Help ##

Questions about `rubin_sim` can be posted on the [sims slack channel](https://lsstc.slack.com/archives/C2LQ5JW9W), or on https://community.lsst.org/

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

The conda create command failed a few times. It looks like creating the conda environment and then installing dependencies in 3-4 batches can be a work-around.

Handy command to get a build node on hyak `srun -p build --time=2:00:00 --mem=20G --pty /bin/bash`


# Developer Guide

If you have push permissions to rubin_sim, you can make changes to the code by checking out a new branch, making edits, push and then make a pull request.
However, we do expect many users who wish to contribute metrics will not have these permissions -- for these contributors the easiest way to do development on rubin_sim may be the following:
 - create a fork of rubin_sim 
 - pip install the fork copy as above (but git clone your own fork, and then use this copy of rubin_sim)
 - edit the code in your fork of rubin_sim, test it, etc.
 - issue a PR from your fork to our original lsst/rubin_sim repository

When contributing code, metrics for MAF can be placed into either rubin_sim/rubin_sim/maf/metrics or rubin_sim/rubin_sim/maf/mafContrib (preferably rubin_sim/maf/metrics). Adding a unit test in the appropriate rubin_sim/tests directory is desirable. For unit tests, all filename should start with `test_` so py.test can automatically find them. An example notebook can be contributed to lsst/rubin_sim_notebooks. 

When contributing to the package, make sure you reformat the code with `black` before commiting.
The package ships with a `pre-commit` configuration file, which allows developers to install a git hook that will reformat the code before commiting.
Most IDEs also contains `black` reformat add-ons.

To install the `pre-commit` hook first install the `pre-commit` package with:
```
conda install -c conda-forge pre-commit
```

Then, install the hook with:
```
pre-commit install
```

## Updating data files

(This must be done by project developers only at this time). 
To update the source contents of the data files:

* Update the files in your local installation
* If you are updating the baseline sim, create a symlink of the new database to baseline.db
* Create a new tar file with a new name, e.g., `tar -chvzf maf_2021_06_01.tgz maf` (no `-h` if symlinks should stay as symlinks)
* Copy your new tar file to NCSA lsst-login01.ncsa.illinois.edu:/lsstdata/user/staff/web_data/sim-data/rubin_sim_data/
* You can check that it is uploaded here: https://lsst.ncsa.illinois.edu/sim-data/rubin_sim_data/
* Update `rubin_sim/data/rs_download_data.py` so the `data_dict` function uses your new filename
* Push and merge the change to `bin/rs_download_data`
* Probably add a new tag.

