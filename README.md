# rubin_sim
Scheduler, survey strategy analysis, and other simulation tools for Rubin Observatory.


[![pypi](https://img.shields.io/pypi/v/rubin-sim.svg)](https://pypi.org/project/rubin-sim/)
 [![Conda Version](https://img.shields.io/conda/vn/conda-forge/rubin-sim.svg)](https://anaconda.org/conda-forge/rubin-sim) <br>
[![Run CI](https://github.com/lsst/rubin_sim/actions/workflows/test_and_build.yaml/badge.svg)](https://github.com/lsst/rubin_sim/actions/workflows/test_and_build.yaml)
[![Build and Upload Docs](https://github.com/lsst/rubin_sim/actions/workflows/build_docs.yaml/badge.svg)](https://github.com/lsst/rubin_sim/actions/workflows/build_docs.yaml)
[![codecov](https://codecov.io/gh/lsst/rubin_sim/branch/main/graph/badge.svg?token=2BUBL8R9RH)](https://codecov.io/gh/lsst/rubin_sim)


[![DOI](https://zenodo.org/badge/365031715.svg)](https://zenodo.org/badge/latestdoi/365031715)


## Installation

### Conda Installation ###

If you are only running `rubin_sim` code and not making changes. If you will be editing the code or need the very latest verison, use the pip instructions below.
```
conda create -n rubin-sim -c conda-forge rubin_sim  ## Create a new environment and install rubin_sim
conda activate rubin-sim
rs_download_data  ## Downloads a few of data to $RUBIN_SIM_DATA_DIR (~/rubin_sim_data if unset)
conda install -c conda-forge jupyter  ## Optional install of jupyter
```
Note that this is not the best option for developers working on their own metrics - a pip installation from their own fork of the repo may work better.

### Pip installation ###

```
pip install rubin-sim
```

Please note that the pip installation of pyoorb does not come with the necessary data files. 
To actually use pyoorb, the data files are most easily installable via conda with
 ```
 conda install -c conda-forge openorb-data
 conda install -c conda-forge openorb-data-de405
 ```
The pip installation of `rubin_sim` will install the pip version of `pyoorb` which is
more up-to-date compared to the conda-forge version of `openorb`. For the purposes of 
`rubin_sim`, the functionality is essentially the same however.


### Developer Installation ###

To install `rubin_sim` from source using pip, with all dependencies (including jupyter):
```
git clone https://github.com/lsst/rubin_sim.git ; cd rubin_sim  ## clone and cd into repo
conda create -n rubin-sim --file=all_req.txt   ## optional (but recommended) new conda env
conda activate rubin-sim   ## substitute mamba for conda if you like
pip install -e . --no-deps
rs_download_data  ## Downloads a few GB of data to $RUBIN_SIM_DATA_DIR (~/rubin_sim_data if unset)
```
Note that external collaborators will likely want to follow similar directions, using a fork of our rubin_sim github repo first (and then clone from there).

### Data download for rubin_sim ###

**Optional: Set $RUBIN_SIM_DATA_DIR data directory.** By default, `rubin_sim` will download needed data files to `$HOME/rubin_sim_data`. If you would like the data to save elsewhere, you should set the `RUBIN_SIM_DATA_DIR` environment variable. In bash  `export RUBIN_SIM_DATA_DIR="/my/preferred/data/path"` (note, always make sure this is set before trying to run `rubin_sim` packages, so put in your .bashrc or whatnot). Another possibility is to set the location via sym-link, `ln -s /my/preferred/data/path ~/rubin_sim_data`.

```
export RUBIN_SIM_DATA_DIR=$HOME/rubin_sim_data  ## Optional. Set the data directory path via env variable
rs_download_data  ## Downloads a few GB of data to $RUBIN_SIM_DATA_DIR
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
jupyter notebook maf/tutorial/Survey_footprint.ipynb  
```


### Downloading additional skybrightness_pre skybrightness files ###

The default skybrightness_pre directory downloaded above contains only one month of pre-calculated skybrightness files.
If you wish to run the scheduler for a longer time period, or need this information outside of the span of that month period,
you will need to download a larger set of pre-computed sky data.

To download the entire optional set all the (43 Gb) of pre-computed sky data. 
```
rs_download_sky
```
Note that subsets of this data can get downloaded via http directly from
```
https://s3df.slac.stanford.edu/data/rubin/sim-data/sims_skybrightness_pre/h5_2023_09_12/
```
(the file names reflect the range of MJD covered within each data file).


## Documentation

Online documentation is available at https://rubin-sim.lsst.io
Example jupyter notebooks can be found at:  https://github.com/lsst/rubin_sim_notebooks

To create a local build of the documentation:
```
conda install -c conda-forge lsst-documenteer-pipelines
cd doc
make html
```

## Getting Help ##

Questions about `rubin_sim` can be posted on the [sims slack channel](https://lsstc.slack.com/archives/C2LQ5JW9W), or on https://community.lsst.org/ (tag @yoachim and/or @ljones so we get notifications about it).
