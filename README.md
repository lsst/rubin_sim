# rubin_sim
Scheduler, survey strategy analysis, and other simulation tools for Rubin Observaotry.



# Installation

Prerequisites:  A working [conda installation ](https://www.anaconda.com/products/individual)

Optional: Set data directory. By default, rubin_sim will look to and download extra data files to `$HOME/rubin_sim_data`. If you would like the data to go somewhere else, you can set the `RUBIN_SIM_DATA_DIR` environement variable, for example in bash with `export RUBIN_SIM_DATA_DIR="/my/preferred/data/path"` (note, always make sure this is set before trying to run `rubin_sim` packages, so put in your .bashrc or whatnot). Another possibility is to set the location via sim-link, `ln -s /my/preferred/data/path ~/rubin_sim_data`. 


Set up a conda envoronment and install rubin_sim from source in development mode:
```
conda create -n rubin -c conda-forge scikit-learn scipy numpy healpy astropy pandas jupyterlab sqlite palpy matplotlib sqlalchemy pytables`
conda activate rubin
pip install pyephem
git clone git@github.com:lsst/rubin_sim.git
cd rubin_sim
pip install -e .
rs_download_data
```
XXX--rs_download_data in progress

Future fast user install should look like:
```
conda create -n rubin rubin_sim
conda activate rubin
rs_download_data 
```

Optional dowload all the pre-computed sky data:
```
XXX--todo
```

## mix and match data files

If someone finds themselves in a situation where they want to use the latest code, but an older version of the data files, one could mix and match by:
```
rm -r ~/rubin_sim_data
git checkout <some old git sha>
rs_download_data
git checkout master
```
And viola, one has the current version of the code, but the data files from a previous version.



# Developer Guide

For unit tests, all filename should start with `test_` so py.test can automatically find them.

XXX--need to put instructions for updating the data sets on NCSA. 

XXX--to make changes to the code, checkout a new branch, make edits, push, make a PR.