# rubin_sim
Scheduler, survey strategy analysis, and other simulation tools for Rubin Observaotry.



# Installation

Prerequisites:  A working [conda installation ](https://www.anaconda.com/products/individual)

Set up a conda envoronment and install rubin_sim from source in development mode:
```
conda create -n rubin scipy numpy healpy astropy pandas jupyterlab sqlite palpy matplotlib sqlalchemy`
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

# Developer Guide

For unit tests, all filename should start with `test_` so py.test can automatically find them.

XXX--need to put instructions for updating the data sets on NCSA. 

XXX--to make changes to the code, checkout a new branch, make edits, push, make a PR.