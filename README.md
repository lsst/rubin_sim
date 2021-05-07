# rubin_sim
Scheduler, survey strategy analysis, and other simulation tools for Rubin Observaotry.



# Installation


Set up a conda envoronment and install rubin_sim from source in development mode:
```
conda create -n rubin scipy numpy healpy astropy pandas jupyterlab sqlite palpy matplotlib sqlalchemy`
conda activate rubin
pip install pyephem
git clone git@github.com:lsst/rubin_sim.git
cd rubin_sim
pip install -e .
```
XXX--next up, downloading the data files. Waiting for NCSA rsync to be up.


Future fast user install should look like:
```
conda create -n rubin rubin_sim
conda activate rubin
download_rubin_data # maybe some flags for all the data, or just the most common
```


# Developer Guide

For unit tests, all filename should start with `test_` so py.test can automatically find them.

