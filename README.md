# rubin_sim
Scheduler, survey strategy analysis, and other simulation tools for Rubin Observatory.



# Installation

Prerequisites:  A working [conda installation ](https://www.anaconda.com/products/individual)

Optional: Set data directory. By default, `rubin_sim` will download needed data files to `$HOME/rubin_sim_data`. If you would like the data to go somewhere else, you can set the `RUBIN_SIM_DATA_DIR` environment variable. In bash  `export RUBIN_SIM_DATA_DIR="/my/preferred/data/path"` (note, always make sure this is set before trying to run `rubin_sim` packages, so put in your .bashrc or whatnot). Another possibility is to set the location via sym-link, `ln -s /my/preferred/data/path ~/rubin_sim_data`. 


Set up a conda environment and install rubin_sim from source in development mode:
```
conda create -n rubin -c conda-forge openorb openorb-data-de405 astroplan george scikit-learn scipy numpy healpy astropy pandas jupyterlab sqlite palpy matplotlib sqlalchemy pytables h5py colorcet setuptools_scm
conda activate rubin
git clone git@github.com:lsst/rubin_sim.git
cd rubin_sim
pip install -e .
rs_download_data  # Downloads ~1.5G of data to $HOME/rubin_sim_data. See README on how to set a different data directory.
```

The installation can be tested by running `py.test` in the github directory. 

Future fast user install should look like:
```
conda create -n rubin rubin_sim
conda activate rubin
rs_download_data 
```

Optional dowload all the (100 Gb) pre-computed sky data. Only needed if you are planning to run full 10 year scheduler simulations. Not needed for MAF, etc.:
```
rs_download_sky
```

# Documentation

Example jupyter notebooks can be found at:  https://github.com/lsst/rubin_sim_notebooks

Building real documentation coming soon!


# Mix and match data files

If someone finds themselves in a situation where they want to use the latest code, but an older version of the data files, one could mix and match by:
```
rm -r ~/rubin_sim_data
git checkout <some old git sha>
rs_download_data
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

Note when developing after running `pip install -e .`, you can edit the python and run to see the changes. However, scripts in the `bin/` directory are copied to the anaconda `envs/<envname>/bin/` folder, so editing those in place will have no effect. To test changes to scripts, re-run `pip install -e .` to re-copy the scripts. I think? Maybe you can edit in place. Anyway, doesn't hurt to re-run pip.

## Updating data files

To update the data files:

* Update the files in your local installation
* Create a new tar file with a new name, e.g., `tar -chvzf maf_june_2021.tgz maf`
* Copy your new tar file to NCSA lsst-login01.ncsa.illinois.edu:/lsstdata/user/staff/web_data/sim-data/rubin_sim_data/
* You can check that it is uploaded here: https://lsst.ncsa.illinois.edu/sim-data/rubin_sim_data/
* Update `bin/rs_download_data` so the `data_dict` function uses your new filename
* push and merge the change to `bin/rs_download_data`
