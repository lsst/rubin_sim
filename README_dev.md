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
* Copy your new tar file to S3DF USDF s3dflogin.slac.stanford.edu:/sdf/group/rubin/web_data/sim-data/rubin_sim_data/
* You can check that it is uploaded here: https://s3df.slac.stanford.edu/data/rubin/sim-data/rubin_sim_data/
* Update `rubin_sim/data/rs_download_data.py` so the `data_dict` function uses your new filename
* Push and merge the change to `bin/rs_download_data`
* Add a new tag, with a message indicating how the data package was changed. 

## Updating throughputs

Process for updating pre-computed files if system throughputs change.

0) update the throughputs in syseng_throughputs (this should be the original trigger to update throughputs anywhere downstream)
1) update the throughputs in lsst/throughputs (including new tag)
2) update rubin_sim_data/throughputs data files
3) update rubin_scheduler.utils.sys_eng_vals.py - there is a notebook in syseng_throughputs which generates this file 
4) recompute sky brightness files with rubin_sim.skybrightness.recalc_mags
5) remake skybrightness_pre files with rubin_sim/rubin_sim/skybrightness_pre/data/generate_hdf5.py
6) remake dark sky map with rubin_sim/rubin_sim/skybrightness_pre/data/generate_dark_sky.py
7) tar and update files at SDF (throughputs, skybrightness, skybrightness_pre)

