.. py:currentmodule:: rubin_sim

.. _installation:

############
Installation
############

Quick Installation
------------------

Installation from PyPI:

::

    pip install rubin-sim
    scheduler_download_data 
    rs_download_data

Note: pip installation of rubin-sim will lack the JPL data (DE405, etc.)
that is needed to actually run ``pyoorb``, used in ``rubin_sim.moving_objects``, as this is not currently available from PyPI.
Please see the `oorb installation instructions <https://github.com/oorb/oorb/wiki/Installation>`_ for more information.

or from conda-forge:

::

    conda install -c conda-forge rubin-sim
    scheduler_download_data 
    rs_download_data

The `scheduler_download_data` and `rs_download_data` commands will
download data files to the default location of `~/rubin_sim_data`.
To store the data elsewhere, see instructions at
:ref:`Data Download<data-download>`.

For Developer Use
-----------------

First, clone the `rubin_sim <https://github.com/lsst/rubin_sim>`_ repository:

::

 git clone git@github.com:lsst/rubin_sim.git
 cd rubin_sim
 conda create --channel conda-forge --name rubin-sim --file requirements.txt python=3.12
 conda activate rubin-sim
 conda install -c conda-forge --file=test-requirements.txt # Optional test requirements
 pip install -e . --no-deps
 scheduler_download_data 
 rs_download_data

The `scheduler_download_data` and `rs_download_data` commands will
download data files to the default location of `~/rubin_sim_data`.
To store the data elsewhere, see instructions at
:ref:`Data Download<data-download>`.

Note conda may override previous installs of 
`rubin_scheduler`, in which case one can uninstall the conda version
and re-run `pip install -e . --no-deps` from the needed git repo directory.

Building Documentation
----------------------

An online copy of the documentation is available at https://rubin-sim.lsst.io,
however building a local copy can be done as follows:

::

 pip install "documenteer[guide]"
 cd docs
 make html


The root of the local documentation will then be ``docs/_build/html/index.html``.

