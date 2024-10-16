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

Note: pip installation of rubin-sim will lack the JPL data (DE405, etc.)
that is needed to actually run ``pyoorb``, used in ``rubin_sim.moving_objects``, as this is not currently available from PyPI.
Please see the `oorb installation instructions <https://github.com/oorb/oorb/wiki/Installation>`_ for more information.

or from conda-forge:

::

    conda install -c conda-forge rubin-sim

Please note that following either installation,
additional data must be downloaded to use the software,
following the instructions at
:ref:`Data Download<data-download>`.

For Developer Use
-----------------

First, clone the `rubin_sim <https://github.com/lsst/rubin_sim>`_ repository:

::

 git clone git@github.com:lsst/rubin_sim.git
 cd rubin_sim


Create a conda environment for it:

::

 conda create --channel conda-forge --name rubin-sim --file requirements.txt python=3.11


If you want to run tests (please do), install the test requirements as well:

::

 conda activate rubin-sim
 conda install -c conda-forge --file=test-requirements.txt


Install the ``rubin_sim`` package into this environment (from the rubin_sim directory):

::

 pip install -e . --no-deps

Please note that following installation,
additional data must be downloaded to use the software,
following the instructions at
:ref:`Data Download<data-download>`.


Building Documentation
----------------------

An online copy of the documentation is available at https://rubin-sim.lsst.io,
however building a local copy can be done as follows:

::

 pip install "documenteer[guide]"
 cd docs
 make html


The root of the local documentation will then be ``docs/_build/html/index.html``.

