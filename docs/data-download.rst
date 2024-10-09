.. py:currentmodule:: rubin_sim.data

.. _data-download:

=============
Data Download
=============

The ``rubin_sim.data`` module provides a script to download the data
required to run various modules in ``rubin_sim``, as well as to check the
expected versions of the data. It also provides utilities to interpret
the location of this $RUBIN_SIM_DATA_DIR on disk and to return the
path to the current baseline simulation output (one of the datafiles
downloaded by this module).

With the split of ``rubin_sim`` into ``rubin_sim`` + ``rubin_scheduler``, the
required data download utilities now live in the
`rubin_scheduler.data <https://rubin-scheduler.lsst.io/data-download.html>`_
package. ``rubin_scheduler`` is a necessary dependency of ``rubin_sim`` and
should have
been installed during the :ref:`installation <installation>` process.
The ``rubin_sim.data`` module simply provides additional information on the
data files necessary for ``rubin_sim``, then calls the scripts from
``rubin_scheduler.data`` to execute the download.


Downloading Necessary Data
^^^^^^^^^^^^^^^^^^^^^^^^^^

Please see the information in the
`rubin-scheduler "Downloading Necessary Data" documentation <https://rubin-scheduler.lsst.io/data-download.html#downloading-necessary-data>`_
for more details on setting up $RUBIN_SIM_DATA_DIR (which is
shared between ``rubin_scheduler``, ``rubin_sim`` and ``schedview``).

Using either the default path to $RUBIN_SIM_DATA_DIR, or after setting it
explicitly, first download the necessary data for ``rubin_scheduler`` and
then add the (larger) data set for ``rubin_sim``:

.. code-block:: bash

    scheduler_download_data
    rs_download_data

This creates a series of directories at $RUBIN_SIM_DATA_DIR (in addition
to the directories originating from `rubin_scheduler <https://rubin-scheduler.lsst.io/data-download.html#downloading-necessary-data>`_):

* maf (containing data used for various metrics)
* maps (containing various stellar density and 2-D and 3-D dust maps)
* movingObjects (containing asteroid SEDs)
* orbits (containing orbits for Solar System population samples)
* orbits_precompute (precomputed daily orbits for the samples above)
* sim_baseline (containing the current baseline simulation output)
* skybrightness (containing information needed for the skybrightness module)
* throughputs (current baseline throughput information)
* test (containing data for unit tests)


Note that the data will only be downloaded for the directories which do
not already exist, regardless of whether the version on disk is up to date.
To force an update to a version which matches the ``rubin_scheduler`` version:

.. code-block:: bash

    rs_download_data --update

This can also be applied only to certain directories, using the
``--dirs`` flag. It may be worth noting that some of the above directories
are more sizeable than others -- the ``maps``, ``maf`` and
``orbits_precompute`` directories are the largest and if not needed, can
be skipped in download by using ``--dirs``.