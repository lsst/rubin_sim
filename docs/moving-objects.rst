.. py:currentmodule:: rubin_sim

.. _moving-objects:


##############
Moving Objects
##############

The ``rubin_sim.movingObjects`` module provides tools to
generate simulated ephemerides of a population of
small bodies throughout an LSST pointing history.
These ephemerides are typically used for further
analysis in :ref:`MAF <maf>` to evaluate the effect of
survey strategy on various populations
of Solar System objects.

There are several populations available in the "orbits" directory of
$RUBIN_SIM_DATA_DIR. Many of these populations were contributed or
enhanced by the LSST Solar System Science Collaboration (SSSC).
Further documentation on these orbital populations is available in the
`LSST-SSSC "SSSC_test_populations" <https://github.com/lsst-sssc/SSSC_test_populations_gitlfs/tree/master/MAF_TEST>`_ repo.