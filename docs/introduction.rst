.. py:currentmodule:: rubin_sim

.. _introduction:

############
Introduction
############

The `Legacy Survey of Space and Time <http://www.lsst.org>`_ (LSST)
is anticipated to encompass around 2 million observations spanning a decade,
averaging 800 visits per night. The ``rubin_sim`` package was built to help
understand the predicted performance of the LSST.

The :ref:`Phot Utils<phot-utils>` module provides synthetic photometry
using provided throughput curves based on current predicted performance.

The :ref:`skybrightness<skybrightness>` module incorporates the ESO
sky model, modified to match measured sky conditions at the LSST site,
including an addition of a model for twilight skybrightness. This is used
to generate the pre-calculated skybrightness data used in
`rubin_scheduler <rubin-scheduler.lsst.io/skybrightness-pre.html>`_.

The :ref:`Moving Objects<moving-objects>` module provides a way to create
synthetic observations of moving objects, based on how they would appear in
pointing databases ("opsims") created by
`rubin_scheduler <rubin-scheduler.lsst.io>`_.

One of the major goals for ``rubin_sim`` is to enable efficient and
scientifically varied evaluation of the LSST survey strategy and progress,
by providing a framework to enable these metrics to run in a
standardized way on opsim outputs.
The :ref:`Metrics Analysis Framework<maf>` module provides these tools.

.. toctree::
    :maxdepth: 2

    User Guide <user-guide>
