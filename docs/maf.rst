.. py:currentmodule:: rubin_sim

.. _maf:

################################
Metrics Analysis Framework (MAF)
################################

The ``rubin_sim.maf`` Metrics Analysis Framework (MAF) module is
intended to make it easier to write code to analyze our simulated LSST
pointing histories (often called "opsim outputs").

As an example: suppose one wanted to evaluate the LSST's performance in
regards to
characterizing a particular kind of periodic variable in a given simulated
survey. As such, you might have particular requirements on the parameters
of the observations at each point in RA/Dec space -- MAF will handle getting
the pointing history from the OpSim output database, splitting them up into
the observations relevant for each RA/Dec point, feeding these observations
into your contributed piece of code, and then consolidating the results of
that evaluation from each RA/Dec location into a map over the entire sky and
presenting those results in a variety of formats (a sky map visualization,
an area-weighted histogram, a power spectrum, or -- with a small bit of additional
analysis -- statistical summaries over the observed sky, such as the mean,
median, RMS, minimum or maximum values). In this case, you would only have
to write a small piece of code (a *metric*) that makes the actual evaluation,
assuming you have the relevant observations for a single piece of sky.
A simple list of all :ref:`available metrics <maf-metric-list>` is available.

A concrete example of this can be found in the KNeMetric - which is illustrated
in depth in a notebook in the github repo at `lsst/rubin_sim_notebooks
<https://github.com/lsst/rubin_sim_notebooks/blob/main/maf/science/KNe%20Metric.ipynb>`_
(see the maf/science directory).

MAF also provides lots of ready to use :ref:`metrics <maf-metric-list>`, as well as
a variety of ways to subdivide the pointing histories using :py:obj:`rubin_sim.maf.slicers`
-- a typical use case is to evaluate a quantity at all points over the sky, which would use
the :py:class:`rubin_sim.maf.slicers.HealpixSlicer` slicer, but there are
other spatially based slicers, such as the
:py:class:`rubin_sim.maf.slicers.UserPointsSlicer` which lets a user define
specific points at which to evaluate a given metric. Some metrics should be
evaluated on a basis which depends on a single data value, rather than spatially
across the sky (such as evaluating the time between filter changes per night, or
counting the number of observations per airmass bin) -- these typically need a
:py:class:`rubin_sim.maf.slicers.OneDSlicer`. Finally, there are metrics should
be applied to all of the observations at once, rather than subdividing them on
any basis; these should use the :py:class:`rubin_sim.maf.slicers.UniSlicer`.
Each slicer pairs with different plotting functions automatically, although
this is user-adjustable. MAF provides the framework to combine metrics, slicers
these plotting functions, and methods to save and restore the results of
each metric calculation.

For more examples of using MAF, please see our `tutorials`_.

.. _tutorials: https://github.com/lsst/rubin_sim_notebooks/tree/main/maf


.. toctree::

    List of Available Metrics <maf-metric-list>