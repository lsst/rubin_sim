.. py:currentmodule:: rubin_sim.maf

.. _maf-api-metricbundles:

==============
Metric Bundles
==============

.. _maf-pandas-constraints:

Pandas Constraints
==================

``MetricBundle`` accepts an optional ``pdconstraint`` containing a
``pandas.DataFrame.query`` expression. It filters the data after any SQL
``constraint`` and before bundle stackers run. ``MetricBundleGroup`` evaluates
each distinct pair of SQL and pandas constraints separately. Omitting
``pdconstraint`` (or setting it to ``None`` or ``""``) adds no pandas filter.

Predicate columns are not discovered or fetched automatically. They must be
stored input columns requested by the metric, slicer, or stacker requirements,
or explicitly added to ``bundle.db_cols`` before constructing the group. For
example, a visit-count metric does not otherwise require depth:

.. code-block:: python

    from rubin_sim import maf

    bundle = maf.MetricBundle(
        maf.CountMetric(col="observationStartMJD"),
        maf.UniSlicer(),
        constraint="",
        pdconstraint="fiveSigmaDepth > 0",
        info_label="positive_depth",
    )
    bundle.db_cols.add("fiveSigmaDepth")
    group = maf.MetricBundleGroup({bundle.file_root: bundle}, "visits.h5")
    group.run_all()

A predicate cannot use a column that will only be produced by a bundle stacker.
For example, ``dayObs`` can be filtered only if it is physically stored and
explicitly requested, not merely scheduled for creation by ``DayObsStacker``.
An omitted column may cause query failure; incidental availability in an
unprojected HDF5 read is not guaranteed. Backtick column names where pandas query
syntax requires it. The bundle interface supplies no external ``@var`` context.

``get_sim_data`` and ``get_visit_data`` also accept ``pdconstraint``. Include
predicate columns in ``dbcols`` when specifying a projection. These direct
utility calls apply explicitly supplied stackers before the pandas query,
unlike bundle execution, where stackers run after filtering.

Use distinct ``info_label`` values for bundles representing different pandas
selections. ``pdconstraint`` is not part of the results-database identity or
automatically generated filenames. Group initialization rejects different
pandas constraints with the same metric name, slicer name, run name, SQL
constraint, and info label, even if custom file roots differ. List inputs with
duplicate file roots raise the existing ``NameError`` during conversion to a
bundle dictionary, before the database-identity check.

.. warning::

    ``run_current(sim_data=...)`` bypasses both SQL and pandas filtering; the
    caller must supply the selected rows. Standalone ``reduce_all``,
    ``summary_all``, and ``plot_all`` select only bundles without a pandas
    constraint. Use the normal ``run_all`` path for reductions and summaries,
    and ``run_all(plot_now=True)`` when plots are needed for pandas-filtered
    bundles.

API Reference
=============

.. automodule:: rubin_sim.maf.metric_bundles
    :imported-members:
    :members:
    :show-inheritance:
