.. _maf-progress:

===============
Survey Progress
===============

MAF can measure LSST survey progress both as completed and extrapolated to a
target date, as described in `RTN-092 <https://rtn-092.lsst.io/>`_. Metrics are
computed in advance and stored in MAF results databases for later visualization,
for example with ``schedview``. These commands do not collect completed visits,
generate baselines, or create progress plots.

Two kinds of visit sequences support different comparisons:

* A **snapshot** contains qualifying visits through an observing day. Running
  the same batch on completed visits and on a baseline measures actual and
  expected progress as a function of date.
* A **chimera** combines completed visits through a transition day with baseline
  visits after that day, through an extrapolation target. Its metrics estimate
  the outcome at that target given the visits already completed.

Chimeras do not model how the scheduler would respond to the actual survey
state. They are therefore a pessimistic extrapolation, not a replacement for a
new simulation initialized with the completed visits.

Inputs and Dates
================

Supply a completed-visits file extracted from consdb and a baseline simulation
file, both in the opsim schema. SQLite and HDF5 inputs are supported; HDF5 visit
files use the key ``observations``. The baseline must be aligned with the actual
survey start so that equal observing days are comparable. Ensure that the input
files contain the columns required by the chosen batch.

Dates are integer ``dayObs`` values formatted as ``YYYYMMDD``, following the
UTC-12 convention in `SITCOMTN-032 <https://sitcomtn-032.lsst.io/>`_. Observing
day D begins at 12:00 UTC on calendar date D and ends at 12:00 UTC the following
day. Steps are numbers of nights, not arithmetic increments to ``YYYYMMDD``.

The commands use end dates differently:

* ``run_progress_batches --end-dayobs D`` includes visits before the end of
  observing day D. It always evaluates D, even if the cadence does not land on
  that date. The start date sets the first evaluation date, not a lower bound
  on the visits in each snapshot.
* ``build_chimeras --end-dayobs E`` limits only the baseline portion of each
  chimera. Transition dates run from ``--start-dayobs`` through the latest
  ``dayObs`` in the completed-visits input, including that last date even when
  it is off cadence. E does not cap transition dates or completed visits.

For start day S, transition day T, and target day E, each chimera contains
completed visits with S <= ``dayObs`` <= T, followed by baseline visits with
T < ``dayObs`` <= E. Both sources exclude visits whose ``fiveSigmaDepth`` is
null or does not exceed ``FIVE_SIGMA_DEPTH_LIMIT`` in ``rubin_sim.maf.progress``
(default 0.0). ``snapshot_batch`` applies the same depth cut and does not require
or filter on a ``simulated`` column.

Chimera construction adds a missing completed-visits ``exposures`` column with
value 1, or fills null values in that column with 1, before keeping only columns
common to the two inputs. It adds a Boolean ``simulated`` column: ``False`` for
completed visits and ``True`` for baseline visits. This flag is useful for
inspection; the progress batches do not depend on it. The exposure normalization
is specific to chimera construction, not snapshot loading.

Command-Line Workflow
=====================

Installing ``rubin_sim`` provides the four commands below; each accepts
``--help``. Replace the example dates and filenames with those appropriate to
your inputs. The baseline snapshot series extends to the target date so that
its final snapshot supplies the baseline reference for extrapolated metrics.

.. code-block:: bash

    CONSDB_VISIT_DB="completed.db"
    BASELINE_DB="baseline.db"
    START_DAYOBS=20260629
    LAST_CONSDB_DAYOBS=20261001
    TARGET_DAYOBS=20360629
    STEP=30
    CHIMERA_DIR=./chimeras
    RESULTS_DIR=./results

    # Build stored sequences, each using baseline visits through the target.
    build_chimeras \
      --consdb-file "$CONSDB_VISIT_DB" \
      --opsim-file "$BASELINE_DB" \
      --start-dayobs "$START_DAYOBS" \
      --end-dayobs "$TARGET_DAYOBS" \
      --step "$STEP" \
      --out-dir "$CHIMERA_DIR"

    # Measure extrapolated progress on every chimera.
    run_chimera_batches \
      --chimera-dir "$CHIMERA_DIR" \
      --batch chimera_batch \
      --out-dir "$RESULTS_DIR"

    # Measure actual progress through the latest completed observing day.
    run_progress_batches \
      --visits-file "$CONSDB_VISIT_DB" \
      --start-dayobs "$START_DAYOBS" \
      --end-dayobs "$LAST_CONSDB_DAYOBS" \
      --step "$STEP" \
      --run-prefix consdb \
      --out-dir "$RESULTS_DIR"

    # Measure baseline progress, including the extrapolation-target reference.
    run_progress_batches \
      --visits-file "$BASELINE_DB" \
      --start-dayobs "$START_DAYOBS" \
      --end-dayobs "$TARGET_DAYOBS" \
      --step "$STEP" \
      --run-prefix baseline \
      --out-dir "$RESULTS_DIR"

    # Extract only chimera summaries from the shared results database.
    make_chimera_summary_table \
      --results-db "$RESULTS_DIR/resultsDb_sqlite.db" \
      --out-file "$RESULTS_DIR/chimera_summary.h5"

All three batch-running invocations use the same output directory and therefore
the same ``resultsDb_sqlite.db``. If only as-completed comparisons are needed,
the baseline series can end at ``LAST_CONSDB_DAYOBS`` instead. If the completed
end date is off cadence, a longer baseline series need not include that exact
date; evaluate it separately when an exact paired endpoint is needed.

Batch Selection
===============

``run_chimera_batches`` defaults to ``glanceBatch``, so specify
``--batch chimera_batch`` for the progress metrics. ``run_progress_batches``
defaults to ``snapshot_batch``. Batch names must resolve to callable functions
in ``rubin_sim.maf.batches``.

Both commands accept repeated ``--batch-kwarg KEY=VALUE`` options. Values are
parsed as Python literals when possible, otherwise kept as strings. For example,
``--batch-kwarg nside=64`` changes the HEALPix resolution. Snapshot batches must
accept both ``run_name`` and ``end_dayobs`` keyword arguments; chimera batches
must accept ``run_name`` or the legacy ``runName`` argument. Unknown batch names
and malformed keyword options exit with a non-zero status. ``--step`` defaults
to 1 for chimera construction and 30 for snapshots; ``--out-dir`` defaults to
the current directory.

Both progress batches accept ``colmap`` for column aliases and coordinate units,
``bands`` for the individual-band selections (an all-band selection is always
included), and ``label_prefix`` for metric info labels. These can be passed in
``batch_kwargs`` in Python or as literal ``--batch-kwarg`` values on the command
line. Chimera construction itself expects the standard opsim column names.

Existing batches can also run on stored chimeras. For example:

.. code-block:: bash

    run_chimera_batches \
      --chimera-dir "$CHIMERA_DIR" \
      --batch science_radar_batch \
      --batch-kwarg dayobs0="$START_DAYOBS" \
      --out-dir ./science_radar_results

``science_radar_batch(dayobs0=D)`` sets the start for its time-dependent metrics
to the beginning of observing day D: MJD at 00:00 UTC on D plus 0.5. For example,
``dayobs0=20260629`` corresponds to ``mjd0=61220.5``. Inconsistent ``dayobs0``
and ``mjd0`` values raise ``ValueError``; omitting both preserves the batch's
default behavior.

Metrics and Outputs
===================

``snapshot_batch`` and ``chimera_batch`` compute the following for each of
u, g, r, i, z, y and for all bands combined:

* Total effective exposure time and visit count.
* HEALPix maps of visit count and coadded five-sigma depth, with standard summary
  statistics, the minimum over the best 18,000 square degrees (``top18k``), and
  the 10th percentile.

``chimera_batch`` additionally computes fOArea and fONv, both raw and normalized
to the 18,000 square degree / 825-visit benchmark, and the area with at least
750 visits. Snapshot fO values are not produced: they can remain near zero long
after substantial progress has been made. The fO baseline reference comes from
the baseline selection and acceptance process, not the baseline snapshot series.
For the other metrics, ``baseline_<target date>`` supplies the extrapolated
baseline reference.

Stored outputs have the following conventions:

* ``chimera_YYYYMMDD.h5`` uses the HDF5 key ``observations``. Each file is an
  ordinary visits source for ``MetricBundleGroup``.
* Runs in ``resultsDb_sqlite.db`` are named ``chimera_YYYYMMDD`` for transition
  dates and ``<prefix>_YYYYMMDD`` for snapshots. The snapshot prefix defaults to
  ``consdb``; use ``baseline`` to distinguish baseline snapshots.
* ``chimera_summary.h5`` uses the key ``summary``. It contains only chimera runs,
  with one row per transition date, an ascending integer ``transition_dayobs``
  index, and four-level columns ``(metric_name, slicer_name, metric_info_label,
  summary_metric)``. When no chimera summaries are available, the command warns
  and does not write a new summary file.

The summary table can be read with pandas:

.. code-block:: python

    import pandas as pd

    summaries = pd.read_hdf("results/chimera_summary.h5", key="summary")

Operational Caveats
===================

Choose ``nside`` high enough that every LSST camera pointing covers at least
one HEALPixel; both progress batches default to 32. Too coarse a resolution can
leave a sparse snapshot's map empty and cause its minimum reduction to fail.
``run_progress_batches`` warns with the affected run name, retains any already
recorded rows, and continues to later dates for this specific error. That
snapshot may have incomplete results and should be rerun at an adequate
resolution. Other ``ValueError`` exceptions propagate; chimera runs do not have
this snapshot recovery behavior.

Dates before the first qualifying visit may have no results-database rows. Do
not interpret a missing date or an incomplete snapshot as a measured zero.

Stored chimeras duplicate visits across transition dates and can consume
substantial disk space. Rebuild them when completed visits or the baseline
change. Use fresh chimera and results directories for a new input revision or
cadence: the chimera runner processes all matching files in its directory, and
the results database is opened rather than cleared. These commands do not
remove stale files or results automatically.

Progress batches select dates and bands using in-memory pandas queries, avoiding
conversion of HDF5 inputs to SQLite for those selections. For custom bundles,
see :ref:`maf-pandas-constraints` for column-request and result-label rules.
The Python orchestration interfaces are in :ref:`maf-api-progress`.
