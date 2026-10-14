.. py:currentmodule:: rubin_sim.clouds

.. _clouds:

Cloud Databases
===============

Historical weather replay is useful for long survey simulations, but an
unusually clear or cloudy historical month can make a poor reference for
short-term survey progress. The clouds module generates synthetic weather
whose individual calendar months reproduce pooled historical cloud-state and
transition frequencies within count tolerances. It is not a forecast and does
not execute or analyze scheduler simulations.

Use the command line for a complete database-generation workflow, or the
Python stages to inspect intermediate products in a notebook. Both work in an
environment with ``rubin_sim`` and its dependencies installed; USDF is a
supported use case, not a platform requirement. See :ref:`installation`,
:ref:`clouds-api`, and :ref:`clouds-architecture`.

For an interactive workflow with historical records and diagnostic plots, see
`generate_average_clouds.ipynb
<https://github.com/lsst/rubin_sim_notebooks/blob/tickets/LSST-64/clouds/generate_average_clouds.ipynb>`_
in the ``rubin_sim_notebooks`` repository's ``clouds`` directory. Supply the
historical input separately and review the notebook's configuration before
running it. The earlier ``generate_average_clouds_prototype.ipynb`` in that
directory is an experimental algorithm prototype, not the current API example.
The self-contained recipes below do not require either notebook.

Historical Input
----------------

Supply one or more readable CTIO record files with whitespace-separated
headings. This is an illustrative format fragment, not a complete dataset::

    sday eday month year q1 q2 q3 q4
    1    2    1     1975 0  0  0  0
    2    3    1     1975 0  2  3  4

``sday`` and ``eday`` are the starting and ending local calendar days of the
night; ``month`` and ``year`` describe its start. Quarters ``q1`` through
``q4`` contain integer cloud cover in eighths, from 0 (clear) to 8 (covered).
Both -1 and 9 mean missing data. Dates before 1975 are unsupported.

Multiple files are combined in input order; the first occurrence of a
duplicate night wins. Across the combined input, all twelve months must have
usable quarters. Generation also requires valid adjacent transitions in every
month. Missing quarters and omitted nights break adjacency, rather than being
interpreted as clear weather. The historical record files must be supplied
separately; these recipes do not obtain them through data-download utilities.

Meaning of Representative
-------------------------

For every generated year/month, state counts differ by at most one quarter
from the pooled historical fractions scaled to that month's length. Transition
counts use pooled historical *joint pair frequencies*, not conditional
row-stochastic probabilities. For expected pair count ``E``, the statistical
limit is one count if ``E < 1`` and ``sqrt(E)`` otherwise; the implementation
also caps that limit at ``MAX_TOLERANCE``, currently 1.0.

Each night has four quarters. States belong to the night-start calendar month.
A transition belongs to the month of its second quarter's starting night,
including fourth-to-first-quarter transitions between consecutive nights.
The first simulated quarter has no artificial incoming transition. Aggregate
diagnostic plots are useful comparisons, but do not independently prove the
contract for every generated year/month. See :ref:`clouds-architecture` for
the precise checks and generation algorithm.

Command-Line Recipe
-------------------

For a real historical dataset, the installed command has the following form
(replace the input paths and create the output directory first)::

    generate_average_clouds historical.txt supplemental.txt \
        --output-database output/average_clouds.db --num-years 20 --seed 6563 \
        --cloud-histogram output/states.png \
        --transition-histograms output/transitions.png \
        --cloud-figure output/sequence.png

``-o`` is an alias for ``--output-database``. Diagnostic paths are optional;
CLI transition plots combine all generated years and show raw counts, whereas
state histograms are normalized per month. Output directories must already
exist. ``--max-retries`` limits candidate attempts per month, including failed
attempts; ``--max-steps`` limits attempts to advance at a single quarter.
Exhausting the limits without a valid candidate raises an error, not a partial
valid result. Use ``--help`` for the complete option list.

``python -m rubin_sim.clouds`` is the equivalent module invocation. The
following executable example uses both invocations and a small synthetic,
all-clear dataset. It demonstrates the mechanics, not realistic weather or
performance. It needs no external records and provides the input used by the
Python recipe below. Run the examples in order in one Python session.

>>> from pathlib import Path
>>> from tempfile import TemporaryDirectory
>>> import subprocess
>>> import sys
>>> workspace = TemporaryDirectory(dir=Path.cwd())
>>> output = Path(workspace.name)
>>> historical_path = output / "historical.txt"
>>> header = "sday eday month year q1 q2 q3 q4\n"
>>> rows = [f"1 2 {month} 1975 0 0 0 0\n" for month in range(1, 13)]
>>> _ = historical_path.write_text(header + "".join(rows), encoding="utf-8")
>>> command = ["generate_average_clouds"]
>>> help_result = subprocess.run(command + ["--help"], capture_output=True, text=True)
>>> help_result.returncode == 0 and "--max-retries" in help_result.stdout
True
>>> result = subprocess.run(command + [str(historical_path),
...     "--output-database", str(output / "cli.db"), "--num-years", "1",
...     "--seed", "6563", "--cloud-histogram", str(output / "states.png"),
...     "--transition-histograms", str(output / "transitions.png"),
...     "--cloud-figure", str(output / "sequence.png")],
...     capture_output=True, text=True)
>>> assert result.returncode == 0, result.stderr
>>> all((output / name).is_file() for name in
...     ["cli.db", "states.png", "transitions.png", "sequence.png"])
True
>>> module_command = [sys.executable, "-m", "rubin_sim.clouds"]
>>> module_help = subprocess.run(module_command + ["--help"], capture_output=True, text=True)
>>> module_help.returncode == 0 and "--output-database" in module_help.stdout
True
>>> module_result = subprocess.run(module_command + [str(historical_path),
...     "-o", str(output / "module.db"), "--num-years", "1", "--seed", "6563"],
...     capture_output=True, text=True)
>>> assert module_result.returncode == 0, module_result.stderr
>>> (output / "module.db").is_file()
True

Staged Python Recipe
--------------------

For real data, replace ``historical_path`` with your supplied file or pass a
list of paths. The stages expose historical quarters, state counts, transition
counts, conditional probabilities, and generated records as pandas dataframes.
The probability matrix is for inspection; the generator consumes counts and
distributions, not that matrix. See :ref:`clouds-schemas` for their layouts.

>>> import matplotlib
>>> matplotlib.use("Agg")  # Noninteractive output; omit for interactive display.
>>> import matplotlib.pyplot as plt
>>> import rubin_sim.clouds as clouds
>>> historical = clouds.read_historical_clouds(historical_path)
>>> distribution = clouds.count_cloud_states(historical)
>>> transitions = clouds.compute_transition_matrix(historical)
>>> probabilities = clouds.compute_stochastic_matrix(transitions)
>>> list(historical.index.names)
['year', 'month', 'sday', 'quarter']
>>> bool((probabilities.sum(axis=1) == 1).all())
True
>>> simulated = clouds.generate_average_clouds(transitions, distribution, 1, seed=6563)
>>> bool(simulated.eighths.eq(0).all())
True
>>> int(simulated.c_date.iloc[-1]) - int(simulated.c_date.iloc[0]) >= 31557600
True
>>> clouds.save_clouds(output / "python.db", simulated)
>>> figure = clouds.plot_cloud_histogram(distribution, clouds.count_cloud_states(simulated))
>>> figure.savefig(output / "python-states.png")
>>> plt.close(figure)

To inspect a single simulated year without losing its incoming boundary pair,
supply the actual adjacent state when counting a slice. Only the beginning of
the complete simulation has no predecessor. This also works for a month slice;
do not concatenate separated months into artificial adjacent records.

>>> year = 1976
>>> selected = simulated.loc[simulated.index.get_level_values("year") == year]
>>> start = simulated.index.get_loc(selected.index[0])
>>> prior = None if start == 0 else int(simulated.eighths.iloc[start - 1])
>>> selected_transitions = clouds.compute_transition_matrix(selected, prior_state=prior)
>>> figure = clouds.plot_transition_histograms(transitions, selected_transitions)
>>> plt.close(figure)

For the shorter orchestration alternative, ``cloud_generation_workflow``
returns the generated dataframe, writes the database, and closes every
diagnostic figure it saves:

>>> generated = clouds.cloud_generation_workflow(historical_path,
...     output / "workflow.db", num_years=1, seed=6563)
>>> generated.equals(simulated)
True
>>> import sqlite3
>>> from contextlib import closing
>>> with closing(sqlite3.connect(output / "workflow.db")) as connection:
...     saved = connection.execute("SELECT COUNT(*), MIN(cloud), MAX(cloud) FROM Cloud").fetchone()
>>> saved == (len(generated), 0.0, 0.0)
True
>>> workspace.cleanup()

Diagnostics and Limitations
---------------------------

All plotting functions return Matplotlib figures owned by the caller. Display
them interactively or save them, and close them when finished.

* ``plot_cloud_histogram`` overlays normalized monthly state distributions;
  a dictionary of year-to-distribution values overlays individual years.
* ``plot_transition_histograms`` compares twelve pairs of raw-count matrices,
  each with its own color scale. Use a correctly counted slice for one year.
* ``make_cloud_figure`` accepts the ``eighths`` series of historical or
  simulated records. Without source labels, Cividis maps clear to dark and
  cloudy to yellow, with missing data red. With an aligned ``origin`` series,
  source determines color, alpha is ``1 - eighths/16``, and missing data white.
* ``observability_acf`` optionally compares one calendar month's historical
  and simulated observability autocorrelation. A quarter is observable only
  if its cloud fraction is strictly below ``cloud_threshold``. Specify
  ``year`` for one simulated year, or omit it to combine years. Calendar gaps,
  missing quarters, and out-of-month quarters remain NaN; the ``missing``
  argument defaults to ``"conservative"`` and controls their treatment.

ACF plotting requires ``statsmodels`` to be importable in the current Python
environment. It is optional and is not a named ``rubin_sim`` installation
extra. Its absence raises an informative ``ImportError``; too little varying
data (including the all-clear synthetic example above) raises ``ValueError``.

Invalid dataframe schemas, absent monthly observations, and absent monthly
transitions raise ``ValueError``. Check the named month and missing-data
coverage before increasing sampling limits. A generation ``RuntimeError``
means no acceptable candidate was found within the limits; increasing limits
may help but does not relax the statistical guarantees. There is no general
runtime guarantee for arbitrary historical inputs or hardware.

Output and Scheduler Compatibility
----------------------------------

The SQLite output has this exact table schema::

    CREATE TABLE Cloud(cloudId INTEGER PRIMARY KEY,c_date INTEGER,cloud DOUBLE, source TEXT);

IDs start at 1 in chronological order. ``cloud`` is fractional cover in
multiples of 1/8, and ``source`` is always ``simulation``. Saving replaces an
existing ``Cloud`` table transactionally and preserves unrelated tables;
missing simulated states cannot be saved.

``c_date`` stores integer quarter-center seconds since 1975-01-01 00:00 TAI.
Nights run between geometric 12-degree evening and morning twilight at Cerro
Pachon. Generation starts in January 1975 and includes leap days. It may add
one full month after the requested calendar years so that the first-to-last
sample span is at least ``num_years * 31557600`` seconds.

The scheduler aligns the stored samples to its configured simulation epoch,
selects the nearest cloud sample, and wraps with a whole number of Julian
years. The 1975 construction epoch is not a requirement to simulate in 1975,
and Julian-year wrapping does not guarantee exact calendar-month alignment
after repetition.
