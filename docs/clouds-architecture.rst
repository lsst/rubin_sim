.. py:currentmodule:: rubin_sim.clouds

.. _clouds-architecture:

Cloud Generation Architecture
=============================

The :ref:`clouds` workflows use a pipe-and-filter sequence: read historical
quarters, count states and transitions, generate representative months, save
the database, and optionally plot diagnostics. The stages share the pandas
schemas in :ref:`clouds-schemas`. ``cloud_generation_workflow`` orchestrates
them for the CLI; notebooks can call the stages separately. Conditional
probabilities from ``compute_stochastic_matrix`` are an inspection product,
not an input to the constrained generator.

Inspecting Shared Schemas
-------------------------

The stages exchange ordinary pandas dataframes with the layouts described in
:ref:`clouds-schemas`. This self-contained example builds one synthetic night
per month, with cloud states 0, 2, 8, and a missing quarter (9). It illustrates
the data contracts, not a realistic historical sample.

>>> from datetime import date
>>> import rubin_sim.clouds as clouds
>>> historical = clouds.make_cloud_frame(
...     [date(1975, month, 1) for month in range(1, 13)], [0, 2, 8, 9] * 12)
>>> list(historical.index.names)
['year', 'month', 'sday', 'quarter']
>>> list(historical.columns)
['eighths', 'cloud', 'c_date']
>>> historical.cloud.iloc[:3].tolist()
[0.0, 0.25, 1.0]
>>> bool(historical.cloud.isna().iloc[3])
True

State counts have one row per month and one column per valid state. The missing
quarter contributes no count, so each synthetic month contains three counted
quarters rather than four.

>>> distribution = clouds.count_cloud_states(historical)
>>> distribution.loc[1].tolist()
[1, 0, 1, 0, 0, 0, 0, 0, 1]

Transition counts use a complete month/origin index and destination columns.
Only the two valid adjacent pairs, 0-to-2 and 2-to-8, contribute in each night;
there is no transition into the missing quarter. The stochastic matrix retains
this layout but contains floating-point conditional probabilities instead of
integer counts.

>>> transitions = clouds.compute_transition_matrix(historical)
>>> transitions.shape, list(transitions.index.names)
((108, 9), ['month', 'origin'])
>>> transitions.columns.name, transitions.columns.tolist()
('destination', [0, 1, 2, 3, 4, 5, 6, 7, 8])
>>> int(transitions.loc[(1, 0), 2])
1
>>> probabilities = clouds.compute_stochastic_matrix(transitions)
>>> bool(probabilities.sum(axis=1).eq(1).all())
True

Origin 1 has no historical transitions. Its fallback priority selects origin 2
before origin 0, so its inspection probabilities favor destination 8.

>>> float(probabilities.loc[(1, 1), 8])
1.0

These examples illustrate representative contracts; the dtype details remain
in :ref:`clouds-schemas`. Public consumers reject malformed records rather than
repairing them:

>>> clouds.count_cloud_states(historical.drop(columns='cloud'))  # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
    ...
ValueError: ...

Statistical Invariants
----------------------

For a calendar month, let :math:`D_i` be historical state counts pooled across
years, :math:`T_{ij}` pooled transition counts, and :math:`Q` the generated
quarter count (four times the calendar day count). Expected state counts are
:math:`S_i = Q D_i / \sum_k D_k`. Each generated month must satisfy
:math:`|N_i-S_i| \leq 1`.

Let :math:`P=Q-1` for the first simulated month and :math:`P=Q` thereafter.
Expected transition counts are :math:`E_{ij}=P T_{ij}/\sum_{kl}T_{kl}`,
not row-normalized transition probabilities. Actual adjacent-pair counts
:math:`C_{ij}` must satisfy

.. math::

   |C_{ij}-E_{ij}| \leq \min\left(\texttt{MAX\_TOLERANCE},
   \begin{cases}1 & E_{ij}<1\\ \sqrt{E_{ij}} & E_{ij}\geq1\end{cases}\right).

The implementation allows numerical slack of :math:`10^{-12}` in both
comparisons. ``MAX_TOLERANCE`` is currently 1.0, so every pair is within one
count by default, including historically zero pairs. These checks apply to
each generated year/month, including any extra coverage month. Aggregate
similarity is not a substitute for them.

Historical missing states, omitted quarters, and calendar gaps break pair
adjacency. Consecutive nights include q4-to-q1 pairs. A pair belongs to its
destination quarter's night-start month. The first simulated month excludes
any artificial predecessor; subsequent months include their real incoming
pair from the previously accepted month.

Here, the January 31 q4-to-February 1 q1 transition belongs to February. Neither
the missing quarter on February 1 nor the absent February 2 night creates a
pair. The first queried cell counts only the real month-boundary pair; the
second excludes the pair separated by missing data.

>>> boundary = clouds.make_cloud_frame(
...     [date(1975, 1, 31), date(1975, 2, 1), date(1975, 2, 3)],
...     [0, 0, 0, 2, 8, 9, 2, 2, 8, 8, 8, 8])
>>> boundary_counts = clouds.compute_transition_matrix(boundary)
>>> int(boundary_counts.loc[(2, 2), 8])
1
>>> int(boundary_counts.loc[(2, 8), 2])
0

Budgets and Fallback
--------------------

State quotas start at the floors of expected counts. Remaining quarters are
allocated to distinct states with fractional-remainder-weighted random draws.
Transition budgets start at expected joint pair counts. Destination columns
are masked when their state quotas are exhausted.

At each quarter, the sampler first tries the current origin's remaining budget
row, then the first usable fallback row. It samples a destination proportional
to that row's remaining weights, consumes a state quota and a selected-row
transition budget, and clips that budget at zero. If all rows are depleted,
it replenishes budgets from historical counts scaled to remaining quarters,
again masking exhausted destinations. ``max_steps`` bounds advancement attempts
at a single quarter, including replenishment. Completed candidates are checked
against actual adjacent pairs, not borrowed fallback origins, and rejected if
either invariant fails.

.. _clouds-fallback:

Fallback Priorities
~~~~~~~~~~~~~~~~~~~

After trying the actual origin, use these origins in order::

    0: 1, 2, 3, 4, 5, 6, 7, 8
    1: 2, 0, 3, 4, 5, 6, 7, 8
    2: 1, 0, 3, 4, 5, 6, 7, 8
    3: 4, 5, 6, 7, 8, 2, 1, 0
    4: 3, 5, 6, 7, 8, 2, 1, 0
    5: 6, 7, 4, 8, 3, 2, 1, 0
    6: 5, 7, 4, 8, 3, 2, 1, 0
    7: 6, 5, 8, 4, 3, 2, 1, 0
    8: 7, 6, 5, 4, 3, 2, 1, 0

The prescribed priorities allow progress when an origin has no usable budget;
they are not strictly nearest-state sorting. Replenishment handles exhaustion
from fractional budgets and rounding. Neither mechanism exempts actual pairs
from the final checks. ``compute_stochastic_matrix`` uses this ordering only
for historically empty rows, copying the first supported row's probabilities;
generation can also borrow when a previously populated budget row is depleted.

Candidate Selection
-------------------

One continuing seeded NumPy random generator serves all attempts and months.
Failed attempts do not reseed it. ``max_retries`` limits all candidate attempts
per year/month. ``NUM_MONTH_CANDIDATES``, currently five, targets valid
candidates; a smaller valid pool is sufficient if attempts run out. A month
with no valid candidate raises ``RuntimeError``. These module constants are
implementation controls, not CLI options or a promised tuning API.

For accepted cumulative counts :math:`A_m` of calendar month :math:`m` and
candidate counts :math:`C`, choose the candidate minimizing

.. math::

   \sum_{ij}\left|A_{m,ij}+C_{ij}
   - (\sum_{kl}A_{m,kl}+P)\frac{T_{m,ij}}{\sum_{kl}T_{m,kl}}\right|.

All cells participate, and ties keep the first candidate. Every candidate
starts from the same accepted predecessor; only the winner enters the
accumulator for that calendar month. This is a greedy cumulative-transition
heuristic, not another correctness threshold or a global optimum. More
candidates need not improve an entire run: choices change later predecessors
and additional draws change the future random stream.

Timing Compatibility
--------------------

Generation starts in January 1975 with actual calendar lengths and leap days.
``make_cloud_frame`` calculates geometric -12-degree solar-altitude boundaries
at Cerro Pachon, divides the elapsed TAI night into quarters, and places centers
at 1/8, 3/8, 5/8, and 7/8 of the duration. Integer seconds are rounded relative
to 1975-01-01 00:00 TAI. Twilight calculations locally disable IERS downloads.

The exact coverage predicate is
``max(c_date) - min(c_date) >= num_years * 31557600``. After the nominal
calendar years, one full January is added if needed. The scheduler's
``CloudData`` uses its configured start-year/offset alignment and nearest
stored samples; its repeat interval is the stored span rounded down to a
whole number of Julian years. This preserves usable duration without
promising exact calendar-month alignment after wrapping.

Verification
------------

From the repository root in an activated development environment, run
``python -m pytest tests/clouds``. The requirements tests in
``tests/clouds/test_average_clouds.py`` use synthetic inputs to check historical
reading, monthly count constraints, intermediate products, timing and scheduler
compatibility, database writing, diagnostics, and CLI/API output agreement.
A single documentation test runs this page's short public-API examples to catch
representative schema, validation, fallback, and adjacency mismatches. It does
not exhaustively test sampler internals or rerun the user-guide workflows.
To run just the architecture doctests, use
``python -m pytest tests/clouds/test_documentation.py``.

The opt-in test
``tests/clouds/test_average_clouds.py::test_historical_data_workflow`` is enabled
by setting ``LSST64_HISTORICAL_DATA`` to a historical record path. It uses the
17,414-night historical input, generates 20 years and diagnostics, and checks
the results and elapsed workflow time. A skipped opt-in test is not a passing
historical-data verification. The one-hour acceptance target is specific to
that workload and the tested USDF resource allocation, not a general performance
guarantee. Run results and operational acceptance evidence belong in issue/PR
history, not in a maintained site-specific batch runner.
