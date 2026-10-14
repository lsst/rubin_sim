.. py:currentmodule:: rubin_sim.clouds

.. _clouds-api:

Clouds API
==========

See :ref:`clouds` for workflows and :ref:`clouds-architecture` for the
generation algorithm. Function signatures, parameter semantics, defaults, and
exceptions below are generated from source docstrings.

.. _clouds-schemas:

Dataframe Schemas
-----------------

``Clouds``, ``TransitionMatrix``, ``StochasticMatrix``, and
``CloudDistribution`` are type aliases for ``pandas.DataFrame``, not separate
container classes. The schema reference is explicit because ordinary comments
beside the aliases are not rendered by autodoc. Functions validate the schemas
they consume and raise ``ValueError`` for malformed input, rather than silently
sorting or repairing it. NumPy ``uintp`` is the platform-sized unsigned integer.

Clouds
~~~~~~

Rows are unique and sorted by the four index levels in the order below. All
calendar fields refer to the local date on which the night began, on or after
1975-01-01. Columns have the following order; ``c_date`` is strictly increasing.

.. list-table:: Clouds fields
   :header-rows: 1

   * - Field
     - Location
     - NumPy dtype
     - Meaning
   * - ``year``
     - Index 1
     - ``uint16``
     - Night-start year
   * - ``month``
     - Index 2
     - ``uint8``
     - Month 1 through 12
   * - ``sday``
     - Index 3
     - ``uint8``
     - Night-start day of month
   * - ``quarter``
     - Index 4
     - ``uint8``
     - Quarter 1 through 4
   * - ``eighths``
     - Column 1
     - ``uint8``
     - State 0 through 8; 9 means missing
   * - ``cloud``
     - Column 2
     - ``float64``
     - ``eighths / 8``; NaN for missing state 9
   * - ``c_date``
     - Column 3
     - ``uintp``
     - Quarter-center seconds since 1975-01-01 00:00 TAI

TransitionMatrix and StochasticMatrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two-level index is ``(month, origin)``, with both levels ``uint8``.
Every combination of months 1 through 12 and origins 0 through 8 occurs
exactly once in that sorted order. Integer destination columns 0 through 8
are in numerical order, and the column-axis name is ``destination``.

``TransitionMatrix`` values are ``uintp`` counts of valid adjacent pairs,
excluding transitions to or from missing data. ``StochasticMatrix`` has the
same layout, with ``float64`` conditional probabilities whose rows sum to
one. Empty historical origin rows borrow probabilities in the fallback order
described in :ref:`clouds-fallback`.

CloudDistribution
~~~~~~~~~~~~~~~~~

The unique integer index is named ``month`` and contains values from 1 through
12. Integer state columns 0 through 8 are in numerical order; values are
nonnegative integer quarter counts, excluding missing state 9.
``count_cloud_states`` returns all twelve months, ``uint8`` labels,
``uintp`` values, and a column-axis name of ``eighths``. Diagnostics can accept
distributions containing a subset of months and reindex absent months to zero.

Public Members
--------------

.. automodule:: rubin_sim.clouds
    :imported-members:
    :members:
    :show-inheritance:
