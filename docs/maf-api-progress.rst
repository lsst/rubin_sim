.. py:currentmodule:: rubin_sim.maf.progress

.. _maf-api-progress:

========
Progress
========

The progress module builds chimera visit files, evaluates batches over date
series, and extracts summary tables. See :ref:`maf-progress` for the command-line
workflow, date conventions, and output formats. The batch definitions are
documented in :ref:`maf-api-batches`.

The Python functions ``build_chimera`` and ``build_chimeras`` require input
DataFrames with a ``dayObs`` column. The ``build_chimeras`` console command adds
this column using ``DayObsStacker`` when it loads the visit files.

.. automodule:: rubin_sim.maf.progress
    :members:
