Concepts of Operations for Chimera Progress Capability
======================================================


1. Scope
--------

1.1 Identification
~~~~~~~~~~~~~~~~~~

This Concepts of Operations (ConOps) document describes the "chimera progress" capability implemented for the Rubin Observatory Simulation Metrics Analysis Framework (rubin_sim/maf) as part of ticket SP-3142. This capability enables the construction of "extrapolated metric vs. time" plots to track survey performance progress, as described in RTN-092.

The capability is implemented in the ``rubin_sim/maf/chimera_progress.py`` module and provides both Python APIs and command-line interfaces for:

- Building chimera visit sequences that combine real operational visits with simulated visits
- Running metric batches on collections of chimera sequences
- Summarizing results across multiple transition dates

1.2 Document Overview
~~~~~~~~~~~~~~~~~~~~~

This document follows the IEEE 1362 standard for Concepts of Operations:

- Section 2: Referenced documents
- Section 3: Current system or situation
- Section 4: Justification for and nature of changes
- Section 5: Concepts for the proposed system
- Section 6: Operational scenarios
- Section 7: Summary of impacts
- Section 8: Analysis of the proposed system
- Section 9: Notes

1.3 System Overview
~~~~~~~~~~~~~~~~~~~

The rubin_sim/maf (Metrics Analysis Framework) provides tools for computing metrics from visit sequence data. The chimera progress capability extends maf's batch processing infrastructure to support analyzing metrics across a series of "chimera" visit sequences, where each sequence combines:

- Real visits from the ConsDB (Operational Database) up to a transition date
- Simulated visits from an OpSim database after the transition date

This approach enables users to evaluate how survey performance metrics would have evolved if observations had been collected up to various points in time, using actual visited fields up to that point and extrapolating with simulated observations afterward.

1.3.1 Implementation Status
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The chimera progress capability has been implemented with the following key features:

- **Simplified dayobs handling**: Internal helper functions consolidated for cleaner code
- **Batch kwargs support**: Additional keyword arguments can be passed to batch functions via ``--batch-kwarg`` CLI option
- **dayobs0 parameter**: ``science_radar_batch`` now accepts ``dayobs0`` as an alternative to ``mjd0`` for specifying the survey start time
- **Flexible batch function signatures**: Supports both ``run_name`` and ``runName`` parameters in batch functions

.. mermaid::

   graph LR
       A[ConsDB Visit Sequence] -->|Up to transition| C[Chimera Visit Sequence]
       B[OpSim Visit Sequence] -->|After transition| C
       C --> D[MAF Batch Processing]
       D --> E[ResultsDb]
       E --> F[Summary Table]

2. Referenced Documents
-----------------------

- RTN-092: "Rubin Observatory Survey Strategy, Progress Monitoring, and Performance Metrics"
- experiments/chimerabatch/chimerabatch.md: Initial change request and prototype description
- IEEE 1362: Standard for Concepts of Operations Analysis
- experiments/chimerabatch/chimerabatch_requirements.rst: Requirements Specification
- experiments/chimerabatch/chimerabatch_sdd.rst: Software Design Document
- rubin_sim/maf documentation: Metrics Analysis Framework user guide
- https://developer.lsst.io/python/numpydoc.html: Rubin Observatory NumPyDoc style
- https://developer.lsst.io/python/style.html: Rubin DM Python Style Guide

3. Current System or Situation
------------------------------

3.1 Background, Objectives, and Scope
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Metrics Analysis Framework (maf) in rubin_sim provides infrastructure for:

- Reading visit sequences from opsim databases or HDF5 files
- Computing metrics from visit data using batch functions
- Storing metric results in a ResultsDb SQLite database
- Generating summary statistics and plots

The existing batch infrastructure processes collections of simulation runs, where each run represents a complete simulated survey. However, there is no built-in capability to analyze metrics for partially-observed surveys that combine real operational data with extrapolated simulations.

3.2 Operational Policies and Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Visits must include a ``dayObs`` field (integer YYYYMMDD in UTC-12 hours)
- OpSim databases follow the standard rubin_sim visit schema
- ConsDB visits are queried from Rubin Observatory's operational database
- ResultsDb stores metrics with run names that identify the simulation

3.3 Description of the Current System or Situation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current maf workflow for batch processing:

1. A user selects a batch function (e.g., ``glanceBatch``, ``science_radar_batch``)
2. The batch function is invoked with a simulation name and parameters
3. Metric bundles are computed for the simulation
4. Results are stored in ResultsDb with the simulation name as the run identifier

This workflow works well for comparing complete simulations but does not support:

- Incremental analysis where the "simulation" is a hybrid of real + extrapolated data
- Time-series analysis of metrics as a function of transition date
- Progressive survey progress tracking

3.4 Modes of Operation for the Current System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current system operates in:

- **Batch mode**: Process a single simulation or collection of simulations
- **Interactive mode**: Run individual metrics in notebooks
- **Command-line mode**: Run pre-defined analysis pipelines

3.5 User Classes and Other Involved Personnel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Survey Strategists**: Analyze survey performance metrics and progress
- **Simulation Analysts**: Compare baseline simulations with observed data
- **Operations Planners**: Evaluate hypothetical observation scenarios
- **Software Developers**: Maintain and extend the maf infrastructure

3.6 Support Environment
~~~~~~~~~~~~~~~~~~~~~~~

- Python 3.12+ runtime environment
- SQLite database for ResultsDb
- HDF5 files for visit sequence storage
- rubin_sim package dependencies (pandas, astropy, sqlalchemy, click)

4. Justification for and Nature of Changes
------------------------------------------

4.1 Justification of Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RTN-092 describes "Progress reports" as a key component of survey strategy monitoring, requiring "extrapolated metric vs. time" plots. These plots show how survey metrics would have evolved if observations had been collected up to various dates, using:

- Real visited fields (from ConsDB) up to each transition date
- Extrapolated remaining fields (from OpSim) after each transition date

Without this capability, there is no systematic way to:

- Track survey progress relative to forecast
- Evaluate the impact of adding real observations to simulations
- Generate progress reports for stakeholder review

4.2 Description of Desired Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The chimera progress capability adds three core functions to rubin_sim/maf:

1. **build_chimera**\ : Construct a single hybrid visit sequence
2. **build_chimeras**\ : Generate multiple chimera sequences for a date range
3. **run_chimera_batches**\ : Process all chimera sequences through maf batches
4. **make_chimera_summary_table**\ : Extract summary metrics into a tabular format

Each function is available as both a Python API and a Click command-line interface.

4.3 Priorities Among Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Priority 1 (Core functionality):
- build_chimera / build_chimeras
- run_chimera_batches
- make_chimera_summary_table

Priority 2 (Enhancements):
- Support for specifying dayobs0 in science_radar_batch (required parameter)
- Changes to sqlconstraint to use "filter" instead of "band" for consistency

4.4 Changes Considered but Not Included
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Direct integration with ObsLocTap service: The current implementation reads from SQLite/HDF5 files. Integration with external services can be added later as a data source layer.
- Online processing mode: The current implementation requires pre-building chimera files. Future versions could support streaming processing.
- Parallel batch execution: The current implementation processes chimera sequences sequentially. Performance enhancements could add parallelization.

5. Concepts for the Proposed System
-----------------------------------

5.1 Background, Objectives, and Scope
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The proposed system extends maf's batch processing infrastructure to support "chimera" visit sequences. A chimera sequence is constructed by combining:

- Real consdb visits from ``start_dayobs`` through ``transition_dayobs``
- Simulated opsim visits from after ``transition_dayobs`` through ``end_dayobs``

The objective is to enable three-step workflows:

1. Build chimera sequences for a range of transition dates
2. Run metric batches on all sequences
3. Extract summary metrics into a time-series table

5.2 Operational Policies and Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``dayObs`` values are integers in YYYYMMDD format (UTC-12 hours)
- All chimera sequences use the same ``start_dayobs``, ``end_dayobs``, and opsim database
- Transition dates must exist in the consdb (or be within the consdb date range)
- Run names follow the pattern ``chimera_YYYYMMDD`` to encode the transition date
- science_radar_batch requires exactly one of ``mjd0`` or ``dayobs0`` parameters

5.3 Description of the Proposed System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The proposed system adds a new module ``rubin_sim/maf/chimera_progress.py`` with the following components:

**Core Python Functions:**

- ``build_chimera(consdb_visits, opsim_visits, start_dayobs, transition_dayobs, end_dayobs)``: Returns a single pandas DataFrame
- ``build_chimeras(...)``: Returns a list of ``(transition_dayobs, hdf5_path)`` tuples
- ``run_chimera_batches(chimera_specs, batch_func, out_dir, batch_kwargs)``: Returns the ResultsDb path
- ``make_chimera_summary_table(results_db)``: Returns a pandas DataFrame with multi-index columns

**Command-Line Interfaces:**

- ``build_chimeras``: Reads visit files, writes chimera HDF5 files
- ``run_chimera_batches``: Runs batches on all chimera files in a directory
- ``make_chimera_summary_table``: Queries ResultsDb, writes summary HDF5

5.4 Modes of Operation
~~~~~~~~~~~~~~~~~~~~~~

The proposed system supports:

- **Interactive mode**: Python API for notebook-based exploration
- **Batch mode**: Command-line execution for pipeline processing
- **Hybrid mode**: Python API with file I/O for reproducible workflows

5.5 User Classes and Other Involved Personnel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Data Analysts**: Use the Python API to build custom analysis workflows
- **Pipeline Operators**: Use CLI commands for automated processing
- **Notebook Authors**: Use the API to demonstrate progress tracking
- **Developers**: Maintain and extend the chimera infrastructure

5.6 Support Environment
~~~~~~~~~~~~~~~~~~~~~~~

The system requires:

- rubin_sim package (with maf and dependencies)
- HDF5 library for visit sequence storage
- SQLite for ResultsDb
- pandas for DataFrame operations
- click for command-line interfaces

6. Operational Scenarios
------------------------

6.1 Scenario 1: Building Chimera Sequences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Example workflow
----------------

.. code-block:: bash

   # Build chimera sequences every 30 nights
   build_chimeras \
       --consdb-file tmp/complete_visits_2026-05-04.db \
       --opsim-file ~/rubin_sim_data/sim_baseline/baseline_v5.0.0_10yrs.db \
       --start-dayobs 20260101 \
       --end-dayobs 20281231 \
       --step 30 \
       --out-dir chimera_sequences/

6.2 Scenario 2: Running Metric Batches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Example workflow
----------------

.. code-block:: bash

   # Run science_radar_batch on all chimera sequences
   run_chimera_batches \
       --chimera-dir chimera_sequences/ \
       --out-dir chimera_results/ \
       --batch science_radar_batch \
       --batch-kwarg srd_only=True \
       --batch-kwarg long_microlensing=False

Example with dayobs0 parameter
------------------------------

.. code-block:: bash

   # Use dayobs0 instead of mjd0 for science_radar_batch
   run_chimera_batches \
       --chimera-dir chimera_sequences/ \
       --out-dir chimera_results/ \
       --batch science_radar_batch \
       --batch-kwarg dayobs0=20260101

6.3 Scenario 3: Creating Summary Table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Example workflow
----------------

.. code-block:: bash

   # Extract summary metrics into a time-series table
   make_chimera_summary_table \
       --results-db chimera_results/resultsDb_sqlite.db \
       --out-file chimera_summary.h5

The resulting HDF5 file contains a DataFrame with:
- Index: ``transition_dayobs`` (integer YYYYMMDD)
- Columns: MultiIndex of ``(metric_name, slicer_name, metric_info_label, summary_metric)``
- Values: Summary statistic values

6.4 Scenario 4: Notebook-Based Progress Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following Python code demonstrates the chimera progress workflow using the Python API. This example can be tested with ``doctest`` or run directly in a Jupyter notebook.

Example: Building a Single Chimera Sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. testcode::

   import pandas as pd
   import datetime

   # Simulated consdb visits with dayObs column
   consdb_visits = pd.DataFrame({
       'visitId': [1, 2, 3],
       'dayObs': [20260101, 20260102, 20260103],
       'filter': ['g', 'r', 'i']
   })

   # Simulated opsim visits with dayObs column
   opsim_visits = pd.DataFrame({
       'visitId': [101, 102, 103],
       'dayObs': [20260104, 20260105, 20260106],
       'filter': ['g', 'r', 'i']
   })

   from rubin_sim.maf.chimera_progress import build_chimera

   # Build a chimera sequence with transition at 2026-01-03
   chimera = build_chimera(
       consdb_visits=consdb_visits,
       opsim_visits=opsim_visits,
       start_dayobs=20260101,
       transition_dayobs=20260103,
       end_dayobs=20260106
   )

   # Verify the chimera contains visits from both sources
   print(len(chimera))  # Should be 6 (3 consdb + 3 opsim)
   print(chimera['visitId'].tolist())  # [1, 2, 3, 101, 102, 103]

Expected output
---------------

.. testoutput::

   6
   [1, 2, 3, 101, 102, 103]

Example: Building Multiple Chimera Sequences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. testcode::

   import os
   import tempfile

   import pandas as pd

   from rubin_sim.maf.chimera_progress import build_chimeras

   # Create temporary directory for chimera files
   with tempfile.TemporaryDirectory() as tmpdir:
       # Build chimera sequences for multiple transition dates
       specs = build_chimeras(
           consdb_visits=consdb_visits,
           opsim_visits=opsim_visits,
           start_dayobs=20260101,
           end_dayobs=20260106,
           step=1,
           out_dir=tmpdir
       )

       # specs contains (transition_dayobs, hdf5_path) tuples
       print(len(specs))  # Number of transition dates
       print(specs[0])  # First spec: (transition_dayobs, path)

       # Verify HDF5 files were created
       h5_files = [f for f in os.listdir(tmpdir) if f.endswith('.h5')]
       print(len(h5_files))  # Should match number of specs

Expected output
---------------

.. testoutput::

   6
   (20260101, '...')
   6

Example: Running Chimera Batches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. testcode::

   import os
   import tempfile

   import pandas as pd

   from rubin_sim.maf.chimera_progress import build_chimeras, run_chimera_batches
   from rubin_sim.maf import batches

   with tempfile.TemporaryDirectory() as tmpdir:
       # Build chimera sequences
       specs = build_chimeras(
           consdb_visits=consdb_visits,
           opsim_visits=opsim_visits,
           start_dayobs=20260101,
           end_dayobs=20260106,
           step=2,
           out_dir=tmpdir
       )

       # Run batches (using a simple batch for demonstration)
       results_db = run_chimera_batches(
           chimera_specs=specs,
           batch_func=batches.glanceBatch,
           out_dir=tmpdir
       )

       # Verify ResultsDb was created
       print(os.path.exists(results_db))  # True

Expected output
---------------

.. testoutput::

   True

Example: Creating a Summary Table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. testcode::

   import os
   import tempfile

   import pandas as pd
   from rubin_sim.maf.chimera_progress import (
       build_chimeras, run_chimera_batches, make_chimera_summary_table
   )

   with tempfile.TemporaryDirectory() as tmpdir:
       # Build and run batches
       specs = build_chimeras(
           consdb_visits=consdb_visits,
           opsim_visits=opsim_visits,
           start_dayobs=20260101,
           end_dayobs=20260106,
           step=2,
           out_dir=tmpdir
       )

       run_chimera_batches(
           specs,
           batch_func=batches.glanceBatch,
           out_dir=tmpdir
       )

       # Create summary table
       summary_df = make_chimera_summary_table(
           os.path.join(tmpdir, 'resultsDb_sqlite.db')
       )

       # Verify the summary table has the expected structure
       print(summary_df.shape[0] > 0)  # Has at least one row
       print(isinstance(summary_df.columns, pd.MultiIndex))  # MultiIndex columns

Expected output
---------------

.. testoutput::

   True
   True

7. Summary of Impacts
---------------------

7.1 Operational Impacts
~~~~~~~~~~~~~~~~~~~~~~~

- **Survey Progress Tracking**: Enables operational teams to track how survey metrics evolve as real observations accumulate
- **Forecast Validation**: Allows comparison of extrapolated metrics against actual survey performance
- **Report Generation**: Supports automated progress reports for stakeholder review
- **Scenario Evaluation**: Enables "what-if" analysis of different observation strategies

7.2 Organizational Impacts
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Data Integration**: Bridges operational data (ConsDB) with simulation data (OpSim)
- **Collaboration**: Enables analysis teams to work with hybrid real+simulated data
- **Workflow Standardization**: Establishes a repeatable three-step pipeline for progress analysis

7.3 Impacts During Development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Testing**: Requires test fixtures with sample consdb and opsim data
- **Documentation**: Need for user guide explaining chimera concepts
- **Maintenance**: New module to maintain alongside existing maf infrastructure

8. Analysis of the Proposed System
----------------------------------

8.1 Summary of Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Enables Progress Monitoring**: The primary goal from RTN-092 - tracking metrics vs. time
- **Leverages Existing Infrastructure**: Reuses maf batch processing and ResultsDb
- **Flexible Data Sources**: Works with SQLite, HDF5, and can be extended to other sources
- **Programmatic and CLI Access**: Supports both interactive and automated workflows

8.2 Disadvantages and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Computational Cost**: Building and processing many chimera sequences can be expensive
- **Storage Requirements**: Each chimera sequence is stored as a separate HDF5 file
- **Memory Usage**: Large visit sequences may require significant memory for concatenation
- **Date Sensitivity**: Results depend on the quality and completeness of consdb data

8.3 Alternatives and Trade-offs Considered
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :class: alt-table

   * - Alternative
     - Pros
     - Cons
   * - Direct integration with ObsLocTap
     - Real-time data, no file management
     - Network dependencies, rate limits
   * - Streaming/online processing
     - Lower storage requirements
     - More complex implementation, less reproducible
   * - Per-transition batch execution
     - Simpler data management
     - More ResultsDb writes, harder to query
   * - Parallel batch execution
     - Faster processing
     - More complex error handling, resource contention

The chosen approach (file-based batch processing) was selected for:
- Simplicity of implementation and debugging
- Reproducibility of results
- Compatibility with existing maf infrastructure
- Flexibility for future extensions

9. Notes
--------

The chimera progress capability is designed as an extension to the existing maf infrastructure rather than a replacement. Key design principles:

- **Reusability**: All functions can be called directly or via CLI
- **Composability**: Chimera sequences are standard HDF5 files compatible with maf
- **Extensibility**: New batch functions can be applied without modifying the core
- **Transparency**: Run names encode transition dates for easy querying

Appendices
----------

A. Example ResultsDb Query
~~~~~~~~~~~~~~~~~~~~~~~~~~

The following example demonstrates querying the ResultsDb to extract specific summary metrics:

.. testcode::

   import os
   import tempfile

   import pandas as pd
   from rubin_sim.maf.chimera_progress import (
       build_chimeras, run_chimera_batches
   )
   from rubin_sim.maf import batches
   from rubin_sim.maf.db import ResultsDb

   with tempfile.TemporaryDirectory() as tmpdir:
       # Create a simple ResultsDb
       specs = build_chimeras(
           consdb_visits=consdb_visits,
           opsim_visits=opsim_visits,
           start_dayobs=20260101,
           end_dayobs=20260106,
           step=2,
           out_dir=tmpdir
       )

       run_chimera_batches(
           specs,
           batch_func=batches.glanceBatch,
           out_dir=tmpdir
       )

       # Query the ResultsDb
       results_db = ResultsDb(database=os.path.join(tmpdir, 'resultsDb_sqlite.db'))
       stats = results_db.get_summary_stats(with_sim_name=True)

       # Filter to chimera runs only
       chimera_stats = stats[
           [r.startswith('chimera_') for r in stats['run_name']]
       ]

       # Convert to DataFrame
       df = pd.DataFrame(chimera_stats)
       df['transition_dayobs'] = df['run_name'].apply(
           lambda r: int(r.replace('chimera_', ''))
       )

       # Verify we have chimera data
       print(len(df) > 0)  # True if we have stats
       print(df['transition_dayobs'].nunique() > 0)  # True if we have dates

**Expected output:**

::

   True
   True

B. DayObs Format Reference
~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``dayObs`` is an integer in YYYYMMDD format
- Represented in UTC-12 hours so it doesn't roll over during observing
- Example: A visit at 2026-01-15 03:00 UTC has ``dayObs = 20260114``

C. Command-Line Interface Reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``build_chimeras``
------------------

.. code-block:: text

   Usage: build_chimeras [OPTIONS]

   Options:
     --consdb-file PATH   [required] SQLite or HDF5 file with consdb visits
     --opsim-file PATH    [required] SQLite or HDF5 file with opsim visits
     --start-dayobs INTEGER  [required] Start date YYYYMMDD
     --end-dayobs INTEGER    [required] End date YYYYMMDD
     --step INTEGER          [default: 1] Nights between transition dates
     --out-dir PATH          [default: .] Output directory for HDF5 files

``run_chimera_batches``
-----------------------

.. code-block:: text

   Usage: run_chimera_batches [OPTIONS]

   Options:
     --chimera-dir PATH     [required] Directory containing chimera_*.h5 files
     --out-dir PATH         [default: .] Output directory for results_db
     --batch TEXT           [default: glanceBatch] Batch function name
     --batch-kwarg TEXT     Additional batch kwarg (KEY=VALUE). May be
                            specified multiple times.

**Example with science_radar_batch dayobs0 parameter:**

.. code-block:: bash
		
    # Use dayobs0 instead of mjd0 for science_radar_batch
    run_chimera_batches \
      --chimera-dir chimera_sequences/ \
      --out-dir chimera_results/ \
      --batch science_radar_batch \
      --batch-kwarg dayobs0=20260101 \
      --batch-kwarg srd_only=True

Note: ``science_radar_batch`` now accepts either ``mjd0`` or ``dayobs0`` (but not both).
When ``dayobs0`` is provided, it is converted to MJD using the relationship
``mjd0 = Time(dayobs0, format='yymmdd').mjd - 0.5``.

``make_chimera_summary_table``
------------------------------

.. code-block:: text

   Usage: make_chimera_summary_table [OPTIONS]

   Options:
     --results-db PATH    [required] Path to resultsDb_sqlite.db
     --out-file PATH      [default: chimera_summary.h5] Output HDF5 file
