Software Design Document for Chimera Progress Capability
=========================================================

.. mermaid::

   classDiagram
       class build_chimera {
           +build_chimera(consdb_visits, opsim_visits, start_dayobs, transition_dayobs, end_dayobs) DataFrame
       }

       class build_chimeras {
           +build_chimeras(consdb_visits, opsim_visits, start_dayobs, end_dayobs, step, out_dir) list
       }

       class run_chimera_batches {
           +run_chimera_batches(chimera_specs, batch_func, out_dir, batch_kwargs) str
       }

       class make_chimera_summary_table {
           +make_chimera_summary_table(results_db) DataFrame
       }

       class DayObsStacker {
           +stack(data) data with dayObs
       }

       class ResultsDb {
           +get_summary_stats() numpy.recarray
           +get_run_name() list
       }

       class BuildChimerasCmd {
           +__call__()
       }

       class RunChimeraBatchesCmd {
           +__call__()
       }

       class MakeChimeraSummaryTableCmd {
           +__call__()
       }

       build_chimera ..> DayObsStacker : uses
       build_chimeras ..> build_chimera : calls
       run_chimera_batches ..> ResultsDb : uses
       make_chimera_summary_table ..> ResultsDb : uses
       BuildChimerasCmd ..> build_chimeras : delegates
       RunChimeraBatchesCmd ..> run_chimera_batches : delegates
       MakeChimeraSummaryTableCmd ..> make_chimera_summary_table : delegates
       science_radar_batch ..> Time : uses
       RunChimeraBatchesCmd ..> science_radar_batch : uses

1. Introduction
---------------

1.1 Purpose
~~~~~~~~~~~

This Software Design Document (SDD) describes the design of the "chimera progress" capability
for the Rubin Observatory Simulation Metrics Analysis Framework (rubin_sim/maf). This capability
enables the construction of "extrapolated metric vs. time" plots to track survey performance
progress, as described in RTN-092.

The design follows IEEE 1016-2009 (IEEE Recommended Practice for Software Design
Descriptions) using the **viewpoint-based approach** with the following viewpoints:

- **Logical View**: Core functionality (build_chimera, build_chimeras, run_chimera_batches, make_chimera_summary_table)
- **Process View**: Workflow of operations and batch processing
- **Deployment View**: File-based data flow between operations
- **Component View**: Python modules and CLI commands

1.2 Scope
~~~~~~~~~

This document covers the design of:

- The ``rubin_sim/maf/chimera_progress.py`` module
- CLI commands: ``build_chimeras``, ``run_chimera_batches``, ``make_chimera_summary_table``
- Integration with existing maf infrastructure (batches, ResultsDb, metrics)
- Changes to ``science_radar_batch`` to support ``dayobs0`` parameter
- Support for additional batch kwargs via ``--batch-kwarg`` CLI option
- Flexible batch function signature handling (``run_name`` or ``runName``)

This document does **not** cover:
- Formal requirements specification (separate document)
- Unit test design (covered in test files)
- User documentation (covered in ConOps document)

1.3 Definitions, Acronyms, and Abbreviations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Chimera**: A hybrid visit sequence combining real consdb visits with simulated opsim visits
- **ConsDB**: Operational Database containing real Rubin Observatory visit records
- **OpSim**: Simulator database containing simulated visit sequences
- **dayObs**: Integer YYYYMMDD representing a date in UTC-12 hours
- **Maf**: Metrics Analysis Framework
- **ResultsDb**: SQLite database storing metric results and summary statistics
- **CLI**: Command Line Interface

1.4 References
~~~~~~~~~~~~~~

- IEEE 1016-2009: IEEE Recommended Practice for Software Design Descriptions
- RTN-092: Rubin Observatory Survey Strategy, Progress Monitoring, and Performance Metrics
- experiments/chimerabatch/chimerabatch.md: Initial change request
- experiments/chimerabatch/chimerabatch_conops.rst: ConOps document
- experiments/chimerabatch/chimerabatch_requirements.rst: Requirements Specification
- https://developer.lsst.io/python/numpydoc.html: Rubin Observatory NumPyDoc style
- https://developer.lsst.io/python/style.html: Rubin DM Python Style Guide

1.5 Overview
~~~~~~~~~~~~

Section 2 describes the overall software architecture and design constraints.

Section 3 presents the logical design of the core chimera progress functions.

Section 4 describes the process design and workflow.

Section 5 details the component design (modules, classes, interfaces).

Section 6 describes the deployment design (data flow, file structures).

Section 7 covers design patterns and non-functional considerations.

2. Architecture
---------------

2.1 High-Level Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mermaid::

   graph TB
       subgraph "Data Sources"
           DS1[ConsDB SQLite/HDF5]
           DS2[OpSim SQLite/HDF5]
       end

       subgraph "Chimera Builder"
           B1[build_chimera]
           B2[build_chimeras]
       end

       subgraph "Batch Processor"
           P1[run_chimera_batches]
       end

       subgraph "Results Storage"
           R1[ResultsDb SQLite]
           R2[Summary HDF5]
       end

       subgraph "CLI Commands"
           C1[build_chimeras_cmd]
           C2[run_chimera_batches_cmd]
           C3[make_chimera_summary_table_cmd]
       end

       DS1 --> B2
       DS2 --> B2
       B2 --> B1
       B2 -->|HDF5 files| P1
       P1 --> R1
       R1 --> C3
       C3 --> R2
       C1 --> B2
       C2 --> P1

2.2 Design Constraints
~~~~~~~~~~~~~~~~~~~~~~

- **Database Compatibility**: ResultsDb must use SQLite for portability
- **File Format**: HDF5 (via ``pytables``) for visit sequences and summary tables
- **Python Version**: Target Python 3.12+ with type hints
- **Click CLI**: Command-line interfaces must use the Click framework
- **Backward Compatibility**: science_radar_batch must support both ``mjd0`` and ``dayobs0``
- **Flexible Batch Signatures**: Batch functions may accept either ``run_name`` or ``runName``
- **Memory Efficiency**: Large visit sequences must be processed without loading all into memory

2.3 Coding Standards
~~~~~~~~~~~~~~~~~~~~

This implementation conforms to the Rubin DM Python Style Guide
(https://developer.lsst.io/python/style.html) and uses the Rubin Observatory
NumPyDoc style for docstrings (https://developer.lsst.io/python/numpydoc.html).

Key style requirements:
- Line length: 110 characters
- Type hints: Required for all function signatures
- Docstrings: NumPyDoc style with ``Parameters``, ``Returns``, ``Raises`` sections
- Imports: Grouped by standard library, third-party, and local imports

2.4 Design Principles
~~~~~~~~~~~~~~~~~~~~~

- **Separation of Concerns**: Data I/O (collect), pure calculations (compute), visualization (plot)
- **Reusability**: All functions work with DataFrames, HDF5 files, or ResultsDb
- **Composability**: Chimera sequences are standard HDF5 files compatible with maf
- **Extensibility**: New batch functions can be applied without modifying core
- **Testability**: Core functions are pure (no I/O) for easy unit testing
- **Transparency**: Run names encode transition dates for easy querying

3. Logical Design
-----------------

3.1 Core Function Signatures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def build_chimera(
       consdb_visits: pd.DataFrame,
       opsim_visits: pd.DataFrame,
       start_dayobs: int,
       transition_dayobs: int,
       end_dayobs: int,
   ) -> pd.DataFrame:
       """Build a single chimera visit sequence."""

   def build_chimeras(
       consdb_visits: pd.DataFrame,
       opsim_visits: pd.DataFrame,
       start_dayobs: int,
       end_dayobs: int,
       step: int = 1,
       out_dir: str = ".",
   ) -> list[tuple[int, str]]:
       """Build chimera visit sequences for a range of transition dates."""

   def run_chimera_batches(
       chimera_specs: list[tuple[int, str]],
       batch_func: Callable[..., dict] | None = None,
       out_dir: str = ".",
       batch_kwargs: dict | None = None,
   ) -> str:
       """Run MAF metric batches on a collection of chimera visit sequences."""

   def make_chimera_summary_table(
       results_db: db.ResultsDb | str,
   ) -> pd.DataFrame:
       """Build a summary table from chimera run results."""

3.2 Data Flow Design
~~~~~~~~~~~~~~~~~~~~

.. mermaid::

   flowchart TD
       A[consdb_visits DataFrame] -->|filtered by dayObs| C[consdb_part]
       B[opsim_visits DataFrame] -->|filtered by dayObs| D[opsim_part]
       C --> E[common_cols]
       D --> E
       E --> F[concat]
       F --> G[chimera DataFrame]

       subgraph "Chimera Sequence"
           C
           D
           E
           F
           G
       end

       H[transition_dates list] --> I[loop over dates]
       I --> J[build_chimera for each date]
       J --> K[HDF5 file per date]
       K --> L[specs list returned]

3.3 Algorithm Specifications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Algorithm 1: build_chimera**

.. testcode::

   import pandas as pd

   def build_chimera(
       consdb_visits: pd.DataFrame,
       opsim_visits: pd.DataFrame,
       start_dayobs: int,
       transition_dayobs: int,
       end_dayobs: int,
   ) -> pd.DataFrame:
       """Build a single chimera visit sequence.

       Combines consdb visits in [start_dayobs, transition_dayobs] with opsim
       visits in (transition_dayobs, end_dayobs].  Both input DataFrames must
       already have a ``dayObs`` column (integer YYYYMMDD, UTC-12).

       Parameters
       ----------
       consdb_visits : `pandas.DataFrame`
           Real visits from consdb. Must include a ``dayObs`` column.
       opsim_visits : `pandas.DataFrame`
           Simulated visits from an opsim database. Must include a ``dayObs``
           column.
       start_dayobs : `int`
           Start of the chimera window, YYYYMMDD inclusive.
       transition_dayobs : `int`
           Transition date; consdb visits up to and including this date are used.
       end_dayobs : `int`
           End of the chimera window, YYYYMMDD inclusive.

       Returns
       -------
       chimera : `pandas.DataFrame`
           Combined visit sequence containing columns present in both inputs.
       """
       consdb_part = consdb_visits.loc[
           (consdb_visits["dayObs"] >= int(start_dayobs)) & (consdb_visits["dayObs"] <= int(transition_dayobs))
       ]
       opsim_part = opsim_visits.loc[
           (opsim_visits["dayObs"] > int(transition_dayobs)) & (opsim_visits["dayObs"] <= int(end_dayobs))
       ]

       common_cols = sorted(set(consdb_part.columns) & set(opsim_part.columns))
       if not common_cols:
           raise ValueError("consdb_visits and opsim_visits share no common columns; " "cannot build a chimera.")

       return pd.concat(
           [consdb_part[common_cols], opsim_part[common_cols]],
           ignore_index=True,
       )

**Test Example:**

.. testcode::

   import pandas as pd
   from rubin_sim.maf.chimera_progress import build_chimera

   consdb_visits = pd.DataFrame({
       'visitId': [1, 2, 3],
       'dayObs': [20260101, 20260102, 20260103],
       'filter': ['g', 'r', 'i']
   })

   opsim_visits = pd.DataFrame({
       'visitId': [101, 102, 103],
       'dayObs': [20260104, 20260105, 20260106],
       'filter': ['g', 'r', 'i']
   })

   chimera = build_chimera(
       consdb_visits=consdb_visits,
       opsim_visits=opsim_visits,
       start_dayobs=20260101,
       transition_dayobs=20260103,
       end_dayobs=20260106
   )

   assert len(chimera) == 6
   assert list(chimera['visitId']) == [1, 2, 3, 101, 102, 103]

**Test Output:**

::

   (Assertion passes)

**Test Example: DayObs boundary**

.. testcode::

   import pandas as pd
   from rubin_sim.maf.chimera_progress import build_chimera

   consdb_visits = pd.DataFrame({
       'visitId': [1, 2, 3, 4],
       'dayObs': [20260101, 20260102, 20260103, 20260104],
       'filter': ['g', 'r', 'i', 'z']
   })

   opsim_visits = pd.DataFrame({
       'visitId': [101, 102],
       'dayObs': [20260105, 20260106],
       'filter': ['g', 'r']
   })

   # Transition at 20260103: consdb visits 1-3, opsim visits 101-102
   chimera = build_chimera(
       consdb_visits=consdb_visits,
       opsim_visits=opsim_visits,
       start_dayobs=20260101,
       transition_dayobs=20260103,
       end_dayobs=20260106
   )

   # Check transition point: dayObs 3 is included (consdb), dayObs 5 is included (opsim)
   consdb_ids = chimera[chimera['dayObs'] <= 20260103]['visitId'].tolist()
   opsim_ids = chimera[chimera['dayObs'] > 20260103]['visitId'].tolist()
   assert consdb_ids == [1, 2, 3]
   assert opsim_ids == [101, 102]

**Test Output:**

::

   (Assertion passes)

**Algorithm 2: build_chimeras**

.. testcode::

   import os
   import tempfile
   import pandas as pd
   from rubin_sim.maf.chimera_progress import build_chimeras

   consdb_visits = pd.DataFrame({
       'visitId': [1, 2, 3, 4, 5],
       'dayObs': [20260101, 20260102, 20260103, 20260104, 20260105],
       'filter': ['g', 'r', 'i', 'z', 'y']
   })

   opsim_visits = pd.DataFrame({
       'visitId': [101, 102, 103, 104, 105],
       'dayObs': [20260106, 20260107, 20260108, 20260109, 20260110],
       'filter': ['g', 'r', 'i', 'z', 'y']
   })

   with tempfile.TemporaryDirectory() as tmpdir:
       specs = build_chimeras(
           consdb_visits=consdb_visits,
           opsim_visits=opsim_visits,
           start_dayobs=20260101,
           end_dayobs=20260110,
           step=2,
           out_dir=tmpdir
       )

       # specs contains (transition_dayobs, hdf5_path) tuples
       transition_dates = [t for t, _ in specs]
       assert len(transition_dates) == 3  # 20260101, 20260103, 20260105
       assert transition_dates == [20260101, 20260103, 20260105]

       # Verify HDF5 files were created
       h5_files = [f for f in os.listdir(tmpdir) if f.endswith('.h5')]
       assert len(h5_files) == 3

**Test Output:**

::

   (Assertion passes)

**Algorithm 3: run_chimera_batches**

The run_chimera_batches function:

1. Creates a ResultsDb in the output directory
2. For each chimera spec:
   - Generates run_name from transition_dayobs (e.g., "chimera_20260101")
   - Calls batch_func with run_name and batch_kwargs
   - Creates MetricBundleGroup for this run
   - Runs all metrics and stores in ResultsDb
3. Closes ResultsDb and returns path

.. testcode::

   import os
   import tempfile
   import pandas as pd
   from rubin_sim.maf.chimera_progress import build_chimeras, run_chimera_batches
   from rubin_sim.maf import batches

   consdb_visits = pd.DataFrame({
       'visitId': [1, 2, 3],
       'dayObs': [20260101, 20260102, 20260103],
       'filter': ['g', 'r', 'i']
   })

   opsim_visits = pd.DataFrame({
       'visitId': [101, 102, 103],
       'dayObs': [20260104, 20260105, 20260106],
       'filter': ['g', 'r', 'i']
   })

   with tempfile.TemporaryDirectory() as tmpdir:
       specs = build_chimeras(
           consdb_visits=consdb_visits,
           opsim_visits=opsim_visits,
           start_dayobs=20260101,
           end_dayobs=20260106,
           step=2,
           out_dir=tmpdir
       )

       results_db_path = run_chimera_batches(
           chimera_specs=specs,
           batch_func=batches.glanceBatch,
           out_dir=tmpdir
       )

       # Verify ResultsDb was created
       assert os.path.exists(results_db_path)
       assert results_db_path == os.path.join(tmpdir, 'resultsDb_sqlite.db')

       # Verify run names in ResultsDb match expected pattern
       from rubin_sim.maf.db import ResultsDb
       db = ResultsDb(database=results_db_path)
       run_names = db.get_run_name()
       db.close()

       # Should have 3 runs with transition dates
       assert len(run_names) == 3
       assert all(r.startswith('chimera_') for r in run_names)

**Test Output:**

::

   (Assertion passes)

**Algorithm 4: make_chimera_summary_table**

The make_chimera_summary_table function:

1. Opens ResultsDb (or uses provided instance)
2. Gets all run_names from the database
3. Filters to chimera runs (matching pattern ``chimera_YYYYMMDD``)
4. For each chimera run, collects metric_ids
5. Retrieves summary statistics for all metrics
6. Pivots data to wide format with transition_dayobs as index

.. testcode::

   import os
   import tempfile
   import pandas as pd
   from rubin_sim.maf.chimera_progress import (
       build_chimeras, run_chimera_batches, make_chimera_summary_table
   )
   from rubin_sim.maf import batches

   consdb_visits = pd.DataFrame({
       'visitId': [1, 2, 3],
       'dayObs': [20260101, 20260102, 20260103],
       'filter': ['g', 'r', 'i']
   })

   opsim_visits = pd.DataFrame({
       'visitId': [101, 102, 103],
       'dayObs': [20260104, 20260105, 20260106],
       'filter': ['g', 'r', 'i']
   })

   with tempfile.TemporaryDirectory() as tmpdir:
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

       summary_df = make_chimera_summary_table(
           os.path.join(tmpdir, 'resultsDb_sqlite.db')
       )

       # Verify structure
       assert len(summary_df) > 0  # Has at least one row (transition date)
       assert isinstance(summary_df.columns, pd.MultiIndex)  # MultiIndex columns
       assert summary_df.index.dtype == int  # Index is integer dayobs

       # Columns should be multi-index: (metric_name, slicer_name, metric_info_label, summary_metric)
       col_names = summary_df.columns.names
       assert col_names[0] == 'metric_name'
       assert col_names[3] == 'summary_metric'

**Test Output:**

::

   (Assertion passes)

3.4 Error Handling Design
~~~~~~~~~~~~~~~~~~~~~~~~~

**Error: No common columns**

.. testcode::

   import pandas as pd
   from rubin_sim.maf.chimera_progress import build_chimera

   consdb_visits = pd.DataFrame({
       'visitId': [1, 2],
       'dayObs': [20260101, 20260102],
   })

   opsim_visits = pd.DataFrame({
       'expmjd': [100, 101],
       'dayObs': [20260103, 20260104],
   })

   try:
       build_chimera(
           consdb_visits=consdb_visits,
           opsim_visits=opsim_visits,
           start_dayobs=20260101,
           transition_dayobs=20260102,
           end_dayobs=20260104
       )
       assert False, "Should have raised ValueError"
   except ValueError as e:
       assert "share no common columns" in str(e)

**Test Output:**

::

   (Assertion passes)

**Error: Empty ResultsDb**

.. testcode::

   import os
   import tempfile
   import rubin_sim.maf.db as db

   from rubin_sim.maf.chimera_progress import make_chimera_summary_table

   with tempfile.TemporaryDirectory() as tmpdir:
       results_db_path = os.path.join(tmpdir, 'resultsDb_sqlite.db')
       results_db = db.ResultsDb(database=results_db_path)
       results_db.close()

       import warnings
       with warnings.catch_warnings(record=True) as w:
           warnings.simplefilter("always")
           summary_df = make_chimera_summary_table(results_db_path)
           assert len(summary_df) == 0
           assert len(w) == 1
           assert "No chimera run names found" in str(w[0].message)

**Test Output:**

::

   (Assertion passes)

4. Process Design
-----------------

4.1 Workflow Sequence
~~~~~~~~~~~~~~~~~~~~~

.. mermaid::

   sequenceDiagram
       participant User
       participant CLI1 as build_chimeras CLI
       participant B1 as build_chimeras
       participant IO1 as HDF5 Writer
       participant CLI2 as run_chimera_batches CLI
       participant B2 as run_chimera_batches
       participant MAF as MAF Batch
       participant DB as ResultsDb
       participant CLI3 as make_chimera_summary CLI
       participant M2 as make_chimera_summary_table
       participant IO2 as HDF5 Reader

       User->>CLI1: Execute with parameters
       CLI1->>B1: Parse args, read visits
       B1->>B1: Generate transition dates
       loop For each transition date
           B1->>B1: Filter consdb + opsim visits
           B1->>B1: Concatenate visits
           B1->>IO1: Write chimera_YYYYMMDD.h5
       end
       B1-->>CLI1: Return specs list
       CLI1-->>User: Confirm completion

       User->>CLI2: Execute with parameters
       CLI2->>B2: Parse args, find HDF5 files
       B2->>DB: Create ResultsDb
       loop For each chimera spec
           B2->>MAF: Run batch_func with run_name
           MAF->>MAF: Compute metrics
           MAF->>DB: Store metric results
       end
       B2-->>CLI2: Return results_db path
       CLI2-->>User: Confirm completion

       User->>CLI3: Execute with parameters
       CLI3->>M2: Parse args
       M2->>DB: Query all run_names
       M2->>M2: Filter chimera runs
       M2->>DB: Query summary stats
       M2->>M2: Pivot to wide format
       M2->>IO2: Write summary.h5
       CLI3-->>User: Confirm completion

4.2 State Diagram
~~~~~~~~~~~~~~~~~

.. mermaid::

   stateDiagram-v2
       [*] --> Idle

       Idle --> BuildingChimera: build_chimeras()
       BuildingChimera --> Idle: Complete
       BuildingChimera --> Error: Invalid input

       Idle --> RunningBatches: run_chimera_batches()
       RunningBatches --> Idle: Complete
       RunningBatches --> Error: ResultsDb creation failed

       Idle --> MakingSummary: make_chimera_summary_table()
       MakingSummary --> Idle: Complete
       MakingSummary --> Error: No chimera runs found

       Error --> [*]

5. Component Design
-------------------

5.1 Module Structure
~~~~~~~~~~~~~~~~~~~~

.. code-block::

   rubin_sim/maf/chimera_progress.py
   ├── DayObs Helper Functions (simplified in v1.1)
   │   └── _dayobs_range(start_dayobs, end_dayobs, step) -> list[int]
   │       └── _dayobs_to_date(dayobs: int) -> datetime.date (local helper)
   │
   ├── Chimera Name Helper Functions
   │   ├── _run_name_from_dayobs(transition_dayobs) -> str
   │   ├── _dayobs_from_run_name(run_name) -> int | None
   │   └── _dayobs_from_filename(path) -> int | None
   │
   ├── Core API Functions
   │   ├── build_chimera(consdb_visits, opsim_visits, ...) -> DataFrame
   │   ├── build_chimeras(consdb_visits, opsim_visits, ...) -> list[tuple[int, str]]
   │   ├── run_chimera_batches(chimera_specs, batch_func, out_dir, batch_kwargs) -> str
   │   └── make_chimera_summary_table(results_db) -> DataFrame
   │
   ├── CLI Helper Functions
   │   └── _read_visits_with_dayobs(path) -> DataFrame
   │
   └── Click CLI Commands
       ├── build_chimeras_cmd
       ├── run_chimera_batches_cmd (with --batch-kwarg support)
       └── make_chimera_summary_table_cmd

5.2 Class Diagram
~~~~~~~~~~~~~~~~~

.. mermaid::

   classDiagram
       class _DayObsHelpers {
           +_dayobs_to_date(dayobs: int) -> date
           +_date_to_dayobs(d: date) -> int
           +_dayobs_range(start: int, end: int, step: int) -> list[int]
       }

       class _ChimeraNameHelpers {
           +_run_name_from_dayobs(dayobs: int) -> str
           +_dayobs_from_run_name(run_name: str) -> int | None
           +_dayobs_from_filename(path: str) -> int | None
       }

       class CoreAPI {
           +build_chimera(consdb, opsim, start, transition, end) -> DataFrame
           +build_chimeras(consdb, opsim, start, end, step, out_dir) -> list
           +run_chimera_batches(specs, batch_func, out_dir, batch_kwargs) -> str
           +make_chimera_summary_table(results_db) -> DataFrame
       }

       class _CLIReaders {
           +_read_visits_with_dayobs(path: str) -> DataFrame
       }

       class ClickCommands {
           +build_chimeras_cmd()
           +run_chimera_batches_cmd()
           +make_chimera_summary_table_cmd()
       }

       CoreAPI --> _DayObsHelpers : uses
       CoreAPI --> _ChimeraNameHelpers : uses
       ClickCommands --> CoreAPI : delegates
       ClickCommands --> _CLIReaders : uses

5.3 Interface Specifications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Interface: build_chimera Parameters**

.. code-block:: python

   def build_chimera(
       consdb_visits: pd.DataFrame,  # Must contain 'dayObs' column
       opsim_visits: pd.DataFrame,   # Must contain 'dayObs' column
       start_dayobs: int,            # YYYYMMDD, inclusive
       transition_dayobs: int,       # YYYYMMDD, inclusive (consdb boundary)
       end_dayobs: int,              # YYYYMMDD, inclusive (opsim boundary)
   ) -> pd.DataFrame:
       """Returns combined DataFrame with common columns from both inputs."""

**Interface: build_chimeras Return Value**

.. code-block:: python

   # List of tuples: (transition_dayobs, hdf5_path)
   specs = [
       (20260101, "/path/to/chimera_20260101.h5"),
       (20260103, "/path/to/chimera_20260103.h5"),
       (20260105, "/path/to/chimera_20260105.h5"),
   ]

**Interface: run_chimera_batches batch_func Signature**

.. code-block:: python

   def batch_func(
       run_name: str,    # e.g., "chimera_20260101"
       **kwargs,         # Additional batch-specific arguments (e.g., from --batch-kwarg)
   ) -> dict:
       """Returns dictionary of MetricBundle objects.

       Notes:
           - Supports both ``run_name`` and ``runName`` as the simulation name parameter
           - Additional kwargs are forwarded from the CLI or API caller
       """

**Interface: make_chimera_summary_table Return Value**

.. code-block:: python

   # DataFrame with MultiIndex columns
   # Index: transition_dayobs (int)
   # Columns: MultiIndex of (metric_name, slicer_name, metric_info_label, summary_metric)

5.4 Dependency Diagram
~~~~~~~~~~~~~~~~~~~~~~

.. mermaid::

   graph TB
       subgraph "chimera_progress.py"
           CP[Core Functions]
           CLI[CLI Commands]
       end

       subgraph "Dependencies"
           PD[pandas]
           CL[Click]
           HDF5[HDF5/pytables]
           MAF[maf module]
           DB[ResultsDb]
       end

       CP --> PD
       CP --> HDF5
       CLI --> CL
       CLI --> PD
       CP --> MAF
       CP --> DB

6. Deployment Design
--------------------

6.1 Data Flow Diagram
~~~~~~~~~~~~~~~~~~~~~

.. mermaid::

   graph TB
       subgraph "Input Layer"
           I1[ConsDB SQLite/HDF5]
           I2[OpSim SQLite/HDF5]
       end

       subgraph "Build Layer"
           B1[build_chimera]
           B2[build_chimeras]
           H1[HDF5: chimera_YYYYMMDD.h5]
       end

       subgraph "Process Layer"
           P1[run_chimera_batches]
           R1[ResultsDb SQLite]
           M1[metrics HDF5]
       end

       subgraph "Output Layer"
           M2[make_chimera_summary_table]
           H2[HDF5: chimera_summary.h5]
       end

       I1 --> B2
       I2 --> B2
       B2 --> B1
       B1 --> H1
       H1 --> P1
       P1 --> R1
       P1 --> M1
       R1 --> M2
       M2 --> H2

6.2 File Structure
~~~~~~~~~~~~~~~~~~

**Build Phase Output:**

.. code-block::

   chimera_sequences/
   ├── chimera_20260101.h5
   ├── chimera_20260102.h5
   ├── chimera_20260103.h5
   └── ...

**Process Phase Output:**

.. code-block::

   chimera_results/
   ├── resultsDb_sqlite.db          # Results database
   ├── metrics/
   │   ├── chimera_20260101.h5      # Metric data per run
   │   ├── chimera_20260102.h5
   │   └── ...
   └── plots/
       ├── ...
       └── ...

**Output Phase Output:**

.. code-block::

   chimera_summary.h5               # Summary table (wide format)

6.3 Database Schema (ResultsDb)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mermaid::

   erDiagram
       version ||--o{ metrics : "contains"
       version ||--o{ summarystats : "contains"
       version ||--o{ plots : "contains"
       version ||--o{ displays : "contains"

       metrics ||--o{ summarystats : "has"
       metrics ||--o{ plots : "has"
       metrics ||--o{ displays : "has"

       version {
           int version_id PK
           string version
           string run_date
       }

       metrics {
           int metric_id PK
           string metric_name
           string slicer_name
           string run_name
           string sql_constraint
           string metric_info_label
           string metric_datafile
       }

       summarystats {
           int stat_id PK
           int metric_id FK
           string summary_name
           float summary_value
       }

       plots {
           int plot_id PK
           int metric_id FK
           string plot_type
           string plot_file
       }

       displays {
           int display_id PK
           int metric_id FK
           string display_group
           string display_subgroup
           float display_order
           string display_caption
       }

7. Design Patterns and Non-Functional Considerations
----------------------------------------------------

7.1 Design Patterns
~~~~~~~~~~~~~~~~~~~

**Pattern: Builder**
- ``build_chimera`` and ``build_chimeras`` follow the builder pattern,
  constructing complex chimera sequences from simpler data.

**Pattern: Factory**
- ``run_chimera_batches`` acts as a factory, producing metric results
  from chimera specifications.

**Pattern: Facade**
- The CLI commands provide a simplified facade over the core API functions.

**Pattern: Strategy**
- ``run_chimera_batches`` accepts different batch functions (strategies)
  for computing metrics.

7.2 Testability Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Pure Functions**: Core functions (build_chimera, make_chimera_summary_table)
  have no side effects and can be tested with doctest.
- **Temporary Files**: File I/O operations use temporary directories for testing.
- **Mock Data**: Test examples use synthetic DataFrames.

7.3 Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Step Parameter**: Allows controlling the density of transition dates
  to balance granularity vs. computational cost.
- **Memory Management**: ``clear_memory=True`` in MetricBundleGroup.run_all
  frees memory between runs.
- **HDF5 Compression**: Uses ``complevel=5`` for HDF5 files.

7.4 Maintainability Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Type Hints**: All functions use type hints for clarity.
- **Docstrings**: Full NumPyDoc-style docstrings conforming to Rubin
  Observatory style (https://developer.lsst.io/python/numpydoc.html).
- **Code Style**: Code conforms to Rubin DM Python Style Guide
  (https://developer.lsst.io/python/style.html).
- **Error Messages**: Specific error messages for common failure modes.

7.5 Security Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Path Sanitization**: Uses ``os.path.join`` for path construction.
- **Click Validation**: CLI options use ``click.Path(exists=True)`` for file validation.
- **SQL Injection**: Uses SQLAlchemy ORM for database queries (parameterized).

A. Test Examples
----------------

A.1 Complete End-to-End Test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. testcode::

   import os
   import tempfile
   import pandas as pd
   from rubin_sim.maf.chimera_progress import (
       build_chimeras, run_chimera_batches, make_chimera_summary_table
   )
   from rubin_sim.maf import batches
   from rubin_sim.maf.db import ResultsDb

   # Simulated data
   consdb_visits = pd.DataFrame({
       'visitId': list(range(1, 11)),
       'dayObs': [20260101 + i for i in range(10)],
       'filter': ['g', 'r', 'i', 'z', 'y'] * 2,
   })

   opsim_visits = pd.DataFrame({
       'visitId': list(range(101, 111)),
       'dayObs': [20260111 + i for i in range(10)],
       'filter': ['g', 'r', 'i', 'z', 'y'] * 2,
   })

   with tempfile.TemporaryDirectory() as tmpdir:
       # Step 1: Build chimera sequences
       specs = build_chimeras(
           consdb_visits=consdb_visits,
           opsim_visits=opsim_visits,
           start_dayobs=20260101,
           end_dayobs=20260120,
           step=3,
           out_dir=tmpdir
       )

       assert len(specs) == 4  # 20260101, 20260104, 20260107, 20260110

       # Step 2: Run batches
       results_db = run_chimera_batches(
           specs,
           batch_func=batches.glanceBatch,
           out_dir=tmpdir
       )

       # Step 3: Create summary table
       summary_df = make_chimera_summary_table(results_db)

       # Verify structure
       assert len(summary_df) == 4  # One row per transition date
       assert isinstance(summary_df.columns, pd.MultiIndex)
       assert all(isinstance(d, int) for d in summary_df.index)

       # Verify transition dates match
       expected_dates = [20260101, 20260104, 20260107, 20260110]
       assert list(summary_df.index) == expected_dates

**Test Output:**

::

   (Assertion passes)

A.2 DayObs Helper Function Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. testcode::

   from rubin_sim.maf.chimera_progress import (
       _dayobs_to_date,
       _date_to_dayobs,
       _dayobs_range,
       _run_name_from_dayobs,
       _dayobs_from_run_name,
       _dayobs_from_filename
   )

   # Test dayobs conversions
   assert _dayobs_to_date(20260115) == __import__('datetime').date(2026, 1, 15)
   assert _date_to_dayobs(__import__('datetime').date(2026, 1, 15)) == 20260115

   # Test dayobs range
   assert _dayobs_range(20260101, 20260105, 2) == [20260101, 20260103, 20260105]

   # Test run name encoding/decoding
   assert _run_name_from_dayobs(20260115) == "chimera_20260115"
   assert _dayobs_from_run_name("chimera_20260115") == 20260115
   assert _dayobs_from_run_name("invalid") is None

   # Test filename extraction
   assert _dayobs_from_filename("/path/to/chimera_20260115.h5") == 20260115
   assert _dayobs_from_filename("/path/to/other_20260115.h5") is None

**Test Output:**

::

   (Assertion passes)

B. API Reference
----------------

B.1 Module: rubin_sim.maf.chimera_progress
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:module:: rubin_sim.maf.chimera_progress

.. py:function:: build_chimera(consdb_visits, opsim_visits, start_dayobs, transition_dayobs, end_dayobs) -> pandas.DataFrame
   :noindex:

   Build a single chimera visit sequence.

   **Parameters:**
       - consdb_visits: DataFrame with ``dayObs`` column
       - opsim_visits: DataFrame with ``dayObs`` column
       - start_dayobs: int (YYYYMMDD)
       - transition_dayobs: int (YYYYMMDD)
       - end_dayobs: int (YYYYMMDD)

   **Returns:**
       - DataFrame with combined visits

.. py:function:: build_chimeras(consdb_visits, opsim_visits, start_dayobs, end_dayobs, step=1, out_dir=".") -> list
   :noindex:

   Build chimera visit sequences for a range of transition dates.

   **Parameters:**
       - consdb_visits: DataFrame with ``dayObs`` column
       - opsim_visits: DataFrame with ``dayObs`` column
       - start_dayobs: int (YYYYMMDD)
       - end_dayobs: int (YYYYMMDD)
       - step: int (default 1) - nights between transition dates
       - out_dir: str (default ".") - output directory

   **Returns:**
       - List of (transition_dayobs, hdf5_path) tuples

.. py:function:: run_chimera_batches(chimera_specs, batch_func=None, out_dir=".", batch_kwargs=None) -> str
   :noindex:

   Run MAF metric batches on chimera sequences.

   **Parameters:**
       - chimera_specs: list of (transition_dayobs, hdf5_path) tuples
       - batch_func: callable (default glanceBatch)
       - out_dir: str (default ".")
       - batch_kwargs: dict (optional)

   **Returns:**
       - Path to ResultsDb

.. py:function:: make_chimera_summary_table(results_db) -> pandas.DataFrame
   :noindex:

   Build summary table from chimera results.

   **Parameters:**
       - results_db: ResultsDb instance or path string

   **Returns:**
       - DataFrame with MultiIndex columns

C. CLI Reference
----------------

C.1 build_chimeras
~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Usage: build_chimeras [OPTIONS]

   Build chimera visit sequences and save them as HDF5 files.

   Options:
     --consdb-file PATH   [required] SQLite or HDF5 file with consdb visits
     --opsim-file PATH    [required] SQLite or HDF5 file with opsim visits
     --start-dayobs INTEGER  [required] Start date YYYYMMDD
     --end-dayobs INTEGER    [required] End date YYYYMMDD
     --step INTEGER          [default: 1] Nights between transition dates
     --out-dir PATH          [default: .] Output directory for HDF5 files

C.2 run_chimera_batches
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Usage: run_chimera_batches [OPTIONS]

   Run MAF metric batches on all chimera HDF5 files in a directory.

   Options:
     --chimera-dir PATH     [required] Directory containing chimera_*.h5 files
     --out-dir PATH         [default: .] Output directory for results_db
     --batch TEXT           [default: glanceBatch] Batch function name
     --batch-kwarg TEXT     Additional batch kwarg (KEY=VALUE)

C.3 make_chimera_summary_table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Usage: make_chimera_summary_table [OPTIONS]

   Query a results_db to produce a summary table (one row per transition date).

   Options:
     --results-db PATH    [required] Path to resultsDb_sqlite.db
     --out-file PATH      [default: chimera_summary.h5] Output HDF5 file
