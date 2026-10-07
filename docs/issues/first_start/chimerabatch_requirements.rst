Formal Requirements Specification for Chimera Progress Capability
==================================================================

Document Number: RTN-092-SP3142-REQ
Version: 1.0
Date: 2026-05-20
Status: Approved

.. mermaid::

   graph LR
       S[Stakeholder Needs] --> R[Functional Requirements]
       S --> NR[Non-Functional Requirements]
       R --> FR[Detailed FRs]
       NR --> P[Performance]
       NR --> S2[Security]
       NR --> M[Maintainability]
       FR --> T[Testability]

1. Introduction
---------------

1.1 Purpose
~~~~~~~~~~~

This document specifies the formal requirements for the Chimera Progress
Capability implemented in the rubin_sim/maf module for Rubin Observatory's
LSST Survey Simulation framework.

The requirements are structured according to IEEE 830-1998 guidelines for
software requirements specifications.

1.2 Scope
~~~~~~~~~

This document defines requirements for:

- Building hybrid "chimera" visit sequences from real (ConsDB) and simulated
  (OpSim) visit data
- Running metric batches on collections of chimera sequences
- Extracting summary metrics across multiple transition dates
- Command-line interfaces for pipeline execution

This document does NOT define requirements for:

- User interface design (covered in user documentation)
- Unit test implementation (covered in test files)
- Operational procedures (covered in ConOps document)

1.3 Definitions, Acronyms, and Abbreviations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Chimera**: A visit sequence combining real observations up to a transition
  date with simulated observations after that date
- **ConsDB**: Rubin Observatory's Operational Database containing real visit records
- **dayObs**: Integer YYYYMMDD representing a date in UTC-12 hours
- **FBS**: Feature Based Scheduler
- **HDF5**: Hierarchical Data Format version 5
- **LSST**: Legacy Survey of Space and Time
- **Maf**: Metrics Analysis Framework
- **OpSim**: Observatory Simulation database
- **RTN**: Rubin Technical Note
- **SP**: Survey Planning ticket
- **SQL**: Structured Query Language

1.4 References
~~~~~~~~~~~~~~

- IEEE 830-1998: IEEE Recommended Practice for Software Requirements Specifications
- RTN-092: Rubin Observatory Survey Strategy, Progress Monitoring, and Performance Metrics
- experiments/chimerabatch/chimerabatch_conops.rst: ConOps document
- experiments/chimerabatch/chimerabatch_sdd.rst: Software Design Document
- https://developer.lsst.io/python/numpydoc.html: Rubin Observatory NumPyDoc style
- https://developer.lsst.io/python/style.html: Rubin DM Python Style Guide

1.5 Overview
~~~~~~~~~~~~

Section 2 presents an overview of the system.

Section 3 defines the overall system requirements.

Section 4 specifies functional requirements in detail.

Section 5 specifies non-functional requirements.

Section 6 describes external interface requirements.

Section 7 covers other requirements.

2. Overview
-----------

2.1 Product Perspective
~~~~~~~~~~~~~~~~~~~~~~~

The Chimera Progress Capability is a component of the rubin_sim/maf module.
It extends the existing Metrics Analysis Framework (maf) to support
progress monitoring workflows that require hybrid real+simulated data.

The capability is designed to be used with:

- Existing maf batch functions (e.g., glanceBatch, science_radar_batch)
- Existing ResultsDb infrastructure
- Existing visit sequence reading utilities

2.2 Product Functions
~~~~~~~~~~~~~~~~~~~~~

The system provides three primary functions:

1. **Chimera Construction**: Build hybrid visit sequences by combining
   real and simulated data at various transition dates

2. **Batch Processing**: Run metric computations on collections of
   chimera sequences

3. **Summary Extraction**: Aggregate results into a time-series format
   for plotting and analysis

2.3 User Classes and Characteristics
------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 25 25

   * - User Class
     - Skills Required
     - Usage Pattern
   * - Survey Strategist
     - Domain expertise, Python basics
     - Interactive notebook analysis
   * - Analyst
     - Python, pandas familiarity
     - Scripted workflows
   * - Developer
     - Python, software engineering
     - Module integration
   * - Pipeline Operator
     - CLI, shell scripting
     - Automated batch processing

2.4 Operating Environment
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Operating System**: Linux (CentOS/RHEL 8+)
- **Python Version**: 3.12 or later
- **Database**: SQLite 3.x for ResultsDb
- **File Format**: HDF5 (via pytables) for visit sequences

2.5 Design and Implementation Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Language**: Python 3.12+
- **Dependencies**: pandas, astropy, sqlalchemy, click, pytables
- **Code Style**: Rubin DM Python Style Guide
  (https://developer.lsst.io/python/style.html)
- **Docstring Style**: Rubin Observatory NumPyDoc style
  (https://developer.lsst.io/python/numpydoc.html)
- **Testing**: doctest examples must pass verification

3. General Requirements
-----------------------

3.1 Functional Requirements

.. list-table::
   :header-rows: 1
   :widths: 10 70 10

   * - Requirement
     - Description
     - Section
   * - FR-001
     - The system shall build a chimera visit sequence by combining consdb visits up to and including the transition_dayobs with opsim visits after the transition_dayobs up to end_dayobs.
     - 5.1
   * - FR-002
     - The system shall validate that both consdb and opsim visit DataFrames contain a dayObs column with integer YYYYMMDD format.
     - 5.2
   * - FR-003
     - The system shall raise a ValueError when consdb and opsim visits share no common columns.
     - 5.3
   * - FR-004
     - The system shall generate one HDF5 file per transition date when building multiple chimera sequences, named chimera_YYYYMMDD.h5.
     - 5.4
   * - FR-005
     - The system shall include the maximum dayObs present in consdb_visits as a transition date, regardless of the step interval.
     - 5.4
   * - FR-006
     - The system shall use run names of the form chimera_YYYYMMDD when processing chimera sequences through metric batches.
     - 5.5
   * - FR-007
     - The system shall store all metric results from chimera batch processing in a single ResultsDb SQLite database.
     - 5.5
   * - FR-008
     - The system shall support batch functions that accept either run_name or runName as the simulation name parameter.
     - 5.5
   * - FR-009
     - The system shall return a pandas DataFrame with transition_dayobs as the index when building a summary table from ResultsDb results.
     - 5.6
   * - FR-010
     - The system shall return a pandas DataFrame with MultiIndex columns (metric_name, slicer_name, metric_info_label, summary_metric).
     - 5.6
   * - FR-011
     - The system shall emit a warning and return an empty DataFrame when no chimera run names are found in the ResultsDb.
     - 5.6
   * - FR-012
     - The system shall provide a command-line interface named build_chimeras with the specified options.
     - 6.1
   * - FR-013
     - The system shall provide a command-line interface named run_chimera_batches with the specified options.
     - 6.2
   * - FR-014
     - The system shall provide a command-line interface named make_chimera_summary_table with the specified options.
     - 6.3
   * - FR-015
     - The science_radar_batch function shall require exactly one of mjd0 or dayobs0 parameters.
     - 5.7
   * - FR-016
     - When dayobs0 is provided to science_radar_batch, the system shall convert it to mjd0 using the relationship mjd0 = Time(dayobs0, format='yymmdd').mjd - 0.5.
     - 5.7

3.2 Non-Functional Requirements

.. list-table::
   :header-rows: 1
   :widths: 10 70 10

   * - Requirement
     - Description
     - Section
   * - NFR-001
     - The build_chimera function shall execute in O(n + m) time where n and m are the numbers of consdb and opsim visits respectively.
     - 7.1
   * - NFR-002
     - The build_chimeras function shall write HDF5 files with compression level 5.
     - 7.1
   * - NFR-003
     - The make_chimera_summary_table function shall process ResultsDb queries in O(k * m) time where k is the number of chimera runs and m is the number of metrics per run.
     - 7.1
   * - NFR-004
     - The system shall handle visit sequences of at least 100,000 visits without memory exhaustion.
     - 7.2
   * - NFR-005
     - All doctest examples in this specification shall pass when executed with Python's doctest module.
     - 7.3
   * - NFR-006
     - The system shall be importable as rubin_sim.maf.chimera_progress.
     - 7.4
   * - NFR-007
     - All public functions shall conform to the Rubin Observatory NumPyDoc style for docstrings and the Rubin DM Python Style Guide for code style.
     - 7.5
   * - NFR-008
     - The system shall be executable in a standalone Python environment with rubin_sim installed.
     - 7.6

4. Detailed Functional Requirements
-----------------------------------

4.1 Chimera Building Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**FR-001**: The system shall build a chimera visit sequence by combining
consdb visits up to and including the transition_dayobs with opsim visits
after the transition_dayobs up to end_dayobs.

*Verification*: The returned DataFrame shall contain visits where:
- dayObs <= transition_dayobs come from consdb_visits
- dayObs > transition_dayobs come from opsim_visits

**FR-002**: The system shall validate that both consdb and opsim visit
DataFrames contain a dayObs column with integer YYYYMMDD format.

*Verification*: When a DataFrame without dayObs column is provided,
the system shall raise a KeyError when attempting to filter by dayObs.

**FR-003**: The system shall raise a ValueError when consdb and opsim
visits share no common columns.

*Verification*: When two DataFrames with disjoint column sets are
provided, the system shall raise ValueError with message containing
"share no common columns".

4.2 Multiple Chimera Building Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**FR-004**: The system shall generate one HDF5 file per transition date
when building multiple chimera sequences, named chimera_YYYYMMDD.h5.

*Verification*: After calling build_chimeras with step=1 and dates
20260101 through 20260105, the output directory shall contain:
- chimera_20260101.h5
- chimera_20260102.h5
- chimera_20260103.h5
- chimera_20260104.h5
- chimera_20260105.h5

**FR-005**: The system shall include the maximum dayObs present in
consdb_visits as a transition date, regardless of the step interval.

*Verification*: With consdb visits up to 20260110 and step=3, the
output shall include transition dates: 20260101, 20260104, 20260107, 20260110.

4.3 Batch Processing Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**FR-006**: The system shall use run names of the form chimera_YYYYMMDD
when processing chimera sequences through metric batches.

*Verification*: After running chimera batches, the ResultsDb metrics
table shall contain run_names matching the pattern ^chimera_\d{8}$.

**FR-007**: The system shall store all metric results from chimera
batch processing in a single ResultsDb SQLite database.

*Verification*: The run_chimera_batches function shall return a path
to a SQLite database file, and subsequent queries shall show metrics
from all processed chimera sequences.

**FR-008**: The system shall support batch functions that accept either
run_name or runName as the simulation name parameter.

*Verification*: When run_chimera_batches is called with a batch_func
that only accepts runName, the system shall successfully execute the
batch and store results.

4.4 Summary Table Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**FR-009**: The system shall return a pandas DataFrame with
transition_dayobs as the index when building a summary table from
ResultsDb results.

*Verification*: The returned DataFrame's index shall be of integer
type containing YYYYMMDD values.

**FR-010**: The system shall return a pandas DataFrame with MultiIndex
columns (metric_name, slicer_name, metric_info_label, summary_metric).

*Verification*: The returned DataFrame.columns shall be a
pandas.MultiIndex with 4 levels corresponding to the specified names.

**FR-011**: The system shall emit a warning and return an empty
DataFrame when no chimera run names are found in the ResultsDb.

*Verification*: When make_chimera_summary_table is called with a
ResultsDb containing no chimera runs, a warning shall be issued
and the return value shall be an empty DataFrame.

4.5 Command-Line Interface Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**FR-012**: The system shall provide a command-line interface named
build_chimeras with the following options:

.. list-table::
   :header-rows: 1
   :widths: 20 10 60

   * - Option
     - Required
     - Description
   * - --consdb-file
     - Yes
     - SQLite or HDF5 file with consdb visits
   * - --opsim-file
     - Yes
     - SQLite or HDF5 file with opsim visits
   * - --start-dayobs
     - Yes
     - Start date YYYYMMDD
   * - --end-dayobs
     - Yes
     - End date YYYYMMDD
   * - --step
     - No
     - Nights between transition dates
   * - --out-dir
     - No
     - Output directory for HDF5 files

**FR-013**: The system shall provide a command-line interface named
run_chimera_batches with the following options:

.. list-table::
   :header-rows: 1
   :widths: 20 10 60

   * - Option
     - Required
     - Description
   * - --chimera-dir
     - Yes
     - Directory containing chimera_*.h5 files
   * - --out-dir
     - No
     - Output directory for results_db
   * - --batch
     - No
     - Batch function name from rubin_sim.maf
   * - --batch-kwarg
     - No
     - Additional batch kwarg (KEY=VALUE). May be specified multiple times.

**FR-014**: The system shall provide a command-line interface named
make_chimera_summary_table with the following options:

.. list-table::
   :header-rows: 1
   :widths: 20 10 60

   * - Option
     - Required
     - Description
   * - --results-db
     - Yes
     - Path to resultsDb_sqlite.db
   * - --out-file
     - No
     - Output HDF5 file for the summary table

4.6 Science Radar Batch Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**FR-015**: The science_radar_batch function shall require exactly one
of mjd0 or dayobs0 parameters.

*Verification*: When science_radar_batch is called without either
parameter, a ValueError shall be raised with message containing
"Eithor mjd0 or dayobs0 must be set."

**FR-016**: When dayobs0 is provided to science_radar_batch, the
system shall convert it to mjd0 using the relationship mjd0 =
Time(dayobs0, format='yymmdd').mjd - 0.5.

*Verification*: Calling science_radar_batch(dayobs0=20260101) shall
produce the same results as calling with mjd0=57388.5.

5. Data Model Requirements
--------------------------

5.1 Visit Sequence Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**DR-001**: A visit sequence shall be represented as a pandas DataFrame.

**DR-002**: A valid visit sequence DataFrame shall contain a ``dayObs``
column with integer values in YYYYMMDD format.

**DR-003**: A visit sequence DataFrame shall contain at least one
common column between consdb and opsim data sources.

5.2 HDF5 File Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~

**DR-004**: Chimera HDF5 files shall be written with key "observations".

**DR-005**: The HDF5 file shall contain all columns present in the
chimera DataFrame at the time of writing.

5.3 ResultsDb Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~

**DR-006**: The ResultsDb shall use SQLite as the backend database.

**DR-007**: The ResultsDb shall contain a ``metrics`` table with columns:
metric_id, metric_name, slicer_name, run_name, sql_constraint,
metric_info_label, metric_datafile.

**DR-008**: The ResultsDb shall contain a ``summarystats`` table with
columns: stat_id, metric_id, summary_name, summary_value.

5.4 Summary Table Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**DR-009**: The summary table shall be stored as an HDF5 file with key
"summary".

**DR-010**: The summary table shall use transition_dayobs as the index.

**DR-011**: The summary table shall use MultiIndex columns with levels:
metric_name, slicer_name, metric_info_label, summary_metric.

6. External Interface Requirements
----------------------------------

6.1 User Interfaces
~~~~~~~~~~~~~~~~~~~

**UI-001**: The command-line interface shall use the Click framework.

**UI-002**: The CLI shall provide help text for all options via the
--help flag.

**UI-003**: Error messages shall include parameter hints for Click
validation errors.

6.3 Software Interfaces
~~~~~~~~~~~~~~~~~~~~~~~

**SI-001**: The system shall import from rubin_sim.maf.batches.

**SI-002**: The system shall import from rubin_sim.maf.db.

**SI-003**: The system shall import from rubin_sim.maf.metric_bundles.

**SI-004**: The system shall import from rubin_sim.maf.stackers.date_stackers.

**SI-005**: The system shall import from rubin_sim.maf.utils.opsim_utils.

6.4 Communication Interfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**CI-001**: All data transfer between functions shall occur via
in-memory DataFrames.

**CI-002**: File I/O shall use standard Python file operations.

7. Other Requirements
---------------------

7.1 Performance Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

7.1.1 Response Time

.. list-table::
   :header-rows: 1
   :widths: 25 15 40

   * - Operation
     - Maximum Time
     - Condition
   * - build_chimera
     - O(n + m)
     - n, m = visit counts
   * - build_chimeras
     - O(k * n)
     - k = number of dates
   * - run_chimera_batches
     - O(k * m * t)
     - t = batch computation
   * - make_chimera_summary_table
     - O(k * m)
     - m = metrics per run

7.1.2 Throughput

.. list-table::
   :header-rows: 1
   :widths: 30 20

   * - Metric
     - Value
   * - Maximum visits per sequence
     - 1,000,000
   * - Maximum transition dates
     - 10,000
   * - Maximum concurrent batches
     - 1

7.2 Security Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~

**SR-001**: The system shall validate that all input file paths exist
before attempting to read.

**SR-002**: The system shall not execute arbitrary code from file
paths without validation.

7.3 Maintainability Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**MR-001**: All public functions shall be documented with Rubin
Observatory NumPyDoc style docstrings (see
https://developer.lsst.io/python/numpydoc.html).

**MR-002**: The system shall conform to the Rubin DM Python Style Guide
(see https://developer.lsst.io/python/style.html).

**MR-003**: The system shall be unit-testable with pytest and doctest.

7.4 Portability Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**PR-001**: The system shall run on Python 3.12 without modification.

**PR-002**: The system shall produce identical output on different
platforms for the same input data.

7.5 Compliance Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**CR-001**: The system shall conform to the Rubin DM Python Style Guide
(see https://developer.lsst.io/python/style.html).

**CR-002**: The system shall pass ruff linting with the project's
configuration.

**CR-003**: The system shall pass black formatting checks.

8. Requirements Traceability
----------------------------

.. mermaid::

   graph LR
       S[Stakeholder Need] --> FR-001
       S --> FR-002
       S --> FR-003
       S --> FR-004
       S --> FR-005
       S --> FR-006
       S --> FR-007
       S --> FR-008
       S --> FR-009
       S --> FR-010
       S --> FR-011
       S --> FR-012
       S --> FR-013
       S --> FR-014
       S --> FR-015
       S --> FR-016

       RTN092[RTN-092] --> FR-001
       RTN092 --> FR-009
       RTN092 --> FR-010

       SDD[Design] --> FR-006
       SDD --> FR-007
       SDD --> FR-008

9. Change History
-----------------

.. list-table::
   :header-rows: 1
   :widths: 10 15 15 30

   * - Version
     - Date
     - Author
     - Change Description
   * - 1.0
     - 2026-05-20
     - Initial
     - Initial requirements spec

10. Approval
------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20

   * - Role
     - Name
     - Signature
     - Date
   * - Product Owner
     -
     -
     -
   * - Technical Lead
     -
     -
     -
   * - QA Reviewer
     -
     -
     -
