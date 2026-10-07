IEEE 829 Style Test Plan for Chimera Progress Capability
=========================================================

Document Information
--------------------

- **Document Identifier**: RTN-092-SP3142-TP-001
- **Version**: 1.0
- **Date**: 2026-05-21
- **Status**: Draft
- **Prepared By**: Survey Strategy Team
- **Approved By**: [Pending]

1. Introduction
---------------

1.1 Purpose
~~~~~~~~~~~

This test plan describes the test approach for the Chimera Progress Capability
implemented in rubin_sim/maf. The capability enables construction of
"extrapolated metric vs. time" plots to track survey performance progress,
as described in RTN-092.

1.2 Scope
~~~~~~~~~

This test plan covers:

- Unit testing of core API functions (build_chimera, build_chimeras,
  run_chimera_batches, make_chimera_summary_table)
- Integration testing of end-to-end workflows
- Command-line interface testing
- Compatibility testing with existing maf infrastructure

This test plan does **not** cover:

- User interface design testing
- Operational procedures
- Performance benchmarking (beyond basic functionality)

1.3 Document Conventions
~~~~~~~~~~~~~~~~~~~~~~~~

The following conventions are used in this document:

- **MUST** - Requirement that is mandatory
- **SHOULD** - Requirement that is recommended but not mandatory
- **MAY** - Requirement that is optional

1.4 Glossary
~~~~~~~~~~~~

+------------------+---------------------------------------------+
| Term             | Definition                                  |
+------------------+---------------------------------------------+
| Chimera          | Hybrid visit sequence combining real consdb |
|                  | visits with simulated opsim visits          |
+------------------+---------------------------------------------+
| ConsDB           | Operational Database containing real Rubin  |
|                  | Observatory visit records                   |
+------------------+---------------------------------------------+
| dayObs           | Integer YYYYMMDD representing a date in     |
|                  | UTC-12 hours                                |
+------------------+---------------------------------------------+
| Maf              | Metrics Analysis Framework                  |
+------------------+---------------------------------------------+
| OpSim            | Observatory Simulation database             |
+------------------+---------------------------------------------+
| ResultsDb        | SQLite database storing metric results      |
|                  | and summary statistics                      |
+------------------+---------------------------------------------+
| Transition Date  | Date that separates consdb visits (up to)   |
|                  | from opsim visits (after) in a chimera      |
+------------------+---------------------------------------------+

1.5 References
~~~~~~~~~~~~~~

- RTN-092: Rubin Observatory Survey Strategy, Progress Monitoring, and Performance Metrics
- experiments/chimerabatch/chimerabatch_requirements.rst: Requirements Specification
- experiments/chimerabatch/chimerabatch_sdd.rst: Software Design Document
- experiments/chimerabatch/chimerabatch_conops.rst: ConOps Document
- IEEE 829-2008: Standard for Software and System Test Documentation
- rubin_sim/maf documentation: Metrics Analysis Framework user guide

2. Test Approach
----------------

2.1 Test Levels
~~~~~~~~~~~~~~~

The test approach includes the following levels:

**Unit Tests (TL-1)**
- Test individual functions with mock/synthetic data
- Focus on input validation, edge cases, and error handling

**Integration Tests (TL-2)**
- Test function combinations and workflows
- Test file I/O operations
- Test database interactions

**System Tests (TL-3)**
- Test complete workflows from CLI
- Test with real sample data from get_baseline()

2.2 Test Types
~~~~~~~~~~~~~~

+---------------+-----------------------------------------------+
| Test Type     | Description                                   |
+---------------+-----------------------------------------------+
| Functional    | Verify functions produce expected outputs     |
|                 for given inputs                              |
+---------------+-----------------------------------------------+
| Structural    | Verify code paths and edge cases              |
+---------------+-----------------------------------------------+
| Interface     | Verify CLI arguments and environment          |
+---------------+-----------------------------------------------+
| Regression    | Verify existing functionality is not broken   |
+---------------+-----------------------------------------------+

2.3 Test Techniques
~~~~~~~~~~~~~~~~~~~

**Equivalence Partitioning**
- Test with valid dayObs ranges
- Test with various step values
- Test with different batch functions

**Boundary Value Analysis**
- Test with minimum step value (1)
- Test with maximum consdb dayObs as transition
- Test with empty DataFrames

**Error Guessing**
- Test with missing dayObs column
- Test with disjoint column sets
- Test with invalid file paths

**Integration Testing**
- Test build_chimera with build_chimeras
- Test run_chimera_batches with make_chimera_summary_table
- Test full end-to-end workflow

2.4 Test Tools
~~~~~~~~~~~~~~

+------------------+------------------------+------------------------+
| Tool             | Purpose                | Version Requirement    |
+------------------+------------------------+------------------------+
| unittest         | Test framework         | Python 3.12+           |
+------------------+------------------------+------------------------+
| pytest           | Alternative test runner| (optional)             |
+------------------+------------------------+------------------------+
| pandas           | DataFrame testing      | Latest stable          |
+------------------+------------------------+------------------------+
| pytables         | HDF5 testing           | Latest stable          |
+------------------+------------------------+------------------------+
| SQLite           | ResultsDb testing      | Python 3 stdlib        |
+------------------+------------------------+------------------------+

2.5 Test Environment
~~~~~~~~~~~~~~~~~~~~

**Hardware Requirements**
- Minimum: 2 GB RAM
- Recommended: 4 GB RAM or more for large visit sequences

**Software Requirements**
- Python 3.12+
- rubin_sim package with all dependencies
- HDF5 library
- SQLite

**Test Data Requirements**
- Sample opsim database (500 visits)
- Sample consdb database (100 visits from first 2 months)
- Generated via get_baseline() sampling

3. Test Schedule
----------------

3.1 Test Phases
~~~~~~~~~~~~~~~

+---------------+-------------------+------------------+------------------+
| Phase         | Duration          | Deliverables     | Milestones       |
+---------------+-------------------+------------------+------------------+
| Unit Tests    | 2 days            | test_chimera.py  | FR-001 to FR-011 |
+---------------+-------------------+------------------+------------------+
| Integration   | 2 days            | test_integration | FR-006 to FR-011 |
+---------------+-------------------+------------------+------------------+
| CLI Tests     | 1 day             | test_cli         | FR-012 to FR-014 |
+---------------+-------------------+------------------+------------------+
| End-to-End    | 1 day             | test_e2e         | FR-001 to FR-016 |
+---------------+-------------------+------------------+------------------+
| Documentation | 0.5 day           | Test plan update | NFR-005          |
+---------------+-------------------+------------------+------------------+

3.2 Test Milestones
~~~~~~~~~~~~~~~~~~~

+------------------+------------------+------------------+
| Milestone        | Target Date      | Status           |
+------------------+------------------+------------------+
| Unit test suite  | 2026-05-23       | Pending          |
+------------------+------------------+------------------+
| Integration test | 2026-05-24       | Pending          |
+------------------+------------------+------------------+
| All tests pass   | 2026-05-25       | Pending          |
+------------------+------------------+------------------+

4. Test Deliverables
--------------------

4.1 Test Suite Files
~~~~~~~~~~~~~~~~~~~~

+---------------------------+----------------------------------+
| File                      | Purpose                          |
+---------------------------+----------------------------------+
| test_chimera.py           | Main test suite                  |
+---------------------------+----------------------------------+
| test_chimera_data.py      | Test data generation helpers     |
+---------------------------+----------------------------------+
| test_chimera_cli.py       | CLI command tests (optional)     |
+---------------------------+----------------------------------+
| resultsDb_sqlite.db       | Test database (generated)        |
+---------------------------+----------------------------------+

4.2 Test Documentation
~~~~~~~~~~~~~~~~~~~~~~

+---------------------------+----------------------------------+
| File                      | Purpose                          |
+---------------------------+----------------------------------+
| test_results.html         | Test results summary             |
+---------------------------+----------------------------------+
| coverage_report/          | Code coverage report             |
+---------------------------+----------------------------------+
| test_log.txt              | Test execution log               |
+---------------------------+----------------------------------+

5. Test Criteria
----------------

5.1 Pass/Fail Criteria
~~~~~~~~~~~~~~~~~~~~~~

**Pass Criteria:**
- All unit tests pass (100% of test cases)
- All integration tests pass (100% of test cases)
- All CLI tests pass (100% of test cases)
- Test coverage >= 80% for core functions
- No memory leaks detected

**Fail Criteria:**
- Any test case fails with AssertionError
- Any test case raises unexpected exception
- Test execution takes more than 5 minutes
- Memory usage exceeds 4 GB during test run

5.2 Exit Criteria
~~~~~~~~~~~~~~~~~

The test suite may be considered complete when:

- All planned test cases have been executed
- No critical or high-priority bugs remain open
- Test coverage meets minimum requirements
- Documentation is complete

5.3 Suspension Criteria
~~~~~~~~~~~~~~~~~~~~~~~

Testing should be suspended if:

- More than 10% of tests fail due to environment issues
- Test data generation fails
- Critical infrastructure is unavailable
- Time budget is exhausted

6. Test Deliverables
--------------------

6.1 Test Output
~~~~~~~~~~~~~~~

Test output shall include:

- Number of test cases executed
- Number of test cases passed
- Number of test cases failed
- Number of test cases skipped
- Execution time
- Code coverage percentage

6.2 Defect Reporting
~~~~~~~~~~~~~~~~~~~~

Defects shall be reported with:

- Unique identifier
- Priority (Critical/High/Medium/Low)
- Severity (Blocker/Cosmetic)
- Steps to reproduce
- Expected vs actual results
- Environment details

7. Test Management
------------------

7.1 Roles and Responsibilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+------------------+----------------------------------+
| Role             | Responsibilities                 |
+------------------+----------------------------------+
| Test Lead        | Coordinate testing activities    |
+------------------+----------------------------------+
| Test Engineer    | Design and execute test cases    |
+------------------+----------------------------------+
| Developer        | Fix defects and provide test data|
+------------------+----------------------------------+

7.2 Test Schedule
~~~~~~~~~~~~~~~~~

+------------------+------------------+------------------+------------------+
| Activity         | Start Date       | End Date         | Duration         |
+------------------+------------------+------------------+------------------+
| Test planning    | 2026-05-21       | 2026-05-21       | 1 day            |
+------------------+------------------+------------------+------------------+
| Test design      | 2026-05-21       | 2026-05-22       | 2 days           |
+------------------+------------------+------------------+------------------+
| Test execution   | 2026-05-23       | 2026-05-25       | 3 days           |
+------------------+------------------+------------------+------------------+
| Test reporting   | 2026-05-25       | 2026-05-25       | 1 day            |
+------------------+------------------+------------------+------------------+

8. Test Environment Requirements
--------------------------------

8.1 Hardware Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~

- CPU: 2+ cores recommended
- RAM: 4 GB minimum, 8 GB recommended
- Disk: 1 GB free space for test data

8.2 Software Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~

- Python 3.12 or later
- rubin_sim package (installed in development mode)
- HDF5 library
- SQLite (Python stdlib)
- pandas, astropy, sqlalchemy, click

8.3 Test Data Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Sample opsim visits: 500 visits from get_baseline()
- Sample consdb visits: 100 visits from first 2 months
- Generated dynamically using random sampling

9. Test Risks and Mitigation
----------------------------

9.1 Test Risks
~~~~~~~~~~~~~~

+------------------+------------------+------------------+
| Risk             | Probability      | Impact           |
+------------------+------------------+------------------+
| Test data        | Medium           | Medium           |
| generation slow  |                  |                  |
+------------------+------------------+------------------+
| MAF batch        | Low              | High             |
| processing slow  |                  |                  |
+------------------+------------------+------------------+
| Environment      | Low              | High             |
| configuration    |                  |                  |
+------------------+------------------+------------------+

9.2 Mitigation
~~~~~~~~~~~~~~

- Use cached test data when possible
- Skip slow tests with @unittest.skipUnless
- Use temporary directories for isolation
- Provide clear error messages

10. Approvals
-------------

+------------------+------------------+------------------+------------------+
| Role             | Name             | Signature        | Date             |
+------------------+------------------+------------------+------------------+
| Test Lead        |                  |                  |                  |
+------------------+------------------+------------------+------------------+
| Project Manager  |                  |                  |                  |
+------------------+------------------+------------------+------------------+
| QA Reviewer      |                  |                  |                  |
+------------------+------------------+------------------+------------------+

11. Appendix
------------

11.1 Test Case Matrix
~~~~~~~~~~~~~~~~~~~~~

+------------------+------------------+------------------+------------------+
| Requirement      | Test Class       | Test Methods     | Status           |
+------------------+------------------+------------------+------------------+
| FR-001           | TestBuildChimera | test_basic,      | Done             |
|                  |                  | test_boundary,   |                  |
|                  |                  | test_empty_*,    |                  |
|                  |                  | test_filter_*    |                  |
+------------------+------------------+------------------+------------------+
| FR-002           | TestBuildChimera | - (no direct    | N/A              |
|                  |                  | equivalent)      |                  |
+------------------+------------------+------------------+------------------+
| FR-003           | TestBuildChimera | - (invalid      | N/A              |
|                  |                  | test - can't    |                  |
|                  |                  | trigger error)   |                  |
+------------------+------------------+------------------+------------------+
| FR-004           | TestBuildChimera | test_hdf5_*,    | Done             |
|                  |                  | test_*,          |                  |
|                  |                  | test_creates_*   |                  |
+------------------+------------------+------------------+------------------+
| FR-005           | TestBuildChimera | test_last_date   | Done             |
+------------------+------------------+------------------+------------------+
| FR-006           | TestRunBatches   | test_multiple_*  | Done             |
+------------------+------------------+------------------+------------------+
| FR-007           | TestRunBatches   | test_*,          | Done             |
|                  |                  | test_stores_*    |                  |
+------------------+------------------+------------------+------------------+
| FR-008           | TestRunBatches   | test_handles_*   | Done             |
+------------------+------------------+------------------+------------------+
| FR-009           | TestSummaryTable | test_*,          | Done             |
|                  |                  | test_has_*,      |                  |
|                  |                  | test_supports_*  |                  |
+------------------+------------------+------------------+------------------+
| FR-010           | TestSummaryTable | test_has_*,      | Done             |
|                  |                  | test_*,          |                  |
|                  |                  | test_has_*       |                  |
+------------------+------------------+------------------+------------------+
| FR-011           | TestSummaryTable | test_handles_*   | Done             |
+------------------+------------------+------------------+------------------+
| FR-012           | TestBuildChimera | - (CLI not      | Not covered      |
|                  |                  | implemented)     |                  |
+------------------+------------------+------------------+------------------+
| FR-013           | TestRunBatches   | - (CLI not      | Not covered      |
|                  |                  | implemented)     |                  |
+------------------+------------------+------------------+------------------+
| FR-014           | TestSummaryTable | - (CLI not      | Not covered      |
|                  |                  | implemented)     |                  |
+------------------+------------------+------------------+------------------+
| FR-015           | TestScienceRadar | - (ScienceRadar | Not covered      |
|                  |                  | tests not        |                  |
|                  |                  | implemented)     |                  |
+------------------+------------------+------------------+------------------+
| FR-016           | TestScienceRadar | - (ScienceRadar | Not covered      |
|                  |                  | tests not        |                  |
|                  |                  | implemented)     |                  |
+------------------+------------------+------------------+------------------+

11.2 Sample Test Cases
~~~~~~~~~~~~~~~~~~~~~~

**Test Case TC-001: Build Chimera Basic Functionality**

- **ID**: TC-001
- **Title**: Build chimera from consdb and opsim visits
- **Objective**: Verify build_chimera combines visits correctly
- **Preconditions**:
  - consdb_visits DataFrame with dayObs column
  - opsim_visits DataFrame with dayObs column
- **Input**:
  - start_dayobs: 20260101
  - transition_dayobs: 20260103
  - end_dayobs: 20260106
- **Expected Result**:
  - Output contains 6 visits (3 from consdb, 3 from opsim)
  - dayObs <= 20260103 from consdb_visits
  - dayObs > 20260103 from opsim_visits
- **Status**: Pending

**Test Case TC-002: Build Chimeras File Generation**

- **ID**: TC-002
- **Title**: Generate HDF5 files for multiple transition dates
- **Objective**: Verify build_chimeras creates correct files
- **Preconditions**:
  - Sample visit DataFrames available
- **Input**:
  - step: 2
  - start_dayobs: 20260101
  - end_dayobs: 20260110
  - consdb max dayObs: 20260109
- **Expected Result**:
  - Files created: chimera_20260101.h5, chimera_20260103.h5,
    chimera_20260105.h5, chimera_20260107.h5, chimera_20260109.h5
  - Last file uses max consdb dayObs regardless of step
- **Status**: Pending

**Test Case TC-003: End-to-End Workflow**

- **ID**: TC-003
- **Title**: Complete workflow from build to summary
- **Objective**: Verify full pipeline works end-to-end
- **Preconditions**:
  - Sample test data generated
- **Input**:
  - Build chimera sequences for 5 transition dates
  - Run glanceBatch on all sequences
- **Expected Result**:
  - ResultsDb created with 5 runs
  - Summary table has 5 rows with correct MultiIndex columns
  - All metrics computed successfully
- **Status**: Pending

11.3 Change History
~~~~~~~~~~~~~~~~~~~

+------------------+------------------+------------------+------------------+
| Version          | Date             | Author           | Changes          |
+------------------+------------------+------------------+------------------+
| 1.0              | 2026-05-21       | Initial          | Draft            |
+------------------+------------------+------------------+------------------+
