# SP-3142 — Long-term regression test suite (design)

| Field | Value |
|-------|--------|
| **Parent issue** | [SP-3142](https://rubinobs.atlassian.net/browse/SP-3142) (`docs/issues/SP-3142.md`) |
| **Branch** | `tickets/SP-3142` |
| **Status** | Implemented 2026-10-01 |
| **Created** | 2026-10-01 |

## 1. Purpose

The tests written while developing SP-3142 were written to drive development and repair. They are
too slow and too large to keep in the permanent suite. This document designs a smaller replacement
that still detects regressions in every External Requirement of SP-3142 (R-1 to R-8). R-9,
performance, remains a manual check. When the replacement is in place, the §6 verification table
and the §7 evidence in `SP-3142.md` are updated to cite it (§7 below).

## 2. Current state (measured 2026-10-01)

The SP-3142 tests were run locally with `--durations`. The results:

| Location | Tests | Lines | Wall time |
|---|---|---|---|
| `tests/progress/test_chimera.py` + `test_chimera_data.py` | 61 (+ subtests) | 1,450 | ≈ 7.5 min |
| SP-3142 additions to `tests/maf/test_metricbundle.py` | 7 | 111 | < 0.1 s |
| SP-3142 additions to `tests/maf/test_opsimutils.py` | 5 | 66 | ≈ 0.5 s |
| SP-3142 additions to `tests/maf/test_batches.py` | 2 | 17 | ≈ 27 s, plus the shared `TestBatches` setup (≈ 140 s) |

The SP-3142 selection ran in 602 s. The cost has the following causes:

- **`glanceBatch` runs.** `TestRunChimeraBatches` (4 tests) and the setup of
  `TestMakeChimeraSummaryTable` run the unrelated `glanceBatch` on 9 chimeras, about 410 s in total.
  Each test also repeats the full run. `glanceBatch` is only the default batch; none of the
  requirements depend on what it computes.
- **Full `science_radar_batch` construction.** `test_science_radar_dayobs0_mjd_mapping` takes
  ≈ 27 s, and the end-to-end `science_radar_batch` (`srd_only`) chimera runs take more. They check
  one MJD conversion.
- **Dependence on the 10-year baseline.** Every sample-data helper reads a subset of the 10-year
  baseline database (`get_baseline()`) through an SQL query. This needs the `sim_baseline` data
  download and repeats the read in each `setUpClass`.
- **Redundancy.** Many tests repeat the same assertion with different fixtures. Examples are the
  five `TestMakeChimeraSummaryTable` structure tests, the three `TestBuildChimeras` tests on step
  and last date, the mocked `run_progress_batches` call-sequence tests, and the 120-line
  `_assert_base_progress_bundles` helper. Mock-heavy tests (`TestRunProgressBatches`, parts of
  `TestRunProgressBatchesCommand`) assert *how* the code calls MAF, not what it produces. They
  break on refactoring without detecting regressions in behavior.
- **A defect.** `test_handles_run_name_and_runname` uses a `**kwargs` batch, so the
  `runName` fallback of R-6 is never exercised.

## 3. Goals and constraints

- **G-1. Requirement coverage.** Every requirement R-1 to R-8 is checked by at least one automated
  test, and the test fails if the stated behavior regresses.
- **G-2. Speed.** The new progress tests take under 15 s in total. The SP-3142 additions to
  existing MAF test files add under 2 s, plus no shared setup beyond what those files already pay.
- **G-3. Size.** The new progress test module is about 300 lines. SP-3142 additions to existing MAF
  test files are roughly halved.
- **G-4. Self-contained data.** Tests use small, deterministic synthetic visits built in the test,
  or the existing `tests/example_v3.4_0yrs.db` that the MAF tests already use. They do not use the
  10-year baseline.
- **G-5. Behavioral tests.** Assert outputs (files, `ResultsDb` rows, summary values, exit codes),
  not internal call sequences. Use no mocks. The only patch is of the module constant
  `FIVE_SIGMA_DEPTH_LIMIT` (T-1, T-3).
- **Constraint.** Test code only. No production code changes are part of this design.

## 4. Layout

- **New:** `tests/maf/test_progress.py`. `rubin_sim/maf/progress.py` and
  `maf/batches/progress_batch.py` are MAF modules, so their tests go with the other MAF tests. The
  file name is unique in the test tree.
- **Deleted:**
  - `tests/progress/` (`__init__.py`, `test_chimera.py`, `test_chimera_data.py`).
  - `tests/__init__.py`, which this branch added only so that `test_chimera.py` could use a
    relative import of `test_chimera_data`. It is not on `main`, and nothing else imports from it.
- **Trimmed:** the SP-3142 additions in `tests/maf/test_metricbundle.py`,
  `tests/maf/test_opsimutils.py`, and `tests/maf/test_batches.py` (§5.3, §5.4).

## 5. Test design

**Amendment 3 update (2026-10-05).** The original parser tests in §5.3 and historical mapping below are superseded by the current parent IWD R-7 mapping. Predicate columns are explicitly requested, not discovered automatically. `test_pdconstraint_required_columns_ignore_literals` and its helper were removed; `test_pdconstraint_db_cols` now exercises omitted-column failure and successful functions, accessors, and stored `dayObs` after explicit requests. `test_pdconstraint_end_to_end` explicitly requests `night`. The original design below remains as implementation history.

### 5.1 Synthetic visits

`test_progress.py` defines a module-level helper of about 20 lines:

```python
def _make_visits(first_dayobs: int, n_nights: int, per_night: int, seed: int) -> pd.DataFrame
```

The helper uses `numpy.random.default_rng(seed)` to make `per_night` visits per night. Each visit
starts between 0.1 and 0.4 days after 00:00 UTC on the night's date, so its `dayObs` is that date.
Bands cycle through `ugrizy`. `fieldRA` is uniform, and `fieldDec` is uniform in sin(dec) over
−90° to +12°. Each visit has `fiveSigmaDepth` ≈ 24 ± 0.3, `visitExposureTime` = 30, and
`rotSkyPos` = 0. The helper writes no `dayObs` column, because `build_chimeras_cmd` adds it with
`DayObsStacker`. These are the only columns the progress batches fetch: `band`, `fieldRA`,
`fieldDec`, `fiveSigmaDepth`, `observationStartMJD`, `rotSkyPos`, and `visitExposureTime`.

A prototype ran the complete four-command workflow (§5.2, T-5) at `nside=16` on these data, with
600 completed and 1,200 baseline visits, in under 5 s. A note for implementers: on sparse sample
data, `chimera_batch` with `nside=8` raised `ValueError` (zero-size array) in the top-18k
summary. `nside=16` did not. Use `nside=16`.

### 5.2 `tests/maf/test_progress.py`

| ID | Test | Requirements | Content |
|---|---|---|---|
| T-1 | `test_build_chimera_selection` | R-1 | Two hand-written DataFrames of 6–8 rows each with explicit `dayObs` values. One call to `build_chimera` checks: (a) the exact list of `observationId` values in order, completed visits first; (b) the S, T, and E boundaries, inclusive and exclusive as specified; (c) completed and baseline rows with `fiveSigmaDepth` of NaN, 0, and negative are dropped from both sources; (d) a missing `exposures` column becomes 1 and a null `exposures` value becomes 1; (e) a column present in only one input is absent from the output; (f) `simulated` is False then True. A second call, inside `patch("rubin_sim.maf.progress.FIVE_SIGMA_DEPTH_LIMIT", 24.1)`, checks that the cut uses the module-level limit. |
| T-2 | `test_build_chimeras_series` | R-2 | The T-1 frames go into a `tmp_path`, with completed visits extending past `end_dayobs` and a step that does not land on the last completed `dayObs`. The test asserts: the returned transition dates are the expected list, crossing a month boundary, and end at max(completed `dayObs`); each `chimera_YYYYMMDD.h5` exists and reads back with key `observations`; the last file includes completed visits after `end_dayobs` and no baseline visits after `end_dayobs`. It also checks that `dayobs_range(..., step=0)` raises `ValueError`. |
| T-3 | `test_snapshot_batch_selection` | R-3 | `snapshot_batch(end_dayobs=20260102, bands=("g",), nside=16)`. Each bundle's `pdconstraint` is applied with `DataFrame.query` to a 5-row frame: just before the end of observing day 20260102 (12:00 UTC on 2026-01-03), exactly at it, depth ≤ 0, wrong band, and an ordinary row. The selected index is asserted for the all-band and g bundles. Run names are asserted. A check with `end_dayobs=None` asserts that the MJD clause is absent. A further check, inside the same `FIVE_SIGMA_DEPTH_LIMIT` patch as T-1, asserts that the depth clause uses the shared limit. |
| T-4 | `test_batch_contents` | R-4 | Table-driven, about 25 lines. For `snapshot_batch` and `chimera_batch` (`nside=16`), it builds the set of `(metric.name, info_label, slicer class)` and checks it equals the expected 4 × 7 set. For each HEALPix bundle, the summary names must include `Count`/`Mean`/`Median`/`Min`/`Max`/`Rms`, `top18k`, and `10th%ile`. `chimera_batch` must have exactly one fO bundle with the five fO summaries and `nside` 16. `snapshot_batch` must have none. |
| T-5 | `TestProgressWorkflow` (shared `setUpClass`, about 5 s) | R-2, R-3, R-4, R-5, R-6 | Writes `completed.h5` (20 nights × 30 visits from 20260105, every 17th depth NaN) and `baseline.h5` (40 nights × 30 visits from 20260101). Through `click.testing.CliRunner`, it runs the §2 workflow commands of `SP-3142.md` with `--batch-kwarg nside=16`, sharing one results directory: `build_chimeras` (start 20260101, end 20260209, step 7), `run_chimera_batches --batch chimera_batch`, `run_progress_batches` on `completed.h5` (start 20251220, end 20260120, step 14), and `make_chimera_summary_table`. All exit codes must be 0. |
| T-5a | `test_run_names` | R-3, R-5 | The `ResultsDb` run names are exactly `chimera_20260101`, `…0108`, `…0115`, `…0122`, `…0124` (off-cadence last completed date) and `consdb_20260117`, `consdb_20260120` (off-cadence end appended). The empty early dates 20251220 and 20260103 are absent. The `run_progress_batches` output reports 4 batches. |
| T-5b | `test_summary_table` | R-5 | The HDF5 table read back has an integer index equal to the five transition dates and no snapshot dates, so `chimera_` filtering holds. Its columns are a 4-level `MultiIndex` with the specified names. |
| T-5c | `test_chimera_metric_values` | R-4 | For each transition date: the all-band and per-band `Numbers of exposures` equal the row counts of the corresponding chimera file. The per-band `Sum t_eff` values sum to the all-band value. The HEALPix `Min` ≤ `10th%ile` ≤ `Max`. The mean coadded depth is in (20, 30). The five fO summary columns are present. Values are masked on sparse data, so only presence is checked. |
| T-6 | `test_legacy_runname_batch` | R-6 | Defines `def legacy(runName="x")`, with no `**kwargs`, that returns one `CountMetric`/`UniSlicer` bundle. `run_chimera_batches` runs it on one T-5 chimera into a separate directory. The stored run name must be `chimera_YYYYMMDD`. This replaces the current test, which never exercised the fallback. |
| T-7 | `test_cli_rejects_bad_options` | R-6 | Uses subtests over both `run_*` commands, without mocks. `--batch not_a_batch`, `--batch-kwarg nside`, and `--batch-kwarg =8` must each give a non-zero exit code. Validation happens before any metric runs, so this is fast with an empty visits file or chimera directory. |
| T-8 | `test_console_scripts_registered` | R-6 | Kept as is: the four entry points resolve to the expected `module:function`. |

T-5 covers literal parsing of `--batch-kwarg` implicitly. If `nside=16` were passed as the string
`"16"`, the HEALPix slicer would fail.

### 5.3 R-7 tests in existing MAF files

`tests/maf/test_metricbundle.py` (4 tests, kept from the current suite or merged):

- `test_pdconstraint_required_columns_ignore_literals`: kept as is. The parser carries the highest
  regression risk, and its table of cases is already compact. Add one row checking that an empty
  `pdconstraint` yields an empty set.
- `test_pdconstraint_db_cols`: merges `_stored`, `_band_value_not_in_group_db_cols`, and
  `_function_not_in_group_db_cols`. One bundle and group with `"@np.isfinite(fiveSigmaDepth) and
  band == 'r'"`; the group's `db_cols` must contain `fiveSigmaDepth` and `band` and must not contain
  `np`, `isfinite`, or `r`. A bundle without the argument must have `pdconstraint == ""`.
- `test_pdconstraint_grouping`: merges `_incompatible` and `_group_structure`. It checks the
  `pdconstraints` mapping and that bundles with different `pdconstraint` values are incompatible.
- `test_pdconstraint_end_to_end`: kept. It checks that a group's filtered count is greater than 0
  and less than the unfiltered count. Add the condition that the filtered count equals the count
  from the SQL constraint `night < 5`. Equality, rather than just "fewer", is what catches a filter
  applied to the wrong data.

`tests/maf/test_opsimutils.py` (2 tests, reduced from 5):

- `test_pdconstraint_sqlite`: `get_sim_data` with `pdconstraint="night < 5"` returns a `recarray`
  in which every row satisfies the constraint. The row count equals the result for the SQL
  constraint `night < 5`. `pdconstraint=None` returns the unfiltered size. `get_visit_data` returns
  a filtered `DataFrame`.
- `test_pdconstraint_hdf5`: kept. HDF5 input is the case that motivated pandas constraints (CDD-2).

### 5.4 R-8 test in `tests/maf/test_batches.py`

Replace both SP-3142 tests with `test_science_radar_dayobs0`, which needs no map data.

- Assert that `science_radar_batch(dayobs0=20260629, mjd0=SURVEY_START_MJD + 1)` raises
  `ValueError`, and that the message reports a difference of `1.0` day. Together these pin the
  mapping m = MJD(D 00:00 UTC) + 0.5 = 61220.5 exactly, because the conflict check runs before any
  bundle is built.
- Also accept `dayobs0="2026-06-29"`, which uses the same path.

The existing no-argument `test_science_radar` already covers "neither given". The path where
consistent arguments are given differs from the conflict path only in not raising. It does not
justify the current 27-second test, which builds the full batch.

*Trade-off.* Matching on the error message couples the test to the message wording. The
alternative is to patch `rubin_sim.maf.TdePopMetric` and capture `mjd0`. That was measured at
≈ 9 s, because the batch builds many bundles before it reaches that metric. The message check
takes ≈ 3 s, nearly all of it the `rubin_sim.maf` import. The message check is preferred. If the
wording changes, update the test in the same commit.

### 5.5 Removed without replacement

These tests are dropped because of the stated overlap. They need no replacement.

| Removed | Reason |
|---|---|
| `TestDayObsHelpers` (except the step ≤ 0 check, kept in T-2) | Private helpers (`_run_name_from_dayobs`, `_dayobs_from_*`) are exercised through the run names and file names in T-2 and T-5. |
| `TestBuildChimera` (9 tests) | Folded into one explicit-expectation test, T-1. |
| `TestBuildChimeras` (6 tests), including `test_hdf5_file_structure` | T-2 and T-5. The HDF5 test tested pandas, not `build_chimeras`. |
| `TestRunChimeraBatches` (5 `glanceBatch` tests) | T-5 and T-6. `glanceBatch` is outside the requirements. |
| `TestRunProgressBatches` (4 mock tests) | T-5a checks the real date series, the appended end date, and empty early dates. |
| `TestRunProgressBatchesCommand` (5) and `TestRunChimeraBatchesCommand` (2) | T-5 (batch selection by name, kwargs, report count) and T-7 (rejections). The `snapshot_batch` default is a single literal in the click option. |
| `TestMakeChimeraSummaryTable` (6) | T-5b. The empty-database warning and `ResultsDb`-instance argument are not requirements. |
| `TestEndToEnd` `science_radar_batch` run | R-8 is §5.4. Legacy `runName` handling is T-6. |
| `_assert_base_progress_bundles` and `TestChimeraBatch`/`TestSnapshotBatch` structural tests | T-4. |

## 6. Expected result

| | Current | Proposed |
|---|---|---|
| Progress test module(s) | 1,450 lines, 61 tests | about 300 lines, 11 tests (T-1 to T-8, T-5 has 3) |
| SP-3142 additions to MAF test files | 194 lines, 14 tests | about 100 lines, 7 tests |
| Wall time (SP-3142 tests) | ≈ 10 min | < 15 s, excluding shared `TestBatches` setup that predates SP-3142 |
| Test data | 10-year baseline download | synthetic, plus existing `example_v3.4_0yrs.db` |

## 7. Requirement-to-verification mapping (replaces §6 table in `SP-3142.md`)

| Req | Verification |
|---|---|
| R-1 | T-1 `test_build_chimera_selection` |
| R-2 | T-2 `test_build_chimeras_series`; T-5 (CLI build, off-cadence last date) |
| R-3 | T-3 `test_snapshot_batch_selection`; T-5a (date series, appended end, empty early dates) |
| R-4 | T-4 `test_batch_contents`; T-5c `test_chimera_metric_values` |
| R-5 | T-5a `test_run_names`; T-5b `test_summary_table` |
| R-6 | T-5 (CLI, `--batch`, `--batch-kwarg`); T-6 `test_legacy_runname_batch`; T-7 `test_cli_rejects_bad_options`; T-8 `test_console_scripts_registered` |
| R-7 | `test_metricbundle.py`: `test_pdconstraint_required_columns_ignore_literals`, `_db_cols`, `_grouping`, `_end_to_end`; `test_opsimutils.py`: `test_pdconstraint_sqlite`, `_hdf5` |
| R-8 | `test_batches.py`: `test_science_radar_dayobs0`; existing `test_science_radar` (no-argument path) |
| R-9 | Manual (unchanged; see `SP-3142.md` §6) |

## 8. Acceptance of the replacement

*Outcome (2026-10-01).* Implemented as designed, with these differences:
the new module has 13 tests (T-5 became six tests, and a snapshot visit-count test,
`test_snapshot_metric_values`, was added); the module is 441 lines; T-3 and T-4 are in class
`TestProgressBatches`; the SP-3142 tests in the three MAF files and `test_progress.py` run in about 10 s.
The mutation spot-check caught ten of twelve edits; the other two (both to the `chimera_` filter in
`make_chimera_summary_table`) are equivalent mutants, because the filter is redundant with
`_dayobs_from_run_name` and with `pivot_table` dropping rows with no transition date. Results are in
the `SP-3142.md` Implementation Notes.

1. `pytest tests/maf/test_progress.py` passes in under 15 s locally. The SP-3142 tests in
   `test_metricbundle.py`, `test_opsimutils.py`, and `test_batches.py` pass, and add under 2 s
   beyond the setup those files already had.
2. **Mutation spot-check, done once by hand and not committed.** Each of the following edits to
   production code makes at least one new test fail:
   - change `<=` to `<` at the transition boundary in `build_chimera`;
   - drop the depth cut on the baseline part;
   - change `+ 1.5` to `+ 0.5` in `snapshot_batch`;
   - remove the appended `end_dayobs` in `run_progress_batches`;
   - remove the `chimera_` filter in `make_chimera_summary_table`;
   - remove the `runName` fallback;
   - change `+ 0.5` to `- 0.5` for `dayobs0`;
   - make `_cols_from_pdconstraint` return string literals;
   - skip `DataFrame.query` in `_local_get_sim_data`.

   Record the results in the `SP-3142.md` Implementation Notes.
3. `tests/progress/` and `tests/__init__.py` are deleted. Ruff, Black, and isort pass on the
   changed test files.
4. The `SP-3142.md` §6 verification table is replaced by §7 above, and the §7 evidence is updated
   with the new test run.
