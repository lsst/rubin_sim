# SP-3142 Targeted Diff Review

| Field | Value |
|---|---|
| Date | 2026-10-05 |
| Implementation reviewed | `9ddb45bd213bfa887885265bb09dfd75da8a4b98` (`tickets/SP-3142`) |
| Base / merge base | `8528c01602de0f9641cc1b58dce0b85dc07e945d` (`origin/main`) |
| Contract | Current `SP-3142.md`, including developer-approved Amendments 1 and 2 and subsequent verification updates |
| Result | Review performed; four findings remain unresolved. Implementation exit is not recommended until findings are resolved or explicitly dispositioned by the developer. |

## Findings

**Refreshed CI follow-up (2026-10-05, developer-reported):** EHN reports passing [CI run 37351686337](https://github.com/lsst/rubin_sim/actions/runs/37351686337) and [lint run 37351587858](https://github.com/lsst/rubin_sim/actions/runs/37351587858) on `a5cd360`, covering the follow-up implementation. All four findings are addressed with local regression and refreshed CI evidence. Only the final implementation exit record remains pending. Earlier follow-up statements about outstanding CI are superseded by this result; original findings remain the review of `9ddb45b`.

**F-4 follow-up (2026-10-05):** Progress batches now pass mapped depth/band/exposure columns to TeffStacker's existing arguments and pass `colmap` to the fO helper's exposure metric and coordinate slicer. Both batches were run on renamed-only input and compared with default-schema maps, masks, and summaries; effective exposure is positive and fO maps cover pixels. Combined suite: 27 tests and 29 subtests passed; Black, isort, and Ruff passed. F-4 is addressed. Together with prior follow-ups, all four findings are addressed locally; refreshed CI and the final implementation exit record remain outstanding. The original findings below remain the review of `9ddb45b`.

**F-1 follow-up (2026-10-05):** EHN approved IWD Amendment 4: R-3/R-4 require nside sufficient for every LSST camera pointing to cover at least one HEALPixel. The snapshot runner catches the reported empty-array minimum-reduction ValueError, warns about incomplete results, and continues later dates without suppressing unrelated ValueErrors. A real sparse-first-snapshot regression confirms recovery and later count/top18k results. Combined regression suite: 26 tests and 27 subtests passed; Black, isort, and Ruff passed. F-1 is addressed under this amended contract; F-4 remains open and refreshed CI is required.

**Follow-up (2026-10-05):** EHN approved removal of automatic predicate-column discovery (IWD Amendment 3). `_cols_from_pdconstraint` was removed; callers explicitly request predicate columns, and progress batches explicitly request depth/date/band. The revised `test_pdconstraint_db_cols` verifies ordinary functions, accessors, and physically stored `dayObs`, with omitted-column failures before explicit requests. Local verification passed (24 tests, 27 subtests); Black, isort, and Ruff passed. F-2 is addressed under the amended contract, and F-3 is also addressed for explicitly requested stored columns. Original findings below are retained as the review of `9ddb45b`. F-1 and F-4 remain unresolved; new CI evidence is required for the follow-up changes.

All four findings are P2 correctness issues. No production fixes or requirement amendments were made during this review.

### F-1: Sparse nonempty snapshots can abort the date series

**Locations:** `rubin_sim/maf/batches/progress_batch.py:80`; propagation through `rubin_sim/maf/progress.py:351`.

The HEALPix progress bundles unconditionally use `AreaSummaryMetric` with a minimum reduction. A valid nonempty visit selection can cover no HEALPix pixel centers at the requested resolution. Its map is then entirely masked, and the summary raises `ValueError: zero-size array to reduction operation minimum which has no identity`. The exception propagates out of the date-series runner, preventing subsequent snapshots from running. R-3 and R-4 state no minimum sky-coverage precondition.

Reproduced at `nside=16` with a single pointing at RA/Dec zero. This is not eliminated by choosing the resolution used in the regression tests. The IWD's acknowledged sparse-input limitation at `nside=8` therefore understates the boundary.

```python
import numpy as np
from rubin_sim.maf.batches import snapshot_batch

bundle = next(
    b for b in snapshot_batch(nside=16, bands=()).values()
    if b.metric.name == "Number of exposure area stats"
)
visits = np.array(
    [(61042.1, 0.0, 0.0, 0.0)],
    dtype=[
        ("observationStartMJD", float), ("fieldRA", float),
        ("fieldDec", float), ("rotSkyPos", float),
    ],
)
bundle.slicer.setup_slicer(visits)
assert not any(len(s["idxs"]) for s in bundle.slicer)
bundle.metric_values = np.ma.masked_all(bundle.slicer.nslice)
bundle.compute_summary_stats()  # Raises ValueError.
```

**Disposition needed:** handle all-masked maps without aborting the workflow, or obtain approval for an explicit input precondition and corresponding behavior. Add a regression using an early nonempty snapshot with no covered centers and demonstrate that later dates still run.

### F-2: Pandas function and method names are fetched as database columns

**Locations:** `rubin_sim/maf/metric_bundles/metric_bundle.py:23-38`, `:330-337`.

The token-based `_cols_from_pdconstraint` treats function and attribute names as column names. Valid queries such as `abs(night) < 3` and `band.str.startswith("g")` therefore request nonexistent SQL columns. R-7 accepts pandas query strings and requires referenced database columns to be fetched; it does not restrict queries to simple comparisons.

Reproduced using an in-memory SQLite `observations` table with `observationId=[1,2,3]`, `night=[1,2,3]`, and `band=["g","r","g"]`. A `CountMetric(col="observationId")` / `UniSlicer` bundle with each predicate was run through `MetricBundleGroup.run_all()`:

| Predicate | Correct pandas count | Erroneous predicate columns | Outcome |
|---|---|---|---|
| `abs(night) < 3` | 2 | `abs`, `night` | `DatabaseError`: no such column `abs` |
| `band.str.startswith("g")` | 2 | `band`, `str`, `startswith` | `DatabaseError`: no such column `str` or `startswith`, depending on set iteration order |

The existing function-token regression uses `@np.isfinite(...)`, which is excluded as an external-reference token; it does not exercise ordinary pandas function or accessor syntax. Basic comparisons and tested numeric literals work.

**Disposition needed:** distinguish referenced columns from callable/accessor names, with end-to-end SQLite regressions for these queries. Restricting the public query syntax instead would require an approved contract amendment, not an undocumented implementation limitation.

### F-3: Stored columns recognized by ColInfo as stacker outputs are omitted

**Locations:** `rubin_sim/maf/metric_bundles/metric_bundle.py:330-353`; `rubin_sim/maf/metric_bundles/metric_bundle_group.py:425-430`, `:473-476`.

Predicate columns are merged into the general required-column set and resolved through `ColInfo`. If a name is registered as a stacker output, its input columns are fetched instead of that name. The stacker runs only after pandas filtering. Consequently, a physically stored `dayObs` column cannot be used in a bundle predicate on SQLite, even though it is a database column available for querying.

Reproduced with an in-memory SQLite table containing `observationId`, `observationStartMJD`, and stored `dayObs=[20230225,20230226,20230227]`. For a CountMetric bundle with `pdconstraint="dayObs < 20230227"`, pandas directly selects two rows, but `bundle.db_cols` contains `observationStartMJD` and `observationId`, not `dayObs`. `run_all()` raises `UndefinedVariableError: name 'dayObs' is not defined`.

R-7 excludes columns that must be produced by stackers, not explicitly names that are already physically stored in the database. The approved design's use of `ColInfo` and statement that DayObsStacker is not usable expose an ambiguity, but do not clearly waive the automatic-fetch requirement for stored database columns. Progress batches avoid this case by using `observationStartMJD` instead.

**Disposition needed:** fetch predicate columns as database columns without generating them before filtering, preserving the approved no-stacker-column-support boundary. Add a regression with stored `dayObs`. If such stored names are intentionally unsupported, resolve that requirement ambiguity with the developer and record an approved amendment.

### F-4: Custom colmap is ignored by effective-exposure and fO components

**Locations:** `rubin_sim/maf/batches/progress_batch.py:96`, `:156`, `:194`, `:242`.

Both public batch functions accept `colmap`, and most bundles use it, but `TeffStacker` is constructed with default column names. The fO helper also uses default exposure and coordinate columns without receiving the mapping. A mapped-only schema fails to compute those metrics; where both default and mapped columns exist, those components can use different data from the other bundles.

Reproduced by mapping depth, band, exposure, and coordinates to `depth`, `passband`, `exptime`, `ra`, and `dec`. `Sum t_eff` still requests `fiveSigmaDepth`, `band`, and `visitExposureTime`; fO still requests `fieldRA`, `fieldDec`, and `visitExposureTime`. Running the effective-exposure stacker on mapped-only data raises `ValueError: no field of name fiveSigmaDepth`.

The standard opsim-schema workflow is unaffected, but the published signatures expose this mapping as a usable interface. Existing regression data only exercise the default mapping.

**Disposition needed:** propagate the mapping through the effective-exposure stacker and fO components, and compare both batches on renamed input against equivalent default-schema input. Removing or narrowing mapping support instead requires an explicit design decision.

## Review Coverage

Review proceeded from scope and outcomes to Context and the approved design, then to the diff and tests. Scope is chimera construction, snapshot and chimera metric execution, date-series ResultsDb storage, four CLIs, pandas predicates, and dayobs0 compatibility. Visualization, visit acquisition, new baselines, bespoke simulations, new metric classes, additional quality cuts, and automation remain excluded. No unauthorized scope expansion was identified in the reviewed changes; the intentional test-artifact ignore entries are already accepted in the IWD.

The reviewed branch diff covers production modules, regression tests, batch exports, console-script registration, dependency declarations, documentation exclusion configuration, and internal design/test documents. The existing uncommitted IWD updates were read as the current contract and preserved. There were no uncommitted production changes at review start.

| Area | Assessment |
|---|---|
| R-1/R-2 chimera construction | Boundaries, shared depth cut, exposures normalization, source flags, common columns, HDF5 key, last transition date, and baseline target semantics agree with the contract and regression coverage. |
| R-3/R-4 snapshots and metrics | Date boundaries, depth selection, batch contents, band labels, and fO inclusion agree on standard data; F-1 and F-4 remain. |
| R-5/R-6 storage and CLI | Shared ResultsDb, date-encoded names, summary pivot, literal kwargs, callable validation, and legacy runName fallback were reviewed. Existing workflow tests pass. |
| R-7 selection flow | SQL then pandas filtering occurs before bundle stackers; bundles are grouped by both predicates, and compatibility checks include pandas predicates. F-2 and F-3 violate or leave ambiguous the general database-column interface. |
| R-7 collision protection | Guard keys match the normal ResultsDb identity fields; dictionary and list conversion error paths agree with Amendment 2 and the collision regression. No persistence/schema or file-naming change was introduced. |
| R-7 sim_data override | Confirmed intentionally unfiltered: a bundle with `night < 2` measured all three supplied rows when `sim_data` was passed. This matches the approved caller-responsibility boundary, rather than a new finding. |
| Empty pandas selection | Confirmed no exception during metric execution, `metric_values=None`, and no ResultsDb run rows. A subsequent temporary-directory cleanup error is environmental, not a selection failure. |
| Standalone reduce/summary/plot | Confirmed `set_current(constraint)` excludes nonempty pandas predicates. Already explicitly deferred in IWD §9; not reopened as a new blocker. |
| R-8 dayobs0 | Conversion uses observing-day start and rejects conflicting MJD. Default behavior remains unchanged. Successful equivalence is not directly tested; see residual risks. |
| Packaging and docs exclusion | Four scripts resolve to progress commands; click is required in both pyproject.toml and requirements.txt and removed from optional requirements. Internal issue documents are excluded while Documenteer defaults are retained. Public documentation remains deferred. |

## Verification Evidence

- Local core regression run: `../.venv/bin/python -m pytest tests/maf/test_metricbundle.py tests/maf/test_opsimutils.py -q`: **12 passed, 18 subtests passed, 1 warning**, 6.31 seconds.
- Independent workflow review run: `tests/maf/test_progress.py`: **13 passed, 16 subtests passed**, 8.76 seconds.
- The existing dayobs0 conflict test was executed directly and passed. A combined progress/batches pytest invocation exceeded 120 seconds in the pre-existing batch setup; the full batch module was not independently completed during this review.
- `git diff --check origin/main...HEAD` passed.
- Ad hoc in-memory SQLite checks reproduced F-2 and F-3, confirmed basic filtering and the sim_data override boundary, and confirmed empty-selection ResultsDb behavior. F-1 and F-4 were reproduced during the independent workflow pass.
- The shared Python environment is Python 3.14.4, not the CI Python versions. A temporary-directory cleanup hit the known shared-mount `OSError: Directory not empty`; the empty review directory was subsequently removed with `rmdir`. No production or test files were changed.
- Developer-provided passing runs: [37338341096](https://github.com/lsst/rubin_sim/actions/runs/37338341096) and [37345298283](https://github.com/lsst/rubin_sim/actions/runs/37345298283), reported on `9ddb45b`. Local `git rev-parse HEAD` independently confirmed the matching full hash. Run results were not independently inspected because the `gh` executable is unavailable.
- Workflow configuration confirms conda installation from requirements.txt and the Linux/macOS, Python 3.12/3.13 test matrix. The test workflow also installs optional requirements; this is not an isolated requirements-only installation. The inspected required click declarations nevertheless agree.

## Residual Risks and Disposition

- The two supplied CI/lint runs are not evidence of a separate documentation build: `test_and_build.yaml` does not build Sphinx, and `build_docs.yaml` is a separate workflow. The IWD records a successful local docs build; verify the appropriate documentation workflow when completing documentation rather than inferring it from unit-test success.
- fO numeric values and normalization are not established by sparse synthetic tests, which primarily assert summary presence. Existing metric implementations are reused, but composition is not numerically checked on dense data.
- The dayobs0 regression intentionally pins conversion through the conflict error; it does not directly demonstrate successful equivalent bundles or consistent dual arguments. This is a documented test-design tradeoff, not a newly discovered production bug.
- R-9's developer-recorded 46-minute workflow upper bound was not independently reproduced. No new isolated benchmark is demanded by the approved contract.
- Collision checks are per group and do not change persistent ResultsDb identity. They do not provide global protection against callers writing different selections through separate groups with reused labels, or arbitrary custom filename reuse; these are outside the approved guard's scope.

**Disposition:** checklist item 1's review has been performed and documented, but its finding-resolution obligation remains open. Resolve F-1 through F-4 with tests and updated evidence, or obtain explicit developer dispositions/amendments where appropriate, before recording implementation exit. Documentation and Closeout has not been declared started or complete, and no pull request was opened.
