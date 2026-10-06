"""Regression tests for chimera and snapshot progress tracking.

All visits are small synthetic data built here.
"""

import os
import tempfile
import unittest
from importlib.metadata import distribution
from unittest.mock import patch

import numpy as np
import pandas as pd
from astropy.time import Time
from click.testing import CliRunner

import rubin_sim.maf.batches as batches
from rubin_sim.maf import MetricBundleGroup
from rubin_sim.maf.db import ResultsDb
from rubin_sim.maf.progress import (
    build_chimera,
    build_chimeras,
    build_chimeras_cmd,
    dayobs_range,
    make_chimera_summary_table_cmd,
    run_chimera_batches,
    run_chimera_batches_cmd,
    run_progress_batches,
    run_progress_batches_cmd,
)

BANDS = "ugrizy"
NSIDE = 16


def _make_visits(first_dayobs, n_nights, per_night, seed):
    """Make deterministic synthetic visits, without a ``dayObs`` column.

    Visits start 0.1 to 0.4 days after 00:00 UTC on the calendar day
    after each ``dayObs`` date, which is within that observing day
    (UTC-12). Only columns the progress batches fetch are included.
    """
    rng = np.random.default_rng(seed)
    start = str(first_dayobs)
    mjd0 = Time(f"{start[:4]}-{start[4:6]}-{start[6:]}T00:00:00").mjd + 1.0
    n_visits = n_nights * per_night
    night = np.repeat(np.arange(n_nights), per_night)
    return pd.DataFrame(
        {
            "observationId": np.arange(n_visits) + 100000 * seed,
            "observationStartMJD": mjd0 + night + rng.uniform(0.1, 0.4, n_visits),
            "fieldRA": rng.uniform(0, 360, n_visits),
            "fieldDec": np.degrees(np.arcsin(rng.uniform(-1, 0.2, n_visits))),
            "band": np.tile(list(BANDS), n_visits // 6 + 1)[:n_visits],
            "fiveSigmaDepth": 24.0 + rng.normal(0, 0.3, n_visits),
            "visitExposureTime": 30.0,
            "rotSkyPos": 0.0,
        }
    )


class TestChimeraConstruction(unittest.TestCase):
    """Construction and date-range tests for chimera visit sequences."""

    def test_build_chimera(self):
        dates = {"start_dayobs": 20260102, "transition_dayobs": 20260104, "end_dayobs": 20260107}
        completed = pd.DataFrame(
            [
                (1, 20260101, 24.0),  # Before start.
                (2, 20260102, 24.0),  # At start: included.
                (3, 20260103, np.nan),
                (4, 20260104, 24.0),  # At transition: included.
                (5, 20260104, 0.0),
                (6, 20260104, -1.0),
                (7, 20260105, 24.0),  # After transition.
            ],
            columns=["observationId", "dayObs", "fiveSigmaDepth"],
        ).assign(band="g", completed_only=range(7))
        baseline = pd.DataFrame(
            [
                (101, 20260104, 23.0),  # At transition: excluded.
                (102, 20260105, 23.0),  # After transition: included.
                (103, 20260105, np.nan),
                (104, 20260106, 0.0),
                (105, 20260107, 23.0),  # At target: included.
                (106, 20260108, 23.0),  # After target.
                (107, 20260105, -1.0),
            ],
            columns=["observationId", "dayObs", "fiveSigmaDepth"],
        ).assign(band="r", exposures=2, baseline_only=range(7))
        result = build_chimera(completed, baseline, **dates)

        # Boundaries (S, T, E inclusive; T exclusive for baseline) and
        # depth cuts (NaN, zero, negative) on both sources.
        self.assertEqual(result["observationId"].tolist(), [2, 4, 102, 105])
        self.assertEqual(result["simulated"].tolist(), [False, False, True, True])
        # Missing completed-visits exposures is 1; baseline value is kept.
        self.assertEqual(result["exposures"].tolist(), [1, 1, 2, 2])
        # Only columns common to both inputs, plus simulated.
        self.assertEqual(
            set(result.columns),
            {"observationId", "dayObs", "fiveSigmaDepth", "band", "exposures", "simulated"},
        )

        # Null exposures in an existing completed-visits column become 1.
        with_exposures = completed.assign(exposures=5.0)
        with_exposures.loc[with_exposures["observationId"] == 2, "exposures"] = 3.0
        with_exposures.loc[with_exposures["observationId"] == 4, "exposures"] = np.nan
        result = build_chimera(with_exposures, baseline, **dates)
        self.assertEqual(result["exposures"].tolist(), [3, 1, 2, 2])

        # A cutoff equal to the baseline depth excludes baseline visits.
        with patch("rubin_sim.maf.progress.FIVE_SIGMA_DEPTH_LIMIT", 23.0):
            result = build_chimera(completed, baseline, **dates)
        self.assertEqual(result["observationId"].tolist(), [2, 4])

    def test_build_chimeras_series(self):
        # Completed visits run past end_dayobs, and the step does not
        # land on the last completed date. Dates cross a month boundary.
        completed = pd.DataFrame(
            {
                "observationId": range(12),
                "dayObs": np.repeat([20260129, 20260130, 20260131, 20260201, 20260202, 20260203], 2),
                "fiveSigmaDepth": 24.0,
            }
        )
        baseline = pd.DataFrame(
            {
                "observationId": range(100, 118),
                "dayObs": np.repeat(
                    [
                        20260129,
                        20260130,
                        20260131,
                        20260201,
                        20260202,
                        20260203,
                        20260204,
                        20260205,
                        20260206,
                    ],
                    2,
                ),
                "fiveSigmaDepth": 23.0,
            }
        )
        with tempfile.TemporaryDirectory() as out_dir:
            specs = build_chimeras(completed, baseline, 20260129, 20260201, step=2, out_dir=out_dir)
            self.assertEqual([t for t, _ in specs], [20260129, 20260131, 20260202, 20260203])
            for transition, path in specs:
                self.assertEqual(os.path.basename(path), f"chimera_{transition}.h5")
            chimeras = {t: pd.read_hdf(path, key="observations") for t, path in specs}

        first, last = chimeras[20260129], chimeras[20260203]
        # Baseline visits stop at end_dayobs ...
        self.assertEqual(first.loc[first["simulated"], "dayObs"].max(), 20260201)
        # ... but completed visits do not, and nothing is simulated
        # once the transition is after end_dayobs.
        self.assertEqual(len(last), len(completed))
        self.assertFalse(last["simulated"].any())
        self.assertEqual(last["dayObs"].max(), 20260203)

    def test_dayobs_range(self):
        self.assertEqual(dayobs_range(20260130, 20260202), [20260130, 20260131, 20260201, 20260202])
        with self.assertRaises(ValueError):
            dayobs_range(20260101, 20260105, 0)


class TestProgressBatches(unittest.TestCase):
    """Regression tests for snapshot and chimera batch definitions."""

    def test_snapshot_batch_selection(self):
        end_mjd = Time("2026-01-03T12:00:00").mjd  # End of observing day 20260102.
        visits = pd.DataFrame(
            [
                (end_mjd - 1.0, 24.0, "g"),
                (end_mjd - 0.001, 24.0, "g"),
                (end_mjd, 24.0, "g"),
                (end_mjd - 0.5, 0.0, "g"),
                (end_mjd - 0.5, 24.0, "r"),
            ],
            columns=["observationStartMJD", "fiveSigmaDepth", "band"],
            index=["ordinary", "just_before_end", "at_end", "zero_depth", "other_band"],
        )
        bundles = batches.snapshot_batch(
            run_name="x_20260102", bands=("g",), nside=NSIDE, end_dayobs=20260102
        )
        constraints = {b.info_label: b.pdconstraint for b in bundles.values()}
        self.assertEqual(
            visits.query(constraints["snapshot_all"]).index.tolist(),
            ["ordinary", "just_before_end", "other_band"],
        )
        self.assertEqual(
            visits.query(constraints["snapshot_g"]).index.tolist(), ["ordinary", "just_before_end"]
        )
        self.assertEqual({b.run_name for b in bundles.values()}, {"x_20260102"})

        # No end date: no date cut, and the shared depth limit still applies.
        future_visits = pd.DataFrame(
            {"observationStartMJD": 1e6, "fiveSigmaDepth": [23.9, 24.0, 24.1, 24.2, 24.3], "band": "g"}
        )
        with patch("rubin_sim.maf.progress.FIVE_SIGMA_DEPTH_LIMIT", 24.0):
            bundles = batches.snapshot_batch(bands=(), nside=NSIDE)
        (pdconstraint,) = {b.pdconstraint for b in bundles.values()}
        self.assertEqual(future_visits.query(pdconstraint)["fiveSigmaDepth"].tolist(), [24.1, 24.2, 24.3])

    def test_colmap_metric_values(self):
        visits = _make_visits(20260101, 1, 600, seed=2)
        visits["visitExposureTime"] = np.resize([15.0, 30.0, 60.0], len(visits))
        renamed = {
            "fiveSigmaDepth": "depth",
            "band": "passband",
            "visitExposureTime": "exptime",
            "fieldRA": "ra",
            "fieldDec": "dec",
            "observationStartMJD": "mjd",
        }
        colmap = batches.col_map_dict()
        colmap.update(
            {key: renamed.get(value, value) for key, value in colmap.items() if isinstance(value, str)}
        )
        with tempfile.TemporaryDirectory() as tmp:
            original_path = os.path.join(tmp, "default.h5")
            mapped_path = os.path.join(tmp, "mapped.h5")
            visits.to_hdf(original_path, key="observations")
            visits.rename(columns=renamed).to_hdf(mapped_path, key="observations")
            for batch, prefix in ((batches.snapshot_batch, "snapshot"), (batches.chimera_batch, "chimera")):
                with self.subTest(batch=batch.__name__):
                    results = []
                    for path, mapping in ((original_path, None), (mapped_path, colmap)):
                        bundles = batch(nside=NSIDE, bands=("g",), colmap=mapping)
                        group = MetricBundleGroup(bundles, path, out_dir=tmp, save_early=False)
                        group.run_all()
                        results.append({(b.metric.name, b.info_label): b for b in bundles.values()})
                    original_bundles, mapped_bundles = results
                    self.assertEqual(original_bundles.keys(), mapped_bundles.keys())
                    for key, original in original_bundles.items():
                        with self.subTest(metric=key):
                            mapped = mapped_bundles[key]
                            np.testing.assert_array_equal(
                                original.metric_values.mask, mapped.metric_values.mask
                            )
                            np.testing.assert_allclose(
                                original.metric_values.compressed(), mapped.metric_values.compressed()
                            )
                            self.assertEqual(original.summary_values.keys(), mapped.summary_values.keys())
                            for name, value in original.summary_values.items():
                                with self.subTest(summary=name):
                                    np.testing.assert_array_equal(value, mapped.summary_values[name])
                            self.assertFalse(set(renamed) & mapped.db_cols)
                    self.assertGreater(mapped_bundles[("Sum t_eff", f"{prefix}_all")].metric_values[0], 0)
                    if batch is batches.chimera_batch:
                        fo = next(b for b in mapped_bundles.values() if b.metric.name == "fO")
                        self.assertGreater(fo.metric_values.count(), 0)

    def test_batch_contents(self):
        required_spatial_summaries = {
            "Mean None",
            "Rms None",
            "Median None",
            "Max None",
            "Min None",
            "N(+3Sigma)",
            "N(-3Sigma)",
            "Count None",
            "top18k",
            "10th%ile metricdata",
        }
        fo_summaries = {"fOArea", "fOArea/benchmark", "fONv", "fONv/benchmark", "fOArea_750"}

        for batch, prefix in ((batches.snapshot_batch, "snapshot"), (batches.chimera_batch, "chimera")):
            with self.subTest(batch=batch.__name__):
                bundles = list(batch(nside=NSIDE).values())
                base_bundles = [b for b in bundles if b.metric.name != "fO"]
                labels = [f"{prefix}_{band}" for band in BANDS] + [f"{prefix}_all"]
                expected = {
                    (name, label, slicer)
                    for label in labels
                    for name, slicer in (
                        ("Sum t_eff", "UniSlicer"),
                        ("Numbers of exposures", "UniSlicer"),
                        ("Number of exposure area stats", "HealpixSlicer"),
                        ("Depth area stats", "HealpixSlicer"),
                    )
                }
                self.assertEqual(len(base_bundles), len(expected))
                self.assertEqual(
                    {(b.metric.name, b.info_label, type(b.slicer).__name__) for b in base_bundles}, expected
                )

                for bundle in base_bundles:
                    if type(bundle.slicer).__name__ == "HealpixSlicer":
                        with self.subTest(metric=bundle.metric.name, label=bundle.info_label):
                            names = {m.name for m in bundle.summary_metrics}
                            self.assertEqual(required_spatial_summaries - names, set())

                fo = [b for b in bundles if b.metric.name == "fO"]
                if prefix == "snapshot":
                    self.assertEqual(fo, [])
                else:
                    self.assertEqual(len(fo), 1)
                    self.assertEqual({m.name for m in fo[0].summary_metrics}, fo_summaries)
                    self.assertEqual(fo[0].slicer.nside, NSIDE)


class TestProgressWorkflow(unittest.TestCase):
    """Exercise progress workflow commands with shared synthetic inputs."""

    @classmethod
    def setUpClass(cls):
        temporary_directory = tempfile.TemporaryDirectory(prefix="progress_test_")
        cls.addClassCleanup(temporary_directory.cleanup)
        cls.tmp = temporary_directory.name
        cls.completed = _make_visits(20260105, 20, 30, seed=1)
        cls.completed.loc[::17, "fiveSigmaDepth"] = np.nan
        baseline = _make_visits(20260101, 40, 30, seed=2)
        completed_file = os.path.join(cls.tmp, "completed.h5")
        baseline_file = os.path.join(cls.tmp, "baseline.h5")
        cls.completed.to_hdf(completed_file, key="observations")
        baseline.to_hdf(baseline_file, key="observations")

        cls.chimera_dir = os.path.join(cls.tmp, "chimeras")
        cls.results_dir = os.path.join(cls.tmp, "results")
        cls.results_db = os.path.join(cls.results_dir, "resultsDb_sqlite.db")
        cls.summary_file = os.path.join(cls.tmp, "chimera_summary.h5")
        kwarg = ["--batch-kwarg", f"nside={NSIDE}"]
        runner = CliRunner()
        steps = {
            "build": (
                build_chimeras_cmd,
                ["--consdb-file", completed_file, "--opsim-file", baseline_file]
                + ["--start-dayobs", "20260101", "--end-dayobs", "20260209", "--step", "7"]
                + ["--out-dir", cls.chimera_dir],
            ),
            "chimera": (
                run_chimera_batches_cmd,
                ["--chimera-dir", cls.chimera_dir, "--out-dir", cls.results_dir] + kwarg,
            ),
            "snapshot": (
                run_progress_batches_cmd,
                ["--visits-file", completed_file, "--start-dayobs", "20251220"]
                + ["--end-dayobs", "20260120", "--step", "14", "--out-dir", cls.results_dir]
                + kwarg,
            ),
            "summary": (
                make_chimera_summary_table_cmd,
                ["--results-db", cls.results_db, "--out-file", cls.summary_file],
            ),
        }
        cls.results = {}
        for name, (command, args) in steps.items():
            result = runner.invoke(command, args)
            if result.exit_code != 0:
                raise AssertionError(
                    f"Workflow step {name!r} failed with exit code {result.exit_code}:\n"
                    f"{result.output}\n{result.exception!r}"
                ) from result.exception
            cls.results[name] = result
        cls.transitions = [20260101, 20260108, 20260115, 20260122, 20260124]
        cls.table = pd.read_hdf(cls.summary_file, key="summary")

    def _column(self, *column):
        return self.table[column]

    def test_commands_succeed(self):
        for name, result in self.results.items():
            with self.subTest(command=name):
                self.assertEqual(result.exit_code, 0, f"{result.output}\n{result.exception!r}")
        # Four batches: the cadence dates plus the appended end date.
        self.assertIn("Ran 4 batch(es)", self.results["snapshot"].output)

    def test_run_names(self):
        results_db = ResultsDb(database=self.results_db)
        run_names = results_db.get_run_name()
        results_db.close()
        # The last completed date and the end date are off cadence. The
        # snapshot dates before the first completed visit have no rows.
        expected = [f"chimera_{t}" for t in self.transitions] + ["consdb_20260117", "consdb_20260120"]
        self.assertEqual(sorted(run_names), sorted(expected))

    def test_summary_table(self):
        # Only chimera runs, although snapshot runs share the database.
        self.assertEqual(list(self.table.index), self.transitions)
        self.assertTrue(np.issubdtype(self.table.index.dtype, np.integer))
        self.assertEqual(
            list(self.table.columns.names),
            ["metric_name", "slicer_name", "metric_info_label", "summary_metric"],
        )

    def test_chimera_metric_values(self):
        count = ("Numbers of exposures", "UniSlicer")
        teff = ("Sum t_eff", "UniSlicer")

        # First: baseline visits after the transition only. Last: all
        # completed visits with a depth, then baseline after 20260124.
        n_completed = int(self.completed["fiveSigmaDepth"].notna().sum())
        counts = self._column(*count, "chimera_all", "Identity")
        self.assertEqual(counts.iloc[0], 39 * 30)
        self.assertEqual(counts.iloc[-1], n_completed + 16 * 30)

        for transition in self.transitions:
            path = os.path.join(self.chimera_dir, f"chimera_{transition}.h5")
            visits = pd.read_hdf(path, key="observations")
            self.assertEqual(counts.loc[transition], len(visits))
            for band in BANDS:
                self.assertEqual(
                    self._column(*count, f"chimera_{band}", "Identity").loc[transition],
                    (visits["band"] == band).sum(),
                )

        band_teff = sum(self._column(*teff, f"chimera_{band}", "Identity") for band in BANDS)
        total_teff = self._column(*teff, "chimera_all", "Identity")
        self.assertTrue((total_teff > 0).all())
        np.testing.assert_allclose(band_teff.to_numpy(), total_teff.to_numpy(), rtol=1e-6)

        for metric in ("Number of exposure area stats", "Depth area stats"):
            for band in ("all",) + tuple(BANDS):
                for summary in ("Count", "Max", "Mean", "Median", "Min", "Rms", "top18k", "10th%ile"):
                    self.assertIn((metric, "HealpixSlicer", f"chimera_{band}", summary), self.table.columns)
            low = self._column(metric, "HealpixSlicer", "chimera_all", "Min")
            tenth = self._column(metric, "HealpixSlicer", "chimera_all", "10th%ile")
            high = self._column(metric, "HealpixSlicer", "chimera_all", "Max")
            self.assertTrue(((low <= tenth) & (tenth <= high)).all())
        depth = self._column("Depth area stats", "HealpixSlicer", "chimera_all", "Mean")
        self.assertTrue(((depth > 20) & (depth < 30)).all())

        # fO values are masked on sparse data; check only presence.
        fo_summaries = {col[3] for col in self.table.columns if col[0] == "fO"}
        self.assertTrue({"fOArea", "fOArea/benchmark", "fOArea_750"} <= fo_summaries)

    def test_snapshot_metric_values(self):
        results_db = ResultsDb(database=self.results_db)
        stats = pd.DataFrame(results_db.get_summary_stats(with_sim_name=True))
        results_db.close()
        stats = stats[
            (stats["metric_name"] == "Numbers of exposures")
            & (stats["metric_info_label"] == "snapshot_all")
            & (stats["summary_metric"] == "Identity")
        ].set_index("run_name")["summary_value"]

        # Completed visits with a depth, through the end of observing day
        # 20260117 (12:00 UTC on 2026-01-18).
        for run_name, end in (("consdb_20260117", "2026-01-18"), ("consdb_20260120", "2026-01-21")):
            visits = self.completed
            selected = visits["fiveSigmaDepth"].notna() & (
                visits["observationStartMJD"] < Time(f"{end}T12:00:00").mjd
            )
            self.assertEqual(stats[run_name], selected.sum())

    def test_legacy_runname_batch(self):
        """A batch that takes only ``runName`` is still supported."""
        from rubin_sim.maf import MetricBundle
        from rubin_sim.maf.metrics import CountMetric
        from rubin_sim.maf.slicers import UniSlicer

        def legacy(runName="x"):
            metric = CountMetric(col="observationStartMJD")
            return {"count": MetricBundle(metric, UniSlicer(), "", run_name=runName)}

        path = os.path.join(self.chimera_dir, "chimera_20260108.h5")
        with tempfile.TemporaryDirectory() as out_dir:
            db_path = run_chimera_batches([(20260108, path)], batch_func=legacy, out_dir=out_dir)
            results_db = ResultsDb(database=db_path)
            run_names = results_db.get_run_name()
            results_db.close()
        self.assertEqual(run_names, ["chimera_20260108"])

    def test_default_batch(self):
        """The Python API defaults to ``chimera_batch``."""
        with tempfile.TemporaryDirectory() as out_dir:
            with patch("rubin_sim.maf.progress.batches.chimera_batch", return_value={}) as default_batch:
                with patch("rubin_sim.maf.progress.mb.MetricBundleGroup"):
                    run_chimera_batches([(20260108, "unused.h5")], out_dir=out_dir)
        default_batch.assert_called_once_with(run_name="chimera_20260108")


class TestProgressCommands(unittest.TestCase):
    """Error-path tests for progress commands."""

    def test_sparse_snapshot_warns_and_continues(self):
        sparse_visits = _make_visits(20260101, 1, 1, seed=1)
        # This pointing misses all camera-footprint pixels at NSIDE=16.
        sparse_visits.loc[:, ["fieldRA", "fieldDec"]] = 0.0
        later_visits = _make_visits(20260102, 1, 600, seed=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "visits.h5")
            pd.concat([sparse_visits, later_visits]).to_hdf(path, key="observations")
            with self.assertWarnsRegex(UserWarning, "Snapshot consdb_20260101 has incomplete results"):
                db_path = run_progress_batches(
                    path,
                    20260101,
                    20260102,
                    step=1,
                    out_dir=tmp,
                    batch_kwargs={"nside": NSIDE, "bands": ()},
                )
            results_db = ResultsDb(database=db_path)
            stats = pd.DataFrame(results_db.get_summary_stats(with_sim_name=True))
            results_db.close()
            later_stats = stats[stats["run_name"] == "consdb_20260102"]
            later_counts = later_stats[
                (later_stats["metric_name"] == "Numbers of exposures")
                & (later_stats["summary_metric"] == "Identity")
            ]
            # The later snapshot includes the sparse visit plus its own 600.
            self.assertEqual(later_counts["summary_value"].tolist(), [601])
            self.assertIn("top18k", later_stats["summary_metric"].tolist())

    def test_unrelated_snapshot_valueerror_propagates(self):
        def invalid_batch(run_name, end_dayobs):
            from rubin_sim.maf import MetricBundle
            from rubin_sim.maf.metrics import CountMetric
            from rubin_sim.maf.slicers import UniSlicer

            class InvalidMetric(CountMetric):
                def run(self, data_slice, slice_point=None):
                    raise ValueError("unrelated metric failure")

            return {"invalid": MetricBundle(InvalidMetric(col="band"), UniSlicer(), run_name=run_name)}

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "visits.h5")
            _make_visits(20260101, 1, 1, seed=1).to_hdf(path, key="observations")
            with self.assertRaisesRegex(ValueError, "unrelated metric failure"):
                run_progress_batches(path, 20260101, 20260102, out_dir=tmp, batch_func=invalid_batch)

    def test_cli_rejects_bad_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            visits_file = os.path.join(tmp, "visits.h5")
            # Option validation runs before any visit data are loaded.
            open(visits_file, "wb").close()
            out_dir = os.path.join(tmp, "out")
            commands = {
                "progress": (
                    run_progress_batches_cmd,
                    ["--visits-file", visits_file, "--start-dayobs", "20260101"]
                    + ["--end-dayobs", "20260101", "--out-dir", out_dir],
                ),
                "chimera": (run_chimera_batches_cmd, ["--chimera-dir", tmp, "--out-dir", out_dir]),
            }
            bad_options = (
                ["--batch", "not_a_batch"],
                ["--batch-kwarg", "nside"],
                ["--batch-kwarg", "=8"],
            )
            for name, (command, args) in commands.items():
                for bad in bad_options:
                    with self.subTest(command=name, option=bad):
                        result = CliRunner().invoke(command, args + bad)
                        self.assertNotEqual(result.exit_code, 0)
                        self.assertFalse(os.path.exists(os.path.join(out_dir, "resultsDb_sqlite.db")))


class TestConsoleScripts(unittest.TestCase):
    """Console-script registration tests for progress commands."""

    def test_console_scripts_registered(self):
        scripts = {
            ep.name: ep.value
            for ep in distribution("rubin-sim").entry_points
            if ep.group == "console_scripts"
        }
        for name in (
            "build_chimeras",
            "run_chimera_batches",
            "run_progress_batches",
            "make_chimera_summary_table",
        ):
            with self.subTest(script=name):
                self.assertEqual(scripts.get(name), f"rubin_sim.maf.progress:{name}_cmd")


if __name__ == "__main__":
    unittest.main()
