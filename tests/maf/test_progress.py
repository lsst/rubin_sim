"""Regression tests for progress tracking (SP-3142): chimeras, snapshots,
and the commands that run progress batches.

All visits are small synthetic data built here; see
docs/issues/SP-3142-regression-tests.md for the design.
"""

import os
import shutil
import tempfile
import unittest
from importlib.metadata import distribution
from unittest.mock import patch

import numpy as np
import pandas as pd
from astropy.time import Time
from click.testing import CliRunner

import rubin_sim.maf.batches as batches
from rubin_sim.maf.db import ResultsDb
from rubin_sim.maf.progress import (
    build_chimera,
    build_chimeras,
    build_chimeras_cmd,
    dayobs_range,
    make_chimera_summary_table_cmd,
    run_chimera_batches,
    run_chimera_batches_cmd,
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
    """R-1 and R-2."""

    def setUp(self):
        # S = 20260102, T = 20260104, E = 20260107 in test_build_chimera.
        self.completed = pd.DataFrame(
            {
                "observationId": [1, 2, 3, 4, 5, 6, 7],
                "dayObs": [20260101, 20260102, 20260103, 20260104, 20260104, 20260104, 20260105],
                "fiveSigmaDepth": [24.0, 24.0, np.nan, 24.0, 0.0, -1.0, 24.0],
                "band": ["g"] * 7,
                "completed_only": range(7),
            }
        )
        self.baseline = pd.DataFrame(
            {
                "observationId": [101, 102, 103, 104, 105, 106, 107],
                "dayObs": [20260104, 20260105, 20260105, 20260106, 20260107, 20260108, 20260105],
                "fiveSigmaDepth": [23.0, 23.0, np.nan, 0.0, 23.0, 23.0, -1.0],
                "band": ["r"] * 7,
                "exposures": [2] * 7,
                "baseline_only": range(7),
            }
        )

    def test_build_chimera(self):
        args = (20260102, 20260104, 20260107)
        result = build_chimera(self.completed, self.baseline, *args)

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
        completed = self.completed.assign(exposures=[5.0, 3.0, 5.0, np.nan, 5.0, 5.0, 5.0])
        result = build_chimera(completed, self.baseline, *args)
        self.assertEqual(result["exposures"].tolist(), [3, 1, 2, 2])

        # The depth cut uses the module-level limit, for both sources.
        with patch("rubin_sim.maf.progress.FIVE_SIGMA_DEPTH_LIMIT", 23.0):
            result = build_chimera(self.completed, self.baseline, *args)
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
                "dayObs": [20260129 + i // 2 if i < 6 else 20260201 + (i - 6) // 2 for i in range(18)],
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
    """R-3 and R-4: the batch definitions."""

    def test_snapshot_batch_selection(self):
        end_mjd = Time("2026-01-03T12:00:00").mjd  # End of observing day 20260102.
        visits = pd.DataFrame(
            {
                "observationStartMJD": [
                    end_mjd - 1.0,
                    end_mjd - 0.001,
                    end_mjd,
                    end_mjd - 0.5,
                    end_mjd - 0.5,
                ],
                "fiveSigmaDepth": [24.0, 24.0, 24.0, 0.0, 24.0],
                "band": ["g", "g", "g", "g", "r"],
            }
        )
        bundles = batches.snapshot_batch(
            run_name="x_20260102", bands=("g",), nside=NSIDE, end_dayobs=20260102
        )
        constraints = {b.info_label: b.pdconstraint for b in bundles.values()}
        self.assertEqual(visits.query(constraints["snapshot_all"]).index.tolist(), [0, 1, 4])
        self.assertEqual(visits.query(constraints["snapshot_g"]).index.tolist(), [0, 1])
        self.assertEqual({b.run_name for b in bundles.values()}, {"x_20260102"})

        # No end date: no date cut, and the shared depth limit still applies.
        late = visits.assign(observationStartMJD=1e6, fiveSigmaDepth=[23.9, 24.0, 24.1, 24.2, 24.3])
        with patch("rubin_sim.maf.progress.FIVE_SIGMA_DEPTH_LIMIT", 24.0):
            bundles = batches.snapshot_batch(bands=(), nside=NSIDE)
        (pdconstraint,) = {b.pdconstraint for b in bundles.values()}
        self.assertEqual(late.query(pdconstraint).index.tolist(), [2, 3, 4])

    def test_batch_contents(self):
        std_summaries = {
            "Mean None",
            "Rms None",
            "Median None",
            "Max None",
            "Min None",
            "N(+3Sigma)",
            "N(-3Sigma)",
            "Count None",
        }
        fo_summaries = {"fOArea", "fOArea/benchmark", "fONv", "fONv/benchmark", "fOArea_750"}

        for batch, prefix in ((batches.snapshot_batch, "snapshot"), (batches.chimera_batch, "chimera")):
            with self.subTest(batch=batch.__name__):
                bundles = list(batch(nside=NSIDE).values())
                other = [b for b in bundles if b.metric.name != "fO"]
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
                self.assertEqual(len(other), len(expected))
                self.assertEqual(
                    {(b.metric.name, b.info_label, type(b.slicer).__name__) for b in other}, expected
                )

                for bundle in other:
                    if type(bundle.slicer).__name__ == "HealpixSlicer":
                        names = {m.name for m in bundle.summary_metrics}
                        self.assertTrue(std_summaries <= names, names)
                        self.assertIn("top18k", names)
                        self.assertIn("10th%ile metricdata", names)

                fo = [b for b in bundles if b.metric.name == "fO"]
                if prefix == "snapshot":
                    self.assertEqual(fo, [])
                else:
                    self.assertEqual(len(fo), 1)
                    self.assertEqual({m.name for m in fo[0].summary_metrics}, fo_summaries)
                    self.assertEqual(fo[0].slicer.nside, NSIDE)


class TestProgressWorkflow(unittest.TestCase):
    """R-2 to R-6: the commands of the SP-3142 example workflow, run on
    synthetic visits sharing one results directory.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="progress_test_")
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
                ["--chimera-dir", cls.chimera_dir, "--batch", "chimera_batch", "--out-dir", cls.results_dir]
                + kwarg,
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
        cls.results = {name: runner.invoke(cmd, args) for name, (cmd, args) in steps.items()}
        cls.transitions = [20260101, 20260108, 20260115, 20260122, 20260124]
        if all(r.exit_code == 0 for r in cls.results.values()):
            cls.table = pd.read_hdf(cls.summary_file, key="summary")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

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

    def test_cli_rejects_bad_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            visits_file = os.path.join(tmp, "visits.h5")
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
    """R-6."""

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
