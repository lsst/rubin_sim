"""Test suite for the chimera progress capability.

Tests the chimera and snapshot APIs in rubin_sim.maf.progress using
synthetic visits sampled from the baseline opsim database.
"""

import datetime
import os
import tempfile
import unittest
from unittest.mock import call, patch

import numpy as np
import pandas as pd
from astropy.time import Time
from click.testing import CliRunner

import rubin_sim.maf.batches as batches
from rubin_sim.maf.db import ResultsDb
from rubin_sim.maf.progress import (
    _dayobs_from_filename,
    _dayobs_from_run_name,
    _run_name_from_dayobs,
    build_chimera,
    build_chimeras,
    dayobs_range,
    make_chimera_summary_table,
    run_chimera_batches,
    run_chimera_batches_cmd,
    run_progress_batches,
    run_progress_batches_cmd,
)

# Use relative imports for test data (works with pytest from rubin_sim)
from .test_chimera_data import make_sample_consdb_visits, make_sample_opsim_visits


class TestDayObsHelpers(unittest.TestCase):
    """Test helper functions for dayObs manipulation."""

    def test_dayobs_range_basic(self):
        """Test dayobs_range with basic inputs."""
        result = dayobs_range(20260101, 20260105, 1)
        expected = [20260101, 20260102, 20260103, 20260104, 20260105]
        self.assertEqual(result, expected)

    def test_dayobs_range_with_step(self):
        """Test dayobs_range with step parameter."""
        result = dayobs_range(20260101, 20260107, 2)
        expected = [20260101, 20260103, 20260105, 20260107]
        self.assertEqual(result, expected)

    def test_dayobs_range_same_start_end(self):
        """Test dayobs_range when start equals end."""
        result = dayobs_range(20260105, 20260105, 1)
        expected = [20260105]
        self.assertEqual(result, expected)

    def test_dayobs_range_crosses_month_boundary(self):
        self.assertEqual(dayobs_range(20260130, 20260202), [20260130, 20260131, 20260201, 20260202])

    def test_dayobs_range_rejects_nonpositive_step(self):
        with self.assertRaises(ValueError):
            dayobs_range(20260101, 20260105, 0)

    def test_run_name_from_dayobs(self):
        """Test _run_name_from_dayobs conversion."""
        self.assertEqual(_run_name_from_dayobs(20260101), "chimera_20260101")
        self.assertEqual(_run_name_from_dayobs(20261231), "chimera_20261231")

    def test_dayobs_from_run_name(self):
        """Test _dayobs_from_run_name extraction."""
        self.assertEqual(_dayobs_from_run_name("chimera_20260101"), 20260101)
        self.assertEqual(_dayobs_from_run_name("chimera_20261231"), 20261231)
        self.assertIsNone(_dayobs_from_run_name("invalid"))
        self.assertIsNone(_dayobs_from_run_name("sim_20260101"))

    def test_dayobs_from_filename(self):
        """Test _dayobs_from_filename extraction."""
        self.assertEqual(_dayobs_from_filename("/path/to/chimera_20260101.h5"), 20260101)
        self.assertEqual(_dayobs_from_filename("chimera_20261231.h5"), 20261231)
        self.assertIsNone(_dayobs_from_filename("other_20260101.h5"))
        self.assertIsNone(_dayobs_from_filename("/path/to/file.txt"))


class TestBuildChimera(unittest.TestCase):
    """Test build_chimera function."""

    def setUp(self):
        """Create test DataFrames with dayObs column."""
        # Use observationId to match baseline data schema
        self.consdb_visits = pd.DataFrame(
            {
                "observationId": [1, 2, 3],
                "dayObs": [20260101, 20260102, 20260103],
                "filter": ["g", "r", "i"],
                "fiveSigmaDepth": [24.1, 24.2, 24.3],
                "observationStartMJD": [61000, 61001, 61002],
            }
        )

        self.opsim_visits = pd.DataFrame(
            {
                "observationId": [101, 102, 103],
                "dayObs": [20260104, 20260105, 20260106],
                "filter": ["g", "r", "i"],
                "fiveSigmaDepth": [23.1, 23.2, 23.3],
                "exposures": [2, 2, 2],
                "observationStartMJD": [61003, 61004, 61005],
            }
        )

    def test_basic_concatenation(self):
        """Test basic chimera concatenation."""
        result = build_chimera(
            self.consdb_visits,
            self.opsim_visits,
            start_dayobs=20260101,
            transition_dayobs=20260103,
            end_dayobs=20260106,
        )
        self.assertEqual(len(result), 6)
        self.assertEqual(result["observationId"].tolist(), [1, 2, 3, 101, 102, 103])

    def test_simulated_flag_and_missing_exposures(self):
        result = build_chimera(
            self.consdb_visits,
            self.opsim_visits,
            start_dayobs=20260101,
            transition_dayobs=20260103,
            end_dayobs=20260106,
        )
        self.assertEqual(result["simulated"].tolist(), [False, False, False, True, True, True])
        self.assertEqual(result["exposures"].tolist()[:3], [1, 1, 1])

    def test_drops_consdb_visits_without_depth(self):
        consdb = self.consdb_visits.copy()
        consdb.loc[1, "fiveSigmaDepth"] = np.nan
        result = build_chimera(
            consdb,
            self.opsim_visits,
            start_dayobs=20260101,
            transition_dayobs=20260103,
            end_dayobs=20260106,
        )
        self.assertEqual(result["observationId"].tolist(), [1, 3, 101, 102, 103])

    def test_shared_depth_limit_filters_both_sources(self):
        consdb = self.consdb_visits.copy()
        opsim = self.opsim_visits.copy()
        consdb["fiveSigmaDepth"] = [24.1, 0.0, -1.0]
        opsim["fiveSigmaDepth"] = [23.1, 0.0, np.nan]

        result = build_chimera(consdb, opsim, 20260101, 20260103, 20260106)
        self.assertEqual(result["observationId"].tolist(), [1, 101])

        with patch("rubin_sim.maf.progress.FIVE_SIGMA_DEPTH_LIMIT", 24.1):
            consdb.loc[0, "fiveSigmaDepth"] = 24.2
            opsim.loc[0, "fiveSigmaDepth"] = 24.1
            result = build_chimera(consdb, opsim, 20260101, 20260103, 20260106)
        self.assertEqual(result["observationId"].tolist(), [1])

    def test_dayobs_boundary(self):
        """Test that dayObs boundary is correct."""
        result = build_chimera(
            self.consdb_visits,
            self.opsim_visits,
            start_dayobs=20260101,
            transition_dayobs=20260103,
            end_dayobs=20260106,
        )

        # dayObs <= transition should be from consdb
        consdb_ids = result[result["dayObs"] <= 20260103]["observationId"].tolist()
        self.assertEqual(consdb_ids, [1, 2, 3])

        # dayObs > transition should be from opsim
        opsim_ids = result[result["dayObs"] > 20260103]["observationId"].tolist()
        self.assertEqual(opsim_ids, [101, 102, 103])

    def test_empty_consdb_returns_empty(self):
        """Test with empty consdb visits where no opsim in range."""
        consdb = self.consdb_visits.iloc[:0].copy()

        # When transition_dayobs is before all opsim visits and end_dayobs is
        # also before, result should be empty
        result = build_chimera(
            consdb,
            self.opsim_visits,
            start_dayobs=20260101,
            transition_dayobs=20251231,  # Before all opsim visits
            end_dayobs=20260101,
        )
        # No consdb visits (empty) and no opsim in (transition, end_dayobs]
        self.assertEqual(len(result), 0)

    def test_empty_consdb_returns_opsim(self):
        """Test with empty consdb visits."""
        consdb = self.consdb_visits.iloc[:0].copy()

        result = build_chimera(
            consdb,
            self.opsim_visits,
            start_dayobs=20260101,
            transition_dayobs=20260106,
            end_dayobs=20260106,
        )
        self.assertEqual(len(result), 0)

    def test_empty_opsim_returns_consdb(self):
        """Test with empty opsim visits."""
        opsim = self.opsim_visits.iloc[:0].copy()

        result = build_chimera(
            self.consdb_visits,
            opsim,
            start_dayobs=20260101,
            transition_dayobs=20260106,
            end_dayobs=20260106,
        )
        self.assertEqual(len(result), 3)

    def test_filter_by_dayobs(self):
        """Test that visits are properly filtered by dayObs."""
        # Test with wider date range - only subset should be included
        result = build_chimera(
            self.consdb_visits,
            self.opsim_visits,
            start_dayobs=20260102,
            transition_dayobs=20260103,
            end_dayobs=20260105,
        )
        # Only visits with dayObs >= 20260102 and <= 20260105
        # consdb: 2, 3 (dayObs 2, 3) - 1 is excluded (< start)
        # opsim: 101, 102 (dayObs 4, 5) - 103 is excluded (> end)
        self.assertEqual(len(result), 4)
        self.assertEqual(result["observationId"].tolist(), [2, 3, 101, 102])


class TestBuildChimeras(unittest.TestCase):
    """Test build_chimeras function."""

    @classmethod
    def setUpClass(cls):
        """Generate sample test data once for all tests."""
        cls.opsim_visits = make_sample_opsim_visits(n_visits=500, random_state=42)
        cls.consdb_visits = make_sample_consdb_visits(n_visits=100, random_state=42)
        cls.out_dir = tempfile.mkdtemp(prefix="chimera_test_")

    @classmethod
    def tearDownClass(cls):
        """Clean up test output directory."""
        import shutil

        shutil.rmtree(cls.out_dir, ignore_errors=True)

    def test_generates_multiple_chimera_files(self):
        """Test that multiple HDF5 files are created."""
        specs = build_chimeras(
            self.consdb_visits,
            self.opsim_visits,
            start_dayobs=20260101,
            end_dayobs=20260228,
            step=7,
            out_dir=self.out_dir,
        )

        # Should have specs for each transition date
        self.assertGreater(len(specs), 0)

        # Each spec should be (transition_dayobs, path) tuple
        for transition_dayobs, path in specs:
            self.assertIsInstance(transition_dayobs, int)
            self.assertIsInstance(path, str)
            self.assertTrue(os.path.exists(path))
            self.assertTrue(path.endswith(".h5"))

    def test_uses_step_parameter(self):
        """Test that step parameter controls date spacing."""
        specs = build_chimeras(
            self.consdb_visits,
            self.opsim_visits,
            start_dayobs=20260101,
            end_dayobs=20260228,
            step=7,
            out_dir=self.out_dir,
        )

        transition_dates = [t for t, _ in specs]

        def _to_date(dayobs):
            s = f"{dayobs:08d}"
            return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:]))

        # Check that dates are spaced by step (except possibly last)
        for i in range(len(transition_dates) - 1):
            diff = (_to_date(transition_dates[i + 1]) - _to_date(transition_dates[i])).days
            self.assertEqual(diff, 7)  # Step should be exactly 7
        # Last date should be max consdb dayObs
        max_consdb = int(self.consdb_visits["dayObs"].max())
        self.assertEqual(transition_dates[-1], max_consdb)

    def test_last_date_is_max_consdb(self):
        """Test that last transition date is always max consdb dayObs."""
        max_consdb = int(self.consdb_visits["dayObs"].max())

        specs = build_chimeras(
            self.consdb_visits,
            self.opsim_visits,
            start_dayobs=20260101,
            end_dayobs=20260228,
            step=30,
            out_dir=self.out_dir,
        )

        last_transition = specs[-1][0]
        self.assertEqual(last_transition, max_consdb)

    def test_hdf5_file_structure(self):
        """Test that HDF5 files have correct structure."""
        specs = build_chimera(
            self.consdb_visits,
            self.opsim_visits,
            start_dayobs=20260101,
            transition_dayobs=20260115,
            end_dayobs=20260228,
        )
        path = os.path.join(self.out_dir, "test_chimera.h5")
        specs.to_hdf(path, key="observations", complevel=5)

        result = pd.read_hdf(path, key="observations")
        self.assertEqual(len(result), len(specs))
        self.assertIn("dayObs", result.columns)

    def test_creates_output_directory(self):
        """Test that output directory is created if it doesn't exist."""
        new_dir = os.path.join(self.out_dir, "new_subdir")
        specs = build_chimeras(
            self.consdb_visits,
            self.opsim_visits,
            start_dayobs=20260101,
            end_dayobs=20260131,
            step=10,
            out_dir=new_dir,
        )

        self.assertTrue(os.path.exists(new_dir))
        self.assertGreater(len(specs), 0)

    def test_consdb_only_chimera(self):
        """Test chimera with only consdb visits in range."""
        # Use a transition date after all consdb visits
        specs = build_chimeras(
            self.consdb_visits,
            self.opsim_visits,
            start_dayobs=20260101,
            end_dayobs=20260228,
            step=1,
            out_dir=self.out_dir,
        )

        for transition_dayobs, path in specs:
            chimera = pd.read_hdf(path, key="observations")
            # All consdb visits should be before transition
            self.assertTrue(
                (
                    chimera[chimera["dayObs"] <= transition_dayobs]["observationId"].isin(
                        self.consdb_visits["observationId"]
                    )
                ).all()
            )


class TestRunChimeraBatches(unittest.TestCase):
    """Test run_chimera_batches function."""

    @classmethod
    def setUpClass(cls):
        """Generate sample test data and build chimera files."""
        cls.opsim_visits = make_sample_opsim_visits(n_visits=500, random_state=42)
        cls.consdb_visits = make_sample_consdb_visits(n_visits=100, random_state=42)
        cls.out_dir = tempfile.mkdtemp(prefix="chimera_batch_test_")

        # Build some chimera files
        cls.chimera_specs = build_chimeras(
            cls.consdb_visits,
            cls.opsim_visits,
            start_dayobs=20260101,
            end_dayobs=20260228,
            step=7,
            out_dir=cls.out_dir,
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up test output directory."""
        import shutil

        shutil.rmtree(cls.out_dir, ignore_errors=True)

    def test_creates_results_db(self):
        """Test that ResultsDb is created."""
        results_db_path = run_chimera_batches(
            self.chimera_specs,
            batch_func=batches.glanceBatch,
            out_dir=self.out_dir,
        )

        self.assertTrue(os.path.exists(results_db_path))
        self.assertIn("resultsDb_sqlite.db", results_db_path)

    def test_uses_batch_kwargs(self):
        """Test that batch_kwargs are passed to batch function."""
        results_db_path = run_chimera_batches(
            self.chimera_specs[:2],  # Only first 2 for speed
            batch_func=batches.glanceBatch,
            out_dir=self.out_dir,
            batch_kwargs={"nyears": 5},  # Use valid batch kwarg
        )

        self.assertTrue(os.path.exists(results_db_path))

    def test_handles_run_name_and_runname(self):
        """Test that run_chimera_batches handles run_name and runName."""

        def batch_func_with_fallback(**kwargs) -> dict:
            """Batch function that handles both run_name and runName."""
            import rubin_sim.maf as maf

            metric = maf.metrics.CountMetric(col="observationStartMJD")
            bundle = maf.MetricBundle(
                metric,
                maf.slicers.UniSlicer(),
                "",
                run_name=kwargs.get("run_name") or kwargs.get("runName"),
            )
            return {"test": bundle}

        # Test with the fallback function (accepts **kwargs)
        results_db_path = run_chimera_batches(
            self.chimera_specs[:1],
            batch_func=batch_func_with_fallback,
            out_dir=self.out_dir,
        )
        self.assertTrue(os.path.exists(results_db_path))

    def test_stores_metrics_in_results_db(self):
        """Test that metrics are stored in ResultsDb."""
        results_db_path = run_chimera_batches(
            self.chimera_specs[:2],  # Only 2 for speed
            batch_func=batches.glanceBatch,
            out_dir=self.out_dir,
        )

        results_db = ResultsDb(database=results_db_path)
        run_names = results_db.get_run_name()
        results_db.close()

        self.assertGreater(len(run_names), 0)
        for run_name in run_names:
            self.assertTrue(run_name.startswith("chimera_"))

    def test_multiple_runs_share_results_db(self):
        """Test that multiple chimera runs share the same ResultsDb."""
        results_db_path = run_chimera_batches(
            self.chimera_specs,
            batch_func=batches.glanceBatch,
            out_dir=self.out_dir,
        )

        results_db = ResultsDb(database=results_db_path)
        # Get all run names from the results DB
        run_names = results_db.get_run_name()
        results_db.close()

        self.assertEqual(len(run_names), len(self.chimera_specs))


class TestRunProgressBatches(unittest.TestCase):
    def test_runs_snapshots_through_end_dayobs(self):
        with tempfile.TemporaryDirectory() as out_dir:
            with (
                patch("rubin_sim.maf.progress.batches.snapshot_batch", return_value={"metric": 1}) as batch,
                patch("rubin_sim.maf.progress.mb.MetricBundleGroup") as group,
                patch("rubin_sim.maf.progress.db.ResultsDb") as results_db,
            ):
                path = run_progress_batches(
                    "visits.h5",
                    start_dayobs=20260130,
                    end_dayobs=20260203,
                    step=2,
                    out_dir=out_dir,
                    run_prefix="baseline",
                    batch_kwargs={"nside": 8},
                )

            self.assertEqual(path, os.path.join(out_dir, "resultsDb_sqlite.db"))
            self.assertEqual(
                batch.call_args_list,
                [
                    call(run_name="baseline_20260130", end_dayobs=20260130, nside=8),
                    call(run_name="baseline_20260201", end_dayobs=20260201, nside=8),
                    call(run_name="baseline_20260203", end_dayobs=20260203, nside=8),
                ],
            )
            self.assertEqual(group.call_count, 3)
            for group_call in group.call_args_list:
                self.assertEqual(group_call.args, (batch.return_value, "visits.h5"))
                self.assertEqual(group_call.kwargs["out_dir"], out_dir)
                self.assertIs(group_call.kwargs["results_db"], results_db.return_value)
            self.assertEqual(group.return_value.run_all.call_args_list, [call(clear_memory=True)] * 3)
            results_db.return_value.close.assert_called_once_with()

    def test_appends_end_dayobs_off_cadence(self):
        with tempfile.TemporaryDirectory() as out_dir:
            with (
                patch("rubin_sim.maf.progress.batches.snapshot_batch", return_value={}) as batch,
                patch("rubin_sim.maf.progress.mb.MetricBundleGroup"),
                patch("rubin_sim.maf.progress.db.ResultsDb"),
            ):
                run_progress_batches("visits.h5", 20260101, 20260106, step=4, out_dir=out_dir)
            self.assertEqual(
                batch.call_args_list,
                [
                    call(run_name="consdb_20260101", end_dayobs=20260101),
                    call(run_name="consdb_20260105", end_dayobs=20260105),
                    call(run_name="consdb_20260106", end_dayobs=20260106),
                ],
            )

    def test_uses_custom_batch_function(self):
        with tempfile.TemporaryDirectory() as out_dir:
            with (
                patch("rubin_sim.maf.progress.mb.MetricBundleGroup") as group,
                patch("rubin_sim.maf.progress.db.ResultsDb"),
            ):
                batch_func = unittest.mock.Mock(return_value={"custom": 1})
                run_progress_batches(
                    "visits.h5",
                    20260101,
                    20260101,
                    out_dir=out_dir,
                    batch_func=batch_func,
                    batch_kwargs={"nside": 8},
                )
            batch_func.assert_called_once_with(run_name="consdb_20260101", end_dayobs=20260101, nside=8)
            self.assertIs(group.call_args.args[0], batch_func.return_value)

    def test_empty_early_snapshot_writes_no_rows(self):
        """Early dates before the first visit create no rows or errors.

        The 2026-09-29 manual check used 100 sampled visits (first dayObs
        20260102), start=20251201, end=20260105, and step=30. This produces
        dates [20251201, 20251231, 20260105]. The first two precede all visits,
        so only consdb_20260105 appears in the ResultsDb.

        Verifies R-3 (empty early snapshots) and R-5 (run names).
        """
        consdb_visits = make_sample_consdb_visits(n_visits=100, random_state=42)
        with tempfile.TemporaryDirectory() as out_dir:
            visits_path = os.path.join(out_dir, "consdb.h5")
            consdb_visits.to_hdf(visits_path, key="observations", complevel=5)

            results_db_path = run_progress_batches(
                visits_path,
                start_dayobs=20251201,
                end_dayobs=20260105,
                step=30,
                out_dir=out_dir,
                batch_kwargs={"nside": 8, "bands": ()},
            )

            results_db = ResultsDb(database=results_db_path)
            run_names = results_db.get_run_name()
            results_db.close()

        # The two dates before the first visit must not appear.
        self.assertNotIn("consdb_20251201", run_names)
        self.assertNotIn("consdb_20251231", run_names)
        # The end date lands on or after the first visit and must appear.
        self.assertIn("consdb_20260105", run_names)


class TestRunProgressBatchesCommand(unittest.TestCase):
    def test_selects_batch_by_name(self):
        with tempfile.TemporaryDirectory() as out_dir:
            visits_file = os.path.join(out_dir, "visits.h5")
            with open(visits_file, "wb"):
                pass
            with (
                patch.object(batches, "custom_progress_batch", create=True) as batch_func,
                patch(
                    "rubin_sim.maf.progress.run_progress_batches", return_value="results.db"
                ) as run_batches,
            ):
                result = CliRunner().invoke(
                    run_progress_batches_cmd,
                    [
                        "--visits-file",
                        visits_file,
                        "--start-dayobs",
                        "20260101",
                        "--end-dayobs",
                        "20260102",
                        "--step",
                        "1",
                        "--batch",
                        "custom_progress_batch",
                        "--batch-kwarg",
                        "nside=8",
                        "--out-dir",
                        out_dir,
                    ],
                )
            self.assertIsNone(result.exception)
            self.assertIn("Ran 2 batch(es)", result.output)
            self.assertIs(run_batches.call_args.kwargs["batch_func"], batch_func)
            self.assertEqual(run_batches.call_args.kwargs["batch_kwargs"], {"nside": 8})

    def test_defaults_to_snapshot_batch(self):
        with tempfile.TemporaryDirectory() as out_dir:
            visits_file = os.path.join(out_dir, "visits.h5")
            with open(visits_file, "wb"):
                pass
            with patch(
                "rubin_sim.maf.progress.run_progress_batches", return_value="results.db"
            ) as run_batches:
                result = CliRunner().invoke(
                    run_progress_batches_cmd,
                    ["--visits-file", visits_file, "--start-dayobs", "20260101", "--end-dayobs", "20260101"],
                )
            self.assertIsNone(result.exception)
            self.assertIs(run_batches.call_args.kwargs["batch_func"], batches.snapshot_batch)

    def test_reports_end_dayobs_when_off_cadence(self):
        with tempfile.TemporaryDirectory() as out_dir:
            visits_file = os.path.join(out_dir, "visits.h5")
            with open(visits_file, "wb"):
                pass
            with patch("rubin_sim.maf.progress.run_progress_batches", return_value="results.db"):
                for end_dayobs, count in (("20260103", 2), ("20260104", 3)):
                    with self.subTest(end_dayobs=end_dayobs):
                        result = CliRunner().invoke(
                            run_progress_batches_cmd,
                            [
                                "--visits-file",
                                visits_file,
                                "--start-dayobs",
                                "20260101",
                                "--end-dayobs",
                                end_dayobs,
                                "--step",
                                "2",
                            ],
                        )
                        self.assertIsNone(result.exception)
                        self.assertIn(f"Ran {count} batch(es)", result.output)

    def test_rejects_unknown_batch(self):
        with tempfile.TemporaryDirectory() as out_dir:
            visits_file = os.path.join(out_dir, "visits.h5")
            with open(visits_file, "wb"):
                pass
            with patch("rubin_sim.maf.progress.run_progress_batches") as run_batches:
                result = CliRunner().invoke(
                    run_progress_batches_cmd,
                    [
                        "--visits-file",
                        visits_file,
                        "--start-dayobs",
                        "20260101",
                        "--end-dayobs",
                        "20260101",
                        "--batch",
                        "not_a_batch",
                    ],
                )
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("not a known batch function", result.output)
            run_batches.assert_not_called()

    def test_rejects_malformed_batch_kwarg(self):
        with tempfile.TemporaryDirectory() as out_dir:
            visits_file = os.path.join(out_dir, "visits.h5")
            with open(visits_file, "wb"):
                pass

            with patch("rubin_sim.maf.progress.run_progress_batches") as run_batches:
                for batch_kwarg in ("nside", "=8"):
                    with self.subTest(batch_kwarg=batch_kwarg):
                        result = CliRunner().invoke(
                            run_progress_batches_cmd,
                            [
                                "--visits-file",
                                visits_file,
                                "--start-dayobs",
                                "20260101",
                                "--end-dayobs",
                                "20260101",
                                "--batch-kwarg",
                                batch_kwarg,
                            ],
                        )
                        self.assertNotEqual(result.exit_code, 0)
                        self.assertIn("Invalid --batch-kwarg", result.output)
                run_batches.assert_not_called()


class TestRunChimeraBatchesCommand(unittest.TestCase):
    def test_rejects_unknown_batch(self):
        with tempfile.TemporaryDirectory() as chimera_dir:
            with patch("rubin_sim.maf.progress.run_chimera_batches") as run_batches:
                result = CliRunner().invoke(
                    run_chimera_batches_cmd,
                    ["--chimera-dir", chimera_dir, "--batch", "not_a_batch"],
                )

            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("not a known batch function", result.output)
            run_batches.assert_not_called()

    def test_rejects_malformed_batch_kwarg(self):
        with tempfile.TemporaryDirectory() as chimera_dir:
            with patch("rubin_sim.maf.progress.run_chimera_batches") as run_batches:
                for batch_kwarg in ("nside", "=8"):
                    with self.subTest(batch_kwarg=batch_kwarg):
                        result = CliRunner().invoke(
                            run_chimera_batches_cmd,
                            [
                                "--chimera-dir",
                                chimera_dir,
                                "--batch-kwarg",
                                batch_kwarg,
                            ],
                        )
                        self.assertNotEqual(result.exit_code, 0)
                        self.assertIn("Invalid --batch-kwarg", result.output)
                run_batches.assert_not_called()


class TestProgressConsoleScripts(unittest.TestCase):
    def test_console_scripts_are_registered(self):
        from importlib.metadata import distribution

        scripts = {
            entry_point.name: entry_point.value
            for entry_point in distribution("rubin-sim").entry_points
            if entry_point.group == "console_scripts"
        }
        expected = {
            "build_chimeras": "rubin_sim.maf.progress:build_chimeras_cmd",
            "run_chimera_batches": "rubin_sim.maf.progress:run_chimera_batches_cmd",
            "run_progress_batches": "rubin_sim.maf.progress:run_progress_batches_cmd",
            "make_chimera_summary_table": "rubin_sim.maf.progress:make_chimera_summary_table_cmd",
        }

        for name, target in expected.items():
            with self.subTest(script=name):
                self.assertEqual(scripts.get(name), target)


def _assert_base_progress_bundles(test_case, bundles, labels):
    """Assert the four required R-4 bundle types for each label.

    Each label in ``labels`` must have exactly these four bundles:
    - ``"Sum t_eff"`` on a UniSlicer (MAF adds only IdentityMetric);
    - ``"Numbers of exposures"`` on a UniSlicer (same);
    - ``"Number of exposure area stats"`` on a HealpixSlicer with standard
      summaries, top-18k, and 10th-percentile summaries;
    - ``"Depth area stats"`` on a HealpixSlicer with the same summaries.

    Parameters
    ----------
    test_case : `unittest.TestCase`
    bundles : `dict`
        Return value of ``chimera_batch`` or ``snapshot_batch``.
    labels : `list` of `str`
        Expected ``info_label`` values, e.g. ``["chimera_g", "chimera_all"]``.
    """
    import rubin_sim.maf.slicers as slicers

    # m.name on standard_summary() instances carries a " None" suffix for
    # column-agnostic metrics.
    STANDARD_SUMMARY_NAMES = {
        "Mean None",
        "Rms None",
        "Median None",
        "Max None",
        "Min None",
        "N(+3Sigma)",
        "N(-3Sigma)",
        "Count None",
    }
    # AreaSummaryMetric(metric_name="top18k") -> m.name == "top18k"
    # PercentileMetric(..., percentile=10) -> m.name == "10th%ile metricdata"
    TOP18K_NAME = "top18k"
    PERCENTILE_NAME = "10th%ile metricdata"
    # MAF auto-adds IdentityMetric to UniSlicer bundles without summaries.
    UNISLICER_AUTO_SUMMARY = {"Identity None"}

    bundle_list = list(bundles.values())
    non_fo = [b for b in bundle_list if b.metric.name != "fO"]

    test_case.assertEqual(
        len(non_fo),
        4 * len(labels),
        f"Expected 4 base bundle types × {len(labels)} labels = {4 * len(labels)} bundles; "
        f"got {len(non_fo)}",
    )

    by_metric_label = {(b.metric.name, b.info_label): b for b in non_fo}

    for label in labels:
        with test_case.subTest(label=label):
            # --- t_eff sum (UniSlicer) ---
            teff_key = ("Sum t_eff", label)
            test_case.assertIn(teff_key, by_metric_label, f"Missing t_eff bundle for {label}")
            teff = by_metric_label[teff_key]
            test_case.assertIsInstance(teff.slicer, slicers.UniSlicer)
            teff_summary_names = {m.name for m in teff.summary_metrics}
            test_case.assertTrue(
                teff_summary_names.issubset(UNISLICER_AUTO_SUMMARY),
                f"t_eff bundle for {label} has unexpected summary metrics: {teff_summary_names}",
            )

            # --- visit count (UniSlicer) ---
            count_uni_key = ("Numbers of exposures", label)
            test_case.assertIn(count_uni_key, by_metric_label, f"Missing visit-count bundle for {label}")
            count_uni = by_metric_label[count_uni_key]
            test_case.assertIsInstance(count_uni.slicer, slicers.UniSlicer)
            count_uni_summary_names = {m.name for m in count_uni.summary_metrics}
            test_case.assertTrue(
                count_uni_summary_names.issubset(UNISLICER_AUTO_SUMMARY),
                f"Visit-count bundle for {label} has unexpected summary metrics: "
                f"{count_uni_summary_names}",
            )

            # --- HEALPix visit-count area stats ---
            count_hp_key = ("Number of exposure area stats", label)
            test_case.assertIn(
                count_hp_key, by_metric_label, f"Missing HEALPix visit-count bundle for {label}"
            )
            count_hp = by_metric_label[count_hp_key]
            test_case.assertIsInstance(count_hp.slicer, slicers.HealpixSlicer)
            count_hp_summary_names = {m.name for m in count_hp.summary_metrics}
            test_case.assertTrue(
                STANDARD_SUMMARY_NAMES.issubset(count_hp_summary_names),
                f"HEALPix visit-count for {label} missing standard summaries; "
                f"got {count_hp_summary_names}",
            )
            test_case.assertIn(
                TOP18K_NAME, count_hp_summary_names, f"HEALPix visit-count for {label} missing top18k summary"
            )
            test_case.assertIn(
                PERCENTILE_NAME,
                count_hp_summary_names,
                f"HEALPix visit-count for {label} missing 10th-percentile summary",
            )

            # --- HEALPix coadded-depth area stats ---
            depth_key = ("Depth area stats", label)
            test_case.assertIn(depth_key, by_metric_label, f"Missing HEALPix depth bundle for {label}")
            depth_hp = by_metric_label[depth_key]
            test_case.assertIsInstance(depth_hp.slicer, slicers.HealpixSlicer)
            depth_hp_summary_names = {m.name for m in depth_hp.summary_metrics}
            test_case.assertTrue(
                STANDARD_SUMMARY_NAMES.issubset(depth_hp_summary_names),
                f"HEALPix depth for {label} missing standard summaries; " f"got {depth_hp_summary_names}",
            )
            test_case.assertIn(
                TOP18K_NAME, depth_hp_summary_names, f"HEALPix depth for {label} missing top18k summary"
            )
            test_case.assertIn(
                PERCENTILE_NAME,
                depth_hp_summary_names,
                f"HEALPix depth for {label} missing 10th-percentile summary",
            )


class TestChimeraBatch(unittest.TestCase):
    def test_fo_bundle_uses_requested_nside(self):
        from rubin_sim.maf.batches.progress_batch import chimera_batch

        bundles = chimera_batch(bands=(), nside=8)
        fo_bundle = next(bundle for bundle in bundles.values() if bundle.metric.name == "fO")
        self.assertEqual(fo_bundle.slicer.nside, 8)
        self.assertEqual({metric.nside for metric in fo_bundle.summary_metrics}, {8})

    def test_covers_required_metrics(self):
        """chimera_batch covers all metrics and summaries required by R-4.

        For each band u,g,r,i,z,y and for all bands combined:
        - a t_eff sum bundle (UniSlicer, no configured summary metrics);
        - a visit-count bundle (UniSlicer, no configured summary metrics);
        - a HEALPix visit-count bundle with standard summary stats,
          the top-18k-deg² minimum, and the 10th-percentile summary;
        - a HEALPix coadded-depth bundle with the same three summary groups.

        Plus exactly the five fO summary metrics (fOArea, fOArea/benchmark,
        fONv, fONv/benchmark, fOArea_750) on a single fO bundle.
        """
        from rubin_sim.maf.batches.progress_batch import chimera_batch

        BANDS = ("u", "g", "r", "i", "z", "y")
        LABELS = [f"chimera_{b}" for b in BANDS] + ["chimera_all"]

        bundles = chimera_batch(bands=BANDS, nside=8)
        bundle_list = list(bundles.values())

        # --- fO bundle ---
        fo_bundles = [b for b in bundle_list if b.metric.name == "fO"]
        self.assertEqual(len(fo_bundles), 1, "Expected exactly one fO bundle")
        fo_summary_names = {m.name for m in fo_bundles[0].summary_metrics}
        self.assertEqual(
            fo_summary_names,
            {
                "fOArea",
                "fOArea/benchmark",
                "fONv",
                "fONv/benchmark",
                "fOArea_750",
            },
        )

        # --- Base bundles (4 types × 7 labels) ---
        _assert_base_progress_bundles(self, bundles, LABELS)


class TestSnapshotBatch(unittest.TestCase):
    def test_uses_shared_depth_limit(self):
        from rubin_sim.maf.batches.progress_batch import snapshot_batch

        with patch("rubin_sim.maf.progress.FIVE_SIGMA_DEPTH_LIMIT", 24.0):
            bundles = snapshot_batch(bands=(), nside=8)
        self.assertEqual({bundle.pdconstraint for bundle in bundles.values()}, {"fiveSigmaDepth > 24.0"})

    def test_filters_real_visits_through_end_dayobs(self):
        from rubin_sim.maf.batches.progress_batch import snapshot_batch

        bundles = snapshot_batch(run_name="baseline_20260102", bands=("g",), nside=8, end_dayobs=20260102)
        constraints = {bundle.info_label: bundle.pdconstraint for bundle in bundles.values()}
        end_mjd = Time("2026-01-03T12:00:00").mjd
        self.assertEqual(
            constraints["snapshot_all"], f"fiveSigmaDepth > 0.0 and observationStartMJD < {end_mjd}"
        )
        self.assertEqual(
            constraints["snapshot_g"],
            f"fiveSigmaDepth > 0.0 and observationStartMJD < {end_mjd} and band == 'g'",
        )
        # Row 0: before end_mjd, depth > 0 -> included
        # Row 1: before end_mjd, depth > 0 -> included
        # Row 2: at end_mjd (not < end_mjd), depth > 0 -> excluded
        # Row 3: before end_mjd, depth <= 0 -> excluded
        visits = pd.DataFrame(
            {
                "observationStartMJD": [end_mjd - 1, end_mjd - 0.5, end_mjd, end_mjd - 0.5],
                "fiveSigmaDepth": [25.0, 24.5, 25.0, -1.0],
                "band": ["g", "g", "g", "g"],
            }
        )
        self.assertEqual(visits.query(constraints["snapshot_all"]).index.tolist(), [0, 1])
        self.assertEqual(visits.query(constraints["snapshot_g"]).index.tolist(), [0, 1])
        self.assertEqual({bundle.run_name for bundle in bundles.values()}, {"baseline_20260102"})

    def test_uses_colmap_mjd_and_supports_no_end_dayobs(self):
        from rubin_sim.maf.batches.col_map_dict import col_map_dict
        from rubin_sim.maf.batches.progress_batch import snapshot_batch

        colmap = col_map_dict()
        colmap["mjd"] = "visitMjd"
        bundles = snapshot_batch(colmap=colmap, bands=(), nside=8, end_dayobs=20260102)
        self.assertEqual(
            {bundle.pdconstraint for bundle in bundles.values()},
            {f"fiveSigmaDepth > 0.0 and visitMjd < {Time('2026-01-03T12:00:00').mjd}"},
        )
        bundles = snapshot_batch(colmap=colmap, bands=(), nside=8)
        self.assertEqual({bundle.pdconstraint for bundle in bundles.values()}, {"fiveSigmaDepth > 0.0"})

    def test_covers_required_metrics(self):
        """snapshot_batch covers all metrics and summaries required by R-4.

        For each band u,g,r,i,z,y and for all bands combined:
        - a t_eff sum bundle (UniSlicer, no configured summary metrics);
        - a visit-count bundle (UniSlicer, no configured summary metrics);
        - a HEALPix visit-count bundle with standard summary stats,
          the top-18k-deg² minimum, and the 10th-percentile summary;
        - a HEALPix coadded-depth bundle with the same three summary groups.

        snapshot_batch does not produce fO bundles (R-4).
        """
        from rubin_sim.maf.batches.progress_batch import snapshot_batch

        BANDS = ("u", "g", "r", "i", "z", "y")
        LABELS = [f"snapshot_{b}" for b in BANDS] + ["snapshot_all"]

        bundles = snapshot_batch(bands=BANDS, nside=8)

        # snapshot_batch produces no fO bundle (R-4).
        fo_bundles = [b for b in bundles.values() if b.metric.name == "fO"]
        self.assertEqual(len(fo_bundles), 0, "snapshot_batch must not produce an fO bundle")

        # 4 base metric types × 7 labels = 28 bundles total.
        self.assertEqual(len(bundles), 4 * len(LABELS))

        _assert_base_progress_bundles(self, bundles, LABELS)


class TestMakeChimeraSummaryTable(unittest.TestCase):
    """Test make_chimera_summary_table function."""

    @classmethod
    def setUpClass(cls):
        """Run a full chimera batch workflow."""
        cls.opsim_visits = make_sample_opsim_visits(n_visits=500, random_state=42)
        cls.consdb_visits = make_sample_consdb_visits(n_visits=100, random_state=42)
        cls.out_dir = tempfile.mkdtemp(prefix="chimera_summary_test_")

        # Build chimera files
        cls.chimera_specs = build_chimeras(
            cls.consdb_visits,
            cls.opsim_visits,
            start_dayobs=20260101,
            end_dayobs=20260228,
            step=7,
            out_dir=cls.out_dir,
        )

        # Run batches
        cls.results_db_path = run_chimera_batches(
            cls.chimera_specs,
            batch_func=batches.glanceBatch,
            out_dir=cls.out_dir,
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up test output directory."""
        import shutil

        shutil.rmtree(cls.out_dir, ignore_errors=True)

    def test_returns_dataframe(self):
        """Test that result is a DataFrame."""
        result = make_chimera_summary_table(self.results_db_path)
        self.assertIsInstance(result, pd.DataFrame)

    def test_has_transition_dayobs_index(self):
        """Test that index is transition_dayobs."""
        result = make_chimera_summary_table(self.results_db_path)
        self.assertEqual(result.index.name, "transition_dayobs")

        # Index should be integer type
        self.assertTrue(np.issubdtype(result.index.dtype, np.integer))

    def test_has_multiindex_columns(self):
        """Test that columns are MultiIndex."""
        result = make_chimera_summary_table(self.results_db_path)

        self.assertIsInstance(result.columns, pd.MultiIndex)
        self.assertEqual(len(result.columns.names), 4)

        expected_names = [
            "metric_name",
            "slicer_name",
            "metric_info_label",
            "summary_metric",
        ]
        self.assertEqual(result.columns.names, expected_names)

    def test_has_expected_number_of_rows(self):
        """Test that result has correct number of rows."""
        result = make_chimera_summary_table(self.results_db_path)

        expected_rows = len(self.chimera_specs)
        self.assertEqual(len(result), expected_rows)

    def test_handles_empty_results_db(self):
        """Test with empty ResultsDb."""
        empty_dir = tempfile.mkdtemp(prefix="empty_test_")
        try:
            empty_db_path = os.path.join(empty_dir, "resultsDb_sqlite.db")
            results_db = ResultsDb(out_dir=empty_dir)
            results_db.close()

            import warnings

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = make_chimera_summary_table(empty_db_path)

                self.assertEqual(len(result), 0)
                self.assertEqual(len(w), 1)
                self.assertIn("No chimera run names found", str(w[0].message))
        finally:
            import shutil

            shutil.rmtree(empty_dir, ignore_errors=True)

    def test_supports_results_db_instance(self):
        """Test that ResultsDb instance is accepted."""
        results_db = ResultsDb(database=self.results_db_path)
        result = make_chimera_summary_table(results_db)
        results_db.close()

        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)


class TestEndToEnd(unittest.TestCase):
    """Test end-to-end chimera workflow.

    The chimera files are built once and shared by two batch runs, each
    writing to its own results directory: ``science_radar_batch`` (a legacy
    batch, using ``dayobs0``) and ``chimera_batch`` (R-4 progress metrics).
    """

    NSIDE = 16
    BANDS = ("u", "g", "r", "i", "z", "y")

    @classmethod
    def setUpClass(cls):
        """Generate sample data, build chimeras, and run both batches."""
        cls.opsim_visits = make_sample_opsim_visits(n_visits=500, random_state=42)
        cls.consdb_visits = make_sample_consdb_visits(n_visits=100, random_state=42)
        cls.out_dir = tempfile.mkdtemp(prefix="e2e_test_")
        cls.radar_out_dir = os.path.join(cls.out_dir, "science_radar")
        cls.progress_out_dir = os.path.join(cls.out_dir, "chimera_batch")

        # Build chimera files once, shared by both batches.
        cls.chimera_specs = build_chimeras(
            cls.consdb_visits,
            cls.opsim_visits,
            start_dayobs=20260101,
            end_dayobs=20260228,
            step=7,
            out_dir=cls.out_dir,
        )

        # science_radar_batch (srd_only for speed) requires dayobs0.
        cls.results_db_path = run_chimera_batches(
            cls.chimera_specs,
            batch_func=batches.science_radar_batch,
            out_dir=cls.radar_out_dir,
            batch_kwargs={"srd_only": True, "dayobs0": 20260101},
        )
        cls.summary_df = make_chimera_summary_table(cls.results_db_path)

        cls.progress_results_db_path = run_chimera_batches(
            cls.chimera_specs,
            batch_func=batches.chimera_batch,
            out_dir=cls.progress_out_dir,
            batch_kwargs={"nside": cls.NSIDE},
        )
        cls.progress_summary_df = make_chimera_summary_table(cls.progress_results_db_path)

    @classmethod
    def tearDownClass(cls):
        """Clean up test output directory."""
        import shutil

        shutil.rmtree(cls.out_dir, ignore_errors=True)

    def _summaries(self):
        """Yield (label, results db path, summary table) for each batch."""
        yield "science_radar_batch", self.results_db_path, self.summary_df
        yield "chimera_batch", self.progress_results_db_path, self.progress_summary_df

    def test_full_workflow_completes(self):
        """Test that full workflow completes successfully."""
        self.assertGreater(len(self.chimera_specs), 0)
        for label, db_path, summary_df in self._summaries():
            with self.subTest(batch=label):
                self.assertTrue(os.path.exists(db_path))
                self.assertGreater(len(summary_df), 0)

    def test_summary_table_structure(self):
        """Test summary table has expected structure."""
        for label, _, summary_df in self._summaries():
            with self.subTest(batch=label):
                self.assertEqual(summary_df.index.name, "transition_dayobs")
                self.assertTrue(np.issubdtype(summary_df.index.dtype, np.integer))
                self.assertIsInstance(summary_df.columns, pd.MultiIndex)
                self.assertEqual(len(summary_df.columns.names), 4)
                self.assertEqual(len(summary_df), len(self.chimera_specs))

    def test_run_names_match_chimera_pattern(self):
        """Test that run names in ResultsDb match chimera pattern."""
        for label, db_path, _ in self._summaries():
            with self.subTest(batch=label):
                results_db = ResultsDb(database=db_path)
                run_names = results_db.get_run_name()
                results_db.close()
                self.assertEqual(
                    sorted(run_names),
                    sorted(_run_name_from_dayobs(dayobs) for dayobs, _ in self.chimera_specs),
                )
                for run_name in run_names:
                    self.assertRegex(run_name, r"^chimera_\d{8}$")

    def test_summary_values_are_numeric(self):
        """Test that summary values are numeric."""
        for label, _, summary_df in self._summaries():
            for col in summary_df.columns:
                values = summary_df[col].dropna()
                if len(values) > 0:
                    self.assertTrue(
                        np.issubdtype(values.dtype, np.number),
                        f"{label} column {col} has non-numeric values: {values.dtype}",
                    )

    def _progress_column(self, metric, slicer, info_label, summary):
        col = (metric, slicer, info_label, summary)
        self.assertIn(col, self.progress_summary_df.columns)
        return self.progress_summary_df[col]

    def test_chimera_batch_visit_counts(self):
        """Visit counts from chimera_batch match the visits in each file."""
        for dayobs, path in self.chimera_specs:
            visits = pd.read_hdf(path, key="observations")
            expected = {"all": len(visits)}
            expected.update({band: int((visits["band"] == band).sum()) for band in self.BANDS})
            for suffix, n_expected in expected.items():
                if n_expected == 0:
                    continue  # Bundles with no visits write no results.
                counts = self._progress_column(
                    "Numbers of exposures", "UniSlicer", f"chimera_{suffix}", "Identity"
                )
                self.assertEqual(counts.loc[dayobs], n_expected, f"{dayobs} {suffix}")

    def test_chimera_batch_teff_is_additive(self):
        """Per-band t_eff is positive and sums to the all-band value."""
        total = self._progress_column("Sum t_eff", "UniSlicer", "chimera_all", "Identity")
        self.assertTrue((total > 0).all())
        band_sum = sum(
            self._progress_column("Sum t_eff", "UniSlicer", f"chimera_{band}", "Identity")
            for band in self.BANDS
        )
        np.testing.assert_allclose(band_sum.to_numpy(), total.to_numpy(), rtol=1e-6)

    def test_chimera_batch_healpix_summaries(self):
        """Count and depth maps have all required, sane summaries."""
        required = ["Count", "Max", "Mean", "Median", "Min", "Rms", "top18k", "10th%ile"]
        for metric in ("Number of exposure area stats", "Depth area stats"):
            for suffix in ("all",) + self.BANDS:
                for summary in required:
                    self._progress_column(metric, "HealpixSlicer", f"chimera_{suffix}", summary)

            counts = self._progress_column(metric, "HealpixSlicer", "chimera_all", "Count")
            self.assertTrue((counts > 0).all())
            mins = self._progress_column(metric, "HealpixSlicer", "chimera_all", "Min")
            maxs = self._progress_column(metric, "HealpixSlicer", "chimera_all", "Max")
            tenth = self._progress_column(metric, "HealpixSlicer", "chimera_all", "10th%ile")
            self.assertTrue((mins <= tenth).all())
            self.assertTrue((tenth <= maxs).all())

        # Coadded depth must be a plausible magnitude; every covered pixel
        # in a visit-count map has at least one visit.
        depth_mean = self._progress_column("Depth area stats", "HealpixSlicer", "chimera_all", "Mean")
        self.assertTrue(((depth_mean > 15) & (depth_mean < 30)).all())
        nvis_min = self._progress_column(
            "Number of exposure area stats", "HealpixSlicer", "chimera_all", "Min"
        )
        self.assertTrue((nvis_min >= 1).all())

    def test_chimera_batch_fo_metrics_present(self):
        """The fO metrics are recorded for every chimera.

        The sample data are too sparse to give meaningful fO values (they
        are masked), so only the presence of the summary columns is checked.
        """
        fo_summaries = {col[3] for col in self.progress_summary_df.columns if col[0] == "fO"}
        required = {
            "fOArea",
            "fOArea/benchmark",
            "fOArea_750",
            "fONv MedianNvis",
            "fONv MinNvis",
            "fONv/benchmark MedianNvis",
            "fONv/benchmark MinNvis",
        }
        self.assertLessEqual(required, fo_summaries)

    def test_chimera_batch_tracks_transition_dates(self):
        """Metrics change as completed visits replace baseline ones."""
        counts = self._progress_column("Numbers of exposures", "UniSlicer", "chimera_all", "Identity")
        self.assertEqual(list(counts.index), [dayobs for dayobs, _ in self.chimera_specs])
        self.assertGreater(counts.nunique(), 1)


if __name__ == "__main__":
    unittest.main()
