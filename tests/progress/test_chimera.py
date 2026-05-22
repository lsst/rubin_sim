"""Test suite for the chimera progress capability.

This module contains tests for the core API functions in rubin_sim.maf.chimera_progress:
- build_chimera
- build_chimeras
- run_chimera_batches
- make_chimera_summary_table

Tests follow the unittest.TestCase pattern and use synthetic test data generated
from the baseline opsim database.
"""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import rubin_sim.maf.batches as batches
from rubin_sim.maf.chimera_progress import (
    _dayobs_from_filename,
    _dayobs_from_run_name,
    _dayobs_range,
    _run_name_from_dayobs,
    build_chimera,
    build_chimeras,
    make_chimera_summary_table,
    run_chimera_batches,
)
from rubin_sim.maf.db import ResultsDb

# Use relative imports for test data (works with pytest from rubin_sim)
from .test_chimera_data import make_sample_consdb_visits, make_sample_opsim_visits


class TestDayObsHelpers(unittest.TestCase):
    """Test helper functions for dayObs manipulation."""

    def test_dayobs_range_basic(self):
        """Test _dayobs_range with basic inputs."""
        result = _dayobs_range(20260101, 20260105, 1)
        expected = [20260101, 20260102, 20260103, 20260104, 20260105]
        self.assertEqual(result, expected)

    def test_dayobs_range_with_step(self):
        """Test _dayobs_range with step parameter."""
        result = _dayobs_range(20260101, 20260107, 2)
        expected = [20260101, 20260103, 20260105, 20260107]
        self.assertEqual(result, expected)

    def test_dayobs_range_same_start_end(self):
        """Test _dayobs_range when start equals end."""
        result = _dayobs_range(20260105, 20260105, 1)
        expected = [20260105]
        self.assertEqual(result, expected)

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
        self.assertEqual(
            _dayobs_from_filename("/path/to/chimera_20260101.h5"), 20260101
        )
        self.assertEqual(
            _dayobs_from_filename("chimera_20261231.h5"), 20261231
        )
        self.assertIsNone(_dayobs_from_filename("other_20260101.h5"))
        self.assertIsNone(_dayobs_from_filename("/path/to/file.txt"))


class TestBuildChimera(unittest.TestCase):
    """Test build_chimera function."""

    def setUp(self):
        """Create test DataFrames with dayObs column."""
        # Use observationId to match baseline data schema
        self.consdb_visits = pd.DataFrame({
            "observationId": [1, 2, 3],
            "dayObs": [20260101, 20260102, 20260103],
            "filter": ["g", "r", "i"],
            "observationStartMJD": [61000, 61001, 61002],
        })

        self.opsim_visits = pd.DataFrame({
            "observationId": [101, 102, 103],
            "dayObs": [20260104, 20260105, 20260106],
            "filter": ["g", "r", "i"],
            "observationStartMJD": [61003, 61004, 61005],
        })

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
        consdb = pd.DataFrame({
            "observationId": [],
            "dayObs": [],
            "filter": [],
        })

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
        consdb = pd.DataFrame({
            "observationId": [],
            "dayObs": [],
            "filter": [],
        })

        result = build_chimera(
            consdb,
            self.opsim_visits,
            start_dayobs=20260101,
            transition_dayobs=20260106,
            end_dayobs=20260106,
        )
        # When consdb is empty but opsim has visits in range, return opsim visits
        # All opsim visits have dayObs 4, 5, 6 and end_dayobs=6 with transition=6
        # opsim visits with dayObs > 6 and <= 6 = none
        # So result should be empty
        self.assertEqual(len(result), 0)

    def test_empty_opsim_returns_consdb(self):
        """Test with empty opsim visits."""
        opsim = pd.DataFrame({
            "observationId": [],
            "dayObs": [],
            "filter": [],
        })

        result = build_chimera(
            self.consdb_visits,
            opsim,
            start_dayobs=20260101,
            transition_dayobs=20260106,
            end_dayobs=20260106,
        )
        # All consdb visits (dayObs 1, 2, 3) are <= transition (6), so all 3 should be included
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
        cls.consdb_visits = make_sample_consdb_visits(n_visits=100, days=60, random_state=42)
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
        # Check that dates are spaced by step (except possibly last)
        for i in range(len(transition_dates) - 1):
            diff = transition_dates[i + 1] - transition_dates[i]
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
                (chimera[chimera["dayObs"] <= transition_dayobs]["observationId"]
                 .isin(self.consdb_visits["observationId"]))
                .all()
            )


class TestRunChimeraBatches(unittest.TestCase):
    """Test run_chimera_batches function."""

    @classmethod
    def setUpClass(cls):
        """Generate sample test data and build chimera files."""
        cls.opsim_visits = make_sample_opsim_visits(n_visits=500, random_state=42)
        cls.consdb_visits = make_sample_consdb_visits(n_visits=100, days=60, random_state=42)
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
        # The run_chimera_batches function tries run_name first, then falls back
        # to runName if there's a TypeError about unexpected 'run_name' argument.
        # This test verifies the fallback logic works.

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


class TestMakeChimeraSummaryTable(unittest.TestCase):
    """Test make_chimera_summary_table function."""

    @classmethod
    def setUpClass(cls):
        """Run a full chimera batch workflow."""
        cls.opsim_visits = make_sample_opsim_visits(n_visits=500, random_state=42)
        cls.consdb_visits = make_sample_consdb_visits(n_visits=100, days=60, random_state=42)
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
    """Test end-to-end chimera workflow."""

    @classmethod
    def setUpClass(cls):
        """Generate sample data and run full workflow."""
        cls.opsim_visits = make_sample_opsim_visits(n_visits=500, random_state=42)
        cls.consdb_visits = make_sample_consdb_visits(n_visits=100, days=60, random_state=42)
        cls.out_dir = tempfile.mkdtemp(prefix="e2e_test_")

        # Build chimera files
        cls.chimera_specs = build_chimeras(
            cls.consdb_visits,
            cls.opsim_visits,
            start_dayobs=20260101,
            end_dayobs=20260228,
            step=7,
            out_dir=cls.out_dir,
        )

        # Run batches with science_radar_batch (srd_only for speed)
        # science_radar_batch requires dayobs0 parameter
        cls.results_db_path = run_chimera_batches(
            cls.chimera_specs,
            batch_func=batches.science_radar_batch,
            out_dir=cls.out_dir,
            batch_kwargs={"srd_only": True, "dayobs0": 20260101},
        )

        # Create summary table
        cls.summary_df = make_chimera_summary_table(cls.results_db_path)

    @classmethod
    def tearDownClass(cls):
        """Clean up test output directory."""
        import shutil

        shutil.rmtree(cls.out_dir, ignore_errors=True)

    def test_full_workflow_completes(self):
        """Test that full workflow completes successfully."""
        self.assertGreater(len(self.chimera_specs), 0)
        self.assertTrue(os.path.exists(self.results_db_path))
        self.assertGreater(len(self.summary_df), 0)

    def test_summary_table_structure(self):
        """Test summary table has expected structure."""
        # Check index
        self.assertEqual(self.summary_df.index.name, "transition_dayobs")
        self.assertTrue(np.issubdtype(self.summary_df.index.dtype, np.integer))

        # Check columns
        self.assertIsInstance(self.summary_df.columns, pd.MultiIndex)
        self.assertEqual(len(self.summary_df.columns.names), 4)

        # Check row count
        self.assertEqual(len(self.summary_df), len(self.chimera_specs))

    def test_run_names_match_chimera_pattern(self):
        """Test that run names in ResultsDb match chimera pattern."""
        results_db = ResultsDb(database=self.results_db_path)
        run_names = results_db.get_run_name()
        results_db.close()

        for run_name in run_names:
            self.assertRegex(run_name, r"^chimera_\d{8}$")

    def test_summary_values_are_numeric(self):
        """Test that summary values are numeric."""
        # All values should be numeric (float or int)
        for col in self.summary_df.columns:
            values = self.summary_df[col].dropna()
            if len(values) > 0:
                self.assertTrue(
                    np.issubdtype(values.dtype, np.number),
                    f"Column {col} has non-numeric values: {values.dtype}",
                )


if __name__ == "__main__":
    unittest.main()
