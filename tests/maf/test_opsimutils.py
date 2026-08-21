import os
import sqlite3
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from rubin_scheduler.data import get_data_dir

import rubin_sim.maf.utils.opsim_utils as opsimUtils
from rubin_sim.sim_archive.util import opsimdb_to_hdf5

TEST_DB = "example_v3.4_0yrs.db"


class TestOpsimUtils(unittest.TestCase):
    def test_scale_benchmarks(self):
        """Test scaling the design and stretch benchmarks for the
        length of the run.
        """
        # First test that method returns expected dictionaries.
        for i in ("design", "stretch"):
            benchmark = opsimUtils.scale_benchmarks(10.0, i)
            self.assertIsInstance(benchmark, dict)
            expectedkeys = (
                "Area",
                "nvisitsTotal",
                "nvisits",
                "seeing",
                "skybrightness",
                "singleVisitDepth",
            )
            expectedfilters = ("u", "g", "r", "i", "z", "y")
            for k in expectedkeys:
                self.assertIn(k, benchmark)
            expecteddictkeys = (
                "nvisits",
                "seeing",
                "skybrightness",
                "singleVisitDepth",
            )
            for k in expecteddictkeys:
                for f in expectedfilters:
                    self.assertIn(f, benchmark[k])

    def test_calc_coadded_depth(self):
        """Test the expected coadded depth calculation."""
        benchmark = opsimUtils.scale_benchmarks(10, "design")
        coadd = opsimUtils.calc_coadded_depth(benchmark["nvisits"], benchmark["singleVisitDepth"])
        for f in coadd:
            self.assertLess(coadd[f], 1000)
        singlevisits = {"u": 1, "g": 1, "r": 1, "i": 1, "z": 1, "y": 1}
        coadd = opsimUtils.calc_coadded_depth(singlevisits, benchmark["singleVisitDepth"])
        for f in coadd:
            self.assertAlmostEqual(coadd[f], benchmark["singleVisitDepth"][f])

    def test_get_sim_data(self):
        """Test that we can get simulation data"""
        database_file = os.path.join(get_data_dir(), "tests", TEST_DB)
        dbcols = ["fieldRA", "fieldDec", "scheduler_note"]
        sql = "night < 10"
        full_sql = "SELECT fieldRA, fieldDec, scheduler_note FROM observations where night < 10;"
        # Check that we get data the usual way
        data = opsimUtils.get_sim_data(database_file, sql, dbcols)
        assert np.size(data) > 0

        # Check that we can pass a connection object
        con = sqlite3.connect(database_file)
        data = opsimUtils.get_sim_data(con, sql, dbcols)
        con.close()
        assert np.size(data) > 0

        # Check that kwarg overrides sqlconstraint and dbcols
        data = opsimUtils.get_sim_data(database_file, "blah blah", ["nocol"], full_sql_query=full_sql)
        assert np.size(data) > 0

        # Check that bad file raises an error
        with self.assertRaises(FileNotFoundError):
            opsimUtils.get_sim_data("not_a_file.db", sql, ["nocol"])

    def test_get_sim_data_hdf5(self):
        """Test that we can get simulation data from HDF5 files."""
        database_file = os.path.join(get_data_dir(), "tests", TEST_DB)

        # Create HDF5 file from database
        with TemporaryDirectory() as tmpdir:
            hdf5_file = os.path.join(tmpdir, "test_visits.h5")
            opsimdb_to_hdf5(database_file, hdf5_file)

            # Test basic HDF5 reading without constraints
            data = opsimUtils.get_sim_data(hdf5_file)
            assert np.size(data) > 0
            # Check that we got reasonable columns
            assert "observationId" in data.dtype.names

            # Test HDF5 with sqlconstraint
            data_filtered = opsimUtils.get_sim_data(hdf5_file, sqlconstraint="observationId < 10")
            assert np.size(data_filtered) > 0
            # Verify all rows satisfy the constraint
            assert np.all(data_filtered["observationId"] < 10)

            # Test HDF5 with full_sql_query
            full_sql = "SELECT observationId, fieldRA FROM observations WHERE observationId < 5;"
            data_query = opsimUtils.get_sim_data(hdf5_file, full_sql_query=full_sql)
            assert np.size(data_query) > 0
            assert "observationId" in data_query.dtype.names
            assert "fieldRA" in data_query.dtype.names
            assert np.all(data_query["observationId"] < 5)

            # Verify HDF5 results match SQLite results
            data_sqlite = opsimUtils.get_sim_data(database_file, sqlconstraint="observationId < 10")

            data_hdf5 = opsimUtils.get_sim_data(hdf5_file, sqlconstraint="observationId < 10")
            assert np.allclose(data_sqlite["fieldRA"], data_hdf5["fieldRA"])

    def test_save_visits_as_parquet_dataframe(self):
        """Test that a DataFrame round-trips through save_visits_as_parquet."""
        database_file = os.path.join(get_data_dir(), "tests", TEST_DB)
        original = opsimUtils.get_sim_data(database_file, return_class=pd.DataFrame)

        with TemporaryDirectory() as tmpdir:
            parquet_file = os.path.join(tmpdir, "visits.parquet")
            opsimUtils.save_visits_as_parquet(original, parquet_file)

            assert os.path.isfile(parquet_file)
            reloaded = pd.read_parquet(parquet_file)

            # Same number of rows
            assert len(reloaded) == len(original)

            # Numeric columns should survive the round-trip
            for col in ("fieldRA", "fieldDec", "observationStartMJD", "night"):
                assert col in reloaded.columns
                np.testing.assert_array_almost_equal(reloaded[col].values, original[col].values, decimal=10)

    def test_save_visits_as_parquet_recarray(self):
        """Test that a numpy recarray round-trips through
        save_visits_as_parquet."""
        database_file = os.path.join(get_data_dir(), "tests", TEST_DB)
        original_rec = opsimUtils.get_sim_data(database_file, return_class=np.recarray)

        with TemporaryDirectory() as tmpdir:
            parquet_file = os.path.join(tmpdir, "visits_rec.parquet")
            opsimUtils.save_visits_as_parquet(original_rec, parquet_file)

            assert os.path.isfile(parquet_file)
            reloaded = pd.read_parquet(parquet_file)

            assert len(reloaded) == len(original_rec)
            np.testing.assert_array_almost_equal(
                reloaded["fieldRA"].values, original_rec["fieldRA"], decimal=10
            )

    def test_save_visits_as_parquet_visit_id_index(self):
        """Test that visit_id becomes the index when present."""
        df = pd.DataFrame(
            {
                "visit_id": [10, 20, 30],
                "fieldRA": [1.0, 2.0, 3.0],
                "fieldDec": [-1.0, -2.0, -3.0],
            }
        )

        with TemporaryDirectory() as tmpdir:
            parquet_file = os.path.join(tmpdir, "visits_id.parquet")
            opsimUtils.save_visits_as_parquet(df, parquet_file)

            reloaded = pd.read_parquet(parquet_file)
            # visit_id should be the index
            np.testing.assert_array_equal(reloaded.index.values, [10, 20, 30])
            # visit_id column should still be present (drop=False)
            assert "visit_id" in reloaded.columns

    def test_save_visits_as_parquet_drops_index_column(self):
        """Test that a spurious 'index' column is removed before writing."""
        df = pd.DataFrame(
            {
                "index": [0, 1, 2],
                "fieldRA": [1.0, 2.0, 3.0],
            }
        )

        with TemporaryDirectory() as tmpdir:
            parquet_file = os.path.join(tmpdir, "visits_index_col.parquet")
            opsimUtils.save_visits_as_parquet(df, parquet_file)

            reloaded = pd.read_parquet(parquet_file)
            assert "index" not in reloaded.columns

    def test_save_visits_as_parquet_all_nan_object_column(self):
        """Test that object columns that are entirely NaN are cast to
        float64."""
        df = pd.DataFrame(
            {
                "fieldRA": [1.0, 2.0, 3.0],
                "all_nan_col": pd.array([None, None, None], dtype=object),
            }
        )

        with TemporaryDirectory() as tmpdir:
            parquet_file = os.path.join(tmpdir, "visits_nan.parquet")
            # Should not raise even though the column has no valid type info.
            opsimUtils.save_visits_as_parquet(df, parquet_file)

            reloaded = pd.read_parquet(parquet_file)
            assert "all_nan_col" in reloaded.columns
            # All values should be NaN in the reloaded data.
            assert reloaded["all_nan_col"].isna().all()

    def test_save_visits_as_parquet_string_nan_filled(self):
        """Test that NaN values in string columns are replaced with empty
        strings."""
        df = pd.DataFrame(
            {
                "fieldRA": [1.0, 2.0, 3.0],
                "note": ["a", None, "c"],
            }
        )

        with TemporaryDirectory() as tmpdir:
            parquet_file = os.path.join(tmpdir, "visits_str.parquet")
            opsimUtils.save_visits_as_parquet(df, parquet_file)

            reloaded = pd.read_parquet(parquet_file)
            assert reloaded["note"].iloc[1] == ""

    def test_save_visits_as_parquet_creates_parent_dirs(self):
        """Test that missing parent directories are created automatically."""
        with TemporaryDirectory() as tmpdir:
            parquet_file = os.path.join(tmpdir, "subdir", "nested", "visits.parquet")
            df = pd.DataFrame({"fieldRA": [1.0, 2.0]})
            opsimUtils.save_visits_as_parquet(df, parquet_file)
            assert os.path.isfile(parquet_file)

    def test_save_visits_as_parquet_bad_input(self):
        """Test that a non-convertible input raises TypeError."""
        with TemporaryDirectory() as tmpdir:
            parquet_file = os.path.join(tmpdir, "visits_bad.parquet")
            with self.assertRaises(TypeError):
                opsimUtils.save_visits_as_parquet("not_a_dataframe", parquet_file)

    def test_get_sim_data_parquet_roundtrip(self):
        """Test that get_sim_data reads a parquet file written by
        save_visits_as_parquet."""
        database_file = os.path.join(get_data_dir(), "tests", TEST_DB)
        original = opsimUtils.get_sim_data(database_file, return_class=pd.DataFrame)

        with TemporaryDirectory() as tmpdir:
            parquet_file = os.path.join(tmpdir, "visits.parquet")
            opsimUtils.save_visits_as_parquet(original, parquet_file)

            # Read back via get_sim_data as a recarray (the default)
            reloaded_rec = opsimUtils.get_sim_data(parquet_file)
            assert isinstance(reloaded_rec, np.recarray)
            assert len(reloaded_rec) == len(original)
            np.testing.assert_array_almost_equal(
                reloaded_rec["fieldRA"], original["fieldRA"].values, decimal=10
            )

            # Read back via get_sim_data as a DataFrame
            reloaded_df = opsimUtils.get_sim_data(parquet_file, return_class=pd.DataFrame)
            assert isinstance(reloaded_df, pd.DataFrame)
            assert len(reloaded_df) == len(original)

    def test_get_sim_data_parquet_with_sqlconstraint(self):
        """Test that get_sim_data applies sqlconstraint when reading
        parquet."""
        database_file = os.path.join(get_data_dir(), "tests", TEST_DB)
        original = opsimUtils.get_sim_data(database_file, return_class=pd.DataFrame)

        with TemporaryDirectory() as tmpdir:
            parquet_file = os.path.join(tmpdir, "visits.parquet")
            opsimUtils.save_visits_as_parquet(original, parquet_file)

            filtered = opsimUtils.get_sim_data(parquet_file, sqlconstraint="night < 5")
            assert isinstance(filtered, np.recarray)
            assert len(filtered) > 0
            assert np.all(filtered["night"] < 5)

    def test_get_visit_data_parquet_roundtrip(self):
        """Test that get_visit_data reads a parquet file as a DataFrame."""
        database_file = os.path.join(get_data_dir(), "tests", TEST_DB)
        original = opsimUtils.get_visit_data(database_file)
        assert isinstance(original, pd.DataFrame)

        with TemporaryDirectory() as tmpdir:
            parquet_file = os.path.join(tmpdir, "visits.parquet")
            opsimUtils.save_visits_as_parquet(original, parquet_file)

            reloaded = opsimUtils.get_visit_data(parquet_file)
            assert isinstance(reloaded, pd.DataFrame)
            assert len(reloaded) == len(original)
            np.testing.assert_array_almost_equal(
                reloaded["fieldRA"].values, original["fieldRA"].values, decimal=10
            )

    def test_save_visits_does_not_mutate_input(self):
        """Test that save_visits_as_parquet does not modify the caller's
        DataFrame."""
        df = pd.DataFrame(
            {
                "visit_id": [1, 2, 3],
                "fieldRA": [10.0, 20.0, 30.0],
                "note": ["a", None, "c"],
                "all_nan": pd.array([None, None, None], dtype=object),
            }
        )
        original_note = df["note"].copy()
        original_all_nan = df["all_nan"].copy()

        with TemporaryDirectory() as tmpdir:
            parquet_file = os.path.join(tmpdir, "visits.parquet")
            opsimUtils.save_visits_as_parquet(df, parquet_file)

        # The original DataFrame should be unchanged.
        pd.testing.assert_series_equal(df["note"], original_note)
        pd.testing.assert_series_equal(df["all_nan"], original_all_nan)


if __name__ == "__main__":
    unittest.main()
