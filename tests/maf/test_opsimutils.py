import os
import sqlite3
import unittest

import numpy as np
from rubin_scheduler.data import get_data_dir

import rubin_sim.maf.utils.opsim_utils as opsimUtils

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
        dbcols = ["fieldRA", "fieldDec", "note"]
        sql = "night < 10"
        full_sql = "SELECT fieldRA, fieldDec, note FROM observations where night < 10;"
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


if __name__ == "__main__":
    unittest.main()
