import matplotlib

matplotlib.use("Agg")
import unittest
import rubin_sim.maf.utils.opsimUtils as opsimUtils
import os
from rubin_sim.data import get_data_dir
import numpy as np
import sqlite3


class TestOpsimUtils(unittest.TestCase):
    def testScaleBenchmarks(self):
        """Test scaling the design and stretch benchmarks for the length of the run."""
        # First test that method returns expected dictionaries.
        for i in ("design", "stretch"):
            benchmark = opsimUtils.scaleBenchmarks(10.0, i)
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

    def testCalcCoaddedDepth(self):
        """Test the expected coadded depth calculation."""
        benchmark = opsimUtils.scaleBenchmarks(10, "design")
        coadd = opsimUtils.calcCoaddedDepth(
            benchmark["nvisits"], benchmark["singleVisitDepth"]
        )
        for f in coadd:
            self.assertLess(coadd[f], 1000)
        singlevisits = {"u": 1, "g": 1, "r": 1, "i": 1, "z": 1, "y": 1}
        coadd = opsimUtils.calcCoaddedDepth(singlevisits, benchmark["singleVisitDepth"])
        for f in coadd:
            self.assertAlmostEqual(coadd[f], benchmark["singleVisitDepth"][f])

    def testGetSimData(self):
        """Test that we can get simulation data"""
        database_file = os.path.join(get_data_dir(), "tests", "example_dbv1.7_0yrs.db")
        dbcols = ["fieldRA", "fieldDec", "note"]
        sql = "night < 10"
        full_sql = "SELECT fieldRA, fieldDec, note FROM observations where night < 10;"
        # Check that we get data the usual way
        data = opsimUtils.getSimData(database_file, sql, dbcols)
        assert np.size(data) > 0

        # Check that we can pass a connection object
        con = sqlite3.connect(database_file)
        data = opsimUtils.getSimData(con, sql, dbcols)
        con.close()
        assert np.size(data) > 0

        # Check that kwarg overrides sqlconstraint and dbcols
        data = opsimUtils.getSimData(
            database_file, "blah blah", ["nocol"], full_sql_query=full_sql
        )
        assert np.size(data) > 0


if __name__ == "__main__":
    unittest.main()
