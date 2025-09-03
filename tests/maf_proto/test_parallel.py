import sqlite3
import unittest

import numpy as np
import pandas as pd

import rubin_sim.maf_proto as maf
from rubin_sim.data import get_baseline


class TestParallel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Read in some observations
        baseline_file = get_baseline()
        con = sqlite3.connect(baseline_file)
        df = pd.read_sql("select * from observations where night < 10;", con)
        cls.visits_array = df.to_records(index=False)
        con.close()

    def test_parallel(self):
        nside = 16

        slicer = maf.Slicer(nside=nside)
        metric = maf.MeanMetric(col="airmass")
        # Run with no info dict
        p1 = maf.metric_parallel(self.visits_array, metric, slicer, processes=2)

        # Also with info dict
        info = maf.empty_info()
        info["run_name"] = "arglebargle"

        p2, info = maf.metric_parallel(self.visits_array, metric, slicer, info=info, processes=2)

        # Check both ways match
        assert np.array_equal(p1, p2, equal_nan=True)

        # Check info dict got things filled in
        assert info["run_name"] == "arglebargle"
        assert info["slicer: nside"] == nside

        # Check we can run on custom point array
        ra = np.arange(0, 360, 1)
        dec = ra * 0 - 20
        slicer = maf.Slicer(ra=ra, dec=dec)

        p3 = maf.metric_parallel(self.visits_array, metric, slicer, processes=2)

        assert np.size(p3) == np.size(ra)


if __name__ == "__main__":
    unittest.main()
