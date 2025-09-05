import sqlite3
import unittest

import numpy as np
import pandas as pd

import rubin_sim.maf_proto as maf
from rubin_sim.data import get_baseline


class TestSlicers(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Read in some observations, compute a quick map
        baseline_file = get_baseline()
        con = sqlite3.connect(baseline_file)
        cls.df = pd.read_sql("select * from observations where night < 10;", con)
        cls.visits_array = cls.df.to_records(index=False)
        con.close()

    def test_slicer(self):

        s1 = maf.Slicer(nside=128)
        s2 = maf.Slicer(nside=16)

        assert s1.shape > s2.shape

    def test_user_slicer(self):
        """Test we can set different points."""

        ra = np.arange(0, 360, 1)
        dec = ra * 0 - 20

        s1 = maf.Slicer()
        s1.setup_slice_points(ra_rad=np.radians(ra), dec_rad=np.radians(dec))

        assert s1.shape == ra.size

    def test_one_point(self):
        slicer = maf.Slicer(nside=128)
        metric = maf.MeanMetric()
        result = slicer(self.visits_array, metric, indx=[0, 1, 2, 200])

        assert len(result) == 4


if __name__ == "__main__":
    unittest.main()
