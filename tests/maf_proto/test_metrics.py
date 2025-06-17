import sqlite3
import unittest

import numpy as np
import pandas as pd

import rubin_sim.maf_proto as maf
from rubin_sim.data import get_baseline


class TestSimple(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Read in some observations, compute a quick map
        baseline_file = get_baseline()
        con = sqlite3.connect(baseline_file)
        df = pd.read_sql("select * from observations where night < 10;", con)
        cls.visits_array = df.to_records(index=False)
        con.close()

    def test_coadd(self):
        """Test that the simple way to run MAF works"""

        filtername = "r"

        nside = 16
        sl = maf.Slicer(nside=nside)
        metric = maf.CoaddM5Metric(filtername)
        coadd_hp = sl(self.visits_array, metric)

        metric = maf.CoaddM5ExtinctionMetric(filtername)
        coadd_hp_extinct = sl(self.visits_array, metric)

        good = np.isfinite(coadd_hp)

        assert np.all(coadd_hp[good] > coadd_hp_extinct[good])

    def test_kuiper(self):
        nside = 16
        sl = maf.Slicer(nside=nside)
        metric = maf.KuiperMetric()
        hp_array = sl(self.visits_array, metric)

        good = np.isfinite(hp_array)

        assert np.all(hp_array[good] >= 0)
        assert np.nanmax(hp_array[good]) > 0


if __name__ == "__main__":

    unittest.main()
