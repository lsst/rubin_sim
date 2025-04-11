import sqlite3
import unittest

import healpy as hp
import numpy as np
import pandas as pd

import rubin_sim.maf_proto as maf
from rubin_sim.data import get_baseline


class TestSimple(unittest.TestCase):

    def test_simple(self):
        """Test that the simple way to run MAF works"""

        # Read in some observations
        baseline_file = get_baseline()
        con = sqlite3.connect(baseline_file)
        df = pd.read_sql("select * from observations where night < 10;", con)
        visits_array = df.to_records(index=False)
        con.close()

        nside = 16
        sl = maf.Slicer(nside=nside)
        metric = maf.MeanMetric(col="airmass")
        hp_array = sl(visits_array, metric)

        assert np.nanmin(hp_array) > 0
        assert hp.npix2nside(hp_array.size) == nside

        # Test we can even do on the fly
        sl = maf.Slicer(nside=16)

        def ontheflymetric(visits, **kwargs):
            return np.mean(visits["airmass"])

        hp_array = sl(visits_array, ontheflymetric)
        assert np.nanmin(hp_array) > 0
        assert hp.npix2nside(hp_array.size) == nside


if __name__ == "__main__":
    unittest.main()
