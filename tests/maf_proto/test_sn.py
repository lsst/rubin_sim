import sqlite3
import unittest

import numpy as np
import pandas as pd

import rubin_sim.maf_proto as maf
from rubin_sim.data import get_baseline


class TestSN(unittest.TestCase):

    def test_n_sn(self):

        # Read in some data
        baseline_file = get_baseline()
        con = sqlite3.connect(baseline_file)
        df = pd.read_sql("select * from observations where night < 300;", con)
        visits_array = df.to_records(index=False)
        con.close()

        # Run on a very low res slicer
        metric = maf.SNNSNMetric()
        slicer = maf.Slicer(nside=4)

        hp_array = slicer(visits_array, metric)

        assert np.nanmax(hp_array["n_sn"]) > 0
        assert np.nanmax(hp_array["zlim"]) > 0

        metric = maf.SNNSNMetric(add_dust=True)
        hp_array = slicer(visits_array, metric)

        assert np.nanmax(hp_array["n_sn"]) > 0
        assert np.nanmax(hp_array["zlim"]) > 0


if __name__ == "__main__":
    unittest.main()
