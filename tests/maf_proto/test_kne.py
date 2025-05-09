import sqlite3
import unittest

import numpy as np
import pandas as pd

import rubin_sim.maf_proto as maf
from rubin_sim.data import get_baseline


class TestKne(unittest.TestCase):

    def test_kne(self):

        baseline_file = get_baseline()
        con = sqlite3.connect(baseline_file)
        df = pd.read_sql("select * from observations where night < 365;", con)
        visits_array = df.to_records(index=False)
        con.close()

        kne_metric = maf.KNePopMetric()
        slicer = maf.Slicer()

        # load some data, run the metric

        result = slicer(visits_array, kne_metric)

        assert slicer.nside is None
        assert np.size(result) == np.size(kne_metric.ra)

        info = maf.empty_info()

        info["run_name"] = "default test"
        result, info = slicer(visits_array, kne_metric, info=info)

        assert slicer.nside is None
        assert info["slicer: nside"] is None

    def test_kne_files(self):
        files = maf.get_kne_filename(inj_params_list=[{'mej_dyn': 0.005, 'mej_wind': 0.050, 'phi': 30, 'theta': 25.8}])
        assert len(files) > 0


if __name__ == "__main__":
    unittest.main()
