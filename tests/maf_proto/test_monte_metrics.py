import sqlite3
import unittest

import numpy as np
import pandas as pd

import rubin_sim.maf_proto as maf
from rubin_sim.data import get_baseline


class TestMonte(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Read in some observations, compute a quick map
        baseline_file = get_baseline()
        con = sqlite3.connect(baseline_file)
        cls.df = pd.read_sql("select * from observations where night < 365;", con)
        cls.visits_array = cls.df.to_records(index=False)
        con.close()

    def test_kne(self):

        kne_metric = maf.KNePopMetric()
        slicer = maf.Slicer()

        # load some data, run the metric

        result = slicer(self.visits_array, kne_metric)

        assert slicer.nside is None
        assert np.size(result) == np.size(kne_metric.ra)

        info = maf.empty_info()

        info["run_name"] = "default test"
        result, info = slicer(self.visits_array, kne_metric, info=info)

        assert slicer.nside is None
        assert info["slicer: nside"] is None

    def test_kne_files(self):
        files = maf.get_kne_filename(
            inj_params_list=[{"mej_dyn": 0.005, "mej_wind": 0.050, "phi": 30, "theta": 25.8}]
        )
        assert len(files) > 0

    def test_microlensing(self):

        mjd0 = self.visits_array["observationStartMJD"].min()

        metric_calcs = ["detect", "Npts", "Fisher"]
        n_events = 20
        for metric_calc in metric_calcs:
            metric = maf.MicrolensingMetric(mjd0=mjd0, metric_calc=metric_calc)

            metric.generate_microlensing_events(n_events=n_events, min_crossing_time=5, max_crossing_time=10)
            sl = maf.Slicer(nside=None, missing=0, ra=np.degrees(metric.ra), dec=np.degrees(metric.dec))
            mic_array = sl(self.visits_array, metric)

            assert len(mic_array) == n_events


if __name__ == "__main__":
    unittest.main()
