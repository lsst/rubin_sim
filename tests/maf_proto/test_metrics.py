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
        cls.df = pd.read_sql("select * from observations where night < 10;", con)
        cls.visits_array = cls.df.to_records(index=False)
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

    def test_fancy(self):
        nside = 4
        sl = maf.Slicer(nside=nside)
        metric = maf.FancyMetric()
        fancy = sl(self.visits_array, metric)

        assert len(fancy["mean"]) > 0
        assert len(fancy["std"]) > 0

    def test_vector(self):
        nside = 4
        n_times = 60
        sl = maf.Slicer(nside=nside)
        times = np.arange(n_times)
        metric = maf.VectorMetric(times=times)
        result = sl(self.visits_array, metric)

        assert result.shape[1] == n_times

    def test_accu(self):
        nside = 4
        n_times = 60
        sl = maf.Slicer(nside=nside)
        times = np.arange(n_times)
        metric = maf.AccumulateCountMetric(times)
        result = sl(self.visits_array, metric)

        assert result.shape[1] == n_times

    def test_kuiper(self):
        nside = 16
        sl = maf.Slicer(nside=nside)
        metric = maf.KuiperMetric()
        hp_array = sl(self.visits_array, metric)

        good = np.isfinite(hp_array)

        assert np.all(hp_array[good] >= 0)
        assert np.nanmax(hp_array[good]) > 0

    def test_bd(self):

        # Add any new columns we need
        ra_pi_amp, dec_pi_amp = maf.parallax_amplitude(
            self.df["fieldRA"].values,
            self.df["fieldDec"].values,
            self.df["observationStartMJD"].values,
            degrees=True,
        )
        self.df["ra_pi_amp"] = ra_pi_amp
        self.df["dec_pi_amp"] = dec_pi_amp

        ra_dcr_amp, dec_dcr_amp = maf.dcr_amplitude(
            90.0 - self.df["altitude"].values,
            self.df["paraAngle"].values,
            self.df["filter"].values,
            degrees=True,
        )
        self.df["ra_dcr_amp"] = ra_dcr_amp
        self.df["dec_dcr_amp"] = dec_dcr_amp

        # But mostly want numpy array for speed.
        visits_array = self.df.to_records(index=False)

        metric = maf.BDParallaxMetric()
        nside = 4
        sl = maf.Slicer(nside=nside)

        result = sl(visits_array, metric)

        assert np.max(result) > 0


if __name__ == "__main__":

    unittest.main()
