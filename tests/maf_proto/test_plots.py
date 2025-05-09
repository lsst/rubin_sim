import sqlite3
import unittest

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

import rubin_sim.maf_proto as maf
from rubin_sim.data import get_baseline


class TestPlots(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Read in some observations, compute a quick map
        baseline_file = get_baseline()
        con = sqlite3.connect(baseline_file)
        df = pd.read_sql("select * from observations where night < 10;", con)
        cls.visits_array = df.to_records(index=False)
        con.close()

        nside = 16
        sl = maf.Slicer(nside=nside)
        metric = maf.MeanMetric(col="airmass")
        cls.hp_array = sl(cls.visits_array, metric)

        # handy info dict
        cls.info = {
            "run_name": "gary",
            "observations_subset": "some of them",
            "metric: unit": "rods / hogs head",
        }

    def test_heal(self):
        """Test that the simple way to run MAF works"""

        # Run with defaults
        pm = maf.PlotMoll()
        fig = pm(self.hp_array)

        # Exercise kwargs
        cb_params = {"shrink": 0.5, "pad": 0.2}
        pm = maf.PlotMoll(info=self.info)
        fig = pm(self.hp_array, cb_params=cb_params, title="ack", min=0.2, log=True)

        # Can it take a figure
        fig, ax = plt.subplots()
        pm = maf.PlotMoll(info=self.info)
        fig = pm(self.hp_array, fig=fig, cb_params=cb_params, title="ack", min=0.2, log=True)

    def test_plothealhist(self):

        phh = maf.PlotHealHist()
        fig = phh(self.hp_array)

        # and with kwargs
        fig, ax = plt.subplots()
        phh = maf.PlotHealHist(info=self.info)
        fig = phh(
            self.hp_array, fig=fig, ax=ax, histtype="bar", bins=np.arange(10), title="ack", ylabel="rods"
        )

    def test_lambert(self):

        # Defaults
        pl = maf.PlotLambert()
        fig = pl(self.hp_array)

        # Non-defaults
        fig, ax = plt.subplots()
        pl = maf.PlotLambert(info=self.info)
        fig = pl(self.hp_array, fig=fig, ax=ax, xlabel="pies", alt_limit=15.0, levels=199)

    def test_hourglass(self):
        # Defaults
        hr = maf.PlotHourglass()
        fig = hr(self.visits_array)

        # Non-defaults
        fig, ax = plt.subplots()

        hr = maf.PlotHourglass(info=self.info)
        fig = hr(self.visits_array, fig=fig, ax=ax, xlabel="ack")

    def test_healbin(self):
        hb = maf.PlotHealbin()
        vals = np.arange(10)
        ra = np.arange(10) / 9.0 * 360
        dec = np.arange(10) / 9.0 * 180 - 90
        fig = hb(ra, dec, vals)

        # Non-defaults
        fig, ax = plt.subplots()

        hb = maf.PlotHealbin(info=self.info)
        fig = hb(ra, dec, vals, fig=fig, nside=64, reduce_func=np.nanmax)


if __name__ == "__main__":
    unittest.main()
