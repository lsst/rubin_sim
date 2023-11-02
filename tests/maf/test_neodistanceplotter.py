import unittest

import numpy as np

import rubin_sim.maf.plots as plots


class TestNeoDistancePlotter(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(61723009)
        names = [
            "eclipLat",
            "eclipLon",
            "MaxGeoDist",
            "NEOHelioX",
            "NEOHelioY",
            "filter",
        ]
        types = [float] * 5
        types.append("|S1")
        npts = 100
        self.metric_values = np.zeros(npts, list(zip(names, types)))
        self.metric_values["MaxGeoDist"] = rng.rand(npts) * 2.0
        self.metric_values["eclipLat"] = rng.rand(npts)
        self.metric_values["NEOHelioX"] = rng.rand(npts) * 3 - 1.5
        self.metric_values["NEOHelioY"] = rng.rand(npts) * 3 - 1.5 + 1
        self.metric_values["filter"] = "g"

    def test_plotter(self):
        """
        Just test that it can make a figure without throwing an error.
        """
        plotter = plots.NeoDistancePlotter()
        # Need to wrap in a list because it will usually go through the
        # UniSlicer, and will thus be an array inside a 1-element masked array
        fig = plotter([self.metric_values], None, {})
        self.assertNotEqual(fig, None)


if __name__ == "__main__":
    unittest.main()
