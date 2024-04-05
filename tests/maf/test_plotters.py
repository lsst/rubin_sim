# imports
import unittest

import numpy as np
from matplotlib.figure import Figure

import rubin_sim.maf as maf


class TestPlotters(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng()

    def test_healpix_plotters(self):
        # Set up a metric bundle to send to plotters
        bundle1 = maf.create_empty_metric_bundle()
        nside = 64
        bundle1.slicer = maf.HealpixSlicer(nside=nside)
        bundle1._setup_metric_values()
        bundle1.metric_values += self.rng.uniform(size=len(bundle1.slicer))
        # First test healpix sky map - just that it runs.
        bundle1.set_plot_funcs([maf.HealpixSkyMap()])
        figs = bundle1.plot()
        self.assertTrue(isinstance(figs["SkyMap"], Figure))
        # Test healpix histogram - just that it runs
        bundle1.set_plot_funcs([maf.HealpixHistogram()])
        figs = bundle1.plot()
        self.assertTrue(isinstance(figs["Histogram"], Figure))
        # Test power spectrum
        bundle1.set_plot_funcs([maf.HealpixPowerSpectrum()])
        figs = bundle1.plot()
        self.assertTrue(isinstance(figs["PowerSpectrum"], Figure))

    def test_base_skymap(self):
        bundle1 = maf.create_empty_metric_bundle()
        npoints = 1000
        ra = self.rng.uniform(low=0, high=360, size=npoints)
        dec = self.rng.uniform(low=-90, high=90, size=npoints)
        bundle1.slicer = maf.UserPointsSlicer(ra, dec)
        bundle1._setup_metric_values()
        bundle1.metric_values += self.rng.uniform(size=len(bundle1.slicer))
        # Test skymap
        bundle1.set_plot_funcs([maf.BaseSkyMap()])
        figs = bundle1.plot()
        self.assertTrue(isinstance(figs["SkyMap"], Figure))
        # Test healpix histogram - just that it runs
        bundle1.set_plot_funcs([maf.BaseHistogram()])
        figs = bundle1.plot()
        self.assertTrue(isinstance(figs["Histogram"], Figure))

    def test_oned_plotter(self):
        bundle1 = maf.create_empty_metric_bundle()
        npoints = 100
        bins = np.arange(0, npoints, 1)
        bundle1.slicer = maf.OneDSlicer(slice_col_name="test", bins=bins)
        bundle1.slicer.slice_points = {"bins": bins}
        bundle1.slicer.nslice = len(bins) - 1
        bundle1.slicer.shape = len(bins) - 1
        bundle1._setup_metric_values()
        bundle1.metric_values += self.rng.uniform(size=len(bundle1.slicer))
        # Test plotter
        bundle1.set_plot_funcs([maf.OneDBinnedData()])
        figs = bundle1.plot()
        self.assertTrue(isinstance(figs["BinnedData"], Figure))


if __name__ == "__main__":
    unittest.main()
