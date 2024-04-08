# imports
import unittest

import numpy as np
from matplotlib.figure import Figure
from rubin_scheduler.scheduler.model_observatory import ModelObservatory

import rubin_sim.maf as maf


class TestPlotters(unittest.TestCase):
    def setUp(self):
        # Set a seed to make the tests reproducible
        self.rng = np.random.default_rng(seed=1234)

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

    def test_hpxmap_plotter(self):
        bundle = maf.create_empty_metric_bundle()
        nside = 64
        bundle.slicer = maf.HealpixSlicer(nside=nside)
        bundle._setup_metric_values()
        bundle.metric_values += self.rng.uniform(size=len(bundle.slicer))

        plotter = maf.HpxmapPlotter()
        bundle.set_plot_funcs([plotter])
        _ = bundle.plot()

    def test_visit_perimeter_plotter(self):
        model_observatory = ModelObservatory(init_load_length=1)

        num_points = 5
        field_ra = np.arange(30, 30 + num_points, dtype=float)
        field_dec = np.arange(-60, -60 + num_points, dtype=float)
        rot_sky_pos = np.arange(num_points, dtype=float) % 360

        unmasked_data = np.empty(dtype=object, shape=(1,))
        unmasked_data[0] = np.core.records.fromarrays(
            (field_ra, field_dec, rot_sky_pos),
            dtype=np.dtype(
                [
                    ("fieldRA", field_ra.dtype),
                    ("fieldDec", field_dec.dtype),
                    ("rotSkyPos", rot_sky_pos.dtype),
                ]
            ),
        )
        masked_data = np.ma.MaskedArray(data=unmasked_data, mask=False, fill_value=-666, dtype=object)

        bundle = maf.create_empty_metric_bundle()
        bundle.slicer = maf.UniSlicer()
        bundle._setup_metric_values()
        bundle.metric_values = masked_data

        def compute_camera_perimeter(ra, decl, rotation):
            # just a quadrangle for this unit test.
            # the math isn't quite right for an actual square,
            # but good enough for a unit test.
            size = 1.0
            ras = [
                ra - 0.5 * size * np.cos(np.radians(decl)) * np.sin(np.radians(rotation)),
                ra - 0.5 * size * np.cos(np.radians(decl)) * np.cos(np.radians(rotation)),
                ra + 0.5 * size * np.cos(np.radians(decl)) * np.sin(np.radians(rotation)),
                ra + 0.5 * size * np.cos(np.radians(decl)) * np.cos(np.radians(rotation)),
            ]
            decls = [
                decl + 0.5 * size * np.cos(np.radians(rotation)),
                decl - 0.5 * size * np.sin(np.radians(rotation)),
                decl - 0.5 * size * np.cos(np.radians(rotation)),
                decl + 0.5 * size * np.sin(np.radians(rotation)),
            ]
            return ras, decls

        plot_dict = {
            "camera_perimeter_func": compute_camera_perimeter,
            "model_observatory": model_observatory,
            "decorations": ["ecliptic", "galactic_plane", "sun", "moon", "horizon"],
        }
        plotter = maf.VisitPerimeterPlotter()
        bundle.set_plot_funcs([plotter])
        bundle.set_plot_dict(plot_dict)
        _ = bundle.plot()


if __name__ == "__main__":
    unittest.main()
