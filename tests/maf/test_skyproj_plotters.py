import unittest
from os import path

import numpy as np
from astropy.time import Time
from rubin_scheduler.data import get_baseline
from rubin_scheduler.scheduler.model_observatory import ModelObservatory

import rubin_sim
import rubin_sim.maf.plots.skyproj_plotters
from rubin_sim import maf


class TestSkyprojPlots(unittest.TestCase):
    def setUp(self):
        self.opsim_fname = get_baseline()
        self.run_name = path.splitext(path.basename(self.opsim_fname))[0]

    def test_hpxmap_plotter(self):
        bundle = maf.MetricBundle(
            metric=maf.Coaddm5Metric(),
            slicer=maf.HealpixSlicer(nside=64),
            constraint="filter='g'",
            run_name=self.run_name,
        )

        bgroup = maf.MetricBundleGroup([bundle], self.opsim_fname)
        bgroup.run_all()

        plot_dict = {"decorations": ["colorbar"]}
        plotter = rubin_sim.maf.plots.skyproj_plotters.HpxmapPlotter()
        bundle.set_plot_funcs([plotter])
        bundle.set_plot_dict(plot_dict)
        _ = bundle.plot()

    def test_visit_perimeter_plotter(self):
        model_observatory = ModelObservatory(init_load_length=1)
        model_observatory.mjd = Time("2025-11-10T06:00:00Z").mjd

        bundle = maf.MetricBundle(
            metric=maf.PassMetric(cols=["fieldRA", "fieldDec", "rotSkyPos"]),
            slicer=maf.UniSlicer(),
            constraint="floor(observationStartMJD) = 61000",
            run_name=self.run_name,
        )

        bgroup = maf.MetricBundleGroup([bundle], self.opsim_fname)
        bgroup.run_all()

        def compute_camera_perimeter(ra, decl, rotation):
            # just a quadrangle for this unit test.
            # the math isn't quite right for an actual square,
            # but good enough for a unit test.
            size = 1.0
            ras = [
                -0.5 * size * np.cos(np.radians(decl)) * np.sin(np.radians(rotation)),
                -0.5 * size * np.cos(np.radians(decl)) * np.cos(np.radians(rotation)),
                0.5 * size * np.cos(np.radians(decl)) * np.sin(np.radians(rotation)),
                0.5 * size * np.cos(np.radians(decl)) * np.cos(np.radians(rotation)),
            ]
            decls = [
                0.5 * size * np.cos(np.radians(rotation)),
                -0.5 * size * np.sin(np.radians(rotation)),
                -0.5 * size * np.cos(np.radians(rotation)),
                0.5 * size * np.sin(np.radians(rotation)),
            ]
            return ras, decls

        plot_dict = {
            "camera_perimeter_func": compute_camera_perimeter,
            "model_observatory": model_observatory,
            "decorations": ["ecliptic", "galactic_plane", "sun", "moon", "horizon"],
        }
        plotter = rubin_sim.maf.plots.skyproj_plotters.VisitPerimeterPlotter()
        bundle.set_plot_funcs([plotter])
        bundle.set_plot_dict(plot_dict)
        _ = bundle.plot()
