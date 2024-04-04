import unittest

import numpy as np
from astropy.time import Time
from rubin_scheduler.scheduler.model_observatory import ModelObservatory

import rubin_sim
import rubin_sim.maf.plots.skyproj_plotters
from rubin_sim import maf


class TestSkyprojPlots(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(seed=6563)

    def test_hpxmap_plotter(self):
        bundle = maf.create_empty_metric_bundle()
        nside = 64
        bundle.slicer = maf.HealpixSlicer(nside=nside)
        bundle._setup_metric_values()
        bundle.metric_values += self.rng.uniform(size=len(bundle.slicer))

        plot_dict = {"decorations": ["colorbar"]}
        plotter = rubin_sim.maf.plots.skyproj_plotters.HpxmapPlotter()
        bundle.set_plot_funcs([plotter])
        bundle.set_plot_dict(plot_dict)
        _ = bundle.plot()

    def test_visit_perimeter_plotter(self):
        model_observatory = ModelObservatory(init_load_length=1)
        model_observatory.mjd = Time("2025-11-10T06:00:00Z").mjd

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
        plotter = rubin_sim.maf.plots.skyproj_plotters.VisitPerimeterPlotter()
        bundle.set_plot_funcs([plotter])
        bundle.set_plot_dict(plot_dict)
        _ = bundle.plot()
