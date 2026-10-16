import unittest

import matplotlib.pylab as plt
import pandas as pd

import rubin_sim.splat as splat


class FigSaverClose:
    """Dummy class to close figs as they come in"""

    def __call__(self, fig, info=None, filename=None):
        plt.close(fig)

    def save_stats(*args, **kwargs):
        pass


class TestReports(unittest.TestCase):

    def test_quick_test(self):

        fs = FigSaverClose()

        kwargs = {"quick_test": True, "fig_saver": fs, "run_name": "test_run"}

        elements = [
            splat.AgnElement(**kwargs),
            splat.AltAzElement(**kwargs),
            splat.AstrometryElement(**kwargs),
            splat.BrownDwarfElement(**kwargs),
            splat.ColorSlopeElement(**kwargs),
            splat.DepthCountElement(**kwargs),
            splat.FoElement(**kwargs),
            splat.KneElement(**kwargs),
            splat.MicrolensingElement(**kwargs),
            splat.PerNoteElement(**kwargs),
            splat.SlewElement(**kwargs),
            splat.HourglassElement(**kwargs),
            splat.TdeElement(**kwargs),
        ]

        # Use the first one to load up the default data
        elements[0].load_visits()
        visits_df = elements[0].visits_df
        # Add columns
        for element in elements:
            visits_df = element.add_cols(visits_df)

        # Have numpy array for speed
        visits_df, visits_array = elements[0].visit_resolver(visits_df, None)

        for element in elements:
            datas, infos, summary_df = element(visits_df=visits_df, visits_array=visits_array)
            assert isinstance(datas, list)
            assert isinstance(infos, list)
            if len(infos) > 0:
                assert isinstance(infos[0], dict)
            assert isinstance(summary_df, pd.DataFrame)
            plt.close()

    def test_sne(self):
        fs = FigSaverClose()
        element = splat.SneElement(quick_test=True, fig_saver=fs)
        datas, infos, summary_df = element(nside=2)
        assert isinstance(datas, list)
        assert isinstance(infos, list)
        if len(infos) > 0:
            assert isinstance(infos[0], dict)
        assert isinstance(summary_df, pd.DataFrame)


if __name__ == "__main__":

    unittest.main()
