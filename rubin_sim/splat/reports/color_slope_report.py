__all__ = ("ColorSlopeElement",)


import numpy as np
import pandas as pd

import rubin_sim.splat as splat
from .base_element import BaseReportElement


class ColorSlopeElement(BaseReportElement):

    def compute_elements(self, visits_df=None, visits_array=None, nside=64):

        summary_stats = []

        stats_to_run = {"mean": np.nanmean, "sum": np.nansum}
        sl = splat.Slicer(nside=nside)

        metrics = []
        metrics.append(splat.ColorSlopeMetric())
        metrics.append(splat.ColorSlope2NightMetric())

        infos = []
        plot_dicts = []
        for m in metrics:
            info = splat.empty_info()
            info["data_source"] = self.run_name
            info["observations_subset"] = self.subset
            infos.append(info)
            plot_dicts.append({})

        hp_arrays, infos = sl(visits_array, metrics, info=infos)

        for hp_array, metric, info, plot_dict in zip(hp_arrays, metrics, infos, plot_dicts):
            pm = splat.PlotMoll(info=info)
            fig = pm(hp_array, **plot_dict)
            self.fig_saver(fig, info=info)
            # Do whatever stats we want on the hp_array
            for stat in stats_to_run:
                summary_stats.append(splat.gen_summary_row(info, stat, stats_to_run[stat](hp_array)))

        summary_dataframe = pd.DataFrame.from_dict(summary_stats)
        return hp_arrays, infos, summary_dataframe
