__all__ = ("SneElement",)


import numpy as np
import pandas as pd

import rubin_sim.splat as splat

from .base_element import BaseReportElement


class SneElement(BaseReportElement):

    def compute_elements(self, visits_df=None, visits_array=None, nside=16):

        summary_stats = []

        info = splat.empty_info()
        info["data_source"] = self.run_name
        info["observations_subset"] = self.subset
        sl = splat.Slicer(nside=nside)
        metric = splat.SNNSNMetric()

        sn_array, info = sl(visits_array, metric, info=info)

        pm = splat.PlotMoll(info=info)

        fig = pm(sn_array["n_sn"], unit="N SNe to z limit")
        self.fig_saver(fig, info=info)
        fig = pm(sn_array["zlim"], unit="z limit")
        self.fig_saver(fig, info=info)

        summary_stats.append(splat.gen_summary_row(info, "sum N SNe", np.nansum(sn_array["n_sn"])))
        summary_stats.append(splat.gen_summary_row(info, "mean z limit", np.nansum(sn_array["zlim"])))
        summary_stats.append(splat.gen_summary_row(info, "median z limit", np.nanmedian(sn_array["zlim"])))

        summary_dataframe = pd.DataFrame.from_dict(summary_stats)
        return [sn_array], [info], summary_dataframe
