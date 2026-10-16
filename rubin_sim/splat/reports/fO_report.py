__all__ = ("FoElement",)

import numpy as np
import pandas as pd

import rubin_sim.splat as splat

from .base_element import BaseReportElement


class FoElement(BaseReportElement):

    def compute_elements(self, visits_df=None, visits_array=None, nside=64):

        summary_stats = []

        info = {"data_source": self.run_name}
        info["observations_subset"] = (self.subset + "\nExptime > 19s").strip()
        info["group"] = "SRD"
        subset = visits_array[np.where(visits_array["visitExposureTime"] > 19)]
        metric = splat.CountMetric()
        sl = splat.Slicer(nside=nside)
        counts_hp_array, info = sl(subset, metric, info=info)
        po = splat.PlotFo(info=info)
        fig = po(counts_hp_array)
        self.fig_saver(fig, info=info)
        # FO stats
        fo_stats = splat.fO_calcs(counts_hp_array)
        for key in fo_stats:
            summary_stats.append(splat.gen_summary_row(info, key, fo_stats[key]))

        summary_dataframe = pd.DataFrame.from_dict(summary_stats)
        return [], [], summary_dataframe
