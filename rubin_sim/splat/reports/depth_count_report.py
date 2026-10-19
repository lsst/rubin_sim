__all__ = ("DepthCountElement",)

import copy

import numpy as np
import pandas as pd

import rubin_sim.splat as splat

from .base_element import BaseReportElement


class DepthCountElement(BaseReportElement):

    def compute_elements(self, visits_df=None, visits_array=None, nside=64, bands="ugrizy"):

        stats_to_run = {"mean": np.nanmean, "median": np.nanmedian}
        sl = splat.Slicer(nside=nside)

        result_hp_arrays = []
        result_infos = []
        summary_stats = []

        # Things calculated per filter
        for i, bandname in enumerate(bands):
            info = {"data_source": self.run_name}
            info["observations_subset"] = self.subset + "\nband=%s" % bandname
            # select subset of data
            sub_data = visits_array[np.where(visits_array["band"] == bandname)[0]]

            #
            metrics = []
            infos = []
            plot_dicts = []

            # Coadd depth
            metrics.append(splat.CoaddM5Metric(bandname))
            temp_info = copy.copy(info)
            temp_info["group"] = "Basics"
            temp_info["subgroup"] = "Coadd"
            infos.append(temp_info)
            plot_dicts.append({})

            # Number of visits
            metrics.append(splat.CountMetric())
            temp_info = copy.copy(info)
            temp_info["group"] = "Basics"
            temp_info["subgroup"] = "Counts"
            infos.append(temp_info)
            plot_dicts.append(
                {"log": True, "min": 1, "norm": "log", "cb_params": {"format": "%i", "n_ticks": 4}}
            )

            # Run the metics through the slicer
            hp_arrays, infos = sl(sub_data, metrics, info=infos)

            result_hp_arrays.extend(hp_arrays)
            result_infos.extend(infos)

            # For each array we've generated, make a vis, summerize it how we like
            for hp_array, metric, info, plot_dict in zip(hp_arrays, metrics, infos, plot_dicts):
                pm = splat.PlotMoll(info=info)
                fig = pm(hp_array, **plot_dict)
                self.fig_saver(fig, info=info)
                # Do whatever stats we want on the hp_array
                for stat in stats_to_run:
                    summary_stats.append(splat.gen_summary_row(info, stat, stats_to_run[stat](hp_array)))

        # Number of obs, all filters
        info = {"data_source": self.run_name}
        info["observations_subset"] = self.subset
        info["group"] = "Basics"
        info["subgroup"] = "Counts"
        metric = splat.CountMetric()
        sl = splat.Slicer(nside=nside)
        counts_hp_array, info = sl(visits_array, metric, info=info)
        result_hp_arrays.append(counts_hp_array)
        result_infos.append(info)
        pm = splat.PlotMoll(info=info)
        fig = pm(counts_hp_array, log=True, norm="log", cb_params={"format": "%i", "n_ticks": 4})
        self.fig_saver(fig, info=info)
        for stat in stats_to_run:
            summary_stats.append(splat.gen_summary_row(info, stat, stats_to_run[stat](counts_hp_array)))

        summary_dataframe = pd.DataFrame.from_dict(summary_stats)
        return result_hp_arrays, result_infos, summary_dataframe
