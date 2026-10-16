__all__ = ("SlewElement",)

import numpy as np
import pandas as pd

import rubin_sim.splat as splat

from .base_element import BaseReportElement


class SlewElement(BaseReportElement):

    def add_cols(self, df):
        df["t_eff_normed"] = splat.t_eff(
            df["fiveSigmaDepth"], df["filter"], exptime=df["visitExposureTime"], normed=True
        )
        return df

    def compute_elements(self, visits_df=None, visits_array=None):

        if visits_df is None:
            raise ValueError("Need visits_df for SlewElement.")

        data_list = []
        infos = []
        summary_stats = []

        info = splat.empty_info()
        info["data_source"] = self.run_name
        info["observations_subset"] = self.subset
        info["name"] = "slew time"
        info["unit"] = "sec"
        # Slew stats
        summary_stats.append(splat.gen_summary_row(info, "max", np.max(visits_df["slewTime"])))
        summary_stats.append(splat.gen_summary_row(info, "min", np.min(visits_df["slewTime"])))
        summary_stats.append(splat.gen_summary_row(info, "mean", np.mean(visits_df["slewTime"])))
        summary_stats.append(splat.gen_summary_row(info, "median", np.median(visits_df["slewTime"])))

        # Plot a histogram of slewtimes
        info = {
            "data_source": self.run_name,
            "unit": "Slew Time (sec)",
            "observations_subset": self.subset,
            "name": "slew time",
        }
        ph = splat.PlotHist(info=info)
        fig = ph(visits_array["slewTime"], log=True)
        self.fig_saver(fig, info)
        data_list.append(None)
        infos.append(info)

        # Hist of slew distances
        info = {"run_name": self.run_name}
        info["group"] = "Slew"
        info["unit"] = "Slew Distance (deg)"
        info["name"] = "slewDistance"
        ph = splat.PlotHist(info=info)
        fig = ph(visits_array["slewDistance"], log=True)
        self.fig_saver(fig, info=info)
        data_list.append(None)
        infos.append(info)

        # Plot filter changes per night
        stats = {"mean": np.mean, "median": np.median, "max": np.max, "min": np.min}
        info = {
            "data_source": self.run_name,
            "name": "filter changes per night",
            "observations_subset": self.subset,
        }

        fpn = visits_df.groupby("night")["filter"].apply(splat.count_value_changes)
        pl = splat.PlotLine(info=info)
        fig = pl(fpn.index, fpn.values, xlabel="Night", ylabel="Filter Changes per Night")
        self.fig_saver(fig, info=info)
        data_list.append(fpn)
        infos.append(info)
        for stat in stats:
            summary_stats.append(splat.gen_summary_row(info, stat, stats[stat](fpn)))

        # Open shutter fraction per night
        info = {"data_source": self.run_name}
        info["group"] = "Slew"
        info["name"] = "open shutter fraction"
        info["caption"] = "Open Shutter Fraction per night"
        osfpn = visits_df.groupby("night").apply(splat.osf_visit_array, include_groups=False)
        pl = splat.PlotLine(info=info)
        fig = pl(osfpn.index, osfpn.values, xlabel="Night", ylabel="Open Shutter Fraction per Night")
        self.fig_saver(fig, info=info)
        data_list.append(osfpn)
        infos.append(info)

        # Open Shutter Fraction overall
        info = {"data_source": self.run_name}
        info["group"] = "Slew"
        info["name"] = "Open Shutter Fraction"
        info["unit"] = "fraction"
        info["observations_subset"] = self.subset
        osf = splat.open_shutter_fraction(
            visits_array["observationStartMJD"], visits_array["visitExposureTime"]
        )
        summary_stats.append(splat.gen_summary_row(info, "OSF", osf))

        # Effective exposure time per filter
        for filtername in "ugrizy":
            info = {"data_source": self.run_name}
            info["group"] = "Slew"
            info["unit"] = "T Effective (normed)"
            info["observations_subset"] = self.subset + "/nfilter=%s" % filtername
            indx = np.where(visits_array["filter"] == filtername)
            summary_stats.append(
                splat.gen_summary_row(info, "mean", np.mean(visits_array["t_eff_normed"][indx]))
            )

        summary_dataframe = pd.DataFrame.from_dict(summary_stats)
        return data_list, infos, summary_dataframe
