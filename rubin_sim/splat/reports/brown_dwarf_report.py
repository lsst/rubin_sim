__all__ = ("BrownDwarfElement",)

import numpy as np
import pandas as pd

import rubin_sim.splat as splat

from .base_element import BaseReportElement


class BrownDwarfElement(BaseReportElement):

    def add_cols(self, df):
        """Add any columns to the dataframe that are needed"""

        if "ra_pi_amp" not in df.columns:
            ra_pi_amp, dec_pi_amp = splat.parallax_amplitude(
                df["fieldRA"].values, df["fieldDec"].values, df["observationStartMJD"].values, degrees=True
            )
            df["ra_pi_amp"] = ra_pi_amp
            df["dec_pi_amp"] = dec_pi_amp

        return df

    def compute_elements(self, visits_df=None, visits_array=None, nside=64):
        summary_stats = []

        stats_to_run = {"mean": np.nanmean, "Volume": splat.hp_sum_volume}
        sl = splat.Slicer(nside=nside)

        metrics = []
        l7_bd_mags = {"i": 20.09, "z": 18.18, "y": 17.13}
        metrics.append(splat.BDParallaxMetric(mags=l7_bd_mags, name="Brown Dwarf, L7"))

        l4_bd_mags = {"i": 18.35, "z": 16.68, "y": 15.66}
        metrics.append(splat.BDParallaxMetric(mags=l4_bd_mags, name="Brown Dwarf, L4"))

        infos = []
        plot_dicts = []
        for m in metrics:
            info = splat.empty_info()
            info["data_source"] = self.run_name
            info["observations_subset"] = (self.subset + "\n %s" % m.name).strip()
            info["population"] = m.name
            infos.append(info)
            plot_dicts.append({})

        hp_arrays, infos = sl(visits_array, metrics, info=infos)

        for hp_array, metric, info in zip(hp_arrays, metrics, infos):

            pm = splat.PlotMoll(info=info)
            fig = pm(hp_array)
            self.fig_saver(fig, info=info)
            for stat in stats_to_run:
                row = splat.gen_summary_row(info, stat, stats_to_run[stat](hp_array))
                if stat == "Volume":
                    row["unit"] = "pc^3"
                summary_stats.append(row)

        summary_dataframe = pd.DataFrame.from_dict(summary_stats)
        return hp_arrays, infos, summary_dataframe
