__all__ = ("AstrometryElement",)

import numpy as np
import pandas as pd

import rubin_sim.splat as splat

from .base_element import BaseReportElement


class AstrometryElement(BaseReportElement):
    """Check parallax, proper motion, etc."""

    def add_cols(self, df):
        """Add any columns to the dataframe that are needed"""
        # Check if needed columns already exist, if not add them

        if "ra_pi_amp" not in df.columns:
            ra_pi_amp, dec_pi_amp = splat.parallax_amplitude(
                df["fieldRA"].values, df["fieldDec"].values, df["observationStartMJD"].values, degrees=True
            )

            df["ra_pi_amp"] = ra_pi_amp
            df["dec_pi_amp"] = dec_pi_amp
        if "ra_dcr_amp" not in df.columns:

            ra_dcr_amp, dec_dcr_amp = splat.dcr_amplitude(
                90.0 - df["altitude"].values, df["paraAngle"].values, df["filter"].values, degrees=True
            )
            df["ra_dcr_amp"] = ra_dcr_amp
            df["dec_dcr_amp"] = dec_dcr_amp

        return df

    def compute_elements(self, visits_df=None, visits_array=None, nside=64):

        summary_stats = []

        stats_to_run = {"mean": np.nanmean, "median": np.nanmedian}
        sl = splat.Slicer(nside=nside)

        metrics = []
        metrics.append(splat.ParallaxMetric())
        metrics.append(splat.ProperMotionMetric())
        metrics.append(splat.ParallaxCoverageMetric())
        metrics.append(splat.ParallaxDcrDegenMetric())

        result_infos = []
        plot_dicts = []
        for m in metrics:
            info = splat.empty_info()
            info["data_source"] = self.run_name
            info["observations_subset"] = self.subset
            result_infos.append(info)
            plot_dicts.append({})

        # Set the first two metrics to get caught and
        # precentile clip the plotted value range.
        plot_dicts[0]["min"] = "percentile"
        plot_dicts[1]["min"] = "percentile"

        result_hp_arrays, result_infos = sl(visits_array, metrics, info=result_infos)

        for hp_array, metric, info, plot_dict in zip(result_hp_arrays, metrics, result_infos, plot_dicts):
            if "min" in plot_dict.keys():
                if plot_dict["min"] == "percentile":
                    min_val, max_val = splat.percentile_clipping(hp_array[np.isfinite(hp_array)])
                    plot_dict["min"] = min_val
                    plot_dict["max"] = max_val

            pm = splat.PlotMoll(info=info)
            fig = pm(hp_array, **plot_dict)
            self.fig_saver(fig, info=info)
            # Do whatever stats we want on the hp_array
            for stat in stats_to_run:
                summary_stats.append(splat.gen_summary_row(info, stat, stats_to_run[stat](hp_array)))

        summary_dataframe = pd.DataFrame.from_dict(summary_stats)

        return result_hp_arrays, result_infos, summary_dataframe
