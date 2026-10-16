__all__ = ("AgnElement",)

import rubin_sim.splat as splat
import numpy as np


from .base_element import BaseReportElement


class AgnElement(BaseReportElement):

    def add_cols(self, df):
        """Add any columns to the dataframe that are needed"""
        # Check if needed columns already exist, if not add them

        if "saturation_mag" not in df.columns:
            saturation_mag = splat.saturation_limit(df.to_records(index=False))
            df["saturation_mag"] = saturation_mag
        return df

    def compute_elements(self, visits_df=None, visits_array=None):

        sl = splat.Slicer(nside=64)

        # Lists to hold data and info
        result_hp_arrays = []
        result_infos = []
        summary_dataframe = None

        for band in "ugrizy":
            metrics = []
            metrics.append(splat.QSONumberCountsMetric(band))
            metrics.append(splat.SFUncertMetric())
            metrics.append(splat.AgnTimeLagMetric(lag=100, unit="AGN Time Lag 100"))
            metrics.append(splat.AgnTimeLagMetric(lag=5, unit="AGN Time Lag 5"))

            indx = np.where(visits_array["band"] == band)
            sub_data = visits_array[indx]

            infos = []
            plot_dicts = []
            for m in metrics:
                info = splat.empty_info()
                info["data_source"] = self.run_name
                info["observations_subset"] = band
                infos.append(info)
                plot_dicts.append({})

            hp_arrays, infos = sl(sub_data, metrics, info=infos)

            result_hp_arrays.extend(hp_arrays)
            result_infos.extend(infos)

            for hp_array, metric, info, plot_dict in zip(hp_arrays, metrics, infos, plot_dicts):
                pm = splat.PlotMoll(info=info)
                fig = pm(hp_array, **plot_dict)
                self.fig_saver(fig, info=info)
  
        return result_hp_arrays, result_infos, summary_dataframe
