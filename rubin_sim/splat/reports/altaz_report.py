__all__ = ("AltAzElement",)

import numpy as np
import pandas as pd

import rubin_sim.splat as splat
from .base_element import BaseReportElement


class AltAzElement(BaseReportElement):

    def compute_elements(self, visits_df=None, visits_array=None, nside=64, bands="ugrizy"):

        result_hp_arrays = []
        result_infos = []

        alt_az_sub = {
            "Near Sun Twilight Observations": np.char.find(
                visits_array["scheduler_note"].astype(str), "_near_sun"
            )
            != -1
        }
        for filtername in bands:
            alt_az_sub["band=%s" % filtername] = np.where(visits_array["band"] == filtername)
        alt_az_sub["all"] = np.arange(visits_array.size)
        alt_az_sub["blob_long"] = np.char.find(visits_array["scheduler_note"].astype(str), "blob_long") != -1
        # Starts with long
        alt_az_sub["long"] = np.char.find(visits_array["scheduler_note"].astype(str), "long") == 0
        alt_az_sub["ToO"] = np.char.find(visits_array["scheduler_note"].astype(str), "ToO") != -1

        for subset in alt_az_sub:
            info = {"data_source": self.run_name}
            info["observations_subset"] = (self.subset + "\n" + subset).strip()
            info["group"] = "AltAz"
            sub_data = visits_array[alt_az_sub[subset]]
            metric = splat.CountMetric()
            sl = splat.Slicer(nside=nside, lat_col="altitude", lon_col="azimuth")
            hp_array, info = sl(sub_data, metric, info=info)
            result_hp_arrays.append(hp_array)
            result_infos.append(info)
            plam = splat.PlotLambert(info=info)
            fig = plam(hp_array)
            self.fig_saver(fig, info=info)

        return result_hp_arrays, result_infos, pd.DataFrame()
