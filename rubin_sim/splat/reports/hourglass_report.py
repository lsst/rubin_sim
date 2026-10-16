__all__ = ("HourglassElement",)

import numpy as np
import pandas as pd

import rubin_sim.splat as splat
from .base_element import BaseReportElement

DAYS_IN_YEAR = 365.25


class HourglassElement(BaseReportElement):

    def compute_elements(self, visits_df=None, visits_array=None):

        info = {"data_source": self.run_name}
        info["group"] = "Hourglass"
        min_year = np.floor(visits_array["night"].min() / DAYS_IN_YEAR)
        max_year = np.ceil(visits_array["night"].max() / DAYS_IN_YEAR)
        hstart = np.arange(min_year, max_year)
        hend = hstart + 1

        for start, end in zip(hstart, hend):
            indx = np.where(
                (visits_array["night"] < end * DAYS_IN_YEAR) & (visits_array["night"] >= start * DAYS_IN_YEAR)
            )
            info["observations_subset"] = "year %i to %i" % (start, end)
            info["metric: name"] = "Hourglass"
            hr = splat.PlotHourglass(info=info)
            fig = hr(visits_array[indx])
            self.fig_saver(fig, info=info)

        return [], [], pd.DataFrame()
