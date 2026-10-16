__all__ = ("TdeElement",)


import numpy as np
import pandas as pd

import rubin_sim.splat as splat

from .base_element import BaseReportElement


class TdeElement(BaseReportElement):

    def compute_elements(self, visits_df=None, visits_array=None, n_events=10000, n_events_quick=100):

        summary_stats = []

        info = splat.empty_info()
        info["data_source"] = self.run_name
        info["observations_subset"] = self.subset
        mjd0 = np.min(visits_array["observationStartMJD"])
        metric = splat.TdePopMetric(mjd0=mjd0)
        if self.quick_test:
            metric.generate_tde_pop(n_events=n_events_quick)
        else:
            metric.generate_tde_pop(n_events=n_events)
        sl = splat.Slicer(nside=None, missing=0, ra=np.degrees(metric.ra), dec=np.degrees(metric.dec))

        tde_array, info = sl(visits_array, metric, info=info)

        ph = splat.PlotHealbin(info=info)

        for key in tde_array.dtype.names:
            summary_stats.append(splat.gen_summary_row(info, "Mean TDE " + key, np.mean(tde_array[key])))
            summary_stats.append(
                splat.gen_summary_row(
                    info,
                    "Uncert in Mean XRB " + key,
                    np.nanmean(tde_array[key]) / np.sqrt(np.sum(tde_array[key])),
                )
            )

        fig = ph(
            np.degrees(metric.ra), np.degrees(metric.dec), tde_array["pre_peak"], unit="TDE mean pre_peak"
        )
        self.fig_saver(fig, info=info)

        summary_dataframe = pd.DataFrame.from_dict(summary_stats)
        return [tde_array], [info], summary_dataframe
