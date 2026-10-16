__all__ = ("KneElement",)


import numpy as np
import pandas as pd

import rubin_sim.splat as splat

from .base_element import BaseReportElement


class KneElement(BaseReportElement):

    def compute_elements(self, visits_df=None, visits_array=None, n_events=500000, n_events_quick=100):

        summary_stats = []

        info = splat.empty_info()
        info["data_source"] = self.run_name
        info["observations_subset"] = self.subset
        mjd0 = np.min(visits_array["observationStartMJD"])
        metric = splat.KNePopMetric(mjd0=mjd0)
        if self.quick_test:
            metric.generate_kn_pop(n_events=n_events_quick, d_min=10, d_max=600)
        else:
            metric.generate_kn_pop(n_events=n_events, d_min=10, d_max=600)
        sl = splat.Slicer(nside=None, missing=0, ra=np.degrees(metric.ra), dec=np.degrees(metric.dec))

        kn_array, info = sl(visits_array, metric, info=info)

        ph = splat.PlotHealbin(info=info)

        for key in kn_array.dtype.names:
            summary_stats.append(splat.gen_summary_row(info, "Mean KNe " + key, np.mean(kn_array[key])))
            summary_stats.append(
                splat.gen_summary_row(
                    info, "Uncert in Mean KNe " + key, np.mean(kn_array[key]) / np.sqrt(np.sum(kn_array[key]))
                )
            )

        fig = ph(
            np.degrees(metric.ra),
            np.degrees(metric.dec),
            kn_array["multi_detect"],
            unit="KNe mean multi_detect",
        )
        self.fig_saver(fig, info=info)

        summary_dataframe = pd.DataFrame.from_dict(summary_stats)
        return [kn_array], [info], summary_dataframe
