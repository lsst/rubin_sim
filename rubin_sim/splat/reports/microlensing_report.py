__all__ = ("MicrolensingElement",)


import numpy as np
import pandas as pd

import rubin_sim.splat as splat
from .base_element import BaseReportElement


class MicrolensingElement(BaseReportElement):

    def compute_elements(self, visits_df=None, visits_array=None, n_events=10000, n_events_quick=20):
        mjd0 = visits_array["observationStartMJD"].min()

        summary_stats = []

        crossing_times = [
            [1, 5],
            [5, 10],
            [10, 20],
            [20, 30],
            [30, 60],
            [60, 90],
            [100, 200],
            [200, 500],
            [500, 1000],
        ]

        for ct in crossing_times:
            info = splat.empty_info()
            info["data_source"] = self.run_name
            info["observations_subset"] = self.subset
            mjd0 = np.min(visits_array["observationStartMJD"])
            metric = splat.MicrolensingMetric(
                mjd0=mjd0, name="Microlensing, crossing %i-%i days" % (ct[0], ct[1])
            )
            if self.quick_test:
                metric.generate_microlensing_events(
                    n_events=n_events_quick, min_crossing_time=ct[0], max_crossing_time=ct[1]
                )
            else:
                metric.generate_microlensing_events(
                    n_events=n_events, min_crossing_time=ct[0], max_crossing_time=ct[1]
                )
            sl = splat.Slicer(nside=None, missing=0, ra=np.degrees(metric.ra), dec=np.degrees(metric.dec))

            mic_array, info = sl(visits_array, metric, info=info)

            ph = splat.PlotHealbin(info=info)

            summary_stats.append(splat.gen_summary_row(info, "Mean Microlensing ", np.mean(mic_array)))
            summary_stats.append(
                splat.gen_summary_row(
                    info, "Uncert in Mean Microlensing ", np.mean(mic_array) / np.sqrt(np.sum(mic_array))
                )
            )
            fig = ph(
                np.degrees(metric.ra),
                np.degrees(metric.dec),
                mic_array,
                unit="Microlensing Crossing %i-%i days Detected (1,0)" % (ct[0], ct[1]),
            )
            self.fig_saver(fig, info=info)

        summary_dataframe = pd.DataFrame.from_dict(summary_stats)
        return [mic_array], [info], summary_dataframe
