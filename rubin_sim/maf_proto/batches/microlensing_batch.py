__all__ = ("microlensing_batch",)

import sqlite3
from os.path import basename

import numpy as np
import pandas as pd

import rubin_sim.maf_proto as maf
from rubin_sim.data import get_baseline


class DoNothing:
    def __call__(*args, **kwargs):
        pass


def microlensing_batch(observations=None, run_name=None, quick_test=False, fig_saver=None):
    if fig_saver is None:
        fig_saver = DoNothing()

    if observations is None:
        observations = get_baseline()
    if run_name is None:
        run_name = basename(observations).replace(".db", "")

    if isinstance(observations, str):
        con = sqlite3.connect(observations)
        # Dataframe is handy for some calcs
        and_string = "scheduler_note not like 'DD%'"
        if quick_test:
            df = pd.read_sql("select * from observations where night < 365 and %s;" % and_string, con)
            subset = "night < 365"
        else:
            df = pd.read_sql("select * from observations where %s;" % and_string, con)
            subset = ""
    else:
        df = observations
    # But mostly want numpy array for speed.
    visits_array = df.to_records(index=False)
    con.close()

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
        info = maf.empty_info()
        info["run_name"] = run_name
        info["observations_subset"] = subset
        mjd0 = np.min(visits_array["observationStartMJD"])
        metric = maf.MicrolensingMetric(mjd0=mjd0, name="Microlensing, crossing %i-%i days" % (ct[0], ct[1]))
        if quick_test:
            metric.generate_microlensing_events(n_events=20, min_crossing_time=ct[0], max_crossing_time=ct[1])
        else:
            metric.generate_microlensing_events(
                n_events=10000, min_crossing_time=ct[0], max_crossing_time=ct[1]
            )
        sl = maf.Slicer(nside=None, missing=0, ra=np.degrees(metric.ra), dec=np.degrees(metric.dec))

        mic_array, info = sl(visits_array, metric, info=info)

        ph = maf.PlotHealbin(info=info)

        summary_stats.append(maf.gen_summary_row(info, "Mean Microlensing ", np.mean(mic_array)))
        summary_stats.append(
            maf.gen_summary_row(
                info, "Uncert in Mean Microlensing ", np.mean(mic_array) / np.sqrt(np.sum(mic_array))
            )
        )

        fig = ph(
            np.degrees(metric.ra),
            np.degrees(metric.dec),
            mic_array,
            unit="Microlensing Crossing %i-%i days Detected (1,0)" % (ct[0], ct[1]),
        )
        fig_saver(fig, info=info)

    return summary_stats
