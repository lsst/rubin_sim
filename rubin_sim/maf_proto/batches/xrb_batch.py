__all__ = ("xrb_batch",)

import sqlite3
from os.path import basename

import numpy as np
import pandas as pd

import rubin_sim.maf_proto as maf
from rubin_sim.data import get_baseline


class DoNothing:
    def __call__(*args, **kwargs):
        pass


def xrb_batch(observations=None, run_name=None, quick_test=False, fig_saver=None):
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

    summary_stats = []

    info = maf.empty_info()
    info["run_name"] = run_name
    info["observations_subset"] = subset
    mjd0 = np.min(visits_array["observationStartMJD"])
    metric = maf.XRBPopMetric(mjd0=mjd0)
    if quick_test:
        metric.generate_xrb_pop(n_events=100)
    else:
        metric.generate_xrb_pop(n_events=10000)
    sl = maf.Slicer(nside=None, missing=0, ra=np.degrees(metric.ra), dec=np.degrees(metric.dec))

    xrb_array, info = sl(visits_array, metric, info=info)

    ph = maf.PlotHealbin(info=info)

    for key in xrb_array.dtype.names:
        summary_stats.append(maf.gen_summary_row(info, "Mean XRB " + key, np.mean(xrb_array[key])))
        summary_stats.append(
            maf.gen_summary_row(
                info,
                "Uncert in Mean XRB " + key,
                np.nanmean(xrb_array[key]) / np.sqrt(np.sum(xrb_array[key])),
            )
        )

    fig = ph(
        np.degrees(metric.ra), np.degrees(metric.dec), xrb_array["early_detect"], unit="XRB mean early_detect"
    )
    fig_saver(fig, info=info)

    return summary_stats
