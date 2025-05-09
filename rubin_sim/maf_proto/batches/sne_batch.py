__all__ = ("sne_batch",)

import sqlite3
from os.path import basename

import numpy as np
import pandas as pd

import rubin_sim.maf_proto as maf
from rubin_sim.data import get_baseline

DAYS_IN_YEAR = 365.25


class DoNothing:
    def __call__(*args, **kwargs):
        pass


def sne_batch(
    observations=None, run_name=None, quick_test=False, fig_saver=None, nside=16, quick_night_limit=61
):
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
            df = pd.read_sql(
                "select * from observations where night < %i and %s;" % (quick_night_limit, and_string), con
            )
            subset = "night < %i and scheduler_note not like DD"
        else:
            df = pd.read_sql("select * from observations where %s;" % and_string, con)
            subset = "scheduler_note not like DD"
    else:
        df = observations

    # But mostly want numpy array for speed.
    visits_array = df.to_records(index=False)
    con.close()

    summary_stats = []

    info = maf.empty_info()
    info["run_name"] = run_name
    info["observations_subset"] = subset
    sl = maf.Slicer(nside=nside)
    metric = maf.SNNSNMetric()

    sn_array, info = sl(visits_array, metric, info=info)

    pm = maf.PlotMoll(info=info)

    fig = pm(sn_array["n_sn"], unit="N SNe to z limit")
    fig_saver(fig, info=info)
    fig = pm(sn_array["zlim"], unit="z limit")
    fig_saver(fig, info=info)

    summary_stats.append(maf.gen_summary_row(info, "sum N SNe", np.nansum(sn_array["n_sn"])))
    summary_stats.append(maf.gen_summary_row(info, "mean z limit", np.nansum(sn_array["zlim"])))
    summary_stats.append(maf.gen_summary_row(info, "median z limit", np.nanmedian(sn_array["zlim"])))

    return summary_stats
