__all__ = ("color_slope_batch",)

import sqlite3
from os.path import basename

import numpy as np
import pandas as pd

import rubin_sim.maf_proto as maf
from rubin_sim.data import get_baseline


class DoNothing:
    def __call__(*args, **kwargs):
        pass


def color_slope_batch(observations=None, run_name=None, quick_test=False, fig_saver=None, nside=64):
    """Run some glance metrics

    Parameters
    ----------
    data : `str` or `pandas.DataFrame`
        Path to database file or . If None, grabs the current baseline
        simulation
    run_name : `str`
        run name to use. Default None pulls name from database file.
    quick_test : `bool`
        If True, grabs just the first 61 days of observations.
        Default False.
    fig_saver : `rubin_sim.maf_proto.db.FigSaver`
        Class that takes matplotlib.Figure objects and saves them.
        Default None.
    """
    if fig_saver is None:
        fig_saver = DoNothing()

    if observations is None:
        observations = get_baseline()
    if run_name is None:
        run_name = basename(observations).replace(".db", "")

    if isinstance(observations, str):
        con = sqlite3.connect(observations)
        # Dataframe is handy for some calcs
        if quick_test:
            df = pd.read_sql("select * from observations where night < 61;", con)
        else:
            df = pd.read_sql("select * from observations;", con)
    else:
        df = observations

    # But mostly want numpy array for speed.
    visits_array = df.to_records(index=False)
    con.close()

    summary_stats = []

    stats_to_run = {"mean": np.nanmean, "sum": np.nansum}
    sl = maf.Slicer(nside=nside)

    metrics = []
    metrics.append(maf.ColorSlopeMetric())
    metrics.append(maf.ColorSlope2NightMetric())

    infos = []
    plot_dicts = []
    for m in metrics:
        info = maf.empty_info()
        info["run_name"] = run_name
        info["observations_subset"] = "all"
        infos.append(info)
        plot_dicts.append({})

    hp_arrays, infos = sl(visits_array, metrics, info=infos)

    for hp_array, metric, info, plot_dict in zip(hp_arrays, metrics, infos, plot_dicts):
        pm = maf.PlotMoll(info=info)
        fig = pm(hp_array, **plot_dict)
        fig_saver(fig, info=info)
        # Do whatever stats we want on the hp_array
        for stat in stats_to_run:
            summary_stats.append(maf.gen_summary_row(info, stat, stats_to_run[stat](hp_array)))

    return summary_stats
