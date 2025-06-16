__all__ = ("color_slope_batch",)


import numpy as np

import rubin_sim.maf_proto as maf


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

    visits_array, df, run_name, subset, fig_saver = maf.batch_preamble(
        observations=observations,
        run_name=run_name,
        quick_test=quick_test,
        fig_saver=fig_saver,
    )

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
        info["observations_subset"] = subset
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
