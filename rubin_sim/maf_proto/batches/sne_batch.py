__all__ = ("sne_batch",)

import numpy as np

import rubin_sim.maf_proto as maf

DAYS_IN_YEAR = 365.25


def sne_batch(
    observations=None, run_name=None, quick_test=False, fig_saver=None, nside=16, quick_night_limit=61
):
    visits_array, df, run_name, subset, fig_saver = maf.batch_preamble(
        observations=observations,
        run_name=run_name,
        quick_test=quick_test,
        fig_saver=fig_saver,
    )

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
