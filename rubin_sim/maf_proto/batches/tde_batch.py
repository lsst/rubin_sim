__all__ = ("tde_batch",)


import numpy as np

import rubin_sim.maf_proto as maf


def tde_batch(observations=None, run_name=None, quick_test=False, fig_saver=None):
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
    mjd0 = np.min(visits_array["observationStartMJD"])
    metric = maf.TdePopMetric(mjd0=mjd0)
    if quick_test:
        metric.generate_tde_pop(n_events=100)
    else:
        metric.generate_tde_pop(n_events=10000)
    sl = maf.Slicer(nside=None, missing=0, ra=np.degrees(metric.ra), dec=np.degrees(metric.dec))

    tde_array, info = sl(visits_array, metric, info=info)

    ph = maf.PlotHealbin(info=info)

    for key in tde_array.dtype.names:
        summary_stats.append(maf.gen_summary_row(info, "Mean TDE " + key, np.mean(tde_array[key])))
        summary_stats.append(
            maf.gen_summary_row(
                info,
                "Uncert in Mean XRB " + key,
                np.nanmean(tde_array[key]) / np.sqrt(np.sum(tde_array[key])),
            )
        )

    fig = ph(np.degrees(metric.ra), np.degrees(metric.dec), tde_array["pre_peak"], unit="TDE mean pre_peak")
    fig_saver(fig, info=info)

    return summary_stats
