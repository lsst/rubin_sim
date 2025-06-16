__all__ = ("xrb_batch",)


import numpy as np

import rubin_sim.maf_proto as maf


def xrb_batch(observations=None, run_name=None, quick_test=False, fig_saver=None):
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
