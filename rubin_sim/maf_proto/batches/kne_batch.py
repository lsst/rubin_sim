__all__ = ("kne_batch",)


import numpy as np

import rubin_sim.maf_proto as maf


def kne_batch(observations=None, run_name=None, quick_test=False, fig_saver=None):
    """KNe batch"""

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
    metric = maf.KNePopMetric(mjd0=mjd0)
    if quick_test:
        metric.generate_kn_pop(n_events=100, d_min=10, d_max=600)
    else:
        metric.generate_kn_pop(n_events=500000, d_min=10, d_max=600)
    sl = maf.Slicer(nside=None, missing=0, ra=np.degrees(metric.ra), dec=np.degrees(metric.dec))

    kn_array, info = sl(visits_array, metric, info=info)

    ph = maf.PlotHealbin(info=info)

    for key in kn_array.dtype.names:
        summary_stats.append(maf.gen_summary_row(info, "Mean KNe " + key, np.mean(kn_array[key])))
        summary_stats.append(
            maf.gen_summary_row(
                info, "Uncert in Mean KNe " + key, np.mean(kn_array[key]) / np.sqrt(np.sum(kn_array[key]))
            )
        )

    fig = ph(
        np.degrees(metric.ra), np.degrees(metric.dec), kn_array["multi_detect"], unit="KNe mean multi_detect"
    )
    fig_saver(fig, info=info)

    return summary_stats
