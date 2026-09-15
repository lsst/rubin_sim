__all__ = ("agn_batch",)

import rubin_sim.splat as splat
import numpy as np


def agn_batch(observations=None, run_name=None, quick_test=False, fig_saver=None):

    if observations is None:
        visits_array, df, run_name, subset, fig_saver = splat.batch_preamble(
            observations=observations,
            run_name=run_name,
            quick_test=quick_test,
            fig_saver=fig_saver,
        )

    # Need to add saturation magnitude
    saturation_mag = splat.saturation_limit(visits_array)
    df["saturation_mag"] = saturation_mag

    # But mostly want numpy array for speed.
    visits_array = df.to_records(index=False)

    sl = splat.Slicer(nside=64)

    for band in "ugrizy":
        metrics = []
        metrics.append(splat.QSONumberCountsMetric(band))
        metrics.append(splat.SFUncertMetric())
        metrics.append(splat.AgnTimeLagMetric(lag=100, unit="AGN Time Lag 100"))
        metrics.append(splat.AgnTimeLagMetric(lag=5, unit="AGN Time Lag 5"))

        indx = np.where(visits_array["band"] == band)
        sub_data = visits_array[indx]

        infos = []
        plot_dicts = []
        for m in metrics:
            info = splat.empty_info()
            info["data_source"] = run_name
            info["observations_subset"] = band
            infos.append(info)
            plot_dicts.append({})

        hp_arrays, infos = sl(sub_data, metrics, info=infos)

        for hp_array, metric, info, plot_dict in zip(hp_arrays, metrics, infos, plot_dicts):
            pm = splat.PlotMoll(info=info)
            fig = pm(hp_array, **plot_dict)
            fig_saver(fig, info=info)
