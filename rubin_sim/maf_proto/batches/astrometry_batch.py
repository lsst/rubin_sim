__all__ = ("astrometry_batch",)

import numpy as np

import rubin_sim.maf_proto as maf


def astrometry_batch(observations=None, run_name=None, quick_test=False, fig_saver=None):
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

    # Add any new columns we need
    ra_pi_amp, dec_pi_amp = maf.parallax_amplitude(
        df["fieldRA"].values, df["fieldDec"].values, df["observationStartMJD"].values, degrees=True
    )

    df["ra_pi_amp"] = ra_pi_amp
    df["dec_pi_amp"] = dec_pi_amp

    ra_dcr_amp, dec_dcr_amp = maf.dcr_amplitude(
        90.0 - df["altitude"].values, df["paraAngle"].values, df["filter"].values, degrees=True
    )
    df["ra_dcr_amp"] = ra_dcr_amp
    df["dec_dcr_amp"] = dec_dcr_amp

    # But mostly want numpy array for speed.
    visits_array = df.to_records(index=False)

    summary_stats = []

    stats_to_run = {"mean": np.nanmean, "median": np.nanmedian}
    sl = maf.Slicer(nside=64)

    metrics = []
    metrics.append(maf.ParallaxMetric())
    metrics.append(maf.ProperMotionMetric())
    metrics.append(maf.ParallaxCoverageMetric())
    metrics.append(maf.ParallaxDcrDegenMetric())

    infos = []
    plot_dicts = []
    for m in metrics:
        info = maf.empty_info()
        info["run_name"] = run_name
        info["observations_subset"] = "all"
        infos.append(info)
        plot_dicts.append({})

    # Set the first two metrics to get caught and
    # precentile clip the plotted value range.
    plot_dicts[0]["min"] = "percentile"
    plot_dicts[1]["min"] = "percentile"

    hp_arrays, infos = sl(visits_array, metrics, info=infos)

    for hp_array, metric, info, plot_dict in zip(hp_arrays, metrics, infos, plot_dicts):
        if "min" in plot_dict.keys():
            if plot_dict["min"] == "percentile":
                min_val, max_val = maf.percentile_clipping(hp_array[np.isfinite(hp_array)])
                plot_dict["min"] = min_val
                plot_dict["max"] = max_val

        pm = maf.PlotMoll(info=info)
        fig = pm(hp_array, **plot_dict)
        fig_saver(fig, info=info)
        # Do whatever stats we want on the hp_array
        for stat in stats_to_run:
            summary_stats.append(maf.gen_summary_row(info, stat, stats_to_run[stat](hp_array)))

    return summary_stats
