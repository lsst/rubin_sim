__all__ = ("bd_batch",)

import numpy as np

import rubin_sim.maf_proto as maf


def bd_batch(observations=None, run_name=None, quick_test=False, fig_saver=None):
    """Brown Dwarf batch.

    Parameters
    ----------
    observations : `str` or `pandas.DataFrame`
        Path to database file or dataframe. If None, grabs the current baseline
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

    visits_array = df.to_records(index=False)

    summary_stats = []

    stats_to_run = {"mean": np.nanmean, "Volume": maf.hp_sum_volume}
    sl = maf.Slicer(nside=64)

    metrics = []
    l7_bd_mags = {"i": 20.09, "z": 18.18, "y": 17.13}
    metrics.append(maf.BDParallaxMetric(mags=l7_bd_mags, name="Brown Dwarf, L7"))

    l4_bd_mags = {"i": 18.35, "z": 16.68, "y": 15.66}
    metrics.append(maf.BDParallaxMetric(mags=l4_bd_mags, name="Brown Dwarf, L4"))

    infos = []
    plot_dicts = []
    for m in metrics:
        info = maf.empty_info()
        info["run_name"] = run_name
        info["observations_subset"] = subset
        infos.append(info)
        plot_dicts.append({})

    hp_arrays, infos = sl(visits_array, metrics, info=infos)

    for hp_array, metric, info in zip(hp_arrays, metrics, infos):

        pm = maf.PlotMoll(info=info)
        fig = pm(hp_array)
        fig_saver(fig, info=info)
        for stat in stats_to_run:
            row = maf.gen_summary_row(info, stat, stats_to_run[stat](hp_array))
            if stat == "Volume":
                row["metric: unit"] = "pc^3"
            summary_stats.append(row)

    return summary_stats
