__all__ = ("glance",)

import copy
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


def glance(observations=None, run_name=None, quick_test=False, fig_saver=None):
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

    # Add any new columns we need
    df["t_eff_normed"] = maf.t_eff(
        df["fiveSigmaDepth"], df["filter"], exptime=df["visitExposureTime"], normed=True
    )

    # But mostly want numpy array for speed.
    visits_array = df.to_records(index=False)
    con.close()

    summary_stats = []

    # ----------------
    # Coadded depth and Number of obs per filter
    # ----------------

    stats_to_run = {"mean": np.nanmean, "median": np.nanmedian}
    sl = maf.Slicer(nside=64)

    # Things calculated per filter
    for i, filtername in enumerate("ugrizy"):
        info = {"run_name": run_name}
        info["observations_subset"] = "filter=%s" % filtername
        # select subset of data
        sub_data = visits_array[np.where(visits_array["filter"] == filtername)[0]]

        #
        metrics = []
        infos = []
        plot_dicts = []

        # Coadd depth
        metrics.append(maf.CoaddM5Metric(filtername))
        infos.append(copy.copy(info))
        plot_dicts.append({})

        # Number of visits
        metrics.append(maf.CountMetric())
        infos.append(copy.copy(info))
        plot_dicts.append({"log": True, "min": 1, "norm": "log", "cb_params": {"format": "%i", "n_ticks": 4}})

        # Run the metics through the slicer
        hp_arrays, infos = sl(sub_data, metrics, info=infos)

        # For each array we've generated, make a vis, summerize it how we like
        for hp_array, metric, info, plot_dict in zip(hp_arrays, metrics, infos, plot_dicts):
            pm = maf.PlotMoll(info=info)
            fig = pm(hp_array, **plot_dict)
            fig_saver(fig, info=info)
            # XXX--send fig to dir and db at this point if desired
            # Do whatever stats we want on the hp_array
            for stat in stats_to_run:
                summary_stats.append(maf.gen_summary_row(info, stat, stats_to_run[stat](hp_array)))

    # Number of obs, all filters
    info = {"run_name": run_name}
    info["observations_subset"] = "All"
    metric = maf.CountMetric()
    sl = maf.Slicer(nside=64)
    counts_hp_array, info = sl(visits_array, metric, info=info)
    pm = maf.PlotMoll(info=info)
    fig = pm(counts_hp_array, log=True, norm="log", cb_params={"format": "%i", "n_ticks": 4})
    fig_saver(fig, info=info)
    for stat in stats_to_run:
        summary_stats.append(maf.gen_summary_row(info, stat, stats_to_run[stat](counts_hp_array)))

    # Now make counts for FO, and make FO plot
    info = {"run_name": run_name}
    info["observations_subset"] = "Exptime > 19s"
    subset = visits_array[np.where(visits_array["visitExposureTime"] > 19)]
    metric = maf.CountMetric()
    sl = maf.Slicer(nside=64)
    counts_hp_array, info = sl(subset, metric, info=info)
    po = maf.PlotFo(info=info)
    fig = po(counts_hp_array)
    fig_saver(fig, info=info)
    # FO stats
    fo_stats = maf.fO_calcs(counts_hp_array)
    for key in fo_stats:
        summary_stats.append(maf.gen_summary_row(info, key, fo_stats[key]))

    # ----------------
    # Number of observations in alt,az
    # ----------------

    alt_az_sub = {
        "Near Sun Twilight Observations": np.char.find(
            visits_array["scheduler_note"].astype(str), "_near_sun"
        )
        != -1
    }
    for filtername in "ugrizy":
        alt_az_sub["filter=%s" % filtername] = np.where(visits_array["filter"] == filtername)
    alt_az_sub["all"] = np.arange(visits_array.size)
    alt_az_sub["blob_long"] = np.char.find(visits_array["scheduler_note"].astype(str), "blob_long") != -1
    # Starts with long
    alt_az_sub["long"] = np.char.find(visits_array["scheduler_note"].astype(str), "long") == 0
    alt_az_sub["ToO"] = np.char.find(visits_array["scheduler_note"].astype(str), "ToO") != -1

    for subset in alt_az_sub:
        info = {"run_name": run_name}
        info["observations_subset"] = subset
        sub_data = visits_array[alt_az_sub[subset]]
        metric = maf.CountMetric()
        sl = maf.Slicer(nside=64, lat_col="altitude", lon_col="azimuth")
        hp_array, info = sl(sub_data, metric, info=info)
        plam = maf.PlotLambert(info=info)
        fig = plam(hp_array)
        fig_saver(fig, info=info)

    # ----------------
    # Roll check
    # ----------------

    year_subs = {"year 1": np.where(visits_array["night"] < DAYS_IN_YEAR)[0]}
    v1 = DAYS_IN_YEAR * 2.5
    v2 = DAYS_IN_YEAR * 3.5
    year_subs["year 2.5-3.5"] = np.where((v1 < visits_array["night"]) & (visits_array["night"] < v2))[0]
    v1 = DAYS_IN_YEAR * 3.5
    v2 = DAYS_IN_YEAR * 4.5
    year_subs["year 3.5-4.5"] = np.where((v1 < visits_array["night"]) & (visits_array["night"] < v2))[0]

    for subset in year_subs:
        info = {"run_name": run_name}
        info["observations_subset"] = subset
        sub_data = visits_array[year_subs[subset]]
        metric = maf.CountMetric()
        sl = maf.Slicer(nside=64)
        hp_array, info = sl(sub_data, metric, info=info)
        pm = maf.PlotMoll(info=info)
        fig = pm(hp_array, max=140, min=5)
        fig_saver(fig, info=info)

    # ----------------
    # Slew stats
    # ----------------
    info = {"run_name": run_name}
    info["metric: unit"] = "Slew Time (sec)"
    info["observations_subset"] = "all"
    info["metric: name"] = "slew time"
    # Maybe this is just a function and not a class
    ph = maf.PlotHist(info=info)
    fig = ph(visits_array["slewTime"], log=True)
    fig_saver(fig, info=info)

    summary_stats.append(maf.gen_summary_row(info, "max", np.max(visits_array["slewTime"])))
    summary_stats.append(maf.gen_summary_row(info, "min", np.min(visits_array["slewTime"])))
    summary_stats.append(maf.gen_summary_row(info, "mean", np.mean(visits_array["slewTime"])))
    summary_stats.append(maf.gen_summary_row(info, "median", np.median(visits_array["slewTime"])))

    info = {"run_name": run_name}
    info["metric: unit"] = "Slew Distance (deg)"
    info["metric: name"] = "slewDistance"
    ph = maf.PlotHist(info=info)
    fig = ph(visits_array["slewDistance"], log=True)
    fig_saver(fig, info=info)

    # Open Shutter Fraction overall
    info = {"run_name": run_name}
    info["metric: unit"] = "Open Shutter Fraction"
    info["observations_subset"] = "all"
    osf = maf.open_shutter_fraction(visits_array["observationStartMJD"], visits_array["visitExposureTime"])
    summary_stats.append(maf.gen_summary_row(info, "OSF", osf))

    # Filter changes per night
    stats = {"mean": np.mean, "median": np.median, "max": np.max, "min": np.min}
    info = {"run_name": run_name}
    info["metric: name"] = "filter changes"
    info["caption"] = "Filter changes per night"
    fpn = df.groupby("night")["filter"].apply(maf.count_value_changes)
    pl = maf.PlotLine(info=info)
    fig = pl(fpn.index, fpn.values, xlabel="Night", ylabel="Filter Changes per Night")
    fig_saver(fig, info=info)
    for stat in stats:
        summary_stats.append(maf.gen_summary_row(info, stat, stats[stat](fpn)))

    # Open shutter fraction per night
    info = {"run_name": run_name}
    info["metric: name"] = "open shutter fraction"
    info["caption"] = "Open Shutter Fraction per night"
    osfpn = df.groupby("night").apply(maf.osf_visit_array, include_groups=False)
    pl = maf.PlotLine(info=info)
    fig = pl(osfpn.index, osfpn.values, xlabel="Night", ylabel="Open Shutter Fraction per Night")
    fig_saver(fig, info=info)

    # Effective exposure time per filter
    for filtername in "ugrizy":
        info = {"run_name": run_name}
        info["metric: unit"] = "T Effective (normed)"
        info["observations_subset"] = "filter=%s" % filtername
        indx = np.where(visits_array["filter"] == filtername)
        summary_stats.append(maf.gen_summary_row(info, "mean", np.mean(visits_array["t_eff_normed"][indx])))

    # Do some stats over each of the scheduler_note values
    # Could do this as a new column at the start instead.
    note_roots = np.array([val.split(",")[0] for val in visits_array["scheduler_note"]])
    n_obs = np.size(note_roots)
    for note in np.unique(note_roots):
        info = {"run_name": run_name}
        info["metric: unit"] = "#"
        info["observations_subset"] = "note start %s" % note
        count = np.size(np.where(note_roots == note)[0])
        summary_stats.append(maf.gen_summary_row(info, "count", count))
        info["metric: unit"] = "fraction"
        summary_stats.append(maf.gen_summary_row(info, "fraction", count / n_obs))

    # Hourglass per year
    info = {"run_name": run_name}
    min_year = np.floor(visits_array["night"].min() / DAYS_IN_YEAR)
    max_year = np.ceil(visits_array["night"].max() / DAYS_IN_YEAR)
    hstart = np.arange(min_year, max_year)
    hend = hstart + 1

    for start, end in zip(hstart, hend):
        indx = np.where(
            (visits_array["night"] < end * DAYS_IN_YEAR) & (visits_array["night"] >= start * DAYS_IN_YEAR)
        )
        info["observations_subset"] = "year %i to %i" % (start, end)
        info["metric: name"] = "Hourglass"
        hr = maf.PlotHourglass(info=info)
        fig = hr(visits_array[indx])
        fig_saver(fig, info=info)

    return summary_stats
