__all__ = ("glance",)

import copy
import sqlite3
from os.path import basename

import numpy as np
import pandas as pd

import rubin_sim.maf_proto as maf
from rubin_sim.data import get_baseline


def glance(path=None, quick_test=False):

    if path is None:
        path = get_baseline()
    run_name = basename(path).replace(".db", "")
    con = sqlite3.connect(path)
    # Dataframe is handy for some calcs
    if quick_test:
        df = pd.read_sql("select * from observations where night < 61;", con)
    else:
        df = pd.read_sql("select * from observations;", con)

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
            # XXX--send fig to dir and db at this point if desired
            # Do whatever stats we want on the hp_array
            for stat in stats_to_run:
                summary_stats.append(maf.gen_summary_row(info, stat, stats_to_run[stat](hp_array)))

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

    # ----------------
    # Roll check
    # ----------------

    year_subs = {"year 1": np.where(visits_array["night"] < 365)[0]}
    v1 = 365.25 * 2.5
    v2 = 365.25 * 3.5
    year_subs["year 2.5-3.5"] = np.where((v1 < visits_array["night"]) & (visits_array["night"] < v2))[0]
    v1 = 365.25 * 3.5
    v2 = 365.25 * 4.5
    year_subs["year 3.5-4.5"] = np.where((v1 < visits_array["night"]) & (visits_array["night"] < v2))[0]

    for subset in year_subs:
        info = {"run_name": run_name}
        info["observations_subset"] = subset
        sub_data = visits_array[year_subs[subset]]
        metric = maf.CountMetric()
        sl = maf.Slicer(nside=64)
        hp_array, info = sl(sub_data, metric, info=info)
        pm = maf.PlotMoll(info=info)
        fig = pm(hp_array, norm="log")

    # ----------------
    # Slew stats
    # ----------------
    info = {"run_name": run_name}
    info["metric: unit"] = "Slew Time (sec)"
    info["observations_subset"] = "all"
    # Maybe this is just a function and not a class
    ph = maf.PlotHist(info=info)
    fig = ph(visits_array["slewTime"], log=True)

    summary_stats.append(maf.gen_summary_row(info, "max", np.max(visits_array["slewTime"])))
    summary_stats.append(maf.gen_summary_row(info, "min", np.min(visits_array["slewTime"])))
    summary_stats.append(maf.gen_summary_row(info, "mean", np.mean(visits_array["slewTime"])))
    summary_stats.append(maf.gen_summary_row(info, "median", np.median(visits_array["slewTime"])))

    info = {"run_name": run_name}
    info["metric: unit"] = "Slew Distance (deg)"
    ph = maf.PlotHist(info=info)
    fig = ph(visits_array["slewDistance"], log=True)

    # Open Shutter Fraction overall
    info = {"run_name": run_name}
    info["metric: unit"] = "Open Shutter Fraction"
    info["observations_subset"] = "all"
    osf = maf.open_shutter_fraction(visits_array["observationStartMJD"], visits_array["visitExposureTime"])
    summary_stats.append(maf.gen_summary_row(info, "OSF", osf))

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

    # let's make an hourglass per year
    info = {"run_name": run_name}
    hstart = np.arange(10)
    hend = hstart + 1

    for start, end in zip(hstart, hend):
        indx = np.where((visits_array["night"] < end * 365.25) & (visits_array["night"] >= start * 365.25))
        info["observations_subset"] = "year %i to %i" % (start, end)
        hr = maf.PlotHourglass(info=info)
        fig = hr(visits_array[indx])

    return summary_stats
