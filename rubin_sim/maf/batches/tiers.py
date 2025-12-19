import argparse
import sqlite3
from os.path import basename

import numpy as np
import pandas as pd

import rubin_sim.maf_proto as maf

if __name__ == "__main__":

    roman_range = [900, 3000]
    hour_max = 730
    altaz_nside = 64

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default=None)
    args = parser.parse_args()

    # Read in the database
    db_file = args.db
    run_name = basename(db_file).replace(".db", "")
    con = sqlite3.connect(db_file)
    df = pd.read_sql("select * from observations;", con)
    con.close()
    # Convert to a numpy array
    visits_array = df.to_records(index=False)

    data_selections = {
        "DDFs": ["DD:" in note and "RGES" not in note for note in visits_array["scheduler_note"]],
        "Pair 33": ["pair_33" in note for note in visits_array["scheduler_note"]],
        "Templates": ["template" in note for note in visits_array["scheduler_note"]],
        "Roman": ["RGES" in note for note in visits_array["scheduler_note"]],
        "Pair 15": ["pair_15" in note for note in visits_array["scheduler_note"]],
        "Greedy": ["greedy" in note for note in visits_array["scheduler_note"]],
        "ToO": ["ToO" in note for note in visits_array["scheduler_note"]],
        "Earth Interior": ["twilight" in note for note in visits_array["scheduler_note"]],
        "Initial Long": ["blob_long" in note for note in visits_array["scheduler_note"]],
        "Final Long": ["long" in note and "blob" not in note for note in visits_array["scheduler_note"]],
        "all": np.arange(visits_array.size),
    }

    fig_saver = maf.FigSaver(close_figs=False, results_file=run_name + "_tiers/maf_results.db", png_dpi=None)

    for key in data_selections:
        info = {"run_name": run_name}
        info["observations_subset"] = key

        # Count in RA,dec space
        sl = maf.Slicer()
        metric = maf.CountMetric()
        hp_array, info = sl(visits_array[data_selections[key]], metric, info=info)

        pm = maf.PlotMoll(info=info)
        fig = pm(hp_array)
        fig_saver(fig, info)

        # Count in alt,az space
        sl = maf.Slicer(lon_col="azimuth", lat_col="altitude", nside=altaz_nside)
        hp_array, info = sl(visits_array[data_selections[key]], metric, info=info)

        pl = maf.PlotLambert(info=info)
        fig = pl(hp_array)
        fig_saver(fig, info)

        # Hourglass plots for select times
        if key == "Roman":
            indx = np.where(
                (data_selections[key])
                & (visits_array["night"] > roman_range[0])
                & (visits_array["night"] < roman_range[1])
            )[0]
            info = {
                "run_name": run_name,
                "observations_subset": key + " %i < night < %i" % (roman_range[0], roman_range[1]),
                "metric: name": "Hourglass",
            }
        else:
            indx = np.where((data_selections[key]) & (visits_array["night"] < hour_max))[0]
            info = {
                "run_name": run_name,
                "observations_subset": key + " night < %i" % hour_max,
                "metric: name": "Hourglass",
            }
        hr = maf.PlotHourglassImage(info=info)
        fig = hr(visits_array[indx])
        fig_saver(fig, info=info)
