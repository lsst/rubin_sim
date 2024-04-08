__all__ = ("combine_result_dbs", "gather_summaries")

import argparse
import glob
import os
import sqlite3

import numpy as np
import pandas as pd


def dirname_to_runname(inpath, replaces=["_glance", "_sci", "_meta", "_ss", "_ddf"]):
    """Given a directory path, construct a plausible runname"""
    result = os.path.basename(os.path.normpath(inpath))
    for rstring in replaces:
        result = result.replace(rstring, "")
    return result


def combine_result_dbs(run_dirs, dbfilename="resultsDb_sqlite.db"):
    """Helper function for gather_summaries

    Parameters
    ----------
    run_dirs : list of str
        A list of directories to search for MAF result databases.
    dbfilename : str (resultsDb_sqlite.db)
        The database filename to look for (default: resultsDb_sqlite.db)."""

    db_files = []
    run_names = []
    for dname in run_dirs:
        fname = os.path.join(dname, dbfilename)
        if os.path.isfile(fname):
            db_files.append(fname)
            run_names.append(dirname_to_runname(dname))

    # querry to grab all the summary stats
    sql_q = "SELECT summarystats.summary_value, "
    sql_q += "metrics.metric_name, metrics.metric_info_label, "
    sql_q += "metrics.slicer_name, summarystats.summary_name "
    sql_q += "FROM summarystats INNER JOIN metrics ON metrics.metric_id=summarystats.metric_id"

    rows = []
    for row_name, fname in zip(run_names, db_files):
        con = sqlite3.connect(fname)
        temp_df = pd.read_sql(sql_q, con)
        con.close()

        # Make column names
        col_names = []
        for summary_name, metric_name, metric_info_label, slicer_name in zip(
            temp_df["summary_name"].values.tolist(),
            temp_df["metric_name"].values.tolist(),
            temp_df["metric_info_label"].values.tolist(),
            temp_df["slicer_name"].values.tolist(),
        ):
            col_name = " ".join(
                [
                    summary_name.strip(),
                    metric_name.strip(),
                    metric_info_label.strip(),
                    slicer_name.strip(),
                ]
            )
            col_names.append(col_name.replace("  ", " "))

        # Make a DataFrame row
        row = pd.DataFrame(
            temp_df["summary_value"].values.reshape([1, temp_df["summary_value"].values.size]),
            columns=col_names,
            index=[row_name],
        )

        # Can have duplicate columns if MAF was run multiple times.
        # Remove duplicates:
        # https://stackoverflow.com/questions/14984119/
        # python-pandas-remove-duplicate-columns
        row = row.loc[:, ~row.columns.duplicated()].copy()

        rows.append(row)
    # Create final large DataFrame to hold everything
    all_cols = np.unique(np.concatenate([r.columns.values for r in rows]))
    u_names = np.unique(run_names)
    result_df = pd.DataFrame(
        np.zeros([u_names.size, all_cols.size]) + np.nan,
        columns=all_cols,
        index=u_names,
    )

    # Put each row into the final DataFrame
    for row_name, row in zip(run_names, rows):
        result_df.loc[row_name, row.columns] = np.ravel(row.values)

    return result_df


def gather_summaries():
    """Find resultsDbs in a series of directories and gather up their summary
    stats into a single CSV or hdf5 file. Outputs one row per unique run name.
    """

    parser = argparse.ArgumentParser(
        description="Find resultsDbs in a series of directories and "
        "gather up their summary stats into a single CSV or hdf5 file. "
        "Intended to run on a set of metrics run on multiple "
        "simulations, so that each results_db has similar summary"
        "statistics."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=".",
        help="Root directory from where to search for MAF (sub)directories.",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="summary",
        help="Output file name. Default (summary)",
    )
    parser.add_argument(
        "--to_hdf",
        dest="to_hdf",
        action="store_true",
        help="Create a .hdf5 file, instead of the default csv file.",
    )
    parser.add_argument(
        "--dirs",
        type=str,
        default=None,
        help="comma seperated list of directories to use, default None",
    )

    args = parser.parse_args()
    if args.dirs is None:
        run_dirs = glob.glob(args.base_dir + "/*/")
    else:
        run_dirs = args.dirs.split(",")

    # Create output file name if needed
    if args.to_hdf:
        outfile = args.outfile + ".h5"
    else:
        outfile = args.outfile + ".csv"

    result_df = combine_result_dbs(run_dirs)

    # Save summary statistics
    if args.to_hdf:
        result_df.to_hdf(outfile, key="stats")
    else:
        # Create a CSV file
        result_df.to_csv(outfile)
