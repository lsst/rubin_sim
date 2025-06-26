__all__ = ("combine_result_dbs", "gather_summaries")

import argparse
import glob
import logging
import os
import sqlite3

import pandas as pd

logger = logging.getLogger(__name__)


def combine_result_dbs(run_dirs, dbfilename="resultsDb_sqlite.db"):
    """Helper function for gather_summaries

    Parameters
    ----------
    run_dirs : `list`  [`str`]
        A list of directories to search for MAF result databases.
    dbfilename : `str`
        The database filename to look for (default: resultsDb_sqlite.db).
    """

    # query to grab all the summary stats
    sql_q = "SELECT summarystats.summary_value, "
    sql_q += "metrics.metric_name, metrics.metric_info_label, "
    sql_q += "metrics.slicer_name, summarystats.summary_name, metrics.run_name "
    sql_q += "FROM summarystats INNER JOIN metrics ON metrics.metric_id=summarystats.metric_id"

    all_summaries = []
    for rdir in run_dirs:
        fname = os.path.join(rdir, dbfilename)
        if not os.path.isfile(fname):
            logger.warning(f"No resultsDb database in {rdir}")

        con = sqlite3.connect(fname)
        temp_df = pd.read_sql(sql_q, con)
        con.close()

        # Make column names
        def make_summary_name(x):
            summary_name = " ".join(
                [
                    x.summary_name.strip(),
                    x.metric_name.strip(),
                    x.metric_info_label.strip(),
                    x.slicer_name.strip(),
                ]
            )
            summary_name = summary_name.replace("  ", " ")
            return summary_name

        temp_df["summary_names"] = temp_df.apply(make_summary_name, axis=1)
        all_summaries.append(temp_df[["summary_names", "summary_value", "run_name"]])

    # Make one big dataframe
    all_summaries = pd.concat(all_summaries)
    # Group by run names and drop duplicates
    g = all_summaries.groupby(["run_name", "summary_names"]).agg({"summary_value": "last"})
    # Convert to one row with all summary stats per run
    result_df = g.reset_index("summary_names").pivot(columns="summary_names")
    # That ended up as a MultiIndex which we didn't need, so fix and rename
    result_df.columns = result_df.columns.droplevel(0).rename("metric")
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
        "--to_csv",
        dest="to_csv",
        action="store_true",
        help="Create a .csv file, instead of the default hdf file.",
    )
    parser.add_argument("--to_hdf", dest="to_hdf", action="store_true")
    parser.add_argument(
        "--dirs",
        type=str,
        default=None,
        help="comma separated list of directories to use, default None",
    )

    args = parser.parse_args()
    if args.dirs is None:
        run_dirs = glob.glob(args.base_dir + "/*/")
    else:
        run_dirs = args.dirs.split(",")

    # Create output file name if needed
    if args.to_csv:
        outfile = args.outfile + ".csv"
    else:
        outfile = args.outfile + ".h5"

    result_df = combine_result_dbs(run_dirs)

    # Save summary statistics
    if args.to_csv:
        result_df.to_csv(outfile)
    else:
        # Create a CSV file
        result_df.to_hdf(outfile, key="stats")
