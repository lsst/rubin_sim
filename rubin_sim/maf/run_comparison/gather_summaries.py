#!/usr/bin/env python

import glob
import argparse

from .runComparison import RunComparison


def gather_summaries():
    """Find resultsDbs in a series of directories and gather up their summary
    stats into a single CSV or hdf5 file. Intended to run on a set of metrics
    run on multiple simulations, so that each resultsDb has similar summary
    statistics.
    """

    parser = argparse.ArgumentParser(
        description="Find resultsDbs in a series of directories and "
        "gather up their summary stats into a single CSV or hdf5 file. "
        "Intended to run on a set of metrics run on multiple "
        "simulations, so that each resultsDb has similar summary"
        "statistics."
    )
    parser.add_argument(
        "--baseDir",
        type=str,
        default=".",
        help="Root directory from where to search for MAF (sub)directories.",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default=None,
        help="Output file name. Default (None) will create a file = [suffix_]summary.csv)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Suffix for directories within which to find resultsDbs. "
        "Default is None, which searches for all potential MAF directories.",
    )
    parser.add_argument(
        "--to_hdf",
        dest="to_hdf",
        action="store_true",
        help="Create a .hdf5 file, instead of the default csv file.",
    )
    args = parser.parse_args()

    # Identify a subset of MAF directories if suffix is set
    if args.suffix is None:
        run_dirs = None
    else:
        run_dirs = glob.glob(f"*{args.suffix}")

    # Create output file name if needed
    if args.outfile is None:
        if args.suffix is None:
            outfile = "summary"
        else:
            outfile = f"{args.suffix.replace('_', '')}_summary"
        if args.to_hdf:
            outfile = outfile + ".h5"
        else:
            outfile = outfile + ".csv"
    else:
        outfile = args.outfile

    # Connect to resultsDbs and pull summary stats into a nice Dataframe
    rc = RunComparison(baseDir=args.baseDir, run_dirs=run_dirs)
    print(f"Found directories {rc.runDirs}")
    mdict = rc.buildMetricDict()
    print(f"And found {len(mdict)} metrics.")
    rc.addSummaryStats(mdict)

    # Save summary statistics
    if args.to_hdf:
        rc.summaryStats.to_hdf(outfile, key="stats")
    else:
        # Create a CSV file
        rc.summaryStats.to_csv(outfile)
