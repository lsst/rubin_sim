__all__ = ("run_moving_fractions",)

import argparse
import glob
import os

import numpy as np
from rubin_scheduler.utils import SURVEY_START_MJD

from . import batches as batches
from . import db as db
from . import metricBundles as mmB


def run_moving_fractions():
    """Calculate completeness and fractions for moving object metrics."""
    parser = argparse.ArgumentParser(description="Run moving object metrics for a particular opsim run.")
    parser.add_argument(
        "--work_dir",
        type=str,
        default=".",
        help="Output (and input) directory for moving object metrics. Default '.'.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Select only files matching this metadata string. Default None (all files).",
    )
    parser.add_argument(
        "--h_mark",
        type=float,
        default=None,
        help="H value at which to calculate cumulative/differential completeness, etc."
        "Default (None) will be set to plot_dict value or median of H range.",
    )
    parser.add_argument(
        "--n_years_max",
        type=int,
        default=10,
        help="Maximum number of years out to which to evaluate completeness." "Default 10.",
    )
    parser.add_argument(
        "--start_time",
        type=float,
        default=None,
        help="Time at start of survey (to set time for summary metrics).",
    )
    args = parser.parse_args()

    # Default parameters for metric setup.
    if args.start_time is None:
        start_time = SURVEY_START_MJD
    else:
        start_time = args.start_time
    stepsize = 365 / 6.0
    times = np.arange(0, args.n_years_max * 365 + stepsize / 2, stepsize)
    times += start_time

    # Create a results Db.
    results_db = db.ResultsDb(out_dir=args.work_dir)

    # Just read in all metrics in the (joint or single) directory,
    # then run completeness and fraction
    # summaries, using the methods in the batches.
    if args.metadata is None:
        matchstring = os.path.join(args.work_dir, "*MOOB.npz")
    else:
        matchstring = os.path.join(args.work_dir, f"*{args.metadata}*MOOB.npz")
    metricfiles = glob.glob(matchstring)
    metric_names = []
    for m in metricfiles:
        mname = os.path.split(m)[-1].replace("_MOOB.npz", "")
        metric_names.append(mname)

    bdict = {}
    for mName, mFile in zip(metric_names, metricfiles):
        bdict[mName] = mmB.create_empty_mo_metric_bundle()
        bdict[mName].read(mFile)

    first = bdict[metric_names[0]]
    figroot = f"{first.run_name}"
    # adding args.metadata to the figroot means that we can narrow
    # down *which* subset of files to work on
    # this lets us add different h_mark values to the output plots,
    # if desired (leaving None/default is ok)
    if args.metadata != ".":
        figroot += f"_{args.metadata}"

    # Calculate completeness. This utility writes these to disk.
    bdict_completeness = batches.run_completeness_summary(
        bdict, args.h_mark, times, args.work_dir, results_db
    )

    # Plot some of the completeness results.
    batches.plot_completeness(
        bdict_completeness,
        figroot=figroot,
        results_db=results_db,
        out_dir=args.work_dir,
    )

    # Calculate fractions of population for characterization.
    # This utility writes these to disk.
    bdict_fractions = batches.run_fraction_summary(bdict, args.h_mark, args.work_dir, results_db)
    # Plot the fractions for colors and lightcurves.
    batches.plot_fractions(bdict_fractions, figroot=figroot, results_db=results_db, out_dir=args.work_dir)

    # Plot nObs and arcLength.
    for k in bdict:
        if "NObs" in k:
            batches.plot_single(bdict[k], results_db=results_db, out_dir=args.work_dir)
        if "ObsArc" in k:
            batches.plot_single(bdict[k], results_db=results_db, out_dir=args.work_dir)

    # Plot the number of chances of discovery metric -
    # this is different than completeness
    # As it plots the metric value directly
    for k in bdict:
        if "DiscoveryNChances" in k and "3_pairs_in_15_nights_detection_loss" in k:
            batches.plot_single(bdict[k], results_db=results_db, out_dir=args.work_dir)
        if "MagicDiscovery" in k:
            batches.plot_single(bdict[k], results_db=results_db, out_dir=args.work_dir)
        if "HighVelocity" in k:
            batches.plot_single(bdict[k], results_db=results_db, out_dir=args.work_dir)

    # Plot likelihood of detecting activity.
    batches.plot_activity(bdict, figroot=figroot, results_db=results_db, out_dir=args.work_dir)
