#!/usr/bin/env python

import os
import argparse
import numpy as np

from . import db as db
from . import metricBundles as mmb
from . import batches as batches
from rubin_sim.utils import survey_start_mjd


def run_moving_calc():
    """Calculate metric values for an input population. Can be used on either
    a split or complete population. If running on a split population for later
    re-combining, use the complete set of orbits as the 'orbit_file'. Assumes
    you have already created the moving object observation files.
    """

    parser = argparse.ArgumentParser(
        description="Run moving object metrics for a particular opsim run."
    )
    parser.add_argument(
        "--orbit_file", type=str, help="File containing the moving object orbits."
    )
    parser.add_argument(
        "--objtype",
        type=str,
        default="",
        help="Object type, to set up default characterization/H value parameters "
        "(if they're not specified by flags) and to label outputs.",
    )
    parser.add_argument(
        "--obs_file",
        type=str,
        help="File containing the observations of the moving objects.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=".",
        help="Output directory for moving object metrics. Default '.'.",
    )
    parser.add_argument(
        "--pointings_db",
        type=str,
        default=None,
        help="Path and filename of opsim db, to write config* files to output directory."
        " Optional: if not provided, config* files won't be created but analysis will run.",
    )
    parser.add_argument(
        "--h_min",
        type=float,
        help="Minimum H value. " "If not set, defaults from objtype will be used.",
    )
    parser.add_argument(
        "--h_max",
        type=float,
        help="Maximum H value. " "If not set, defaults from objtype will be used.",
    )
    parser.add_argument(
        "--h_step",
        type=float,
        help="Stepsizes in H values. "
        "If not set, defaults from objtype will be used.",
    )
    parser.add_argument(
        "--h_mark",
        type=float,
        default=None,
        help="Add vertical lines at H=h_mark on plots. Default None.",
    )
    parser.add_argument(
        "--characterization",
        type=str,
        help="Inner/Outer solar system characterization. "
        "If unset, defaults from objtype will be used.",
    )
    parser.add_argument(
        "--constraint_info_label",
        type=str,
        default="",
        help="Metadata to add to the output files beyond objtype. Typically translation of "
        "the sql constraint into something more readable.",
    )
    parser.add_argument(
        "--constraint",
        type=str,
        default=None,
        help="(sql-style) constraint to apply to the solar system observations.",
    )
    parser.add_argument(
        "--albedo",
        type=float,
        default=None,
        help="Albedo value, to add diameters to upper scales on plots. Default None.",
    )

    parser.add_argument(
        "--n_years_max",
        type=int,
        default=10,
        help="Maximum number of years out to which to evaluate completeness."
        "Default 10.",
    )
    parser.add_argument(
        "--start_time",
        type=float,
        default=None,
        help="Time at start of survey (to set time for summary metrics).",
    )
    args = parser.parse_args()

    run_name = args.pointings_db.replace('.db', '')

    if args.orbit_file is None or args.obs_file is None:
        print("Must specify an orbit_file and an obs_file to calculate the metrics.")
        exit()

    # Get default H and other values:
    defaults = batches.ss_population_defaults(args.objtype)
    h_min = defaults["Hrange"][0]
    h_max = defaults["Hrange"][1]
    h_step = defaults["Hrange"][2]
    h_mark = defaults["h_mark"]
    characterization = defaults["char"]
    magtype = defaults["magtype"]
    # If H info is specified from parser, use that
    if args.h_min is not None:
        h_min = args.h_min
    if args.h_max is not None:
        h_max = args.h_max
    if args.h_step is not None:
        h_step = args.h_step
    if args.h_mark is not None:
        h_mark = args.h_mark
    if args.characterization is not None:
        characterization = args.characterization
    Hrange = np.arange(h_min, h_max + h_step, h_step)

    # Default parameters for metric setup.
    if args.start_time is None:
        start_time = survey_start_mjd()
    else:
        start_time = args.start_time

    stepsize = 365 / 2.0
    times = np.arange(0, args.n_years_max * 365 + stepsize / 2, stepsize)
    times += start_time

    # Set up resultsDb.
    if not (os.path.isdir(args.outDir)):
        try:
            os.makedirs(args.outDir)
        except FileExistsError:
            # This can happen if you are running these in parallel and two scripts try to make
            # the same directory.
            pass
    resultsDb = db.ResultsDb(out_dir=args.outDir)

    colmap = batches.ColMapDict()
    slicer = batches.setup_mo_slicer(args.orbit_file, Hrange, obs_file=args.obs_file)
    # Run discovery metrics using 'trailing' losses
    bdictT = batches.quickDiscoveryBatch(
        slicer,
        colmap=colmap,
        run_name=run_name,
        objtype=args.objtype,
        constraint_info_label=args.constraint_info_label,
        constraint=args.constraint,
        detection_losses="trailing",
        albedo=args.albedo,
        h_mark=h_mark,
    )
    # Run these discovery metrics
    print("Calculating quick discovery metrics with simple trailing losses.")
    bg = mmb.MoMetricBundleGroup(bdictT, out_dir=args.outDir, resultsDb=resultsDb)
    bg.runAll()

    # Run all discovery metrics using 'detection' losses
    bdictD = batches.quickDiscoveryBatch(
        slicer,
        colmap=colmap,
        run_name=run_name,
        objtype=args.objtype,
        constraint_info_label=args.constraint_info_label,
        constraint=args.constraint,
        detection_losses="detection",
        albedo=args.albedo,
        h_mark=h_mark,
    )
    bdict = batches.discoveryBatch(
        slicer,
        colmap=colmap,
        run_name=run_name,
        objtype=args.objtype,
        constraint_info_label=args.constraint_info_label,
        constraint=args.constraint,
        detection_losses="detection",
        albedo=args.albedo,
        h_mark=h_mark,
    )
    bdictD.update(bdict)

    # Run these discovery metrics
    print("Calculating full discovery metrics with detection losses.")
    bg = mmb.MoMetricBundleGroup(bdictD, out_dir=args.outDir, resultsDb=resultsDb)
    bg.runAll()

    # Run all characterization metrics
    if characterization.lower() == "inner":
        bdictC = batches.characterizationInnerBatch(
            slicer,
            colmap=colmap,
            run_name=run_name,
            objtype=args.objtype,
            albedo=args.albedo,
            constraint_info_label=args.constraint_info_label,
            constraint=args.constraint,
            h_mark=h_mark,
        )
    elif characterization.lower() == "outer":
        bdictC = batches.characterizationOuterBatch(
            slicer,
            colmap=colmap,
            run_name=run_name,
            objtype=args.objtype,
            albedo=args.albedo,
            constraint_info_label=args.constraint_info_label,
            constraint=args.constraint,
            h_mark=h_mark,
        )
    # Run these characterization metrics
    print("Calculating characterization metrics.")
    bg = mmb.MoMetricBundleGroup(bdictC, out_dir=args.outDir, resultsDb=resultsDb)
    bg.runAll()
