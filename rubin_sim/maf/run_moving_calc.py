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
    re-combining, use the complete set of orbits as the 'orbitFile'. Assumes
    you have already created the moving object observation files.
    """

    parser = argparse.ArgumentParser(
        description="Run moving object metrics for a particular opsim run."
    )
    parser.add_argument(
        "--orbitFile", type=str, help="File containing the moving object orbits."
    )
    parser.add_argument(
        "--objtype",
        type=str,
        default="",
        help="Object type, to set up default characterization/H value parameters "
        "(if they're not specified by flags) and to label outputs.",
    )
    parser.add_argument(
        "--obsFile",
        type=str,
        help="File containing the observations of the moving objects.",
    )
    parser.add_argument(
        "--opsimRun",
        type=str,
        default="opsim",
        help="Name of opsim run. Default 'opsim'.",
    )
    parser.add_argument(
        "--outDir",
        type=str,
        default=".",
        help="Output directory for moving object metrics. Default '.'.",
    )
    parser.add_argument(
        "--opsimDb",
        type=str,
        default=None,
        help="Path and filename of opsim db, to write config* files to output directory."
        " Optional: if not provided, config* files won't be created but analysis will run.",
    )
    parser.add_argument(
        "--hMin",
        type=float,
        help="Minimum H value. " "If not set, defaults from objtype will be used.",
    )
    parser.add_argument(
        "--hMax",
        type=float,
        help="Maximum H value. " "If not set, defaults from objtype will be used.",
    )
    parser.add_argument(
        "--hStep",
        type=float,
        help="Stepsizes in H values. "
        "If not set, defaults from objtype will be used.",
    )
    parser.add_argument(
        "--hMark",
        type=float,
        default=None,
        help="Add vertical lines at H=hMark on plots. Default None.",
    )
    parser.add_argument(
        "--characterization",
        type=str,
        help="Inner/Outer solar system characterization. "
        "If unset, defaults from objtype will be used.",
    )
    parser.add_argument(
        "--constraintInfoLabel",
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
        "--nYearsMax",
        type=int,
        default=10,
        help="Maximum number of years out to which to evaluate completeness."
        "Default 10.",
    )
    parser.add_argument(
        "--startTime",
        type=float,
        default=None,
        help="Time at start of survey (to set time for summary metrics).",
    )
    args = parser.parse_args()

    if args.orbitFile is None or args.obsFile is None:
        print("Must specify an orbitFile and an obsFile to calculate the metrics.")
        exit()

    # Get default H and other values:
    defaults = batches.ss_population_defaults(args.objtype)
    hMin = defaults["Hrange"][0]
    hMax = defaults["Hrange"][1]
    hStep = defaults["Hrange"][2]
    hMark = defaults["Hmark"]
    characterization = defaults["char"]
    magtype = defaults["magtype"]
    # If H info is specified from parser, use that
    if args.hMin is not None:
        hMin = args.hMin
    if args.hMax is not None:
        hMax = args.hMax
    if args.hStep is not None:
        hStep = args.hStep
    if args.hMark is not None:
        hMark = args.hMark
    if args.characterization is not None:
        characterization = args.characterization
    Hrange = np.arange(hMin, hMax + hStep, hStep)

    # Default parameters for metric setup.
    if args.startTime is None:
        startTime = survey_start_mjd()
    else:
        startTime = args.startTime

    stepsize = 365 / 2.0
    times = np.arange(0, args.nYearsMax * 365 + stepsize / 2, stepsize)
    times += startTime

    # Set up resultsDb.
    if not (os.path.isdir(args.outDir)):
        try:
            os.makedirs(args.outDir)
        except FileExistsError:
            # This can happen if you are running these in parallel and two scripts try to make
            # the same directory.
            pass
    resultsDb = db.ResultsDb(outDir=args.outDir)

    colmap = batches.ColMapDict()
    slicer = batches.setupMoSlicer(args.orbitFile, Hrange, obsFile=args.obsFile)
    # Run discovery metrics using 'trailing' losses
    bdictT = batches.quickDiscoveryBatch(
        slicer,
        colmap=colmap,
        runName=args.opsimRun,
        objtype=args.objtype,
        constraintInfoLabel=args.constraintInfoLabel,
        constraint=args.constraint,
        detectionLosses="trailing",
        albedo=args.albedo,
        Hmark=hMark,
    )
    # Run these discovery metrics
    print("Calculating quick discovery metrics with simple trailing losses.")
    bg = mmb.MoMetricBundleGroup(bdictT, outDir=args.outDir, resultsDb=resultsDb)
    bg.runAll()

    # Run all discovery metrics using 'detection' losses
    bdictD = batches.quickDiscoveryBatch(
        slicer,
        colmap=colmap,
        runName=args.opsimRun,
        objtype=args.objtype,
        constraintInfoLabel=args.constraintInfoLabel,
        constraint=args.constraint,
        detectionLosses="detection",
        albedo=args.albedo,
        Hmark=hMark,
    )
    bdict = batches.discoveryBatch(
        slicer,
        colmap=colmap,
        runName=args.opsimRun,
        objtype=args.objtype,
        constraintInfoLabel=args.constraintInfoLabel,
        constraint=args.constraint,
        detectionLosses="detection",
        albedo=args.albedo,
        Hmark=hMark,
    )
    bdictD.update(bdict)

    # Run these discovery metrics
    print("Calculating full discovery metrics with detection losses.")
    bg = mmb.MoMetricBundleGroup(bdictD, outDir=args.outDir, resultsDb=resultsDb)
    bg.runAll()

    # Run all characterization metrics
    if characterization.lower() == "inner":
        bdictC = batches.characterizationInnerBatch(
            slicer,
            colmap=colmap,
            runName=args.opsimRun,
            objtype=args.objtype,
            albedo=args.albedo,
            constraintInfoLabel=args.constraintInfoLabel,
            constraint=args.constraint,
            Hmark=hMark,
        )
    elif characterization.lower() == "outer":
        bdictC = batches.characterizationOuterBatch(
            slicer,
            colmap=colmap,
            runName=args.opsimRun,
            objtype=args.objtype,
            albedo=args.albedo,
            constraintInfoLabel=args.constraintInfoLabel,
            constraint=args.constraint,
            Hmark=hMark,
        )
    # Run these characterization metrics
    print("Calculating characterization metrics.")
    bg = mmb.MoMetricBundleGroup(bdictC, outDir=args.outDir, resultsDb=resultsDb)
    bg.runAll()
