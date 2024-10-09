__all__ = ("run_moving_calc",)

import argparse
import os

import numpy as np
from rubin_scheduler.utils import SURVEY_START_MJD

from rubin_sim.maf.slicers import MoObjSlicer

from . import batches as batches
from . import db as db
from . import metricBundles as mmB


def run_moving_calc():
    """Calculate metric values for an input population. Can be used on either
    a split or complete population. If running on a split population for later
    re-combining, use the complete set of orbits as the 'orbit_file'. Assumes
    you have already created the moving object observation files.
    """

    parser = argparse.ArgumentParser(description="Run moving object metrics for a particular scheduler run.")
    parser.add_argument("--orbit_file", type=str, help="File containing the moving object orbits.")
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
        "--simulation_db",
        type=str,
        default=None,
        help="Path and filename of opsim output sqlite file, to write config* files to output directory."
        " Optional: if not provided, files won't have useful names.",
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
        help="Stepsizes in H values. " "If not set, defaults from objtype will be used.",
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
        help="Inner/Outer solar system characterization. " "If unset, defaults from objtype will be used.",
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
        help="Maximum number of years out to which to evaluate completeness." "Default 10.",
    )
    parser.add_argument(
        "--start_time",
        type=float,
        default=None,
        help="Time at start of survey (to set time for summary metrics).",
    )
    args = parser.parse_args()

    run_name = os.path.split(args.simulation_db)[-1].replace(".db", "")

    if args.orbit_file is None or args.obs_file is None:
        print("Must specify an orbit_file and an obs_file to calculate the metrics.")
        exit()

    # Get default H and other values:
    defaults = batches.ss_population_defaults(args.objtype)
    h_min = defaults["h_range"][0]
    h_max = defaults["h_range"][1]
    h_step = defaults["h_range"][2]
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
        start_time = SURVEY_START_MJD
    else:
        start_time = args.start_time

    stepsize = 365 / 2.0
    times = np.arange(0, args.n_years_max * 365 + stepsize / 2, stepsize)
    times += start_time

    # Set up results_db.
    if not (os.path.isdir(args.out_dir)):
        try:
            os.makedirs(args.out_dir)
        except FileExistsError:
            # This can happen if you are running these in parallel
            # and two scripts try to make
            # the same directory.
            pass
    results_db = db.ResultsDb(out_dir=args.out_dir)

    colmap = batches.col_map_dict()
    slicer = MoObjSlicer(h_range=Hrange)
    slicer.setup_slicer(orbit_file=args.orbit_file, obs_file=args.obs_file)
    # Run discovery metrics using 'trailing' losses
    bdictT = batches.quick_discovery_batch(
        slicer,
        colmap=colmap,
        run_name=run_name,
        objtype=args.objtype,
        constraint_info_label=args.constraint_info_label,
        constraint=args.constraint,
        detection_losses="trailing",
        albedo=args.albedo,
        h_mark=h_mark,
        magtype=magtype,
    )
    # Run these discovery metrics
    print("Calculating quick discovery metrics with simple trailing losses.")
    bg = mmB.MoMetricBundleGroup(bdictT, out_dir=args.out_dir, results_db=results_db)
    bg.run_all()

    # Run all discovery metrics using 'detection' losses
    bdictD = batches.quick_discovery_batch(
        slicer,
        colmap=colmap,
        run_name=run_name,
        objtype=args.objtype,
        constraint_info_label=args.constraint_info_label,
        constraint=args.constraint,
        detection_losses="detection",
        albedo=args.albedo,
        h_mark=h_mark,
        magtype=magtype,
    )
    bdict = batches.discovery_batch(
        slicer,
        colmap=colmap,
        run_name=run_name,
        objtype=args.objtype,
        constraint_info_label=args.constraint_info_label,
        constraint=args.constraint,
        detection_losses="detection",
        albedo=args.albedo,
        h_mark=h_mark,
        magtype=magtype,
    )
    bdictD.update(bdict)

    # Run these discovery metrics
    print("Calculating full discovery metrics with detection losses.")
    bg = mmB.MoMetricBundleGroup(bdictD, out_dir=args.out_dir, results_db=results_db)
    bg.run_all()

    # Run all characterization metrics
    if characterization.lower() == "inner":
        bdictC = batches.characterization_inner_batch(
            slicer,
            colmap=colmap,
            run_name=run_name,
            objtype=args.objtype,
            albedo=args.albedo,
            constraint_info_label=args.constraint_info_label,
            constraint=args.constraint,
            h_mark=h_mark,
            magtype=magtype,
        )
    elif characterization.lower() == "outer":
        bdictC = batches.characterization_outer_batch(
            slicer,
            colmap=colmap,
            run_name=run_name,
            objtype=args.objtype,
            albedo=args.albedo,
            constraint_info_label=args.constraint_info_label,
            constraint=args.constraint,
            h_mark=h_mark,
            magtype=magtype,
        )
    # Run these characterization metrics
    print("Calculating characterization metrics.")
    bg = mmB.MoMetricBundleGroup(bdictC, out_dir=args.out_dir, results_db=results_db)
    bg.run_all()
