#!/usr/bin/env python

import os
import argparse
import logging

import rubin_sim.moving_objects as mo
from rubin_sim.maf.batches import ColMapDict

__all__ = ["setup_args"]


def setup_args(parser=None):
    """Parse the command line arguments.

    Parameters
    ----------
    parser: argparse.ArgumentParser, optional
        Generally left at the default (None), but a user could set up their own parser if desired.

    Returns
    -------
    argparse.Namespace
        The argument options.
    """

    if parser is None:
        parser = argparse.ArgumentParser(
            description="Generate moving object detections."
        )
    parser.add_argument(
        "--simulation_db",
        type=str,
        default=None,
        help="Simulation output db file (example: kraken_2026.db). Default None.",
    )
    parser.add_argument(
        "--orbit_file",
        type=str,
        default=None,
        help="File containing the moving object orbits. "
        "See https://github.com/lsst/oorb/blob/lsst-dev/python/README.rst for "
        "additional documentation on the orbit file format. Default None.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=".",
        help="Output directory for moving object detections. Default '.'",
    )
    parser.add_argument(
        "--obs_file",
        type=str,
        default=None,
        help="Output file name for moving object observations."
        " Default will build out_dir/simulation_db_orbitFile_obs.txt.",
    )
    parser.add_argument(
        "--sql_constraint",
        type=str,
        default="",
        help="SQL constraint to use to select data from simulation_db. Default no constraint.",
    )
    parser.add_argument(
        "--obs_metadata",
        type=str,
        default=None,
        help="Additional metadata to write into output file. "
        "The default metadata will combine the simulation_db name, the sqlconstraint, and "
        "the name of the orbit file; obs_metadata is an optional addition.",
    )
    parser.add_argument(
        "--footprint",
        type=str,
        default="camera",
        help="Type of footprint to use to identify observations of each object. "
        "Options are 'circle' (r=1.75 deg), 'rectangle', or 'camera' (camera footprint). "
        "Default is 'camera'.",
    )
    parser.add_argument(
        "--r_fov",
        type=float,
        default=1.75,
        help="If using a circular footprint, this is the radius of the FOV (in degrees). "
        "Default 1.75 deg.",
    )
    parser.add_argument(
        "--x_tol",
        type=float,
        default=5,
        help="If using a rectangular footprint, this is the tolerance in the RA direction "
        "(in degrees). Default is 5 degrees.",
    )
    parser.add_argument(
        "--y_tol",
        type=float,
        default=3,
        help="If using a rectangular footprint, this is the tolerance in the Dec direction "
        "(in degrees). Default is 3 degrees.",
    )
    parser.add_argument(
        "--rough_tol",
        type=float,
        default=10,
        help="If using direct/exact ephemeris generation, this is the tolerance for the "
        "preliminary matches between ephemerides and pointings (in degrees). "
        "Default 10 degrees.",
    )
    parser.add_argument(
        "--obs_type",
        type=str,
        default="direct",
        help="Method for generating observations: 'direct' or 'linear'. "
        "Linear will use linear interpolation between a grid of ephemeris points. "
        "Direct will first generate rough ephemerides, look for observations within "
        "roughTol of these points, and then generate exact ephemerides at those times. "
        "Default 'direct'.",
    )
    parser.add_argument(
        "--obs_code",
        type=str,
        default="I11",
        help="Observatory code for generating observations. "
        "Default is I11 (Cerro Pachon).",
    )
    parser.add_argument(
        "--t_step",
        type=float,
        default=1.0,
        help="Timestep between ephemeris generation for either the first (rough) stage of "
        "direct ephemeris generation or the grid for linear interpolation "
        "ephemerides. Default 1 day.",
    )
    parser.add_argument(
        "--eph_mode",
        type=str,
        default="nbody",
        help="2body or nbody mode for ephemeris generation. Default is nbody.",
    )
    parser.add_argument(
        "--prelim_eph_mode",
        type=str,
        default="nbody",
        help="Use either 2body or nbody for preliminary ephemeris generation in the rough "
        "stage for DirectObs. Default 2body.",
    )
    parser.add_argument(
        "--eph_type",
        type=str,
        default="basic",
        help="Generate either 'basic' or 'full' ephemerides from OOrb. "
        "See https://github.com/lsst/oorb/blob/lsst-dev/python/README.rst for details"
        "of the contents of 'full' or 'basic' ephemerides. "
        "Default basic.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Send log output to log_file, instead of to console. (default = console)",
    )
    args = parser.parse_args()

    if args.simulation_db is None:
        raise ValueError("Must specify an simulation database output file.")

    if args.orbit_file is None:
        raise ValueError("Must specify an orbit file.")

    # Check interpolation type.
    if args.obs_type not in ("linear", "direct"):
        raise ValueError(
            "Must choose linear or direct observation generation method (obsType)."
        )

    # Add these useful pieces to args.
    args.orbitbase = ".".join(os.path.split(args.orbitFile)[-1].split(".")[:-1])
    args.simulation_db = (
        os.path.split(args.simulation_db)[-1].replace("_sqlite.db", "").replace(".db", "")
    )

    # Set up obs_file if not specified.
    if args.obs_file is None:
        args.obs_file = os.path.join(
            args.outDir, "%s__%s_obs.txt" % (args.simulation_db, args.orbitbase)
        )
    else:
        args.obs_file = os.path.join(args.outDir, args.obs_file)

    # Build some provenance metadata to add to output file.
    obs_metadata = args.simulation_db
    if len(args.sql_constraint) > 0:
        obs_metadata += " selected with sqlconstraint %s" % args.sql_constraint
    obs_metadata += " + Orbitfile %s" % args.orbitbase
    if args.obs_metadata is not None:
        obs_metadata += "\n# %s" % args.obs_metadata
    args.obs_metadata = obs_metadata

    return args


def make_lsst_obs():

    # Parser command
    args = setup_args()

    # Send info and above logging messages to the console or logfile.
    if args.logFile is not None:
        logging.basicConfig(filename=args.logFile, level=logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)

    # Read orbits.
    orbits = mo.read_orbits(args.orbitFile)

    # Read pointing data
    colmap = ColMapDict("fbs")
    pointing_data = mo.read_opsim(
        args.simulation_db,
        colmap,
        constraint=args.sql_constraint,
        footprint=args.footprint,
        dbcols=None,
    )
    # Generate ephemerides.
    mo.run_obs(
        orbits,
        pointing_data,
        colmap,
        args.obs_file,
        args.footprint,
        args.r_fov,
        args.x_tol,
        args.y_tol,
        args.eph_mode,
        args.prelim_eph_mode,
        args.obs_code,
        args.eph_type,
        args.t_step,
        args.rough_tol,
        args.obs_metadata,
    )

    logging.info("Completed successfully.")
