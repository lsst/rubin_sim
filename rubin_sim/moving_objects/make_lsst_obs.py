#!/usr/bin/env python

import os
import argparse
import logging

import rubin_sim.movingObjects as mo
from rubin_sim.maf.batches import ColMapDict

__all__ = ["setupArgs"]


def setupArgs(parser=None):
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
        "--opsimDb",
        type=str,
        default=None,
        help="Opsim output db file (example: kraken_2026.db). Default None.",
    )
    parser.add_argument(
        "--orbitFile",
        type=str,
        default=None,
        help="File containing the moving object orbits. "
        "See https://github.com/lsst/oorb/blob/lsst-dev/python/README.rst for "
        "additional documentation on the orbit file format. Default None.",
    )
    parser.add_argument(
        "--outDir",
        type=str,
        default=".",
        help="Output directory for moving object detections. Default '.'",
    )
    parser.add_argument(
        "--obsFile",
        type=str,
        default=None,
        help="Output file name for moving object observations."
        " Default will build outDir/opsimRun_orbitFile_obs.txt.",
    )
    parser.add_argument(
        "--sqlConstraint",
        type=str,
        default="",
        help="SQL constraint to use to select data from opsimDb. Default no constraint.",
    )
    parser.add_argument(
        "--obsMetadata",
        type=str,
        default=None,
        help="Additional metadata to write into output file. "
        "The default metadata will combine the opsimDb name, the sqlconstraint, and "
        "the name of the orbit file; obsMetadata is an optional addition.",
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
        "--rFov",
        type=float,
        default=1.75,
        help="If using a circular footprint, this is the radius of the FOV (in degrees). "
        "Default 1.75 deg.",
    )
    parser.add_argument(
        "--xTol",
        type=float,
        default=5,
        help="If using a rectangular footprint, this is the tolerance in the RA direction "
        "(in degrees). Default is 5 degrees.",
    )
    parser.add_argument(
        "--yTol",
        type=float,
        default=3,
        help="If using a rectangular footprint, this is the tolerance in the Dec direction "
        "(in degrees). Default is 3 degrees.",
    )
    parser.add_argument(
        "--roughTol",
        type=float,
        default=10,
        help="If using direct/exact ephemeris generation, this is the tolerance for the "
        "preliminary matches between ephemerides and pointings (in degrees). "
        "Default 10 degrees.",
    )
    parser.add_argument(
        "--obsType",
        type=str,
        default="direct",
        help="Method for generating observations: 'direct' or 'linear'. "
        "Linear will use linear interpolation between a grid of ephemeris points. "
        "Direct will first generate rough ephemerides, look for observations within "
        "roughTol of these points, and then generate exact ephemerides at those times. "
        "Default 'direct'.",
    )
    parser.add_argument(
        "--obsCode",
        type=str,
        default="I11",
        help="Observatory code for generating observations. "
        "Default is I11 (Cerro Pachon).",
    )
    parser.add_argument(
        "--tStep",
        type=float,
        default=1.0,
        help="Timestep between ephemeris generation for either the first (rough) stage of "
        "direct ephemeris generation or the grid for linear interpolation "
        "ephemerides. Default 1 day.",
    )
    parser.add_argument(
        "--ephMode",
        type=str,
        default="nbody",
        help="2body or nbody mode for ephemeris generation. Default is nbody.",
    )
    parser.add_argument(
        "--prelimEphMode",
        type=str,
        default="nbody",
        help="Use either 2body or nbody for preliminary ephemeris generation in the rough "
        "stage for DirectObs. Default 2body.",
    )
    parser.add_argument(
        "--ephType",
        type=str,
        default="basic",
        help="Generate either 'basic' or 'full' ephemerides from OOrb. "
        "See https://github.com/lsst/oorb/blob/lsst-dev/python/README.rst for details"
        "of the contents of 'full' or 'basic' ephemerides. "
        "Default basic.",
    )
    parser.add_argument(
        "--logFile",
        type=str,
        default=None,
        help="Send log output to logFile, instead of to console. (default = console)",
    )
    args = parser.parse_args()

    if args.opsimDb is None:
        raise ValueError("Must specify an opsim database output file.")

    if args.orbitFile is None:
        raise ValueError("Must specify an orbit file.")

    # Check interpolation type.
    if args.obsType not in ("linear", "direct"):
        raise ValueError(
            "Must choose linear or direct observation generation method (obsType)."
        )

    # Add these useful pieces to args.
    args.orbitbase = ".".join(os.path.split(args.orbitFile)[-1].split(".")[:-1])
    args.opsimRun = (
        os.path.split(args.opsimDb)[-1].replace("_sqlite.db", "").replace(".db", "")
    )

    # Set up obsFile if not specified.
    if args.obsFile is None:
        args.obsFile = os.path.join(
            args.outDir, "%s__%s_obs.txt" % (args.opsimRun, args.orbitbase)
        )
    else:
        args.obsFile = os.path.join(args.outDir, args.obsFile)

    # Build some provenance metadata to add to output file.
    obsMetadata = "Opsim %s" % args.opsimRun
    if len(args.sqlConstraint) > 0:
        obsMetadata += " selected with sqlconstraint %s" % (args.sqlConstraint)
    obsMetadata += " + Orbitfile %s" % args.orbitbase
    if args.obsMetadata is not None:
        obsMetadata += "\n# %s" % args.obsMetadata
    args.obsMetadata = obsMetadata

    return args


def make_lsst_obs():

    # Parser command
    args = setupArgs()

    # Send info and above logging messages to the console or logfile.
    if args.logFile is not None:
        logging.basicConfig(filename=args.logFile, level=logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)

    # Read orbits.
    orbits = mo.readOrbits(args.orbitFile)

    # Read opsim data
    colmap = ColMapDict("fbs")
    opsimdata = mo.readOpsim(
        args.opsimDb,
        colmap,
        constraint=args.sqlConstraint,
        footprint=args.footprint,
        dbcols=None,
    )
    # Generate ephemerides.
    mo.runObs(
        orbits,
        opsimdata,
        colmap,
        args.obsFile,
        args.footprint,
        args.rFov,
        args.xTol,
        args.yTol,
        args.ephMode,
        args.prelimEphMode,
        args.obsCode,
        args.ephType,
        args.tStep,
        args.roughTol,
        args.obsMetadata,
    )

    logging.info("Completed successfully.")
