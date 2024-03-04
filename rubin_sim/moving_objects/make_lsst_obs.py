#!/usr/bin/env python

__all__ = ("setup_args",)

import argparse
import logging
import os

import numpy as np

# So things don't fail on hyak
from astropy.utils import iers

import rubin_sim.moving_objects as mo
from rubin_sim.maf.batches import col_map_dict

iers.conf.auto_download = False


def setup_args(parser=None):
    """Parse the command line arguments.

    Parameters
    ----------
    parser: argparse.ArgumentParser, optional
        Generally left at the default (None), but a user could set up
        their own parser if desired.

    Returns
    -------
    argparse.Namespace
        The argument options.
    """

    if parser is None:
        parser = argparse.ArgumentParser(description="Generate moving object detections.")
    parser.add_argument(
        "--simulation_db",
        type=str,
        default=None,
        help="Simulation output db file (example: baseline_v2.1_10yrs.db). Default None.",
    )
    parser.add_argument(
        "--orbit_file",
        type=str,
        default=None,
        help="File containing the moving object orbits. "
        "See https://github.com/oorb/oorb/tree/master/python#defining-orbits for "
        "additional documentation on the orbit file format. Default None.",
    )
    parser.add_argument(
        "--positions_file",
        type=str,
        default=None,
        help="File with pre-computed 'rough' ephemerides for the objects in the orbit file."
        " Default None, which will trigger computation of these rough ephemerides on-the-fly.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=".",
        help="Output directory for moving object detections. Default '.'",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file name for moving object observations."
        " Default will build out_dir/simulation_db_orbitFile_obs.npz.",
    )
    parser.add_argument(
        "--sql_constraint",
        type=str,
        default="",
        help="SQL constraint to use to select data from simulation_db. Default no constraint.",
    )
    parser.add_argument(
        "--obs_info",
        type=str,
        default=None,
        help="Additional info to write into output file. "
        "The default info will combine the simulation_db name, the sqlconstraint, and "
        "the name of the orbit file; obs_info is an optional addition.",
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
        "--obs_code",
        type=str,
        default="I11",
        help="Observatory code for generating observations. " "Default is I11 (Cerro Pachon).",
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
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print more output")
    parser.set_defaults(verbose=False)

    args = parser.parse_args()

    if args.simulation_db is None:
        raise ValueError("Must specify an simulation database output file.")

    if args.orbit_file is None:
        raise ValueError("Must specify an orbit file.")

    run_name = os.path.split(args.simulation_db)[-1].replace(".db", "")

    # Add these useful pieces to args.
    args.orbitbase = ".".join(os.path.split(args.orbit_file)[-1].split(".")[:-1])

    # Set up obs_file if not specified.
    if args.output_file is None:
        args.output_file = os.path.join(args.out_dir, "%s__%s_obs.npz" % (run_name, args.orbitbase))
    else:
        args.output_file = os.path.join(args.out_dir, args.output_file)

    # Build some provenance to add to output file.
    obs_info = args.simulation_db
    if len(args.sql_constraint) > 0:
        obs_info += " selected with sqlconstraint %s" % args.sql_constraint
    obs_info += " + orbit_file %s" % args.orbitbase
    if args.obs_info is not None:
        obs_info += "\n# %s" % args.obs_info
    args.obs_info = obs_info

    return args


def make_lsst_obs():
    # Parser command
    args = setup_args()

    # Send info and above logging messages to the console or logfile.
    if args.log_file is not None:
        logging.basicConfig(filename=args.log_file, level=logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)

    # Read orbits.
    orbits = mo.Orbits()
    orbits.read_orbits(args.orbit_file)

    if args.positions_file is not None:
        position_data = np.load(args.positions_file)
        object_positions = position_data["positions"].copy()
        object_mjds = position_data["mjds"].copy()
        position_data.close()
    else:
        object_positions = None
        object_mjds = None
    # Read pointing data
    colmap = col_map_dict("fbs")
    pointing_data = mo.read_observations(
        args.simulation_db,
        colmap,
        constraint=args.sql_constraint,
        dbcols=None,
    )

    d_obs = mo.DirectObs(
        footprint=args.footprint,
        r_fov=args.r_fov,
        x_tol=args.x_tol,
        y_tol=args.y_tol,
        eph_mode=args.eph_mode,
        prelim_eph_mode=args.prelim_eph_mode,
        obs_code=args.obs_code,
        eph_file=None,
        eph_type=args.eph_type,
        obs_time_col=colmap["mjd"],
        obs_time_scale="TAI",
        seeing_col=colmap["seeingGeom"],
        visit_exp_time_col=colmap["exptime"],
        obs_ra=colmap["ra"],
        obs_dec=colmap["dec"],
        obs_rot_sky_pos=colmap["rotSkyPos"],
        obs_degrees=colmap["raDecDeg"],
        outfile_name=None,
        tstep=args.t_step,
        rough_tol=args.rough_tol,
        obs_info=args.obs_info,
        verbose=args.verbose,
    )
    filterlist = np.unique(pointing_data["filter"])
    d_obs.read_filters(filterlist=filterlist)
    # Calculate all colors ahead of time.
    sednames = np.unique(orbits.orbits["sed_filename"])
    for sedname in sednames:
        d_obs.calc_colors(sedname)

    # Generate object observations.
    object_observations = d_obs.run(
        orbits,
        pointing_data,
        object_positions=object_positions,
        object_mjds=object_mjds,
    )

    np.savez(
        os.path.join(args.out_dir, args.output_file),
        object_observations=object_observations,
        info=d_obs.info,
    )

    # logging.info("Completed successfully.")
