import os
import logging
import numpy as np
from rubin_sim.maf.utils import get_sim_data
from .orbits import Orbits
from .direct_obs import DirectObs

__all__ = ["read_observations", "read_orbits", "setup_colors", "run_obs"]


def read_observations(
    simfile, colmap, constraint=None, footprint="camera", dbcols=None
):
    """Read the opsim database.

    Parameters
    ----------
    simfile : `str`
        Name (& path) of the opsim database file.
    colmap : `dict`
        colmap dictionary (from rubin_sim.maf.batches.ColMapDict)
    constraint : `str`, optional
        Optional SQL constraint (minus 'where') on the opsim data to read from db.
        Default is None.
    footprint : `str`, optional
        Footprint option for the final matching of object against OpSim FOV.
        Default 'camera' means that 'rotSkyPos' must be fetched from the db.
        Any other value will not require rotSkyPos.
    dbcols : `list` of `str`, optional
        List of additional columns to query from the db and add to the output observations.
        Default None.

    Returns
    -------
    np.ndarray, dictionary
        The OpSim data read from the database, and the dictionary mapping the column names to the data.
    """
    if "rotSkyPos" not in colmap:
        colmap["rotSkyPos"] = "rotSkyPos"

    # Set the minimum required columns.
    min_cols = [
        colmap["mjd"],
        colmap["night"],
        colmap["ra"],
        colmap["dec"],
        colmap["filter"],
        colmap["exptime"],
        colmap["seeingGeom"],
        colmap["fiveSigmaDepth"],
    ]
    if footprint == "camera":
        min_cols.append(colmap["rotSkyPos"])
    if dbcols is not None:
        min_cols += dbcols

    more_cols = [colmap["rotSkyPos"], colmap["seeingEff"], "solarElong"]

    cols = min_cols + more_cols
    cols = list(set(cols))
    logging.info("Querying for columns:\n %s" % (cols))

    # Go ahead and query for all of the observations.
    simdata = get_sim_data(simfile, constraint, cols)
    logging.info(
        "Queried data from opsim %s, fetched %d visits." % (simfile, len(simdata))
    )
    return simdata


def read_orbits(orbit_file):
    """Read the orbits from the orbit_file.

    Parameters
    ----------
    orbit_file: str
        Name (and path) of the orbit file.

    Returns
    -------
    rubin_sim.movingObjects.Orbits
        The orbit object.
    """
    if not os.path.isfile(orbit_file):
        logging.critical("Could not find orbit file %s" % (orbit_file))
    orbits = Orbits()
    orbits.read_orbits(orbit_file)
    logging.info("Read orbit information from %s" % (orbit_file))
    return orbits


def setup_colors(obs, filterlist, orbits):
    # Set up filters
    obs.read_filters(filterlist=filterlist)
    # Calculate all colors ahead of time.
    sednames = np.unique(orbits.orbits["sed_filename"])
    for sedname in sednames:
        obs.calc_colors(sedname)
    return obs


def run_obs(
    orbits,
    simdata,
    colmap,
    obs_file,
    footprint="camera",
    r_fov=1.75,
    x_tol=5,
    y_tol=3,
    eph_mode="nbody",
    prelim_eph_mode="nbody",
    obs_code="I11",
    eph_type="basic",
    t_step=1,
    rough_tol=10,
    obs_metadata=None,
):
    """Generate the observations.

    Parameters
    ----------
    orbits : `rubin_sim.movingObjects.Orbit`
        Orbits for which to calculate observations
    simdata : `np.ndarray`
        The simulated pointing history data from OpSim
    colmap : `dict`
        Dictionary of the column mappings (from column names here to opsim columns).
    obs_file : `str`
        Output file for the observations
    footprint : `str`, opt
        Footprint - camera, circle or rectangle. Default camera footprint.
    r_fov : `float`, opt
        If using a circular FOV, this is the radius of that circle.
        Default 1.75, but only used if footprint is 'circle'.
    x_tol : `float`, opt
        If using a rectangular footprint, this is the tolerance in the RA direction.
        Default 5.
    y_tol : `float`, opt
        If using a rectangular footprint, this is the tolerance in the Dec direction
        Default 3.
    eph_mode : `str`, opt
        Ephemeris generation mode (2body or nbody) for exact matching. Default nbody.
    prelim_eph_mode : `str`, opt
        Preliminary (rough grid) ephemeris generation mode (2body or nbody). Default nbody.
    obs_code : `str`, opt
        Observatory code for ephemeris generation. Default I11 = Cerro Pachon.
    eph_type : `str`, opt
        ephemeris type (from oorb.generate_ephemeris) to return ('full' or 'basic'). Default 'basic'.
    t_step : `float`, opt
        Time step for rough grid, in days. Default 1 day.
    rough_tol : `float`, opt
        Tolerance in degrees between rough grid position and opsim pointings. Default 10 deg.
    obs_metadata : `str`, opt
        Metadata to write into output file header.
    """
    obs = DirectObs(
        footprint=footprint,
        r_fov=r_fov,
        x_tol=x_tol,
        y_tol=y_tol,
        eph_mode=eph_mode,
        prelim_eph_mode=prelim_eph_mode,
        obs_code=obs_code,
        eph_file=None,
        eph_type=eph_type,
        obs_time_col=colmap["mjd"],
        obs_time_scale="TAI",
        seeing_col=colmap["seeingGeom"],
        visit_exp_time_col=colmap["exptime"],
        obs_ra=colmap["ra"],
        obs_dec=colmap["dec"],
        obs_rot_sky_pos=colmap["rotSkyPos"],
        obs_degrees=colmap["raDecDeg"],
        outfile_name=obs_file,
        tstep=t_step,
        rough_tol=rough_tol,
        obs_metadata=obs_metadata,
    )
    filterlist = np.unique(simdata["filter"])
    obs = setup_colors(obs, filterlist, orbits)
    obs.run(orbits, simdata)
