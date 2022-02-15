import os
import logging
import numpy as np
from rubin_sim.maf.utils import getSimData
from .orbits import Orbits
from .directObs import DirectObs

__all__ = ["readOpsim", "readOrbits", "setupColors", "runObs"]


def readOpsim(opsimfile, colmap, constraint=None, footprint="camera", dbcols=None):
    """Read the opsim database.

    Parameters
    ----------
    opsimfile : `str`
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
    simdata = getSimData(opsimfile, constraint, cols)
    logging.info(
        "Queried data from opsim %s, fetched %d visits." % (opsimfile, len(simdata))
    )
    return simdata


def readOrbits(orbitfile):
    """Read the orbits from the orbitfile.

    Parameters
    ----------
    orbitfile: str
        Name (and path) of the orbit file.

    Returns
    -------
    rubin_sim.movingObjects.Orbits
        The orbit object.
    """
    if not os.path.isfile(orbitfile):
        logging.critical("Could not find orbit file %s" % (orbitfile))
    orbits = Orbits()
    orbits.readOrbits(orbitfile)
    logging.info("Read orbit information from %s" % (orbitfile))
    return orbits


def setupColors(obs, filterlist, orbits):
    # Set up filters
    obs.readFilters(filterlist=filterlist)
    # Calculate all colors ahead of time.
    sednames = np.unique(orbits.orbits["sed_filename"])
    for sedname in sednames:
        obs.calcColors(sedname)
    return obs


def runObs(
    orbits,
    simdata,
    colmap,
    obsFile,
    footprint="camera",
    rFov=1.75,
    xTol=5,
    yTol=3,
    ephMode="nbody",
    prelimEphMode="nbody",
    obsCode="I11",
    ephType="basic",
    tStep=1,
    roughTol=10,
    obsMetadata=None,
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
    obsFile : `str`
        Output file for the observations
    footprint : `str`, opt
        Footprint - camera, circle or rectangle. Default camera footprint.
    rFov : `float`, opt
        If using a circular FOV, this is the radius of that circle.
        Default 1.75, but only used if footprint is 'circle'.
    xTol : `float`, opt
        If using a rectangular footprint, this is the tolerance in the RA direction.
        Default 5.
    yTol : `float`, opt
        If using a rectangular footprint, this is the tolerance in the Dec direction
        Default 3.
    ephMode : `str`, opt
        Ephemeris generation mode (2body or nbody) for exact matching. Default nbody.
    prelimEphMode : `str`, opt
        Preliminary (rough grid) ephemeris generation mode (2body or nbody). Default nbody.
    obsCode : `str`, opt
        Observatory code for ephemeris generation. Default I11 = Cerro Pachon.
    ephType : `str`, opt
        ephemeris type (from oorb.generate_ephemeris) to return ('full' or 'basic'). Default 'basic'.
    tStep : `float`, opt
        Time step for rough grid, in days. Default 1 day.
    roughTol : `float`, opt
        Tolerance in degrees between rough grid position and opsim pointings. Default 10 deg.
    obsMetadata : `str`, opt
        Metadata to write into output file header.
    """
    obs = DirectObs(
        footprint=footprint,
        rFov=rFov,
        xTol=xTol,
        yTol=yTol,
        ephMode=ephMode,
        prelimEphMode=prelimEphMode,
        obsCode=obsCode,
        ephFile=None,
        ephType=ephType,
        obsTimeCol=colmap["mjd"],
        obsTimeScale="TAI",
        seeingCol=colmap["seeingGeom"],
        visitExpTimeCol=colmap["exptime"],
        obsRA=colmap["ra"],
        obsDec=colmap["dec"],
        obsRotSkyPos=colmap["rotSkyPos"],
        obsDegrees=colmap["raDecDeg"],
        outfileName=obsFile,
        tstep=tStep,
        roughTol=roughTol,
        obsMetadata=obsMetadata,
    )
    filterlist = np.unique(simdata["filter"])
    obs = setupColors(obs, filterlist, orbits)
    obs.run(orbits, simdata)
