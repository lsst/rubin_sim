__all__ = ("read_observations",)

import logging
import os

from rubin_sim.maf.utils import get_sim_data

from .orbits import Orbits


def read_observations(simfile, colmap, constraint=None, dbcols=None):
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
    if dbcols is not None:
        min_cols += dbcols

    more_cols = [
        colmap["rotSkyPos"],
        colmap["seeingEff"],
        "solarElong",
        "observationId",
    ]

    cols = min_cols + more_cols
    cols = list(set(cols))
    logging.info("Querying for columns:\n %s" % (cols))

    # Go ahead and query for all of the observations.
    simdata = get_sim_data(simfile, constraint, cols)
    logging.info("Queried data from opsim %s, fetched %d visits." % (simfile, len(simdata)))
    return simdata
