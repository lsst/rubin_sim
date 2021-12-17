# Collection of utilities for MAF that relate to Opsim specifically.

import numpy as np
import pandas as pd
import sqlite3

__all__ = [
    "getSimData",
    "scaleBenchmarks",
    "calcCoaddedDepth",
]


def getSimData(
    opsimDb, sqlconstraint, dbcols, stackers=None, tableName='observations'
):
    """Query an opsim database for the needed data columns and run any required stackers.

    Parameters
    ----------
    opsimDb : `str` or database connection object
        A string that is the path to a sqlite3 file or a  
    sqlconstraint : `str`
        SQL constraint to apply to query for observations.
    dbcols : `list` [`str`]
        Columns required from the database.
    stackers : `list` [`rubin_sim.maf.stackers`], optional
        Stackers to be used to generate additional columns. Default None.
    tableName : `str` (observations)
        Name of the table to query. Default None uses the opsimDb default.

    Returns
    -------
    simData: `np.ndarray`
        A numpy structured array with columns resulting from dbcols + stackers, for observations matching
        the SQLconstraint.
    """
    # Get data from database.

    if type(opsimDb) == str:
        con = sqlite3.connect(opsimDb)
    else:
        con = opsimDb

    col_str = ''
    for colname in dbcols:
        col_str += colname+', '
    col_str = col_str[0:-2] + ' '

    query = 'SELECT %s FROM %s' % (col_str, tableName)
    if len(sqlconstraint) > 0:
        query += ' WHERE %s' % (sqlconstraint)
    query += ';'

    simData = pd.read_sql(query, con).to_records(index=False)

    if len(simData) == 0:
        raise UserWarning("No data found matching sqlconstraint %s" % (sqlconstraint))
    # Now add the stacker columns.
    if stackers is not None:
        for s in stackers:
            simData = s.run(simData)
    return simData


def scaleBenchmarks(runLength, benchmark="design"):
    """
    Set the design and stretch values of the number of visits, area of the footprint,
    seeing values, FWHMeff values, skybrightness, and single visit depth (based on SRD values).
    Scales number of visits for the length of the run, relative to 10 years.

    Parameters
    ----------
    runLength : float
        The length (in years) of the run.
    benchmark : str
        design or stretch - which version of the SRD values to return.
        requested is another option, in which case the values of the number of visits requested
        by the OpSim run (recorded in the Config table) is returned.

    Returns
    -------
    benchmarks: `dict` of floats
       A dictionary containing the number of visits, area of footprint, seeing and FWHMeff values,
       skybrightness and single visit depth for either the design or stretch SRD values.
    """
    # Set baseline (default) numbers for the baseline survey length (10 years).
    baseline = 10.0

    design = {}
    stretch = {}

    design["nvisitsTotal"] = 825
    stretch["nvisitsTotal"] = 1000
    design["Area"] = 18000
    stretch["Area"] = 20000

    design["nvisits"] = {"u": 56, "g": 80, "r": 184, "i": 184, "z": 160, "y": 160}
    stretch["nvisits"] = {"u": 70, "g": 100, "r": 230, "i": 230, "z": 200, "y": 200}

    # mag/sq arcsec
    design["skybrightness"] = {
        "u": 21.8,
        "g": 22.0,
        "r": 21.3,
        "i": 20.0,
        "z": 19.1,
        "y": 17.5,
    }
    stretch["skybrightness"] = {
        "u": 21.8,
        "g": 22.0,
        "r": 21.3,
        "i": 20.0,
        "z": 19.1,
        "y": 17.5,
    }

    # arcsec - old seeing values
    design["seeing"] = {"u": 0.77, "g": 0.73, "r": 0.7, "i": 0.67, "z": 0.65, "y": 0.63}
    stretch["seeing"] = {
        "u": 0.77,
        "g": 0.73,
        "r": 0.7,
        "i": 0.67,
        "z": 0.65,
        "y": 0.63,
    }

    # arcsec - new FWHMeff values (scaled from old seeing)
    design["FWHMeff"] = {
        "u": 0.92,
        "g": 0.87,
        "r": 0.83,
        "i": 0.80,
        "z": 0.78,
        "y": 0.76,
    }
    stretch["FWHMeff"] = {
        "u": 0.92,
        "g": 0.87,
        "r": 0.83,
        "i": 0.80,
        "z": 0.78,
        "y": 0.76,
    }

    design["singleVisitDepth"] = {
        "u": 23.9,
        "g": 25.0,
        "r": 24.7,
        "i": 24.0,
        "z": 23.3,
        "y": 22.1,
    }
    stretch["singleVisitDepth"] = {
        "u": 24.0,
        "g": 25.1,
        "r": 24.8,
        "i": 24.1,
        "z": 23.4,
        "y": 22.2,
    }

    # Scale the number of visits.
    if runLength != baseline:
        scalefactor = float(runLength) / float(baseline)
        # Calculate scaled value for design and stretch values of nvisits, per filter.
        for f in design["nvisits"]:
            design["nvisits"][f] = int(np.floor(design["nvisits"][f] * scalefactor))
            stretch["nvisits"][f] = int(np.floor(stretch["nvisits"][f] * scalefactor))

    if benchmark == "design":
        return design
    elif benchmark == "stretch":
        return stretch
    else:
        raise ValueError(
            "Benchmark value %s not understood: use 'design' or 'stretch'" % (benchmark)
        )


def calcCoaddedDepth(nvisits, singleVisitDepth):
    """
    Calculate the coadded depth expected for a given number of visits and single visit depth.

    Parameters
    ----------
    nvisits : dict of ints or floats
        Dictionary (per filter) of number of visits
    singleVisitDepth : dict of floats
        Dictionary (per filter) of the single visit depth

    Returns
    -------
    dict of floats
        Dictionary of coadded depths per filter.
    """
    coaddedDepth = {}
    for f in nvisits:
        if f not in singleVisitDepth:
            raise ValueError("Filter keys in nvisits and singleVisitDepth must match")
        coaddedDepth[f] = float(
            1.25 * np.log10(nvisits[f] * 10 ** (0.8 * singleVisitDepth[f]))
        )
        if not np.isfinite(coaddedDepth[f]):
            coaddedDepth[f] = singleVisitDepth[f]
    return coaddedDepth
