# Collection of utilities for MAF that relate to Opsim specifically.

import numpy as np
import pandas as pd
import sqlite3
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import make_url

__all__ = [
    "getSimData",
    "scaleBenchmarks",
    "calcCoaddedDepth",
]


def getSimData(
    db_con,
    sqlconstraint,
    dbcols,
    stackers=None,
    tableName=None,
    full_sql_query=None,
):
    """Query an opsim database for the needed data columns and run any required stackers.

    Parameters
    ----------
    db_con : `str` or SQLAlchemy connectable, or sqlite3 connection
        Filename to a sqlite3 file, or a connection object that can be used by pandas.read_sql
    sqlconstraint : `str`
        SQL constraint to apply to query for observations. Ignored if full_sql_query is set.
    dbcols : `list` [`str`]
        Columns required from the database. Ignored if full_sql_query is set.
    stackers : `list` [`rubin_sim.maf.stackers`], optional
        Stackers to be used to generate additional columns. Default None.
    tableName : `str` (None)
        Name of the table to query. Default None will try "observations" and "SummaryAllProps".
        Ignored if full_sql_query is set.
    full_sql_query : `str`
        The full SQL query to use. Overrides sqlconstraint, dbcols, tablename.

    Returns
    -------
    simData: `np.ndarray`
        A numpy structured array with columns resulting from dbcols + stackers, for observations matching
        the SQLconstraint.
    """

    # Check if table is "observations" or "SummaryAllProps"
    if (tableName is None) & (full_sql_query is None) & (type(db_con) == str):
        url = make_url("sqlite:///" + db_con)
        eng = create_engine(url)
        inspector = inspect(eng)
        tables = [
            inspector.get_table_names(schema=schema)
            for schema in inspector.get_schema_names()
        ]
        if "observations" in tables[0]:
            tableName = "observations"
        elif "SummaryAllProps" in tables[0]:
            tableName = "SummaryAllProps"
        elif "summary" in tables[0]:
            tableName = "summary"
        else:
            ValueError(
                "Could not guess tableName, set with tableName or full_sql_query kwargs"
            )
    elif (tableName is None) & (full_sql_query is None):
        # If someone passes in a connection object with an old tableName things will fail
        # that's probably fine, keep people from getting fancy with old sims
        tableName = "observations"

    if type(db_con) == str:
        con = sqlite3.connect(db_con)
    else:
        con = db_con

    if full_sql_query is None:
        col_str = ""
        for colname in dbcols:
            col_str += colname + ", "
        col_str = col_str[0:-2] + " "

        # Need to guess "observations" and "SummaryAllProps" for the table name
        # to be backwards compatible I guess
        query = "SELECT %s FROM %s" % (col_str, tableName)
        if len(sqlconstraint) > 0:
            query += " WHERE %s" % (sqlconstraint)
        query += ";"
        simData = pd.read_sql(query, con).to_records(index=False)

    else:
        query = full_sql_query
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
