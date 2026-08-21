__all__ = (
    "get_sim_data",
    "get_visit_data",
    "scale_benchmarks",
    "calc_coadded_depth",
)

import os
import sqlite3
import urllib
from contextlib import closing
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import make_url


def _local_get_sim_data(
    db_con,
    sqlconstraint=None,
    dbcols=None,
    stackers=None,
    table_name=None,
    full_sql_query=None,
    *,
    return_class=np.recarray,
):
    if sqlconstraint is None:
        sqlconstraint = ""

    need_sql: bool = full_sql_query is not None or len(sqlconstraint) > 0

    # Check that file exists
    if isinstance(db_con, str):
        if os.path.isfile(db_con) is False:
            raise FileNotFoundError("No file %s" % db_con)

    # Check what type of source was provided, defaulting
    # to an sqlite3 file if it is a string without an extinsion
    # that identifies it as a parquet or hdf5 file name.
    source_type: str = "sqlite3"
    if isinstance(db_con, str) and db_con.lower().endswith((".h5", ".hdf5")):
        source_type = "hdf5"
    elif isinstance(db_con, str) and db_con.lower().endswith((".parquet", ".pq", ".parq")):
        source_type = "parquet"
    elif isinstance(db_con, sqlite3.Connection):
        source_type = "connection"

    # Check if table is "observations" or "SummaryAllProps"
    if (table_name is None) & (full_sql_query is None) & (isinstance(db_con, str)):
        if source_type == "hdf5":
            # For HDF5, detect table names by reading the store keys
            with pd.HDFStore(db_con, mode="r") as store:
                table_keys = [key.lstrip("/") for key in store.keys()]
            if "observations" in table_keys:
                table_name = "observations"
            elif "SummaryAllProps" in table_keys:
                table_name = "SummaryAllProps"
            elif "summary" in table_keys:
                table_name = "summary"
            else:
                raise ValueError("Could not guess table_name, set with table_name or full_sql_query kwargs")
        elif source_type == "parquet":
            table_name = "observations"
        else:
            url = make_url("sqlite:///" + db_con)
            eng = create_engine(url)
            inspector = inspect(eng)
            tables = [inspector.get_table_names(schema=schema) for schema in inspector.get_schema_names()]
            if "observations" in tables[0]:
                table_name = "observations"
            elif "SummaryAllProps" in tables[0]:
                table_name = "SummaryAllProps"
            elif "summary" in tables[0]:
                table_name = "summary"
            else:
                raise ValueError("Could not guess table_name, set with table_name or full_sql_query kwargs")
    elif (table_name is None) & (full_sql_query is None):
        # If someone passes in a connection object with an old table_name
        # things will fail
        # that's probably fine, keep people from getting fancy with old sims
        table_name = "observations"

    if full_sql_query is None:
        col_str = "*" if dbcols is None else ", ".join(dbcols)

        query = "SELECT %s FROM %s" % (col_str, table_name)
        if len(sqlconstraint) > 0:
            query += " WHERE %s" % (sqlconstraint)
        query += ";"
    else:
        query = full_sql_query

    sim_data: np.recarray | pd.DataFrame | None = None

    match source_type:
        case "connection":
            sim_data = pd.read_sql(query, db_con)
        case "sqlite3":
            with closing(sqlite3.connect(db_con)) as con:
                sim_data = pd.read_sql(query, con)
        case "hdf5":
            if need_sql:
                with closing(sqlite3.connect(":memory:")) as con:
                    with pd.HDFStore(db_con, mode="r") as store:
                        for key in store.keys():
                            table_name = key.lstrip("/")
                            table_values = pd.read_hdf(db_con, key=key)
                            table_values.to_sql(table_name, con, index=False)
                    sim_data = pd.read_sql(query, con)
            else:
                sim_data = pd.read_hdf(db_con, key=table_name)
        case "parquet":
            if need_sql:
                with closing(sqlite3.connect(":memory:")) as con:
                    raw_observations = pd.read_parquet(db_con)
                    raw_observations.to_sql("observations", con, index=False)
                    sim_data = pd.read_sql(query, con)
            else:
                sim_data = pd.read_parquet(db_con)
        case _:
            raise RuntimeError(f"Cannot find {db_con}.")

    if len(sim_data) == 0:
        raise UserWarning("No data found matching sqlconstraint %s" % (sqlconstraint))

    # Now add the stacker columns.
    # This fails for pandas.DataFrames, so convert to recarray
    # if necessary.
    if stackers is not None:
        if not isinstance(sim_data, np.recarray):
            assert isinstance(sim_data, pd.DataFrame)
            sim_data = sim_data.to_records(index=False)
        for s in stackers:
            sim_data = s.run(sim_data).view(np.recarray)

    if return_class is np.recarray:
        if isinstance(sim_data, pd.DataFrame):
            sim_data = sim_data.to_records(index=False)
    if return_class is pd.DataFrame:
        if isinstance(sim_data, np.recarray):
            sim_data = pd.DataFrame(sim_data)

    assert isinstance(sim_data, return_class)

    return sim_data


def save_visits_as_parquet(
    visits: Union[pd.DataFrame, Any],
    parquet_file_path: Union[str, Path],
) -> None:
    """Save visit data to a Parquet file.

    Parameters
    ----------
    visits : `pandas.DataFrame` or Any
        Visit-like tabular data. If not already a `pandas.DataFrame`, the input
        must be convertible via ``pandas.DataFrame(visits)``.
    parquet_file_path : `str` or `pathlib.Path`
        Destination path for the output Parquet file. Parent directories are
        created automatically if they do not exist.

    Notes
    -----
    This function works around a few issues with pandas auto-detection
    of column types for columns with NaN or None values.
    """

    # Normalize path and ensure parent directory exists.
    parquet_path = Path(parquet_file_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert input to DataFrame, and ensure our adjustments
    # do not alter the originally passed data.
    if isinstance(visits, pd.DataFrame):
        cleaned_visits = visits.copy(deep=True)
    else:
        try:
            cleaned_visits = pd.DataFrame(visits).copy(deep=True)
        except Exception as exc:
            raise TypeError("visits must be a pandas.DataFrame or a DataFrame-compatible object.") from exc

    if "visit_id" in cleaned_visits.columns:
        cleaned_visits = cleaned_visits.set_index("visit_id", drop=False).rename_axis(index=None)

    if "index" in cleaned_visits.columns:
        cleaned_visits = cleaned_visits.drop(columns=["index"])

    # Convert object columns that are entirely NaN to float for parquet.
    obj_cols = cleaned_visits.select_dtypes(include=["object"]).columns
    all_nan_obj_cols = [c for c in obj_cols if cleaned_visits[c].isna().all()]
    if all_nan_obj_cols:
        cleaned_visits[all_nan_obj_cols] = cleaned_visits[all_nan_obj_cols].astype("float64")

    # For object columns that contain only strings (ignoring nulls),
    # replace NaN with "".
    string_columns = []
    for col in cleaned_visits.select_dtypes(include=["object"]).columns:
        non_null = cleaned_visits[col].dropna()
        if non_null.empty or non_null.map(lambda v: isinstance(v, str)).all():
            string_columns.append(col)

    if string_columns:
        cleaned_visits[string_columns] = cleaned_visits[string_columns].fillna("")

    cleaned_visits.to_parquet(parquet_path, index=True)


def get_sim_data(
    db_con,
    sqlconstraint=None,
    dbcols=None,
    stackers=None,
    table_name=None,
    full_sql_query=None,
    *,
    return_class=np.recarray,
):
    """Query an opsim database for the needed data columns
    and run any required stackers.

    Parameters
    ----------
    db_con : `str` or SQLAlchemy connectable, or sqlite3 connection
        Filename or lsst.resources path to an hdf5 or sqlite3 file,
        or a connection object that can be used by pandas.read_sql
    sqlconstraint : `str` or None
        SQL constraint to apply to query for observations.
        Ignored if full_sql_query is set.
    dbcols : `list` [`str`]
        Columns required from the database. Ignored if full_sql_query is set.
    stackers : `list` [`rubin_sim.maf.stackers`], optional
        Stackers to be used to generate additional columns. Default None.
    table_name : `str` (None)
        Name of the table to query.
        Default None will try "observations".
        Ignored if full_sql_query is set.
    full_sql_query : `str`
        The full SQL query to use. Overrides sqlconstraint, dbcols, tablename.

    Returns
    -------
    sim_data : `np.ndarray`
        A numpy structured array with columns resulting from dbcols + stackers,
        for observations matching the SQLconstraint.
    """
    if (not isinstance(db_con, str)) or urllib.parse.urlparse(db_con).scheme == "":
        # Already have a local copy
        sim_data = _local_get_sim_data(
            db_con, sqlconstraint, dbcols, stackers, table_name, full_sql_query, return_class=return_class
        )
    else:
        try:
            from lsst.resources import ResourcePath

            with ResourcePath(db_con).as_local() as local_db_path:
                sim_data = _local_get_sim_data(
                    local_db_path,
                    sqlconstraint,
                    dbcols,
                    stackers,
                    table_name,
                    full_sql_query,
                    return_class=return_class,
                )
        except ModuleNotFoundError:
            raise RuntimeError(
                f"Cannot read visits from {db_con}."
                "Maybe it does not exist, or maybe you need to install lsst.resources."
            )
    return sim_data


# This almost-alias to get_sim_data is named to be less misleading,
# in that the same get_*_data function can be used for real visits
# from consdb as well. Return a DF instead of a recarray by default
# because much of the code designed for reading consdb output
# wants that.
def get_visit_data(
    db_con,
    sqlconstraint=None,
    dbcols=None,
    stackers=None,
    table_name=None,
    full_sql_query=None,
    *,
    return_class=pd.DataFrame,
):
    """Query an opsim database, returning a `pandas.DataFrame` by default.

    Thin wrapper around `get_sim_data` with ``return_class`` defaulting to
    `pandas.DataFrame` instead of `numpy.recarray`.  All parameters are
    forwarded unchanged; see `get_sim_data` for full documentation.

    Parameters
    ----------
    db_con : `str` or SQLAlchemy connectable, or sqlite3 connection
        Filename or lsst.resources path to an hdf5 or sqlite3 file,
        or a connection object that can be used by pandas.read_sql
    sqlconstraint : `str` or None
        SQL constraint to apply to query for observations.
        Ignored if full_sql_query is set.
    dbcols : `list` [`str`]
        Columns required from the database. Ignored if full_sql_query is set.
    stackers : `list` [`rubin_sim.maf.stackers`], optional
        Stackers to be used to generate additional columns. Default None.
    table_name : `str` (None)
        Name of the table to query.
        Default None will try "observations".
        Ignored if full_sql_query is set.
    full_sql_query : `str`
        The full SQL query to use. Overrides sqlconstraint, dbcols, tablename.
    return_class : `type`, optional
        Class of the returned data. Default `pandas.DataFrame`.

    Returns
    -------
    sim_data : `pandas.DataFrame`
        A `pandas.DataFrame` with columns resulting from dbcols + stackers,
        for observations matching the sqlconstraint.

    See Also
    --------
    get_sim_data : Equivalent function with ``return_class`` defaulting to
        `numpy.recarray`.
    """
    return get_sim_data(
        db_con,
        sqlconstraint=sqlconstraint,
        dbcols=dbcols,
        stackers=stackers,
        table_name=table_name,
        full_sql_query=full_sql_query,
        return_class=return_class,
    )


def scale_benchmarks(run_length, benchmark="design"):
    """Set design and stretch values of the number of visits or
    area of the footprint or seeing/Fwhmeff/skybrightness and single visit
    depth (based on SRD values).
    Scales number of visits for the length of the run, relative to 10 years.

    Parameters
    ----------
    run_length : `float`
        The length (in years) of the run.
    benchmark : `str`
        design or stretch - which version of the SRD values to return.

    Returns
    -------
    benchmarks: `dict` of floats
       A dictionary containing the number of visits, area of footprint,
       seeing and FWHMeff values, skybrightness and single visit depth
       for either the design or stretch SRD values.
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
    if run_length != baseline:
        scalefactor = float(run_length) / float(baseline)
        # Calculate scaled value for design and stretch values of nvisits,
        # per filter.
        for f in design["nvisits"]:
            design["nvisits"][f] = int(np.floor(design["nvisits"][f] * scalefactor))
            stretch["nvisits"][f] = int(np.floor(stretch["nvisits"][f] * scalefactor))

    if benchmark == "design":
        return design
    elif benchmark == "stretch":
        return stretch
    else:
        raise ValueError("Benchmark value %s not understood: use 'design' or 'stretch'" % (benchmark))


def calc_coadded_depth(nvisits, single_visit_depth):
    """Calculate the coadded depth expected for a given number of visits
    and single visit depth.

    Parameters
    ----------
    nvisits : `dict` of `int` or `float`
        Dictionary (per filter) of number of visits
    single_visit_depth : `dict` of `float`
        Dictionary (per filter) of the single visit depth

    Returns
    -------
    coadded_depth : `dict` of `float`
        Dictionary of coadded depths per filter.
    """
    coadded_depth = {}
    for f in nvisits:
        if f not in single_visit_depth:
            raise ValueError("Filter keys in nvisits and single_visit_depth must match")
        coadded_depth[f] = float(1.25 * np.log10(nvisits[f] * 10 ** (0.8 * single_visit_depth[f])))
        if not np.isfinite(coadded_depth[f]):
            coadded_depth[f] = single_visit_depth[f]
    return coadded_depth
