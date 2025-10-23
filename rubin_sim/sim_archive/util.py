import hashlib
import sqlite3
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time


def dayobs_to_date(dayobs: str | date | int | Time) -> date:
    """Convert dayobs in as a str, date, int, or astropyTime
    to a python datetime.date.

    Parameters
    ----------
    dayobs: `str` or `datetime.date` or `int` or `astropy.time.Time`
        The date of observation an flexible format.

    Return
    ------
    dayobs_date : `datetime.date`
        The date of observation as a ``datetime.date``.
    """
    match dayobs:
        case int():
            year = dayobs // 10000
            month = (dayobs // 100) % 100
            day = dayobs % 100
            dayobs = date(year, month, day)
        case str():
            dayobs = datetime.fromisoformat(dayobs).date()
        case Time():
            # Pacify type checkers
            assert isinstance(dayobs, Time)
            dayobs_dt = dayobs.to_datetime(timezone=timezone(timedelta(hours=-12)))
            if isinstance(dayobs_dt, datetime):
                dayobs = dayobs_dt.date()
            else:
                assert isinstance(dayobs_dt, np.ndarray)
                assert len(dayobs_dt) == 1
                dayobs = dayobs_dt.item().date()
                assert isinstance(dayobs, date)
        case datetime():
            dayobs = dayobs.date()
        case _:
            assert isinstance(dayobs, date)

    assert isinstance(dayobs, date)
    return dayobs


def opsimdb_to_hdf5(opsimdb_path: str | Path, hdf5_path: str | Path | None = None) -> str:
    """Convert an opsim sqlite3 format database into an hdf5
    format table store.

    Parameters
    ----------
    opsimdb_path : `str` or `Path`
        Path to an opsim sqlite3-format database.
    hdf5_path : `str` or `Path`
        Path to the output HDF5 file to be created.

    Returns
    -------
    hdf5_path : `str`
        The path of the written file.
    """

    if hdf5_path is None:
        hdf5_path = Path(opsimdb_path).with_suffix(".h5")

    conn = sqlite3.connect(str(opsimdb_path))

    try:
        # Get all table names
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        # Write each table to the HDF5 file
        hdf5_mode = "w"
        for table in tables:
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            df.to_hdf(str(hdf5_path), key=table, mode=hdf5_mode)
            hdf5_mode = "a"

    finally:
        conn.close()

    return str(hdf5_path)


def hdf5_to_opsimdb(hdf5_path: str | Path, opsimdb_path: str | Path | None = None) -> str:
    """Convert an hdf5 datastore with opsim output into an opsim sqlite3 file.

    Parameters
    ----------
    hdf5_path : `str`
        Path to the input HDF5 file.
    opsimdb_path : `str`
        Path to the output SQLite database file to be created.

    Returns
    -------
    opsimdb_path : `str`
        The path of the written file.
    """
    if opsimdb_path is None:
        opsimdb_path = Path(hdf5_path).with_suffix(".db")

    conn = sqlite3.connect(str(opsimdb_path))

    try:
        with pd.HDFStore(str(hdf5_path), mode="r") as store:
            for key in store.keys():
                # Remove leading '/' from key name
                table_name = key.lstrip("/")
                df = pd.read_hdf(hdf5_path, key=key)
                df.to_sql(table_name, conn, index=False)

    finally:
        conn.close()

    return str(opsimdb_path)


def compute_conda_env() -> tuple:
    """Find the current conda environment and its SHA‑256 hash.

    Returns
    -------
    tuple
        A two‑element tuple ``(conda_env_hash, conda_env_json)`` where
        ``conda_env_hash`` is a `bytes` instance containing the
        SHA‑256 digest, and ``conda_env_json`` is a `str` holding
        the JSON output of ``conda list --json``.
    """

    conda_list_result = subprocess.run(
        ["conda", "list", "--json"], capture_output=True, text=True, check=True
    )
    conda_env_json = conda_list_result.stdout
    conda_env_hash = bytes.fromhex(hashlib.sha256(conda_env_json.encode()).hexdigest())
    return conda_env_hash, conda_env_json
