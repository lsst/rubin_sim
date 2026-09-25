"""Test data generation helpers for chimera progress tests.

This module provides functions to generate sample test data by sampling from
the baseline opsim database.
"""

import numpy as np
import pandas as pd
from rubin_sim.data import get_baseline
from rubin_sim.maf.utils.opsim_utils import get_sim_data
from rubin_sim.maf.stackers.date_stackers import DayObsStacker


def _dayobs_to_mjd(dayobs: int) -> float:
    """Convert YYYYMMDD dayObs to MJD.

    Parameters
    ----------
    dayobs : int
        dayObs in YYYYMMDD format (e.g., 20260101).

    Returns
    -------
    mjd : float
        Modified Julian Date.
    """
    import datetime
    # Parse YYYYMMDD
    year = dayobs // 10000
    month = (dayobs % 10000) // 100
    day = dayobs % 100

    # Convert to datetime
    dt = datetime.datetime(year, month, day)

    # Calculate days since MJD epoch (1858-11-17)
    mjd_epoch = datetime.datetime(1858, 11, 17)
    delta = dt - mjd_epoch
    days_since_epoch = delta.days

    # MJD = days since epoch + fractional day (noon UTC)
    mjd = days_since_epoch + 0.5

    return mjd


def make_sample_opsim_visits(n_visits: int = 500, random_state: int | None = 42) -> pd.DataFrame:
    """Generate synthetic opsim visits by sampling from baseline database.

    Samples visits with dayObs in the range 20260101-20260228.

    Parameters
    ----------
    n_visits : int, optional
        Number of visits to generate. Default 500.
    random_state : int or None, optional
        Random state for reproducible sampling. Default 42.

    Returns
    -------
    visits : pandas.DataFrame
        Synthetic visits with dayObs column.
    """
    rng = np.random.default_rng(random_state)

    # Get baseline database
    baseline_path = get_baseline()

    # Define the dayObs range we want
    start_dayobs = 20260101
    end_dayobs = 20260228

    # Convert to MJD range for SQL filtering
    start_mjd = _dayobs_to_mjd(start_dayobs)
    end_mjd = _dayobs_to_mjd(end_dayobs)

    # Load visits within the desired dayObs range
    sql = f"observationStartMJD >= {start_mjd} AND observationStartMJD <= {end_mjd}"
    stackers = [DayObsStacker()]
    sim_data = get_sim_data(baseline_path, sqlconstraint=sql, stackers=stackers)

    # Convert to DataFrame
    df = pd.DataFrame(sim_data)

    # Randomly sample n_visits
    if len(df) > n_visits:
        df = df.sample(n=n_visits, random_state=random_state)

    return df


def make_sample_consdb_visits(
    n_visits: int = 100, days: int = 60, random_state: int | None = 42
) -> pd.DataFrame:
    """Generate synthetic consdb visits from first N days of baseline.

    Samples visits with dayObs in the range 20260101-20260130.

    Parameters
    ----------
    n_visits : int, optional
        Number of visits to generate. Default 100.
    days : int, optional
        Number of days to generate visits within. Default 60 (2 months).
    random_state : int or None, optional
        Random state for reproducible sampling. Default 42.

    Returns
    -------
    visits : pandas.DataFrame
        Synthetic visits with dayObs column.
    """
    rng = np.random.default_rng(random_state)

    # Get baseline database
    baseline_path = get_baseline()

    # Define the dayObs range we want (shorter range for consdb)
    start_dayobs = 20260101
    end_dayobs = 20260130

    # Convert to MJD range for SQL filtering
    start_mjd = _dayobs_to_mjd(start_dayobs)
    end_mjd = _dayobs_to_mjd(end_dayobs)

    # Load visits within the desired dayObs range
    sql = f"observationStartMJD >= {start_mjd} AND observationStartMJD <= {end_mjd}"
    stackers = [DayObsStacker()]
    sim_data = get_sim_data(baseline_path, sqlconstraint=sql, stackers=stackers)

    # Convert to DataFrame
    df = pd.DataFrame(sim_data)

    # Randomly sample n_visits
    if len(df) > n_visits:
        df = df.sample(n=n_visits, random_state=random_state)

    return df


def add_dayobs_column(df: pd.DataFrame, mjd_col: str = "observationStartMJD") -> pd.DataFrame:
    """Add a dayObs column to a DataFrame using DayObsStacker.

    The dayObs is computed as defined by SITCOMTN-32 (UTC-12 hours).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with an MJD column.
    mjd_col : str, optional
        Name of the MJD column. Default "observationStartMJD".

    Returns
    -------
    df : pandas.DataFrame
        Copy of input DataFrame with dayObs column added.
    """
    # Convert to recarray for stacker
    rec_array = df.to_records(index=False)

    # Run DayObsStacker
    stacker = DayObsStacker(mjd_col=mjd_col)
    result = stacker.run(rec_array)

    # Convert back to DataFrame
    return pd.DataFrame(result)
