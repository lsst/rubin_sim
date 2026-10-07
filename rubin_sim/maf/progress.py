__all__ = (
    "dayobs_range",
    "build_chimera",
    "build_chimeras",
)

import ast
import datetime
import glob
import os
import re
import warnings
from collections.abc import Callable

import click
import pandas as pd

import rubin_sim.maf.batches as batches
import rubin_sim.maf.db as db
import rubin_sim.maf.metric_bundles as mb
from rubin_sim.maf.stackers.date_stackers import DayObsStacker
from rubin_sim.maf.utils.opsim_utils import get_sim_data

# Default values for consdb columns without valid values.
CONSDB_DEFAULTS = {"exposures": 1}


def dayobs_range(start_dayobs: int, end_dayobs: int, step: int = 1) -> list[int]:
    """Generate a list of integer ``dayObs`` values between two dates.

    ``dayObs`` values are integer dates formatted as ``YYYYMMDD`` in the
    UTC-12 observing day convention. The returned list begins at
    ``start_dayobs`` and includes each subsequent date separated by ``step``
    nights, up to and including ``end_dayobs`` whenever the step lands exactly
    on or before it.

    Parameters
    ----------
    start_dayobs : `int`
        First date in the range, formatted as ``YYYYMMDD``.
    end_dayobs : `int`
        Last date in the range, formatted as ``YYYYMMDD``.
    step : `int`, optional
        Number of days to advance between successive values. Must be a
        positive integer. Default is 1.

    Returns
    -------
    dayobs_list : `list` [`int`]
        List of integer ``YYYYMMDD`` dayObs values from ``start_dayobs`` to
        ``end_dayobs`` (inclusive), spaced by ``step`` days.

    Raises
    ------
    ValueError
        If ``step`` is not a positive integer.
    """

    if step <= 0:
        raise ValueError("step must be a positive integer.")

    def _dayobs_to_date(dayobs: int) -> datetime.date:
        s = f"{int(dayobs):08d}"
        return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:]))

    current = _dayobs_to_date(start_dayobs)
    end_date = _dayobs_to_date(end_dayobs)
    result = []
    while current <= end_date:
        dayobs = int(current.strftime("%Y%m%d"))
        result.append(dayobs)
        current += datetime.timedelta(days=step)
    return result


def _run_name_from_dayobs(transition_dayobs: int) -> str:
    """Return the run name string for a given transition dayobs."""
    return f"chimera_{int(transition_dayobs):08d}"


def _dayobs_from_run_name(run_name: str) -> int | None:
    """Extract transition dayobs integer from a chimera run name, or None."""
    m = re.match(r"^chimera_(\d{8})$", run_name)
    return int(m.group(1)) if m else None


def _dayobs_from_filename(path: str) -> int | None:
    """Extract transition dayobs integer from a chimera HDF5
    filename, or None."""
    basename = os.path.basename(path)
    m = re.match(r"^chimera_(\d{8})\.h5$", basename)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Core Python API
# ---------------------------------------------------------------------------


def build_chimera(
    consdb_visits: pd.DataFrame,
    opsim_visits: pd.DataFrame,
    start_dayobs: int,
    transition_dayobs: int,
    end_dayobs: int,
) -> pd.DataFrame:
    """Build a single chimera visit sequence.

    Combines consdb visits in [start_dayobs, transition_dayobs] with opsim
    visits in (transition_dayobs, end_dayobs].  Both input DataFrames must
    already have a ``dayObs`` column (integer YYYYMMDD, UTC-12).

    Parameters
    ----------
    consdb_visits : `pandas.DataFrame`
        Real visits from consdb. Must include a ``dayObs`` column.
    opsim_visits : `pandas.DataFrame`
        Simulated visits from an opsim database. Must include a ``dayObs``
        column.
    start_dayobs : `int`
        Start of the chimera window, YYYYMMDD inclusive.
    transition_dayobs : `int`
        Transition date; consdb visits up to and including this date are used.
    end_dayobs : `int`
        End of the chimera window, YYYYMMDD inclusive.

    Returns
    -------
    chimera : `pandas.DataFrame`
        Combined visit sequence containing columns present in both inputs.
    """
    consdb_part = consdb_visits.loc[
        (consdb_visits["dayObs"] >= int(start_dayobs)) & (consdb_visits["dayObs"] <= int(transition_dayobs))
    ].copy()

    consdb_part = consdb_part.loc[consdb_part["fiveSigmaDepth"] > FIVE_SIGMA_DEPTH_LIMIT].copy()

    # Mark which visits were simulated, and which not
    consdb_part["simulated"] = False

    # Fix columns from consdb that can be missing or have bad values
    for column in CONSDB_DEFAULTS:
        if column not in consdb_part.columns:
            consdb_part[column] = CONSDB_DEFAULTS[column]
        else:
            consdb_part[column] = consdb_part[column].fillna(CONSDB_DEFAULTS[column])

    opsim_part = opsim_visits.loc[
        (opsim_visits["dayObs"] > int(transition_dayobs))
        & (opsim_visits["dayObs"] <= int(end_dayobs))
        & (opsim_visits["fiveSigmaDepth"] > FIVE_SIGMA_DEPTH_LIMIT)
    ].copy()
    opsim_part["simulated"] = True

    common_cols = sorted(set(consdb_part.columns) & set(opsim_part.columns))
    if not common_cols:
        raise ValueError("consdb_visits and opsim_visits share no common columns; " "cannot build a chimera.")

    return pd.concat(
        [consdb_part[common_cols], opsim_part[common_cols]],
        ignore_index=True,
    )


def build_chimeras(
    consdb_visits: pd.DataFrame,
    opsim_visits: pd.DataFrame,
    start_dayobs: int,
    end_dayobs: int,
    step: int = 1,
    out_dir: str = ".",
) -> list[tuple[int, str]]:
    """Build chimera visit sequences for a range of transition dates.

    For each transition date, a chimera is constructed by combining real
    consdb visits up to that date with simulated opsim visits after it, then
    saved as an HDF5 file named ``chimera_YYYYMMDD.h5``.

    Both input DataFrames must already have a ``dayObs`` column (integer
    YYYYMMDD, UTC-12).

    Parameters
    ----------
    consdb_visits : `pandas.DataFrame`
        Real visits from consdb. Must include a ``dayObs`` column.
    opsim_visits : `pandas.DataFrame`
        Simulated visits from an opsim database. Must include a ``dayObs``
        column.
    start_dayobs : `int`
        First date in the chimera window, YYYYMMDD.
    end_dayobs : `int`
        End of the opsim extension window used for every chimera, YYYYMMDD.
    step : `int`, optional
        Number of nights between successive transition dates.  Default 1.
    out_dir : `str`, optional
        Directory in which to write HDF5 files.  Created if absent.

    Returns
    -------
    chimera_specs : `list` of `(int, str)`
        List of ``(transition_dayobs, hdf5_path)`` tuples, one per chimera.
        The last transition date is always the maximum dayobs present in
        ``consdb_visits``, even if it does not fall on the step cadence.
    """
    os.makedirs(out_dir, exist_ok=True)
    last_consdb = int(consdb_visits["dayObs"].max())
    transition_dates = dayobs_range(start_dayobs, last_consdb, step)
    if not transition_dates or transition_dates[-1] != last_consdb:
        transition_dates.append(last_consdb)
    chimera_specs = []
    for t in transition_dates:
        chimera = build_chimera(consdb_visits, opsim_visits, start_dayobs, t, end_dayobs)
        fname = os.path.join(out_dir, f"chimera_{t:08d}.h5")
        chimera.to_hdf(fname, key="observations", complevel=5)
        chimera_specs.append((t, fname))
    return chimera_specs




# ---------------------------------------------------------------------------
# CLI helpers: read visit sequences from files
# ---------------------------------------------------------------------------


def _read_visits_with_dayobs(path: str) -> pd.DataFrame:
    """Read a visit sequence (SQLite or HDF5) and add a dayObs column."""
    sim_data = get_sim_data(path, sqlconstraint="", stackers=[DayObsStacker()])
    return pd.DataFrame(sim_data)


# ---------------------------------------------------------------------------
# Click CLI commands
# ---------------------------------------------------------------------------


@click.command(name="build_chimeras")
@click.option(
    "--consdb-file",
    required=True,
    type=click.Path(exists=True),
    help="SQLite or HDF5 file with consdb visits.",
)
@click.option(
    "--opsim-file",
    required=True,
    type=click.Path(exists=True),
    help="SQLite or HDF5 file with opsim visits.",
)
@click.option("--start-dayobs", required=True, type=int, help="Start date YYYYMMDD.")
@click.option("--end-dayobs", required=True, type=int, help="End date YYYYMMDD.")
@click.option("--step", default=1, show_default=True, type=int, help="Nights between transition dates.")
@click.option("--out-dir", default=".", show_default=True, help="Output directory for chimera HDF5 files.")
def build_chimeras_cmd(consdb_file, opsim_file, start_dayobs, end_dayobs, step, out_dir):
    """Build chimera visit sequences and save them as HDF5 files.

    Each HDF5 file is named chimera_YYYYMMDD.h5, where YYYYMMDD is the
    transition date that separates real consdb visits from simulated
    opsim visits.
    """
    consdb_visits = _read_visits_with_dayobs(consdb_file)
    opsim_visits = _read_visits_with_dayobs(opsim_file)
    specs = build_chimeras(consdb_visits, opsim_visits, start_dayobs, end_dayobs, step, out_dir)
    click.echo(f"Wrote {len(specs)} chimera files to {out_dir}.")

