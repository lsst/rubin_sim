__all__ = (
    "build_chimera",
    "build_chimeras",
    "run_chimera_batches",
    "make_chimera_summary_table",
)

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

# ---------------------------------------------------------------------------
# DayObs helpers
# DayObs is an integer YYYYMMDD in UTC-12, so it doesn't roll over mid-night.
# ---------------------------------------------------------------------------


def _dayobs_to_date(dayobs: int) -> datetime.date:
    """Convert an integer YYYYMMDD dayobs to a datetime.date."""
    s = f"{int(dayobs):08d}"
    return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:]))


def _date_to_dayobs(d: datetime.date) -> int:
    """Convert a datetime.date to an integer YYYYMMDD dayobs."""
    return int(d.strftime("%Y%m%d"))


def _dayobs_range(start_dayobs: int, end_dayobs: int, step: int = 1) -> list[int]:
    """Return a list of integer dayobs values from start to end (inclusive)."""
    current = _dayobs_to_date(start_dayobs)
    end_date = _dayobs_to_date(end_dayobs)
    result = []
    while current <= end_date:
        result.append(_date_to_dayobs(current))
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
    ]
    opsim_part = opsim_visits.loc[
        (opsim_visits["dayObs"] > int(transition_dayobs)) & (opsim_visits["dayObs"] <= int(end_dayobs))
    ]

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
    transition_dates = _dayobs_range(start_dayobs, last_consdb, step)
    if not transition_dates or transition_dates[-1] != last_consdb:
        transition_dates.append(last_consdb)
    chimera_specs = []
    for t in transition_dates:
        chimera = build_chimera(consdb_visits, opsim_visits, start_dayobs, t, end_dayobs)
        fname = os.path.join(out_dir, f"chimera_{t:08d}.h5")
        chimera.to_hdf(fname, key="observations", complevel=5)
        chimera_specs.append((t, fname))
    return chimera_specs


def run_chimera_batches(
    chimera_specs: list[tuple[int, str]],
    batch_func: Callable[..., dict] | None = None,
    out_dir: str = ".",
) -> str:
    """Run MAF metric batches on a collection of chimera visit sequences.

    Each chimera is processed with ``batch_func``, which should return a
    dictionary of ``MetricBundle`` objects.  All runs share a single
    ``ResultsDb`` in ``out_dir``, with run names of the form
    ``chimera_YYYYMMDD`` encoding the transition date.

    Parameters
    ----------
    chimera_specs : `list` of `(int, str)`
        List of ``(transition_dayobs, hdf5_path)`` tuples as returned by
        `build_chimeras`.
    batch_func : callable, optional
        Function with signature ``batch_func(run_name=...) -> dict``
        or ``batch_func(runName=...) -> dict``
        Defaults to `rubin_sim.maf.batches.glanceBatch`.
    out_dir : `str`, optional
        Directory for results_db and metric output files.

    Returns
    -------
    results_db_path : `str`
        Path to the shared ``resultsDb_sqlite.db`` file.
    """
    if batch_func is None:
        batch_func = batches.glanceBatch

    os.makedirs(out_dir, exist_ok=True)
    results_db = db.ResultsDb(out_dir=out_dir)

    for transition_dayobs, hdf5_path in chimera_specs:
        run_name = _run_name_from_dayobs(transition_dayobs)
        try:
            bdict = batch_func(run_name=run_name)
        except TypeError as batch_error:
            if "got an unexpected keyword argument 'run_name'" not in str(batch_error):
                # we got some other exception, just pass it along.
                raise
            # We have a batch that uses runName instead of run_name.
            bdict = batch_func(runName=run_name)

        group = mb.MetricBundleGroup(
            bdict,
            hdf5_path,
            out_dir=out_dir,
            results_db=results_db,
            save_early=False,
        )
        group.run_all(clear_memory=True)

    results_db.close()
    return os.path.join(out_dir, "resultsDb_sqlite.db")


def make_chimera_summary_table(results_db: db.ResultsDb | str) -> pd.DataFrame:
    """Build a summary table from chimera run results.

    Queries the ``ResultsDb`` for all runs whose names match the
    ``chimera_YYYYMMDD`` pattern and returns a wide-format DataFrame with
    one row per transition date and one column per summary metric.

    Parameters
    ----------
    results_db : `rubin_sim.maf.db.ResultsDb` or `str`
        An open ``ResultsDb`` instance, or a path to a ``resultsDb_sqlite.db``
        file.

    Returns
    -------
    summary_table : `pandas.DataFrame`
        DataFrame indexed by ``transition_dayobs`` (integer YYYYMMDD) with a
        ``MultiIndex`` column of
        ``(metric_name, slicer_name, metric_info_label, summary_metric)``.
    """
    close_after = False
    if isinstance(results_db, str):
        results_db = db.ResultsDb(database=results_db)
        close_after = True

    # Get all run_names that look like chimera runs and find their metric IDs.
    all_run_names = results_db.get_run_name()
    chimera_run_names = [r for r in all_run_names if _dayobs_from_run_name(r) is not None]

    if not chimera_run_names:
        warnings.warn("No chimera run names found in results_db.")
        if close_after:
            results_db.close()
        return pd.DataFrame()

    # Collect all metric IDs for chimera runs.
    metric_ids = []
    for run_name in chimera_run_names:
        results_db.open()
        ids = (
            results_db.session.query(db.results_db.MetricRow.metric_id)
            .filter(db.results_db.MetricRow.run_name == run_name)
            .all()
        )
        results_db.close()
        metric_ids.extend(i[0] for i in ids)

    if not metric_ids:
        if close_after:
            results_db.close()
        return pd.DataFrame()

    # Retrieve summary stats with run_name included.
    stats = results_db.get_summary_stats(metric_id=metric_ids, with_sim_name=True)

    if close_after:
        results_db.close()

    if stats.size == 0:
        return pd.DataFrame()

    df = pd.DataFrame(stats)
    df["transition_dayobs"] = df["run_name"].apply(_dayobs_from_run_name)

    pivot = df.pivot_table(
        index="transition_dayobs",
        columns=["metric_name", "slicer_name", "metric_info_label", "summary_metric"],
        values="summary_value",
        aggfunc="first",
    )
    pivot.index = pivot.index.astype(int)
    pivot.sort_index(inplace=True)
    return pivot


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


@click.command(name="run_chimera_batches")
@click.option(
    "--chimera-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing chimera_*.h5 files.",
)
@click.option("--out-dir", default=".", show_default=True, help="Output directory for results_db.")
@click.option(
    "--batch",
    default="glanceBatch",
    show_default=True,
    help="Batch function name from rubin_sim.maf.batches.",
)
def run_chimera_batches_cmd(chimera_dir, out_dir, batch):
    """Run MAF metric batches on all chimera HDF5 files in a directory."""
    batch_func = getattr(batches, batch, None)
    if batch_func is None:
        raise click.BadParameter(
            f"'{batch}' is not a known batch function in rubin_sim.maf.batches.",
            param_hint="--batch",
        )
    h5_files = sorted(glob.glob(os.path.join(chimera_dir, "chimera_*.h5")))
    if not h5_files:
        raise click.UsageError(f"No chimera_*.h5 files found in {chimera_dir}.")
    chimera_specs = []
    for path in h5_files:
        t = _dayobs_from_filename(path)
        if t is not None:
            chimera_specs.append((t, path))
    results_db_path = run_chimera_batches(chimera_specs, batch_func=batch_func, out_dir=out_dir)
    click.echo(f"Results written to {results_db_path}.")


@click.command(name="make_chimera_summary_table")
@click.option(
    "--results-db",
    required=True,
    type=click.Path(exists=True),
    help="Path to resultsDb_sqlite.db.",
)
@click.option(
    "--out-file",
    default="chimera_summary.h5",
    show_default=True,
    help="Output HDF5 file for the summary table.",
)
def make_chimera_summary_table_cmd(results_db, out_file):
    """Query a results_db to produce a summary table
    (one row per transition date)."""
    table = make_chimera_summary_table(results_db)
    if table.empty:
        click.echo("Warning: summary table is empty.")
    else:
        table.to_hdf(out_file, key="summary")
        click.echo(f"Summary table ({table.shape[0]} rows x {table.shape[1]} cols) written to {out_file}.")
