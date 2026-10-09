"""Historical cloud records, transition statistics, and diagnostics."""

__all__ = [
    "Clouds",
    "TransitionMatrix",
    "StochasticMatrix",
    "CloudDistribution",
    "make_cloud_frame",
    "read_historical_clouds",
    "count_cloud_states",
    "compute_transition_matrix",
    "compute_stochastic_matrix",
    "save_clouds",
    "plot_cloud_histogram",
    "plot_transition_histograms",
]

import calendar
import datetime
import os
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from typing import Any, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy.utils import iers
from matplotlib.figure import Figure
from rubin_scheduler.utils import Site
from rubin_scheduler.utils.riseset import riseset_times

# MultiIndex: year (uint16), month, sday, quarter (uint8); quarters 1..4.
# Columns: eighths (uint8, 0..8, missing=9), cloud (float64, eighths/8,
# missing=NaN), c_date (uintp, quarter-center seconds since 1975-01-01 TAI).
# All calendar fields describe the local date at the start of the night.
# Rows are unique and sorted by date and quarter; columns are ordered.
Clouds: TypeAlias = pd.DataFrame

# MultiIndex: month, origin (uint8); columns: destination (uint8, 0..8).
# All 12 months and 9 origins occur exactly once in sorted order, with
# ordered destination columns 0..8. Values: counts (uintp), excluding missing.
TransitionMatrix: TypeAlias = pd.DataFrame

# Same index and columns as TransitionMatrix; values float64, rows sum to 1.
StochasticMatrix: TypeAlias = pd.DataFrame

# Index: unique integer month (1..12); ordered integer columns: states 0..8.
# Values: nonnegative integer quarter counts, excluding missing state 9.
# count_cloud_states returns all months, uint8 labels and uintp values.
CloudDistribution: TypeAlias = pd.DataFrame

_MONTHS = pd.Index(np.arange(1, 13, dtype=np.uint8), name="month")
_STATES = np.arange(9, dtype=np.uint8)
_FALLBACK_ORIGINS = (
    (1, 2, 3, 4, 5, 6, 7, 8),
    (2, 0, 3, 4, 5, 6, 7, 8),
    (1, 0, 3, 4, 5, 6, 7, 8),
    (4, 5, 6, 7, 8, 2, 1, 0),
    (3, 5, 6, 7, 8, 2, 1, 0),
    (6, 7, 4, 8, 3, 2, 1, 0),
    (5, 7, 4, 8, 3, 2, 1, 0),
    (6, 5, 8, 4, 3, 2, 1, 0),
    (7, 6, 5, 4, 3, 2, 1, 0),
)


def _validate_clouds(clouds: Clouds) -> None:
    """Reject malformed records without sorting or repairing caller data."""
    if not isinstance(clouds, pd.DataFrame) or not clouds.columns.equals(
        pd.Index(["eighths", "cloud", "c_date"])
    ):
        raise ValueError("Clouds columns must be eighths, cloud, c_date, in that order")
    index = clouds.index
    if not isinstance(index, pd.MultiIndex) or not pd.Index(index.names).equals(
        pd.Index(["year", "month", "sday", "quarter"])
    ):
        raise ValueError("Clouds index levels must be year, month, sday, quarter, in that order")
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Clouds rows must be unique and sorted by date and quarter")
    for name, dtype in zip(index.names, [np.uint16, np.uint8, np.uint8, np.uint8]):
        if index.get_level_values(name).dtype != np.dtype(dtype):
            raise ValueError(f"Clouds index level {name} must have dtype {np.dtype(dtype)}")
    for name, dtype in (("eighths", np.uint8), ("cloud", np.float64), ("c_date", np.uintp)):
        if clouds[name].dtype != np.dtype(dtype):
            raise ValueError(f"Clouds column {name} must have dtype {np.dtype(dtype)}")
    fields = index.to_frame(index=False)
    if ((fields.quarter < 1) | (fields.quarter > 4)).any():
        raise ValueError("Clouds quarters must be from 1 through 4")
    for year, month, day in fields[["year", "month", "sday"]].drop_duplicates().itertuples(index=False):
        if datetime.date(int(year), int(month), int(day)) < datetime.date(1975, 1, 1):
            raise ValueError("Clouds dates must be on or after 1975-01-01")
    states = clouds.eighths.to_numpy()
    if (states > 9).any():
        raise ValueError("Clouds eighths must be from 0 through 9")
    expected = np.where(states == 9, np.nan, states / 8)
    if not np.array_equal(clouds.cloud.to_numpy(), expected, equal_nan=True):
        raise ValueError("Clouds cloud values must equal eighths/8, or NaN for missing state 9")
    seconds = clouds.c_date.to_numpy()
    if np.any(seconds[1:] <= seconds[:-1]):
        raise ValueError("Clouds c_date values must be strictly increasing")


def _validate_transition_matrix(matrix: TransitionMatrix) -> None:
    """Validate the complete, ordered empirical count schema."""
    if not isinstance(matrix, pd.DataFrame):
        raise ValueError("TransitionMatrix must be a pandas DataFrame")
    index = pd.MultiIndex.from_product([_MONTHS, _STATES], names=["month", "origin"])
    if (
        not isinstance(matrix.index, pd.MultiIndex)
        or not pd.Index(matrix.index.names).equals(pd.Index(index.names))
        or not matrix.index.equals(index)
        or any(matrix.index.get_level_values(name).dtype != np.dtype(np.uint8) for name in index.names)
    ):
        raise ValueError(
            "TransitionMatrix requires sorted, unique months 1..12 and origins 0..8, with uint8 indices"
        )
    if not pd.Index([matrix.columns.name]).equals(pd.Index(["destination"])) or not matrix.columns.equals(
        pd.Index(_STATES)
    ):
        raise ValueError("TransitionMatrix destination columns must be ordered integer states 0 through 8")
    if not pd.api.types.is_integer_dtype(matrix.columns.dtype):
        raise ValueError("TransitionMatrix destination labels must be integers")
    if any(dtype != np.dtype(np.uintp) for dtype in matrix.dtypes):
        raise ValueError("TransitionMatrix counts must have dtype uintp")


def _validate_cloud_distribution(distribution: CloudDistribution) -> None:
    """Validate monthly counts and their state-column ordering."""
    if not isinstance(distribution, pd.DataFrame):
        raise ValueError("CloudDistribution must be a pandas DataFrame")
    if (
        not pd.Index([distribution.index.name]).equals(pd.Index(["month"]))
        or not pd.api.types.is_integer_dtype(distribution.index.dtype)
        or distribution.index.hasnans
        or not distribution.index.is_unique
        or not set(distribution.index) <= set(range(1, 13))
    ):
        raise ValueError(
            "CloudDistribution requires a unique integer month index with values from 1 through 12"
        )
    if not distribution.columns.equals(pd.Index(_STATES)) or not pd.api.types.is_integer_dtype(
        distribution.columns.dtype
    ):
        raise ValueError("CloudDistribution columns must be ordered integer states 0 through 8")
    if (
        any(not pd.api.types.is_integer_dtype(dtype) for dtype in distribution.dtypes)
        or distribution.isna().to_numpy().any()
        or (distribution.to_numpy() < 0).any()
    ):
        raise ValueError("CloudDistribution counts must be nonnegative integers")


def make_cloud_frame(dates: Sequence[datetime.date], eighths: Sequence[int] | np.ndarray) -> Clouds:
    """Build cloud records for four quarters of each supplied night.

    Parameters
    ----------
    dates : sequence of `datetime.date`
        Local calendar dates of the evenings, no earlier than 1975-01-01.
    eighths : sequence of `int` or `numpy.ndarray`
        Flat, night-major sequence of four states (0..9) per date.

    Returns
    -------
    clouds : `Clouds`
        Records sorted by local date and quarter, with the documented
        schema. Missing states have NaN fractional cloud cover.

    Notes
    -----
    Night boundaries are geometric -12 degree solar altitude at Cerro
    Pachon, using the scheduler's vectorized twilight solver. Quarter
    centers divide the elapsed TAI night into four equal intervals.
    Times are rounded to the nearest second. IERS downloads are disabled
    locally; Astropy's bundled tables and the solver's sidereal-time
    fallback suffice for these weather sampling times.
    """
    dates = list(dates)
    values = np.asarray(eighths)
    if values.ndim != 1 or values.size != 4 * len(dates):
        raise ValueError("eighths must contain exactly four flat values per date")
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 9) | (values != np.floor(values))):
        raise ValueError("Cloud states must be integers from 0 through 9")
    if any(not isinstance(day, datetime.date) or day < datetime.date(1975, 1, 1) for day in dates):
        raise ValueError("dates must be calendar dates on or after 1975-01-01")
    if len(set(dates)) != len(dates):
        raise ValueError("dates must not contain duplicate nights")

    index = pd.MultiIndex.from_arrays(
        [
            np.repeat(np.array([day.year for day in dates], dtype=np.uint16), 4),
            np.repeat(np.array([day.month for day in dates], dtype=np.uint8), 4),
            np.repeat(np.array([day.day for day in dates], dtype=np.uint8), 4),
            np.tile(np.arange(1, 5, dtype=np.uint8), len(dates)),
        ],
        names=["year", "month", "sday", "quarter"],
    )
    seconds = np.empty(values.size, dtype=np.uintp)
    if dates:
        site = Site("LSST")
        location = EarthLocation.from_geodetic(
            site.longitude * u.deg, site.latitude * u.deg, site.height * u.m
        )
        epoch = Time("1975-01-01T00:00:00", scale="tai")
        # 04:00 UTC the following day is always within the local night,
        # even when evening twilight falls after the UTC date rollover.
        anchors = Time([day.isoformat() for day in dates], scale="utc").mjd + 1 + 4 / 24
        with iers.conf.set_temp("auto_download", False):
            # Bound temporary coordinate arrays for long historical inputs.
            for start in range(0, len(dates), 512):
                stop = min(start + 512, len(dates))
                evening = Time(
                    riseset_times(anchors[start:stop], "down", alt=-12, location=location),
                    format="mjd",
                    scale="utc",
                )
                morning = Time(
                    riseset_times(anchors[start:stop], "up", alt=-12, location=location),
                    format="mjd",
                    scale="utc",
                )
                duration = (morning - evening).to_value(u.s)
                if np.any(~np.isfinite(duration)) or np.any((duration <= 0) | (duration >= 86400)):
                    raise ValueError("Could not determine valid twilight boundaries")
                centers = (evening - epoch).to_value(u.s)[:, None] + duration[:, None] * (
                    np.arange(4) + 0.5
                ) / 4
                seconds[4 * start : 4 * stop] = np.rint(centers).ravel().astype(np.uintp)
    fractional = values.astype(np.float64) / 8
    fractional[values == 9] = np.nan
    return pd.DataFrame(
        {"eighths": values.astype(np.uint8), "cloud": fractional, "c_date": seconds}, index=index
    ).sort_index()


def read_historical_clouds(
    paths: str | os.PathLike | Sequence[str | os.PathLike],
) -> Clouds:
    """Read whitespace-separated CTIO records from one or more paths.

    Files must have sday, eday, month, year, and q1..q4 headings.
    Duplicate nights retain their first occurrence in path order.
    Raises ValueError if any month has no valid quarters across all files.
    The returned Clouds includes quarter-center c_date values.

    Both supported missing sentinels, -1 and 9, are stored as state 9.
    Other invalid states and dates before 1975 are rejected.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    if not paths:
        raise ValueError("At least one historical cloud file is required")
    required = ["sday", "eday", "month", "year", "q1", "q2", "q3", "q4"]
    nights = pd.concat(
        [pd.read_csv(path, sep=r"\s+", usecols=required, dtype=np.int64) for path in paths],
        ignore_index=True,
    ).drop_duplicates(["year", "month", "sday"], keep="first")
    nights = nights.sort_values(["year", "month", "sday"])
    dates = [datetime.date(int(y), int(m), int(d)) for y, m, d in nights[["year", "month", "sday"]].values]
    states = nights[["q1", "q2", "q3", "q4"]].to_numpy()
    states[states == -1] = 9
    if np.any((states < 0) | (states > 9)):
        raise ValueError("Cloud states must be integers from 0 through 9")
    present = nights.loc[np.any(states < 9, axis=1), "month"]
    missing = sorted(set(range(1, 13)) - set(present))
    if missing:
        raise ValueError(f"Historical records have no valid cloud data for months {missing}")
    return make_cloud_frame(dates, states.ravel())


def count_cloud_states(clouds: Clouds) -> CloudDistribution:
    """Count valid quarters per month and state across all supplied years.

    Returns all twelve months and states 0..8, filling absent counts with
    zero. Missing states are excluded without bridging missing quarters.
    """
    _validate_clouds(clouds)
    valid = clouds.loc[clouds.eighths < 9]
    counts = valid.groupby(["month", "eighths"]).size().unstack("eighths", fill_value=0)
    return counts.reindex(index=_MONTHS, columns=pd.Index(_STATES, name="eighths"), fill_value=0).astype(
        np.uintp
    )


def compute_transition_matrix(clouds: Clouds, prior_state: int | None = None) -> TransitionMatrix:
    """Count adjacent valid-state transitions, grouped by destination month.

    Calendar gaps and omitted or missing quarters break adjacency.
    Consecutive nights include the fourth-to-first-quarter transition.
    A valid prior_state adds a transition into the first chronological
    quarter; the caller asserts that it immediately precedes that quarter.
    None and 9 omit that initial transition. All months and states are
    returned, including zero-count rows.
    """
    _validate_clouds(clouds)
    if prior_state is not None and (
        isinstance(prior_state, bool)
        or not isinstance(prior_state, (int, np.integer))
        or prior_state not in range(10)
    ):
        raise ValueError("prior_state must be None or an integer from 0 through 9")
    ordered = clouds
    counts = np.zeros((12, 9, 9), dtype=np.uintp)
    if len(ordered):
        fields = ordered.index.to_frame(index=False)
        days = np.array(
            [datetime.date(int(y), int(m), int(d)).toordinal() for y, m, d in fields.iloc[:, :3].values],
            dtype=np.int64,
        )
        positions = 4 * days + fields.quarter.to_numpy(dtype=np.int64) - 1
        states = ordered.eighths.to_numpy(dtype=np.int64)
        origins = np.empty_like(states)
        origins[0] = 9 if prior_state is None else prior_state
        origins[1:] = states[:-1]
        adjacent = np.r_[prior_state is not None, np.diff(positions) == 1]
        valid = adjacent & (origins >= 0) & (origins < 9) & (states >= 0) & (states < 9)
        months = fields.month.to_numpy(dtype=np.int64) - 1
        np.add.at(counts, (months[valid], origins[valid], states[valid]), 1)
    return pd.DataFrame(
        counts.reshape(108, 9),
        index=pd.MultiIndex.from_product([_MONTHS, _STATES], names=["month", "origin"]),
        columns=pd.Index(_STATES, name="destination"),
    )


def compute_stochastic_matrix(matrix: TransitionMatrix) -> StochasticMatrix:
    """Normalize transition rows, using the issue's ordered fallbacks.

    Empty origins copy the first nonempty empirical row in their month's
    fallback list. An entirely empty month raises ValueError. All twelve
    months and all origins must be supplied in the documented order.
    """
    _validate_transition_matrix(matrix)
    probabilities = matrix.astype(np.float64).copy()
    for month in matrix.index.get_level_values("month").unique():
        counts = matrix.loc[month].to_numpy(dtype=np.float64)
        totals = counts.sum(axis=1)
        if not np.any(totals):
            raise ValueError(f"Month {month} has no valid transitions")
        for origin in matrix.loc[month].index:
            source = int(origin)
            if totals[source] == 0:
                source = next(candidate for candidate in _FALLBACK_ORIGINS[source] if totals[candidate] > 0)
            probabilities.loc[(month, origin)] = counts[source] / totals[source]
    return probabilities


def save_clouds(path: str | os.PathLike, clouds: Clouds) -> None:
    """Write a scheduler-compatible SQLite Cloud table.

    The Cloud table is replaced transactionally, preserving other tables.
    cloudId starts at one in chronological order; source is 'simulation'.
    Missing cloud states cannot be written as simulated cloud values.
    """
    _validate_clouds(clouds)
    ordered = clouds
    states = ordered.eighths.to_numpy()
    if np.any((states < 0) | (states > 8)):
        raise ValueError("Simulated cloud data must not contain missing or invalid states")
    records = (
        (index, int(seconds), float(state) / 8, "simulation")
        for index, (seconds, state) in enumerate(zip(ordered.c_date, states), start=1)
    )
    with closing(sqlite3.connect(os.fspath(path))) as connection, connection:
        connection.execute("BEGIN")
        connection.execute("DROP TABLE IF EXISTS Cloud")
        connection.execute(
            "CREATE TABLE Cloud(cloudId INTEGER PRIMARY KEY,c_date INTEGER,cloud DOUBLE, source TEXT)"
        )
        connection.executemany("INSERT INTO Cloud VALUES (?, ?, ?, ?)", records)


def plot_cloud_histogram(
    historical_distribution: CloudDistribution,
    simulated_distribution: CloudDistribution | dict[int, CloudDistribution],
    **kwargs: Any,
) -> Figure:
    """Compare normalized monthly state distributions in twelve panels.

    A simulated dictionary overlays separate curves labelled by year.
    figsize, dpi, and layout are passed to figure creation; remaining
    keyword arguments are passed to matplotlib Axes.step for simulated
    curves. Historical counts are plotted as a filled distribution.
    Empty months display zeros rather than NaNs.
    """
    _validate_cloud_distribution(historical_distribution)
    simulated = (
        simulated_distribution
        if isinstance(simulated_distribution, dict)
        else {"Simulated": simulated_distribution}
    )
    for distribution in simulated.values():
        _validate_cloud_distribution(distribution)
    historical_distribution = historical_distribution.reindex(index=_MONTHS, fill_value=0)
    simulated = {label: frame.reindex(index=_MONTHS, fill_value=0) for label, frame in simulated.items()}
    figure_kwargs = {key: kwargs.pop(key) for key in ("figsize", "dpi", "layout") if key in kwargs}
    figure_kwargs.setdefault("figsize", (14, 10))
    figure_kwargs.setdefault("layout", "constrained")
    figure, axes = plt.subplots(3, 4, sharex=True, sharey=True, **figure_kwargs)
    edges = (np.arange(10) - 0.5) / 8
    for month, axis in enumerate(axes.flat, start=1):
        historical = historical_distribution.loc[month].to_numpy(dtype=float)
        historical /= historical.sum() or 1
        axis.stairs(historical, edges, fill=True, alpha=0.25, color="black", label="Historical")
        for label, distribution in simulated.items():
            values = distribution.loc[month].to_numpy(dtype=float)
            values /= values.sum() or 1
            options = {"where": "mid", "label": str(label), **kwargs}
            axis.step(_STATES / 8, values, **options)
        axis.set_title(calendar.month_name[month])
        axis.set_xticks([0, 0.25, 0.5, 0.75, 1])
    axes.flat[0].legend()
    figure.supxlabel("Fractional cloud cover")
    figure.supylabel("Fraction of quarters")
    return figure


def plot_transition_histograms(
    historical_matrix: TransitionMatrix, simulated_matrix: TransitionMatrix, **kwargs: Any
) -> Figure:
    """Plot twelve pairs of historical and simulated transition counts.

    Each pair has independently scaled colorbars labelled with counts.
    Callers can compute simulated_matrix from a selected Clouds year to
    limit the diagnostic, or combine all years before computing it.
    figsize, dpi, and layout affect the figure; remaining keywords are
    passed to Axes.imshow (for example cmap, vmin, vmax, or norm).
    """
    _validate_transition_matrix(historical_matrix)
    _validate_transition_matrix(simulated_matrix)
    figure_kwargs = {key: kwargs.pop(key) for key in ("figsize", "dpi", "layout") if key in kwargs}
    figure_kwargs.setdefault("figsize", (18, 12))
    figure_kwargs.setdefault("layout", "constrained")
    figure, axes = plt.subplots(4, 6, **figure_kwargs)
    historical = historical_matrix
    simulated = simulated_matrix
    options = {"origin": "lower", "interpolation": "nearest", "cmap": "viridis", **kwargs}
    for month in range(1, 13):
        row, pair = divmod(month - 1, 3)
        for offset, (label, matrix) in enumerate((("Historical", historical), ("Simulated", simulated))):
            axis = axes[row, 2 * pair + offset]
            image = axis.imshow(matrix.loc[month].to_numpy(), **options)
            axis.set_title(f"{calendar.month_abbr[month]}: {label}")
            axis.set_xticks([0, 2, 4, 6, 8])
            axis.set_yticks([0, 2, 4, 6, 8])
            figure.colorbar(image, ax=axis, label="Transitions", shrink=0.8)
    figure.supxlabel("Destination cloud state (eighths)")
    figure.supylabel("Origin cloud state (eighths)")
    return figure
