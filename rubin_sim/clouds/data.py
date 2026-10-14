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
    "make_cloud_figure",
    "observability_acf",
]

import calendar
import datetime
import os
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from typing import Any, TypeAlias

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy.utils import iers
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Patch
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

# Shared label arrays for the month and cloud-state axes. Defining them once
# guarantees every matrix uses identical uint8 dtypes, which the validators
# compare against exactly.
_MONTHS: pd.Index = pd.Index(np.arange(1, 13, dtype=np.uint8), name="month")
_STATES: npt.NDArray[np.uint8] = np.arange(9, dtype=np.uint8)

# Origin states to borrow transition probabilities from, in priority order,
# when a month has no observed transitions out of the origin given by the
# row index. This is the fallback priority list for transition sampling.
_FALLBACK_ORIGINS: tuple[tuple[int, ...], ...] = (
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
    """Check that a frame conforms to the `Clouds` schema.

    Parameters
    ----------
    clouds : `Clouds`
        Frame to validate.

    Raises
    ------
    ValueError
        If the columns, index levels, dtypes, ordering, or values differ
        from the `Clouds` data model.

    Notes
    -----
    Malformed input is rejected rather than sorted or repaired: silently
    fixing caller data would hide upstream bugs, and downstream code
    (for example the transition counter) relies on chronological order.
    """
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

    # Check each distinct night once rather than every quarter.
    for year, month, day in fields[["year", "month", "sday"]].drop_duplicates().itertuples(index=False):
        if datetime.date(int(year), int(month), int(day)) < datetime.date(1975, 1, 1):
            raise ValueError("Clouds dates must be on or after 1975-01-01")

    states = clouds.eighths.to_numpy()
    if (states > 9).any():
        raise ValueError("Clouds eighths must be from 0 through 9")

    # NaN != NaN, so the comparison must treat NaNs as equal.
    expected = np.where(states == 9, np.nan, states / 8)
    if not np.array_equal(clouds.cloud.to_numpy(), expected, equal_nan=True):
        raise ValueError("Clouds cloud values must equal eighths/8, or NaN for missing state 9")
    seconds = clouds.c_date.to_numpy()
    if np.any(seconds[1:] <= seconds[:-1]):
        raise ValueError("Clouds c_date values must be strictly increasing")


def _validate_transition_matrix(matrix: TransitionMatrix) -> None:
    """Check that a frame conforms to the `TransitionMatrix` schema.

    Parameters
    ----------
    matrix : `TransitionMatrix`
        Frame to validate.

    Raises
    ------
    ValueError
        If the index, columns, or dtypes differ from the data model.

    Notes
    -----
    The index is compared against the full 12 x 9 product of months and
    origins, so a missing, duplicated, or unsorted row is also rejected.
    """
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
    """Check that a frame conforms to the `CloudDistribution` schema.

    Parameters
    ----------
    distribution : `CloudDistribution`
        Frame to validate.

    Raises
    ------
    ValueError
        If the month index, state columns, or count values are invalid.

    Notes
    -----
    Unlike `TransitionMatrix`, a distribution may cover only a subset of
    months, because diagnostics may compare a single simulated year or
    season. Plotting code reindexes it to all twelve months.
    """
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
    Times are seconds since 1975-01-01 00:00 TAI, rounded to the nearest
    second. IERS downloads are disabled
    locally; Astropy's bundled tables and the solver's sidereal-time
    fallback suffice for these weather sampling times.

    Raises
    ------
    ValueError
        If the number of states is not four per date, a state is not an
        integer from 0 through 9, a date is before 1975-01-01 or
        duplicated, or twilight times cannot be determined.
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

    # Four rows per night: calendar fields repeat, quarter numbers cycle.
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

        # 04:00 UTC the following day is always within the local night
        # (Chile is UTC-3 or UTC-4), even when evening twilight falls
        # after the UTC date rollover. It anchors the solver to the night
        # that began on the given local date.
        anchors = Time([day.isoformat() for day in dates], scale="utc").mjd + 1 + 4 / 24
        with iers.conf.set_temp("auto_download", False):
            # Bound temporary coordinate arrays for long historical inputs
            # (the solver is vectorized and memory grows with batch size).
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

                # Quarter centers sit at 1/8, 3/8, 5/8 and 7/8 of the night.
                centers = (evening - epoch).to_value(u.s)[:, None] + duration[:, None] * (
                    np.arange(4) + 0.5
                ) / 4
                seconds[4 * start : 4 * stop] = np.rint(centers).ravel().astype(np.uintp)

    fractional = values.astype(np.float64) / 8
    fractional[values == 9] = np.nan  # Missing data is NaN, not 9/8.

    # Dates may arrive unsorted; the schema requires chronological order.
    return pd.DataFrame(
        {"eighths": values.astype(np.uint8), "cloud": fractional, "c_date": seconds}, index=index
    ).sort_index()


def read_historical_clouds(
    paths: str | os.PathLike | Sequence[str | os.PathLike],
) -> Clouds:
    """Read whitespace-separated CTIO records from one or more paths.

    Parameters
    ----------
    paths : `str`, `os.PathLike`, or sequence of those
        Historical record file or files. Each must have sday, eday, month,
        year, and q1 through q4 column headings.

    Returns
    -------
    clouds : `Clouds`
        Records for every night, including quarter-center ``c_date``
        values.

    Raises
    ------
    ValueError
        If no path is given, a state is invalid, a date is before 1975,
        or any month has no valid quarters across all files.

    Notes
    -----
    Duplicate nights retain their first occurrence in path order, so
    earlier files take precedence. Both missing-data sentinels, -1 and 9,
    are stored as state 9.
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
    states[states == -1] = 9  # Normalize to a single sentinel.
    if np.any((states < 0) | (states > 9)):
        raise ValueError("Cloud states must be integers from 0 through 9")

    # A month with no usable data in any year cannot be simulated.
    present = nights.loc[np.any(states < 9, axis=1), "month"]
    missing = sorted(set(range(1, 13)) - set(present))
    if missing:
        raise ValueError(f"Historical records have no valid cloud data for months {missing}")
    return make_cloud_frame(dates, states.ravel())


def count_cloud_states(clouds: Clouds) -> CloudDistribution:
    """Count valid quarters per month and state across all supplied years.

    Parameters
    ----------
    clouds : `Clouds`
        Records to count. Each quarter is attributed to the month of the
        local date on which its night began.

    Returns
    -------
    distribution : `CloudDistribution`
        Counts for all twelve months and states 0 through 8, with absent
        combinations filled with zero. Missing data (state 9) is excluded.

    Raises
    ------
    ValueError
        If ``clouds`` does not conform to the `Clouds` schema.
    """
    _validate_clouds(clouds)

    valid = clouds.loc[clouds.eighths < 9]
    counts = valid.groupby(["month", "eighths"]).size().unstack("eighths", fill_value=0)

    # Reindex so that months or states absent from the data still appear.
    return counts.reindex(index=_MONTHS, columns=pd.Index(_STATES, name="eighths"), fill_value=0).astype(
        np.uintp
    )


def compute_transition_matrix(
    clouds: Clouds, prior_state: int | None = None, **kwargs: Any
) -> TransitionMatrix:
    """Count adjacent valid-state transitions, grouped by destination month.

    Parameters
    ----------
    clouds : `Clouds`
        Records whose transitions are counted.
    prior_state : `int` or `None`, optional
        Cloud state of the quarter immediately preceding the first quarter
        of ``clouds``; the caller asserts that it is adjacent. `None` or 9
        omits the transition into the first quarter.
    **kwargs : `Any`
        Accepted by the documented interface; currently unused.

    Returns
    -------
    matrix : `TransitionMatrix`
        Counts for all twelve months and nine origins, including rows
        with no transitions.

    Raises
    ------
    ValueError
        If ``clouds`` is invalid or ``prior_state`` is not `None` or an
        integer from 0 through 9.

    Notes
    -----
    A pair is counted in the month in which its second quarter's night
    began. Calendar gaps and omitted or missing quarters break
    adjacency. Consecutive nights include the fourth-to-first-quarter
    transition.
    """
    _validate_clouds(clouds)

    # bool is a subclass of int but is never a meaningful state here.
    if prior_state is not None and (
        isinstance(prior_state, bool)
        or not isinstance(prior_state, (int, np.integer))
        or prior_state not in range(10)
    ):
        raise ValueError("prior_state must be None or an integer from 0 through 9")

    # Validation guarantees chronological order.
    ordered = clouds
    counts: npt.NDArray[np.uintp] = np.zeros((12, 9, 9), dtype=np.uintp)

    if len(ordered):
        fields = ordered.index.to_frame(index=False)
        days = np.array(
            [datetime.date(int(y), int(m), int(d)).toordinal() for y, m, d in fields.iloc[:, :3].values],
            dtype=np.int64,
        )

        # Absolute quarter number: two quarters are adjacent exactly when
        # their positions differ by one, including across night and month
        # boundaries, and a missing row breaks the chain automatically.
        positions = 4 * days + fields.quarter.to_numpy(dtype=np.int64) - 1
        states = ordered.eighths.to_numpy(dtype=np.int64)

        origins = np.empty_like(states)
        origins[0] = 9 if prior_state is None else prior_state
        origins[1:] = states[:-1]

        # The first quarter is adjacent to its prior state only if the
        # caller supplied one.
        adjacent = np.r_[prior_state is not None, np.diff(positions) == 1]
        valid = adjacent & (origins >= 0) & (origins < 9) & (states >= 0) & (states < 9)
        months = fields.month.to_numpy(dtype=np.int64) - 1

        # add.at (unlike fancy-index +=) accumulates repeated index triples.
        np.add.at(counts, (months[valid], origins[valid], states[valid]), 1)

    return pd.DataFrame(
        counts.reshape(108, 9),
        index=pd.MultiIndex.from_product([_MONTHS, _STATES], names=["month", "origin"]),
        columns=pd.Index(_STATES, name="destination"),
    )


def compute_stochastic_matrix(matrix: TransitionMatrix) -> StochasticMatrix:
    """Normalize transition counts into row-stochastic probabilities.

    Parameters
    ----------
    matrix : `TransitionMatrix`
        Transition counts for all months and origins.

    Returns
    -------
    probabilities : `StochasticMatrix`
        Probability of each destination given the month and origin. Every
        row sums to one.

    Raises
    ------
    ValueError
        If ``matrix`` is invalid or a month has no transitions at all.

    Notes
    -----
    An origin with no observed transitions copies the row of the first
    origin with data in its fallback list, since a stochastic row
    cannot be empty. See :ref:`clouds-fallback` for the ordered priorities.
    This inspection product is not an input to `generate_average_clouds`,
    which uses raw counts and remaining budgets instead.
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

    Parameters
    ----------
    path : `str` or `os.PathLike`
        Database file to write; created if it does not exist.
    clouds : `Clouds`
        Simulated records. ``cloudId`` starts at one in chronological
        order and ``source`` is always "simulation".

    Returns
    -------
    None
        Writes the database without returning a value.

    Raises
    ------
    ValueError
        If ``clouds`` is invalid or contains missing states, which have
        no meaningful simulated cloud value.

    Notes
    -----
    The Cloud table is replaced inside a single transaction, so a failure
    leaves any existing table intact, and other tables are preserved.
    Its schema is::

        Cloud(cloudId INTEGER PRIMARY KEY, c_date INTEGER,
              cloud DOUBLE, source TEXT)

    ``c_date`` stores quarter-center seconds
    since 1975-01-01 00:00 TAI; ``cloud`` is fractional cover in eighths.
    """
    _validate_clouds(clouds)

    # Validation guarantees chronological order, which cloudId follows.
    ordered = clouds
    states = ordered.eighths.to_numpy()
    if np.any((states < 0) | (states > 8)):
        raise ValueError("Simulated cloud data must not contain missing or invalid states")

    # A generator keeps memory flat for databases with millions of rows.
    records = (
        (index, int(seconds), float(state) / 8, "simulation")
        for index, (seconds, state) in enumerate(zip(ordered.c_date, states), start=1)
    )
    with closing(sqlite3.connect(os.fspath(path))) as connection, connection:
        # Explicit BEGIN so the DDL participates in the transaction.
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

    Parameters
    ----------
    historical_distribution : `CloudDistribution`
        Historical counts, drawn as a filled distribution.
    simulated_distribution : `CloudDistribution` or `dict`
        Simulated counts, drawn as a step curve. A dictionary overlays one
        curve per key, labelled by year.
    **kwargs
        ``figsize``, ``dpi``, and ``layout`` are passed to figure
        creation; the rest are passed to `matplotlib.axes.Axes.step` for
        the simulated curves.

    Returns
    -------
    figure : `matplotlib.figure.Figure`
        Figure with one panel per month. The caller owns it and should
        close it.

    Raises
    ------
    ValueError
        If any distribution does not conform to `CloudDistribution`.

    Notes
    -----
    Counts are normalized per month so that months with different numbers
    of quarters (and historical vs. simulated totals) are comparable.
    Months with no counts are drawn as zeros rather than NaNs.
    """
    _validate_cloud_distribution(historical_distribution)

    simulated = (
        simulated_distribution
        if isinstance(simulated_distribution, dict)
        else {"Simulated": simulated_distribution}
    )
    for distribution in simulated.values():
        _validate_cloud_distribution(distribution)

    # Fill absent months so every panel has data.
    historical_distribution = historical_distribution.reindex(index=_MONTHS, fill_value=0)
    simulated = {label: frame.reindex(index=_MONTHS, fill_value=0) for label, frame in simulated.items()}

    # Separate figure options from artist options so that the remaining
    # kwargs can be forwarded wholesale to the plotting call.
    figure_kwargs = {key: kwargs.pop(key) for key in ("figsize", "dpi", "layout") if key in kwargs}
    figure_kwargs.setdefault("figsize", (14, 10))
    figure_kwargs.setdefault("layout", "constrained")
    figure, axes = plt.subplots(3, 4, sharex=True, sharey=True, **figure_kwargs)

    # Bin edges centred on each eighth, spanning -1/16 to 17/16.
    edges = (np.arange(10) - 0.5) / 8

    for month, axis in enumerate(axes.flat, start=1):
        historical = historical_distribution.loc[month].to_numpy(dtype=float)
        historical /= historical.sum() or 1  # "or 1" avoids 0/0 for empty months.
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

    Parameters
    ----------
    historical_matrix : `TransitionMatrix`
        Historical transition counts.
    simulated_matrix : `TransitionMatrix`
        Simulated transition counts. Callers can compute it from a
        selected year of `Clouds` to limit the diagnostic, or from all
        years combined.
    **kwargs
        ``figsize``, ``dpi``, and ``layout`` affect the figure; the rest
        are passed to `matplotlib.axes.Axes.imshow` (for example ``cmap``,
        ``vmin``, ``vmax``, or ``norm``).

    Returns
    -------
    figure : `matplotlib.figure.Figure`
        Figure with a historical and simulated panel for each month.
        The caller owns it and should close it.

    Raises
    ------
    ValueError
        If either matrix does not conform to `TransitionMatrix`.

    Notes
    -----
    Each panel has its own colorbar scale: historical counts are far
    larger than those of a single simulated year, so a shared scale
    would make the simulated panels unreadable.
    """
    _validate_transition_matrix(historical_matrix)
    _validate_transition_matrix(simulated_matrix)

    figure_kwargs = {key: kwargs.pop(key) for key in ("figsize", "dpi", "layout") if key in kwargs}
    figure_kwargs.setdefault("figsize", (18, 12))
    figure_kwargs.setdefault("layout", "constrained")
    figure, axes = plt.subplots(4, 6, **figure_kwargs)

    historical = historical_matrix
    simulated = simulated_matrix
    # origin="lower" puts state 0 at the bottom-left.
    options = {"origin": "lower", "interpolation": "nearest", "cmap": "viridis", **kwargs}

    for month in range(1, 13):
        # Three month-pairs per row of six panels.
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


def make_cloud_figure(eighths: pd.Series, origin: pd.Series | None = None) -> Figure:
    """Show a complete cloud sequence as quarter-night strips by year.

    Parameters
    ----------
    eighths : `pandas.Series`
        Cloud cover in integer eighths, indexed by year (uint16), month,
        sday, and quarter (uint8), in that order, as in `Clouds`. Rows must
        be unique and sorted. Absent quarters, NaNs, -1, and 9 are missing.
    origin : `pandas.Series`, optional
        Text source names with exactly the same index. Source colors match
        the notebook prototype for ctio, blanco, satellite, and
        stochastic_matrix; other names receive distinct palette colors.
        Alpha is 1 - eighths/16, and missing quarters are white. Without
        sources, Cividis maps 0..8 to dark..yellow and missing data are red.

    Returns
    -------
    figure : `matplotlib.figure.Figure`
        One strip per calendar year, with day of year on the horizontal
        axis. The caller owns the figure and should close it.

    Raises
    ------
    ValueError
        If the series is empty or its index, states, or sources are invalid.
    """
    if not isinstance(eighths, pd.Series) or eighths.empty:
        raise ValueError("eighths must be a nonempty pandas Series")
    index = eighths.index
    names = ["year", "month", "sday", "quarter"]
    if not isinstance(index, pd.MultiIndex) or list(index.names) != names:
        raise ValueError("eighths index levels must be year, month, sday, quarter, in that order")
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("eighths rows must be unique and sorted by date and quarter")
    for name, dtype in zip(names, [np.uint16, np.uint8, np.uint8, np.uint8]):
        if index.get_level_values(name).dtype != np.dtype(dtype):
            raise ValueError(f"eighths index level {name} must have dtype {np.dtype(dtype)}")
    fields = index.to_frame(index=False)
    if not fields.quarter.between(1, 4).all():
        raise ValueError("quarters must be from 1 through 4")
    try:
        dates = pd.to_datetime(fields.rename(columns={"sday": "day"})[["year", "month", "day"]])
    except (ValueError, OverflowError) as error:
        raise ValueError("eighths must have valid calendar dates") from error
    if (fields.year < 1975).any():
        raise ValueError("eighths dates must be on or after 1975-01-01")
    if (
        not pd.api.types.is_numeric_dtype(eighths.dtype)
        or pd.api.types.is_bool_dtype(eighths.dtype)
        or pd.api.types.is_complex_dtype(eighths.dtype)
    ):
        raise ValueError("eighths must contain integer cloud states 0..8, or missing values")
    values = eighths.to_numpy(dtype=float, na_value=np.nan)
    if not np.all(np.isnan(values) | np.isin(values, np.arange(-1, 10))):
        raise ValueError("eighths must contain integer cloud states 0..8, or missing values")
    if origin is not None:
        if not isinstance(origin, pd.Series) or not origin.index.identical(index):
            raise ValueError("origin must be a pandas Series with the same index as eighths")
        if not origin.dropna().map(lambda value: isinstance(value, str)).all():
            raise ValueError("origin must contain text source names")

    years = np.arange(int(fields.year.min()), int(fields.year.max()) + 1)
    # Allocate every calendar quarter so gaps never become artificial runs.
    lengths = np.array([4 * (366 if calendar.isleap(int(year)) else 365) for year in years])
    offsets = np.r_[0, np.cumsum(lengths)]
    y = np.repeat(years, lengths)
    x = np.concatenate([1 + np.arange(length) / 4 for length in lengths])
    vertices = np.stack(
        [
            np.column_stack([x, y - 0.4]),
            np.column_stack([x + 0.25, y - 0.4]),
            np.column_stack([x + 0.25, y + 0.4]),
            np.column_stack([x, y + 0.4]),
        ],
        axis=1,
    )
    colors = np.tile(to_rgba("red" if origin is None else "white"), (len(x), 1))
    positions = offsets[fields.year.to_numpy(dtype=int) - years[0]]
    positions += (dates.dt.dayofyear.to_numpy() - 1) * 4 + fields.quarter.to_numpy() - 1
    valid = (values >= 0) & (values <= 8)
    cmap = plt.get_cmap("cividis")
    norm = Normalize(0, 8)
    handles = []
    if origin is None:
        colors[positions[valid]] = cmap(norm(values[valid]))
    else:
        source_indices = {"ctio": 0, "blanco": 9, "satellite": 4, "stochastic_matrix": 3}
        available = iter(i for i in range(len(cc.glasbey_category10)) if i not in source_indices.values())
        for source in sorted(origin.dropna().unique()):
            if source not in source_indices:
                source_indices[source] = next(available, len(source_indices))
            color = cc.glasbey_category10[source_indices[source] % len(cc.glasbey_category10)]
            selected = valid & origin.eq(source).to_numpy(dtype=bool, na_value=False)
            colors[positions[selected], :3] = color
            colors[positions[selected], 3] = 1 - values[selected] / 16
            handles.append(Patch(color=color, label=source))

    figure, axis = plt.subplots(figsize=(10, max(2.5, 0.25 * len(years))), layout="constrained")
    axis.add_collection(PolyCollection(vertices, facecolors=colors, edgecolors="none"))
    axis.set(
        xlim=(0, 367),
        ylim=(years[-1] + 1, years[0] - 1),
        xlabel="Day of year",
        ylabel="Year",
        title="Cloud cover by year" if origin is None else "Cloud data source by year",
    )
    axis.set_yticks(years[::5] if len(years) > 20 else years)
    handles.append(Patch(facecolor="red" if origin is None else "white", edgecolor="black", label="Missing"))
    axis.legend(
        handles=handles,
        title="Source" if origin is not None else None,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=min(5, len(handles)),
    )
    if origin is None:
        figure.colorbar(
            ScalarMappable(norm=norm, cmap=cmap), ax=axis, label="Cloud cover (eighths)", ticks=np.arange(9)
        )
    return figure


def observability_acf(
    cloud_threshold: float,
    historical_clouds: Clouds,
    simulated_clouds: Clouds,
    month: int,
    year: int | None = None,
    **kwargs: Any,
) -> Figure:
    """Overlay historical and simulated monthly observability ACFs.

    Parameters
    ----------
    cloud_threshold : `float`
        A quarter is observable if its fractional cloud cover is strictly
        below this finite threshold.
    historical_clouds, simulated_clouds : `Clouds`
        Historical and simulated records in chronological order.
    month : `int`
        Calendar month to compare (1 through 12).
    year : `int` or `None`, optional
        Limit the simulation to this year and month. By default, combine
        all simulated years; historical years are always combined.
    **kwargs
        Passed to `statsmodels.tsa.stattools.acf`, with ``missing`` set to
        ``"conservative"`` by default. If ``alpha`` or ``qstat`` is supplied,
        only the returned ACF values are plotted.

    Returns
    -------
    figure : `matplotlib.figure.Figure`
        Historical and simulated curves versus lag in quarter nights.
        The caller owns the figure and should close it.

    Raises
    ------
    ImportError
        If the optional ``statsmodels`` package cannot be imported.
    ValueError
        If either frame or a selection argument is invalid, or either
        selection has insufficient varying data to plot an ACF.

    Notes
    -----
    Missing quarters, omitted nights, and out-of-month quarters are NaN,
    not unobservable zeros. Multi-year sequences retain calendar spacing
    rather than concatenating the selected months. The ``missing`` option
    determines how statsmodels treats these NaNs.
    """
    try:
        from statsmodels.tsa.stattools import acf
    except ImportError as error:
        raise ImportError(
            "observability_acf requires statsmodels to be importable in this environment"
        ) from error

    _validate_clouds(historical_clouds)
    _validate_clouds(simulated_clouds)
    if isinstance(month, bool) or not isinstance(month, (int, np.integer)) or month not in range(1, 13):
        raise ValueError("month must be an integer from 1 through 12")
    if year is not None and (
        isinstance(year, bool) or not isinstance(year, (int, np.integer)) or not 1975 <= year <= 65535
    ):
        raise ValueError("year must be None or an integer from 1975 through 65535")
    if (
        isinstance(cloud_threshold, bool)
        or not isinstance(cloud_threshold, (int, float, np.integer, np.floating))
        or not np.isfinite(cloud_threshold)
    ):
        raise ValueError("cloud_threshold must be a finite number")

    kwargs.setdefault("missing", "conservative")
    curves = []
    for label, clouds in (("Historical", historical_clouds), ("Simulated", simulated_clouds)):
        if label == "Simulated" and year is not None:
            clouds = clouds.loc[
                (clouds.index.get_level_values("year") == year)
                & (clouds.index.get_level_values("month") == month)
            ]
        if clouds.empty:
            raise ValueError(f"{label} selection has too little data to plot an ACF")

        selected = (clouds.index.get_level_values("month") == month) & clouds.cloud.notna()
        observable = np.where(selected, (clouds.cloud < cloud_threshold).astype(float), np.nan)
        fields = clouds.index.to_frame(index=False)
        days = np.array(
            [datetime.date(int(y), int(m), int(d)).toordinal() for y, m, d in fields.iloc[:, :3].values],
            dtype=np.int64,
        )
        positions = 4 * days + fields.quarter.to_numpy(dtype=np.int64) - 1
        # Reinsert absent rows so a missing night cannot become a lag-one pair.
        observable = (
            pd.Series(observable, index=positions)
            .reindex(pd.RangeIndex(int(positions[0]), int(positions[-1]) + 1))
            .to_numpy()
        )
        valid = observable[np.isfinite(observable)]
        if len(valid) < 2 or np.unique(valid).size < 2:
            raise ValueError(f"{label} selection has too little varying data to plot an ACF")

        values = acf(observable, **kwargs)
        if isinstance(values, tuple):
            values = values[0]
        if len(values) < 2 or not np.isfinite(values[1:]).any():
            raise ValueError(f"{label} selection has too little data to plot an ACF at nonzero lags")
        curves.append((label, values))

    figure, axis = plt.subplots(layout="constrained")
    for label, values in curves:
        axis.plot(np.arange(len(values)), values, label=label)
    suffix = "" if year is None else f" (simulation: {year})"
    axis.set_title(f"{calendar.month_name[month]} observability ACF{suffix}")
    axis.set_xlabel("Lag (quarter nights)")
    axis.set_ylabel("Autocorrelation")
    axis.legend()
    return figure
