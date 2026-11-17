"""Black-box tests of the user-visible behavior of the average-cloud generator.

The generator reads historical CTIO cloud records (one row per night, four
quarter-night cloud levels in eighths), simulates a database in which every
month is statistically typical of that calendar month, writes it in the
SQLite format read by ``rubin_scheduler``, and produces diagnostic plots.

These tests describe the behavior the generator must have:

* Reading: historical records are read from a space- or tab-separated text
  file; missing data (sentinels 9 and -1) are preserved but excluded from
  statistics; a month with no usable data in any year is an error.
* Output database: the SQLite ``Cloud`` table has the schema and time
  convention expected by ``rubin_scheduler.site_models.CloudData``, covers
  at least the requested number of years, and repeats correctly when the
  scheduler wraps past its end.
* Monthly distribution: in every simulated month, the number of quarters in
  each cloud state matches the historical distribution for that calendar
  month to within one quarter.
* Monthly transitions: in every simulated month, the number of transitions
  between each pair of states matches the historical frequency, scaled to
  the length of the month, to within the square root of the expected count
  (or one, if the expected count is below one). Transitions involving
  missing quarters are excluded, and a transition belongs to the month of
  its destination quarter.
* Intermediate products: the historical records, transition counts,
  stochastic matrix, and simulated records are available to callers as
  ``pandas`` objects with documented indexes.
* Diagnostics: histogram, transition-matrix, full-sequence, and (optionally,
  if ``statsmodels`` is installed) autocorrelation figures.

Ordinary tests use small synthetic CTIO records, so no historical data are
needed. Set the environment variable ``LSST64_HISTORICAL_DATA`` to the path
of the real historical record file to enable one additional, opt-in test
that runs the full 20-year workflow on real data and checks that it
finishes within one hour (the intended performance target when run as a
batch job).
"""

import builtins
import calendar
import os
import sqlite3
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter

import colorcet as cc
import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time
from astropy.utils import iers
from click.testing import CliRunner
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from rubin_scheduler.site_models import CloudData

from rubin_sim.clouds import (
    cloud_generation_workflow,
    compute_stochastic_matrix,
    compute_transition_matrix,
    count_cloud_states,
    generate_average_clouds_cli,
    make_cloud_figure,
    observability_acf,
    plot_cloud_histogram,
    plot_transition_histograms,
    read_historical_clouds,
)


@pytest.fixture(autouse=True)
def offline_plots():
    """Use a non-interactive backend, avoid network access, and close figures.

    Astropy would otherwise try to download IERS tables while computing
    twilight times; the bundled tables are accurate enough for these tests.
    """
    plt.switch_backend("Agg")
    with iers.conf.set_temp("auto_download", False):
        try:
            yield
        finally:
            plt.close("all")


@pytest.fixture(scope="module")
def records():
    """Create two years of synthetic CTIO nightly records in raw file format.

    Quarters 1, 2 and 4 are constant (0, 8, 8) in every night. Quarter 3 is
    8 in odd-numbered months and 0 in even-numbered months, so each month has
    a distinct, easily calculated distribution of states and transitions.
    Only the states 0 and 8 occur.
    """
    nights = pd.date_range("1975-01-01", "1976-12-31")
    return pd.DataFrame(
        {
            "sday": nights.day,
            "eday": (nights + pd.Timedelta(days=1)).day,
            "month": nights.month,
            "year": nights.year,
            "q1": 0,
            "q2": 8,
            "q3": np.where(nights.month % 2, 8, 0),
            "q4": 8,
        }
    )


@pytest.fixture(scope="module")
def historical_path(tmp_path_factory, records):
    """Write the synthetic records to a space-separated file with a header."""
    path = tmp_path_factory.mktemp("requirements-clouds") / "synthetic.txt"
    records.to_csv(path, sep=" ", index=False)
    return path


@pytest.fixture(scope="module")
def simulated(tmp_path_factory, historical_path):
    """Run the full workflow for two years; return records and database."""
    directory = tmp_path_factory.mktemp("requirements-output")
    clouds = cloud_generation_workflow(historical_path, directory / "clouds.db", 2, seed=6563)
    return clouds, directory / "clouds.db"


def reference_counts(records):
    """Count states and transitions directly from raw CTIO rows.

    This is a deliberately simple reference implementation, independent of the
    cloud API, used to check what the API computes.

    Parameters
    ----------
    records : `pandas.DataFrame`
        Raw rows with columns ``sday``, ``month``, ``year``, ``q1`` ... ``q4``.

    Returns
    -------
    states : `numpy.ndarray`, (12, 9)
        Number of quarters in each cloud state (eighths 0..8) for each month.
        Quarters with missing data (9 or -1) are not counted.
    transitions : `numpy.ndarray`, (12, 9, 9)
        Number of transitions from each origin state to each destination
        state, indexed by the month of the destination quarter. A pair counts
        only if both quarters are valid and adjacent in time (including from
        quarter 4 of one night to quarter 1 of the next night); a missing
        quarter or a missing night breaks the pair.
    """
    states = np.zeros((12, 9), dtype=int)
    transitions = np.zeros((12, 9, 9), dtype=int)
    previous_position, previous_state = None, None
    for row in records.sort_values(["year", "month", "sday"]).itertuples(index=False):
        night = date(row.year, row.month, row.sday)
        for quarter, state in enumerate([row.q1, row.q2, row.q3, row.q4]):
            position = 4 * night.toordinal() + quarter
            if 0 <= state <= 8:
                states[row.month - 1, state] += 1
                if previous_position == position - 1 and previous_state in range(9):
                    transitions[row.month - 1, previous_state, state] += 1
            previous_position, previous_state = position, state
    return states, transitions


def assert_typical_months(clouds, historical_states, historical_transitions):
    """Check that every simulated month is typical of its calendar month.

    For each simulated (year, month), compare against the historical counts
    for that calendar month, scaled to the length of the simulated month:

    * the number of quarters in each cloud state must be within one of the
      scaled historical count;
    * the number of transitions for each (origin, destination) pair must be
      within the square root of the scaled historical count, or within one
      if that count is below one.

    The transition into the first quarter of a month is counted in that
    month, using the last quarter of the previous month as origin. The very
    first quarter of the simulation has no predecessor, so the first month
    has one fewer transition than quarters.

    Parameters
    ----------
    clouds : `pandas.DataFrame`
        Simulated records, indexed by year, month, sday, and quarter.
    historical_states : `numpy.ndarray`, (12, 9)
        Historical quarter counts per month and state.
    historical_transitions : `numpy.ndarray`, (12, 9, 9)
        Historical transition counts per month, origin, and destination.
    """
    previous = None
    for (year, month), group in clouds.groupby(level=["year", "month"], sort=True):
        states = group.eighths.to_numpy(dtype=int)
        quarters = 4 * calendar.monthrange(int(year), int(month))[1]
        assert len(states) == quarters
        assert set(states) <= set(range(9))
        expected = historical_states[month - 1] * quarters / historical_states[month - 1].sum()
        actual = np.bincount(states, minlength=9)
        assert np.all(np.abs(actual - expected) <= 1 + 1e-12), (year, month, "state counts")

        pairs = list(zip(states[:-1], states[1:]))
        if previous is not None:
            pairs.insert(0, (previous, states[0]))
        actual_pairs = np.zeros((9, 9), dtype=int)
        for origin, destination in pairs:
            actual_pairs[origin, destination] += 1
        historical = historical_transitions[month - 1]
        expected_pairs = historical * len(pairs) / historical.sum()
        tolerance = np.where(expected_pairs < 1, 1, np.sqrt(expected_pairs))
        assert np.all(np.abs(actual_pairs - expected_pairs) <= tolerance + 1e-12), (
            year,
            month,
            "transition counts",
        )
        previous = int(states[-1])


def test_read_ctio_quarters_and_exclude_missing_data(tmp_path, records):
    """Preserve quarter values and exclude missing data from counts.

    Quarters 3 and 4 of the first night are set to the missing-data sentinels
    9 and -1, and the third night is dropped from the file entirely. Both
    sentinels must read as eighths=9 with NaN fractional cloud cover, and
    neither the missing quarters nor the missing night may contribute to the
    state counts or transition counts, nor bridge a transition across the gap.
    """
    records = records.copy()
    records.loc[0, ["q1", "q2", "q3", "q4"]] = [0, 2, 9, -1]
    records = records.drop(index=2)  # Missing night must not create a transition.
    path = tmp_path / "mixed.txt"
    records.to_csv(path, sep="\t", index=False)
    clouds = read_historical_clouds(path)
    assert len(clouds) == len(records) * 4
    np.testing.assert_array_equal(clouds.loc[(1975, 1, 1), "eighths"], [0, 2, 9, 9])
    np.testing.assert_allclose(clouds.loc[(1975, 1, 1), "cloud"], [0, 0.25, np.nan, np.nan])
    states, transitions = reference_counts(records)
    np.testing.assert_array_equal(count_cloud_states(clouds), states)
    np.testing.assert_array_equal(compute_transition_matrix(clouds).to_numpy().reshape(12, 9, 9), transitions)


@pytest.mark.parametrize("all_missing", [False, True])
def test_no_usable_data_for_a_month_is_an_error(tmp_path, records, all_missing):
    """Require the CLI to fail when a month has no usable data in any year.

    February is either missing from the file entirely, or present with every
    quarter marked missing. In both cases there is nothing from which to build
    February statistics, so the CLI must fail with an explanatory message.
    """
    records = records.copy()
    if all_missing:
        records.loc[records.month == 2, ["q1", "q2", "q3", "q4"]] = 9
    else:
        records = records.loc[records.month != 2]
    path = tmp_path / "missing-february.txt"
    records.to_csv(path, sep=" ", index=False)
    result = CliRunner().invoke(generate_average_clouds_cli, [str(path), "-o", str(tmp_path / "clouds.db")])
    assert result.exit_code != 0
    assert "no valid cloud data" in result.output


def test_each_generated_month_is_typical(records, simulated):
    """Check that every simulated month is typical of its calendar month.

    The synthetic data contain only states 0 and 8 and a distinct mix in each
    month, so each month has different targets. The statistics are checked
    month by month across the simulation, including the leap-year February
    (29 days) and the transitions across month boundaries.
    """
    clouds, _ = simulated
    assert set(clouds.eighths) == {0, 8}
    assert len(clouds.loc[(1976, 2)]) == 29 * 4
    assert_typical_months(clouds, *reference_counts(records))


def test_database_schema_values_and_scheduler_repetition(simulated):
    """Write SQLite output with required schema, values, and duration.

    Checks that:

    * the ``Cloud`` table has the columns ``cloudId`` (integer primary key),
      ``c_date`` (integer), ``cloud`` (double), and ``source`` (text);
    * ``cloudId`` increments from 1, ``c_date`` and ``cloud`` match the
      simulated records, and ``source`` is always "simulation";
    * the simulation spans at least ``num_years * 31557600`` seconds, which
      requires an extra month (the first of the following year);
    * ``rubin_scheduler``'s ``CloudData`` reads the database and returns the
      stored cloud values at the stored times, and the same values one database
      duration later, so that the scheduler's wrapping logic does not fail.
    """
    clouds, database = simulated
    with closing(sqlite3.connect(database)) as connection:
        schema = connection.execute("PRAGMA table_info(Cloud)").fetchall()
        assert [(row[1], row[2], row[5]) for row in schema] == [
            ("cloudId", "INTEGER", 1),
            ("c_date", "INTEGER", 0),
            ("cloud", "DOUBLE", 0),
            ("source", "TEXT", 0),
        ]
        saved = connection.execute(
            "SELECT cloudId,c_date,cloud,source FROM Cloud ORDER BY cloudId"
        ).fetchall()
    np.testing.assert_array_equal([row[0] for row in saved], np.arange(1, len(clouds) + 1))
    np.testing.assert_array_equal([row[1] for row in saved], clouds.c_date)
    np.testing.assert_array_equal([row[2] for row in saved], clouds.cloud)
    assert {row[3] for row in saved} == {"simulation"}
    assert int(clouds.c_date.iloc[-1]) - int(clouds.c_date.iloc[0]) >= 2 * 31557600
    assert clouds.index[-1][:2] == (1977, 1)  # Additional month supplies repeat coverage.
    reader = CloudData(Time("2025-01-01", scale="tai"), cloud_db=str(database))
    selected = clouds.iloc[[10, 17, 500]]
    times = reader.start_time + selected.c_date.to_numpy(dtype=float) * u.s
    np.testing.assert_allclose(reader(times), selected.cloud)
    np.testing.assert_allclose(reader(times + reader.time_range * u.s), selected.cloud)


def test_quarter_times_use_tai_epoch_and_cerro_pachon_twilight(simulated):
    """Check quarter-center times relative to the 1975-01-01 TAI epoch.

    A night is the period between evening and morning 12-degree twilight at
    Cerro Pachon, and is divided into four equal quarters. For several nights
    (including a leap day), the test checks that the four quarter centers are
    evenly spaced, that the Sun is 12 degrees below the horizon at the start
    of the first quarter and end of the last (inferred from the centers), and
    that the middle of the night falls on the UTC date after the local evening.
    """
    clouds, _ = simulated
    location = EarthLocation.from_geodetic(-70.74941 * u.deg, -30.244628 * u.deg, 2650 * u.m)
    epoch = Time("1975-01-01T00:00:00", scale="tai")
    for night in [date(1975, 1, 1), date(1975, 6, 21), date(1976, 2, 29)]:
        centers = clouds.loc[(night.year, night.month, night.day), "c_date"].to_numpy(dtype=float)
        spacing = (centers[-1] - centers[0]) / 3
        np.testing.assert_allclose(np.diff(centers), spacing, atol=1)
        endpoints = epoch + np.array([centers[0] - spacing / 2, centers[-1] + spacing / 2]) * u.s
        altitudes = get_sun(endpoints).transform_to(AltAz(obstime=endpoints, location=location)).alt.deg
        np.testing.assert_allclose(altitudes, [-12, -12], atol=0.02)
        assert (epoch + centers.mean() * u.s).utc.datetime.date() == night + timedelta(days=1)


def test_intermediate_products_are_accessible(historical_path, records, simulated):
    """Expose intermediate products as pandas objects with documented indexes.

    Historical and simulated records are ``DataFrame``s indexed by year, month,
    sday, and quarter, with ``eighths``, ``cloud``, and ``c_date`` columns. The
    transition counts and stochastic matrix are ``DataFrame``s indexed by
    month and origin state; the counts match an independent count of the raw
    records, and each row of the stochastic matrix sums to one. In the
    synthetic January, a clear (0) quarter is always followed by a cloudy (8)
    one, so the 0 -> 8 probability is 1.
    """
    historical = read_historical_clouds(historical_path)
    transitions = compute_transition_matrix(historical)
    probabilities = compute_stochastic_matrix(transitions)
    clouds, _ = simulated
    for frame in [historical, clouds]:
        assert isinstance(frame, pd.DataFrame)
        assert frame.index.names == ["year", "month", "sday", "quarter"]
        assert {"eighths", "cloud", "c_date"} <= set(frame.columns)
    for frame in [transitions, probabilities, compute_transition_matrix(clouds)]:
        assert isinstance(frame, pd.DataFrame)
        assert frame.index.names == ["month", "origin"]
    _, expected_pairs = reference_counts(records)
    np.testing.assert_array_equal(transitions.to_numpy().reshape(12, 9, 9), expected_pairs)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    np.testing.assert_array_equal(probabilities.loc[(1, 0)], [0, 0, 0, 0, 0, 0, 0, 0, 1])


@pytest.mark.parametrize("origin", range(9))
def test_missing_transition_origins_follow_required_fallback_order(historical_path, origin):
    """Use the highest-priority available state for each empty origin.

    If a month has no recorded transitions out of some origin state, that
    row of the stochastic matrix is copied from the first state with data in
    that origin's priority list. The priorities favor nearby states with a
    bias that depends on the origin; they are defined in the test below.

    Progressively remove data from the most-preferred remaining substitute
    and check that the next available state is used. Give each substitute
    state ``s`` a recognizable transition pattern (``s`` -> ``s`` three
    times, ``s`` -> ``s+1`` once) to identify the copied row.
    """
    priorities = [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [2, 0, 3, 4, 5, 6, 7, 8],
        [1, 0, 3, 4, 5, 6, 7, 8],
        [4, 5, 6, 7, 8, 2, 1, 0],
        [3, 5, 6, 7, 8, 2, 1, 0],
        [6, 7, 4, 8, 3, 2, 1, 0],
        [5, 7, 4, 8, 3, 2, 1, 0],
        [6, 5, 8, 4, 3, 2, 1, 0],
        [7, 6, 5, 4, 3, 2, 1, 0],
    ][origin]
    template = compute_transition_matrix(read_historical_clouds(historical_path))
    for position, preferred in enumerate(priorities):
        counts = template.copy()
        counts.loc[1] = 0
        for source in priorities[position:]:
            counts.loc[(1, source), source] = 3
            counts.loc[(1, source), (source + 1) % 9] = 1
        probabilities = compute_stochastic_matrix(counts)
        expected = np.zeros(9)
        expected[preferred], expected[(preferred + 1) % 9] = 0.75, 0.25
        np.testing.assert_allclose(probabilities.loc[(1, origin)], expected)


def test_diagnostic_plots_show_monthly_counts_and_normalized_distributions(historical_path, simulated):
    """Check monthly counts and normalized distributions in diagnostic plots.

    Cloud-state histograms: one panel per month. The historical distribution
    is a normalized histogram, and the simulated distribution is overplotted
    as normalized line(s): either one line for all simulated years combined,
    or one per simulated year if a dictionary of year to distribution is
    supplied.

    Transition histograms: two panels per month (historical and simulated),
    each showing the transition count matrix as an image, for all simulated
    years combined or for a single selected year.
    """
    historical = read_historical_clouds(historical_path)
    clouds, _ = simulated
    historical_counts = count_cloud_states(historical)
    by_year = {int(year): count_cloud_states(group) for year, group in clouds.groupby(level="year")}
    for counts in [count_cloud_states(clouds), by_year]:
        figure = plot_cloud_histogram(historical_counts, counts)
        assert isinstance(figure, Figure)
        assert len(figure.axes) == 12
        series = counts if isinstance(counts, dict) else {"Simulated": counts}
        for month, axis in enumerate(figure.axes, 1):
            historical_values = historical_counts.loc[month].to_numpy(dtype=float)
            np.testing.assert_allclose(
                axis.patches[0].get_data().values, historical_values / historical_values.sum()
            )
            for line, (year, distribution) in zip(axis.lines, series.items()):
                values = distribution.loc[month].to_numpy(dtype=float)
                np.testing.assert_allclose(line.get_xdata(), np.arange(9) / 8)
                np.testing.assert_allclose(line.get_ydata(), values / (values.sum() or 1))
                assert line.get_label() == str(year)
        plt.close(figure)
    historical_pairs = compute_transition_matrix(historical)
    for selected in [clouds, clouds.loc[clouds.index.get_level_values("year") == 1976]]:
        simulated_pairs = compute_transition_matrix(selected)
        figure = plot_transition_histograms(historical_pairs, simulated_pairs)
        image_axes = [axis for axis in figure.axes if axis.images]
        assert len(image_axes) == 24
        for month in range(1, 13):
            for offset, counts in enumerate([historical_pairs, simulated_pairs]):
                np.testing.assert_array_equal(
                    image_axes[2 * (month - 1) + offset].images[0].get_array(), counts.loc[month]
                )
        plt.close(figure)


@pytest.mark.parametrize("dataset", ["historical", "simulated"])
def test_full_cloud_sequence_accepts_either_dataset(historical_path, simulated, dataset):
    """Plot either historical or simulated records in the full-sequence figure.

    Each quarter is drawn as one cell, colored with the cividis color map by
    cloud level (clear is dark, cloudy is yellow), with one row per year.
    """
    clouds = read_historical_clouds(historical_path) if dataset == "historical" else simulated[0]
    figure = make_cloud_figure(clouds.eighths)
    assert isinstance(figure, Figure)
    colors = figure.axes[0].collections[0].get_facecolors()
    np.testing.assert_allclose(colors[: len(clouds)], plt.get_cmap("cividis")(clouds.eighths / 8))
    assert figure.axes[0].get_ylabel() == "Year"
    assert figure.axes[0].yaxis_inverted()


def test_full_cloud_sequence_missing_quarters_and_leap_dates():
    """Render missing quarters and leap days in their expected positions.

    The input has an absent quarter, a NaN, and both missing-data sentinels (9
    and -1) around February 28-29 of a leap year. With no ``origin`` given,
    valid quarters use the cividis colors and every kind of missing quarter is
    red, not silently skipped. The leap day occupies its own column (day 60).

    If ``origin`` is given, color represents the data source and transparency
    the cloud level, and missing data are white.
    """
    index = pd.MultiIndex.from_arrays(
        [
            np.array([1976] * 5, dtype=np.uint16),
            np.array([2] * 5, dtype=np.uint8),
            np.array([28, 28, 28, 29, 29], dtype=np.uint8),
            np.array([1, 2, 4, 1, 2], dtype=np.uint8),
        ],
        names=["year", "month", "sday", "quarter"],
    )
    eighths = pd.Series([0, 8, np.nan, 9, -1], index=index)
    figure = make_cloud_figure(eighths)
    colors = figure.axes[0].collections[0].get_facecolors()
    start = 58 * 4  # Use February 28's zero-based quarter position in a leap year.
    np.testing.assert_allclose(colors[start : start + 2], plt.get_cmap("cividis")([0.0, 1.0]))
    np.testing.assert_allclose(colors[start + 2 : start + 6], np.tile(to_rgba("red"), (4, 1)))
    assert len(colors) == 366 * 4
    vertices = figure.axes[0].collections[0].get_paths()[start + 4].vertices
    assert vertices[:, 0].min() == 60  # February 29 is day 60.
    assert vertices[:, 0].max() == 60.25

    origin = pd.Series("ctio", index=index)
    source_figure = make_cloud_figure(eighths, origin)
    colors = source_figure.axes[0].collections[0].get_facecolors()
    np.testing.assert_allclose(colors[start : start + 2, :3], np.tile(cc.glasbey_category10[0], (2, 1)))
    np.testing.assert_allclose(colors[start : start + 2, 3], [1, 0.5])
    np.testing.assert_allclose(colors[start + 2 : start + 6], np.tile(to_rgba("white"), (4, 1)))


def test_cloud_source_palette_matches_prototype(historical_path):
    """Color each source consistently and include it in the legend.

    Map ``ctio``, ``blanco``, ``satellite``, and ``stochastic_matrix`` to
    palette indices 0, 9, 4, and 3 in ``glasbey_category10``. Set opacity to
    ``1 - eighths / 16`` so it decreases with cloud cover, and include
    "Missing" in the legend.
    """
    clouds = read_historical_clouds(historical_path).iloc[:4]
    origins = pd.Series(["ctio", "blanco", "satellite", "stochastic_matrix"], index=clouds.index)
    figure = make_cloud_figure(clouds.eighths, origins)
    colors = figure.axes[0].collections[0].get_facecolors()[:4]
    np.testing.assert_allclose(colors[:, :3], np.array(cc.glasbey_category10)[[0, 9, 4, 3]])
    np.testing.assert_allclose(colors[:, 3], 1 - clouds.eighths.to_numpy() / 16)
    assert {text.get_text() for text in figure.axes[0].get_legend().get_texts()} == {
        "ctio",
        "blanco",
        "satellite",
        "stochastic_matrix",
        "Missing",
    }


def test_cli_writes_database_and_all_diagnostics(tmp_path, historical_path, records):
    """Exercise the CLI end to end with every diagnostic output option.

    Runs the CLI with every output option, then checks that all outputs exist
    and the images are readable. The sequence figure and the database must be
    identical to those from calling the Python workflow with the same seed,
    and the simulated months must be typical of the historical data.
    """
    database, histogram, pairs = [tmp_path / name for name in ["cli.db", "clouds.png", "pairs.png"]]
    sequence = tmp_path / "sequence.png"
    result = CliRunner().invoke(
        generate_average_clouds_cli,
        [
            str(historical_path),
            "-o",
            str(database),
            "--num-years",
            "1",
            "--cloud-histogram",
            str(histogram),
            "--transition-histograms",
            str(pairs),
            "--cloud-figure",
            str(sequence),
        ],
    )
    assert result.exit_code == 0, result.output
    assert all(path.stat().st_size > 0 for path in [database, histogram, pairs, sequence])
    for path in [histogram, pairs, sequence]:
        assert plt.imread(path).ndim == 3
    api_sequence = tmp_path / "api-sequence.png"
    expected = cloud_generation_workflow(
        historical_path, tmp_path / "api.db", 1, seed=6563, cloud_figure=api_sequence
    )
    np.testing.assert_array_equal(plt.imread(sequence), plt.imread(api_sequence))
    assert not plt.get_fignums()
    assert_typical_months(expected, *reference_counts(records))
    with closing(sqlite3.connect(database)) as connection:
        saved = connection.execute("SELECT c_date,cloud FROM Cloud ORDER BY cloudId").fetchall()
    np.testing.assert_array_equal(saved, expected[["c_date", "cloud"]].to_numpy())


@pytest.mark.parametrize("year", [None, 1976])
def test_optional_observability_acf(simulated, year):
    """Plot historical and simulated ACFs when ``statsmodels`` is installed.

    Observability is 1 where the cloud fraction is below the threshold and 0
    otherwise. Quarters outside the selected month, or with missing data,
    become NaN (not dropped) so that no artificial transitions are created
    across gaps. The plotted curves are compared with ``statsmodels``' ACF
    of independently prepared observability series, for all simulated years
    (``year=None``) or one selected year. Skipped if ``statsmodels`` is not
    installed.
    """
    stattools = pytest.importorskip("statsmodels.tsa.stattools")
    clouds, _ = simulated
    clouds = clouds.copy()
    clouds.loc[(1975, 1, 1, 2), ["eighths", "cloud"]] = [9, np.nan]
    figure = observability_acf(0.5, clouds, clouds, 1, year, nlags=6)
    assert isinstance(figure, Figure)
    for line, selected_year in zip(figure.axes[0].lines, [None, year]):
        selected = clouds if selected_year is None else clouds.loc[(selected_year, 1)]
        in_month = clouds.index.get_level_values("month") == 1 if selected_year is None else True
        values = np.where(in_month & selected.cloud.notna(), (selected.cloud < 0.5).astype(float), np.nan)
        np.testing.assert_allclose(line.get_ydata(), stattools.acf(values, nlags=6, missing="conservative"))
    assert [line.get_label() for line in figure.axes[0].lines] == ["Historical", "Simulated"]


def test_acf_dependency_is_optional_and_insufficient_data_is_rejected(monkeypatch, simulated):
    """Keep other diagnostic plots available without ``statsmodels``.

    Simulates a missing ``statsmodels`` and verifies that the other
    diagnostic plots still work, and that the ACF function raises an
    ``ImportError`` naming the package. When the package is present, also
    checks that a ``ValueError`` is raised if there is too little data
    to compute an ACF.
    """
    clouds, _ = simulated
    original_import = builtins.__import__

    def without_statsmodels(name, *args, **kwargs):
        if name.startswith("statsmodels"):
            raise ModuleNotFoundError("statsmodels unavailable")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", without_statsmodels)
        assert isinstance(
            plot_cloud_histogram(count_cloud_states(clouds), count_cloud_states(clouds)), Figure
        )
        with pytest.raises(ImportError, match="statsmodels"):
            observability_acf(0.5, clouds, clouds, 1)
    pytest.importorskip("statsmodels.tsa.stattools")
    with pytest.raises(ValueError, match="too little"):
        observability_acf(0.5, clouds.iloc[:1], clouds.iloc[:1], 1)


@pytest.mark.skipif(not os.environ.get("LSST64_HISTORICAL_DATA"), reason="Opt in with LSST64_HISTORICAL_DATA")
def test_historical_data_workflow(tmp_path, record_property):
    """Run the opt-in 20-year workflow against real historical data.

    Intended to be run as a batch job (see ``LSST64_HISTORICAL_DATA``) on the
    historical record file with 17,414 nights. Verifies that generating 20
    years and all diagnostic plots takes under an hour, that every month is
    typical of its calendar month, that the simulation covers at least 20
    years, and that the output is readable by ``CloudData``.
    """
    path = Path(os.environ["LSST64_HISTORICAL_DATA"])
    records = pd.read_csv(path, sep=r"\s+")
    assert len(records) == 17414, "This test requires the standard 17,414-night historical input"
    database, histogram, pairs = [tmp_path / name for name in ["average.db", "clouds.png", "pairs.png"]]
    sequence = tmp_path / "sequence.png"
    start = perf_counter()
    clouds = cloud_generation_workflow(
        path, database, 20, cloud_histogram=histogram, transition_histograms=pairs, cloud_figure=sequence
    )
    elapsed = perf_counter() - start
    record_property("workflow_seconds", elapsed)
    assert elapsed < 3600, f"Generation and diagnostics took {elapsed:.1f} seconds, over the one hour limit"
    assert_typical_months(clouds, *reference_counts(records))
    assert int(clouds.c_date.iloc[-1]) - int(clouds.c_date.iloc[0]) >= 20 * 31557600
    assert all(output.stat().st_size > 0 for output in [database, histogram, pairs, sequence])
    reader = CloudData(Time("2025-01-01", scale="tai"), cloud_db=str(database))
    assert reader.time_range >= 20 * 31557600
