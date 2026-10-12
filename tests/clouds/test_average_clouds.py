"""Black-box verification of LSST-64 external requirements.

Ordinary tests use synthetic CTIO records. Set LSST64_HISTORICAL_DATA to
enable the separate 20-year historical-data acceptance test for R-5/R-6.
"""

import builtins
import calendar
import os
import sqlite3
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time
from astropy.utils import iers
from click.testing import CliRunner
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from rubin_scheduler.site_models import CloudData

from rubin_sim.clouds import (
    cloud_generation_workflow,
    compute_stochastic_matrix,
    compute_transition_matrix,
    count_cloud_states,
    generate_average_clouds_cli,
    observability_acf,
    plot_cloud_histogram,
    plot_transition_histograms,
    read_historical_clouds,
)


@pytest.fixture(autouse=True)
def offline_plots():
    plt.switch_backend("Agg")
    with iers.conf.set_temp("auto_download", False):
        try:
            yield
        finally:
            plt.close("all")


@pytest.fixture(scope="module")
def records():
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
    path = tmp_path_factory.mktemp("requirements-clouds") / "synthetic.txt"
    records.to_csv(path, sep=" ", index=False)
    return path


@pytest.fixture(scope="module")
def simulated(tmp_path_factory, historical_path):
    directory = tmp_path_factory.mktemp("requirements-output")
    clouds = cloud_generation_workflow(historical_path, directory / "clouds.db", 2, seed=6563)
    return clouds, directory / "clouds.db"


def reference_counts(records):
    """Count from raw CTIO rows, independently of the cloud API (R-3/R-4)."""
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
    """Verify every month, not merely the combined simulated distribution."""
    previous = None
    for (year, month), group in clouds.groupby(level=["year", "month"], sort=True):
        states = group.eighths.to_numpy(dtype=int)
        quarters = 4 * calendar.monthrange(int(year), int(month))[1]
        assert len(states) == quarters
        assert set(states) <= set(range(9))
        expected = historical_states[month - 1] * quarters / historical_states[month - 1].sum()
        actual = np.bincount(states, minlength=9)
        assert np.all(np.abs(actual - expected) <= 1 + 1e-12), (year, month, "R-3")

        pairs = list(zip(states[:-1], states[1:]))
        if previous is not None:
            pairs.insert(0, (previous, states[0]))
        actual_pairs = np.zeros((9, 9), dtype=int)
        for origin, destination in pairs:
            actual_pairs[origin, destination] += 1
        historical = historical_transitions[month - 1]
        expected_pairs = historical * len(pairs) / historical.sum()
        tolerance = np.where(expected_pairs < 1, 1, np.sqrt(expected_pairs))
        assert np.all(np.abs(actual_pairs - expected_pairs) <= tolerance + 1e-12), (year, month, "R-4")
        previous = int(states[-1])


def test_read_ctio_quarters_and_exclude_missing_data(tmp_path, records):
    """R-1/R-3/R-4/R-7: preserve values and exclude missing pairs."""
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
    """R-1: a month absent in every year cannot be simulated."""
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
    """R-3/R-4: check each year, leap February, and month boundaries."""
    clouds, _ = simulated
    assert set(clouds.eighths) == {0, 8}
    assert len(clouds.loc[(1976, 2)]) == 29 * 4
    assert_typical_months(clouds, *reference_counts(records))


def test_database_schema_values_and_scheduler_repetition(simulated):
    """R-2: inspect the actual SQLite output and use the scheduler reader."""
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
    """R-2: infer twilight endpoints from evenly spaced quarter centers."""
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
    """R-7: callers can inspect records, counts, probabilities, and output."""
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
    """R-4: exhaust every priority in the specified fallback table."""
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
    """R-8: verify plot data for all years and a selected year."""
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


def test_cli_writes_database_and_both_diagnostics(tmp_path, historical_path, records):
    """R-1/R-2/R-3/R-4/R-8: exercise the user-facing command end to end."""
    database, histogram, pairs = [tmp_path / name for name in ["cli.db", "clouds.png", "pairs.png"]]
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
        ],
    )
    assert result.exit_code == 0, result.output
    assert all(path.stat().st_size > 0 for path in [database, histogram, pairs])
    for path in [histogram, pairs]:
        assert plt.imread(path).ndim == 3
    expected = cloud_generation_workflow(historical_path, tmp_path / "api.db", 1, seed=6563)
    assert_typical_months(expected, *reference_counts(records))
    with closing(sqlite3.connect(database)) as connection:
        saved = connection.execute("SELECT c_date,cloud FROM Cloud ORDER BY cloudId").fetchall()
    np.testing.assert_array_equal(saved, expected[["c_date", "cloud"]].to_numpy())


@pytest.mark.parametrize("year", [None, 1976])
def test_optional_observability_acf(simulated, year):
    """R-8: compare ACF curves with independently prepared inputs."""
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
    """R-8: missing statsmodels must not disable the other diagnostics."""
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
    """R-5/R-6: 20 years and both diagnostics with the specified CTIO input."""
    path = Path(os.environ["LSST64_HISTORICAL_DATA"])
    records = pd.read_csv(path, sep=r"\s+")
    assert len(records) == 17414, "R-5 requires the specified 17,414-night historical input"
    database, histogram, pairs = [tmp_path / name for name in ["average.db", "clouds.png", "pairs.png"]]
    start = perf_counter()
    clouds = cloud_generation_workflow(
        path, database, 20, cloud_histogram=histogram, transition_histograms=pairs
    )
    elapsed = perf_counter() - start
    record_property("workflow_seconds", elapsed)
    assert elapsed < 3600, f"R-5: generation and diagnostics took {elapsed:.1f} seconds"
    assert_typical_months(clouds, *reference_counts(records))
    assert int(clouds.c_date.iloc[-1]) - int(clouds.c_date.iloc[0]) >= 20 * 31557600
    assert all(output.stat().st_size > 0 for output in [database, histogram, pairs])
    reader = CloudData(Time("2025-01-01", scale="tai"), cloud_db=str(database))
    assert reader.time_range >= 20 * 31557600
