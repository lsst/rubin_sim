"""Verify LSST-64 cloud records, monthly statistics, and scheduler output."""

import calendar
import os
import sqlite3
import warnings
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import matplotlib
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
    generate_average_clouds,
    generate_average_clouds_cli,
    make_cloud_frame,
    plot_cloud_histogram,
    plot_transition_histograms,
    read_historical_clouds,
    save_clouds,
)
from rubin_sim.clouds.generate import _generate_month

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def offline_astronomy():
    """Avoid network access and close figures even after a failed assertion."""
    with iers.conf.set_temp("auto_download", False):
        yield
    plt.close("all")


@pytest.fixture(scope="module")
def clear_records():
    """A complete historical year guarantees feasible all-clear generation."""
    nights = pd.date_range("1975-01-01", "1975-12-31", freq="D")
    return pd.DataFrame(
        {
            "sday": nights.day,
            "eday": (nights + pd.Timedelta(days=1)).day,
            "month": nights.month,
            "year": nights.year,
            "q1": 0,
            "q2": 0,
            "q3": 0,
            "q4": 0,
        }
    )


@pytest.fixture(scope="module")
def historical_path(tmp_path_factory, clear_records):
    path = tmp_path_factory.mktemp("historical-clouds") / "historical.txt"
    clear_records.to_csv(path, sep="\t", index=False)
    return path


@pytest.fixture(scope="module")
def historical(historical_path):
    return read_historical_clouds(historical_path)


@pytest.fixture(scope="module")
def clear_statistics(historical):
    return compute_transition_matrix(historical), count_cloud_states(historical)


@pytest.fixture(scope="module")
def generated(clear_statistics):
    transitions, distribution = clear_statistics
    return generate_average_clouds(transitions, distribution, 2, max_retries=1)


def assert_cloud_schema(clouds):
    assert clouds.index.names == ["year", "month", "sday", "quarter"]
    assert clouds.index.is_unique
    assert clouds.index.is_monotonic_increasing
    for name, dtype in zip(clouds.index.names, [np.uint16, np.uint8, np.uint8, np.uint8]):
        assert clouds.index.get_level_values(name).dtype == np.dtype(dtype)
    assert list(clouds.columns) == ["eighths", "cloud", "c_date"]
    assert clouds.eighths.dtype == np.dtype(np.uint8)
    assert clouds.cloud.dtype == np.dtype(np.float64)
    assert clouds.c_date.dtype == np.dtype(np.uintp)
    np.testing.assert_allclose(
        clouds.cloud, np.where(clouds.eighths == 9, np.nan, clouds.eighths / 8), equal_nan=True
    )
    assert np.all(np.diff(clouds.c_date.to_numpy(dtype=np.int64)) > 0)


def assert_monthly_requirements(clouds, historical_transitions, historical_distribution):
    """Count independently; never trust the implementation's own validation."""
    previous = None
    for (year, month), group in clouds.groupby(level=["year", "month"], sort=True):
        states = group.eighths.to_numpy(dtype=int)
        quarters = 4 * calendar.monthrange(int(year), int(month))[1]
        assert len(states) == quarters
        assert np.all((states >= 0) & (states <= 8))
        expected_states = historical_distribution.loc[month].to_numpy(dtype=float)
        expected_states *= quarters / expected_states.sum()
        actual_states = np.bincount(states, minlength=9)
        assert np.all(np.abs(actual_states - expected_states) <= 1 + 1e-12), (year, month, "R-3")
        origins = states[:-1] if previous is None else np.r_[previous, states[:-1]]
        destinations = states[1:] if previous is None else states
        actual_transitions = np.bincount(origins * 9 + destinations, minlength=81).reshape(9, 9)
        expected_transitions = historical_transitions.loc[month].to_numpy(dtype=float)
        expected_transitions *= len(origins) / expected_transitions.sum()
        assert np.all(
            np.abs(actual_transitions - expected_transitions)
            <= np.maximum(1, np.sqrt(expected_transitions)) + 1e-12
        ), (year, month, "R-4")
        previous = int(states[-1])


def test_read_schema_and_missing_values(tmp_path, clear_records):
    historical_path = tmp_path / "mixed.txt"
    clear_records = clear_records.copy()
    clear_records.loc[0, ["q1", "q2", "q3", "q4"]] = [0, 2, 8, 9]
    clear_records.to_csv(historical_path, sep=" ", index=False)
    clouds = read_historical_clouds(str(historical_path))
    assert_cloud_schema(clouds)
    np.testing.assert_array_equal(clouds.eighths.iloc[:4], [0, 2, 8, 9])
    assert np.isnan(clouds.cloud.iloc[3])
    counts = count_cloud_states(clouds)
    assert counts.index.name == "month"
    assert counts.index.dtype == np.dtype(np.uint8)
    assert list(counts.columns) == list(range(9))
    assert all(dtype == np.dtype(np.uintp) for dtype in counts.dtypes)
    assert counts.loc[1].sum() == 31 * 4 - 1
    assert counts.loc[1, 2] == counts.loc[1, 8] == 1


def test_multiple_inputs_deduplicate_whole_night(tmp_path, clear_records):
    first, second = tmp_path / "first.txt", tmp_path / "second.txt"
    early = clear_records.loc[clear_records.month <= 6].copy()
    late = clear_records.loc[clear_records.month >= 6].copy()
    early.loc[early.month == 6, "q2"] = 9
    late.loc[late.month == 6, ["q1", "q2", "q3", "q4"]] = 8
    early.iloc[::-1].to_csv(first, sep=" ", index=False)
    late.to_csv(second, sep="\t", index=False)
    clouds = read_historical_clouds([first, second])
    assert len(clouds) == 365 * 4
    assert clouds.index.is_monotonic_increasing
    assert clouds.loc[(1975, 6, 1, 2), "eighths"] == 9
    assert clouds.loc[(1975, 6, 1, 1), "eighths"] == 0
    reversed_clouds = read_historical_clouds([second, first])
    assert reversed_clouds.loc[(1975, 6, 1, 2), "eighths"] == 8


@pytest.mark.parametrize(
    "invalid", ["missing_month", "all_missing_month", "state", "negative_state", "date", "header"]
)
def test_invalid_historical_records(tmp_path, clear_records, invalid):
    clear_records = clear_records.copy()
    if invalid == "missing_month":
        clear_records = clear_records.loc[clear_records.month != 2]
    elif invalid == "all_missing_month":
        clear_records.loc[clear_records.month == 2, ["q1", "q2", "q3", "q4"]] = 9
    elif invalid == "state":
        clear_records.loc[0, "q1"] = 10
    elif invalid == "negative_state":
        clear_records.loc[0, "q1"] = -2
    elif invalid == "date":
        clear_records.loc[0, "sday"] = 32
    else:
        clear_records = clear_records.drop(columns="q4")
    path = tmp_path / "invalid.txt"
    clear_records.to_csv(path, sep=" ", index=False)
    with pytest.raises(ValueError):
        read_historical_clouds(path)


def test_supplied_ctio_negative_missing_sentinel_without_warning(tmp_path, clear_records):
    records = clear_records.copy()
    records.loc[0, ["q2", "q3"]] = -1
    path = tmp_path / "negative-missing.txt"
    records.to_csv(path, sep=" ", index=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        clouds = read_historical_clouds(path)
    np.testing.assert_array_equal(clouds.eighths.iloc[:4], [0, 9, 9, 0])
    assert clouds.cloud.iloc[1:3].isna().all()
    assert compute_transition_matrix(clouds).to_numpy().sum() == len(clouds) - 4


def test_no_historical_paths():
    with pytest.raises(ValueError):
        read_historical_clouds([])


def invalid_schema(frame, invalid):
    """Corrupt one invariant without relying on implementation validators."""
    frame = frame.copy()
    if invalid == "not_dataframe":
        return frame.to_numpy()
    if invalid == "column_order":
        return frame.iloc[:, ::-1]
    if invalid == "missing_column":
        return frame.iloc[:, :-1]
    if invalid == "extra_column":
        frame["extra"] = 0
    elif invalid == "duplicate_column":
        return pd.concat([frame, frame.iloc[:, [0]]], axis=1)
    elif invalid == "column_names":
        frame.columns = pd.Index(["wrong", *frame.columns[1:]], name=frame.columns.name)
    elif invalid == "column_missing_label":
        frame.columns = pd.Index([*frame.columns[:-1], pd.NA], name=frame.columns.name)
    elif invalid == "destination_name":
        frame.columns.name = "wrong"
    elif invalid == "column_float_labels":
        frame.columns = frame.columns.astype(float)
    elif invalid == "nullable_counts":
        frame = frame.astype("Int64")
        frame.iloc[0, 0] = pd.NA
    elif invalid == "nullable_month":
        frame.index = pd.Index([pd.NA, *frame.index[1:]], dtype="Int64", name="month")
    elif invalid == "row_order":
        return frame.iloc[::-1]
    elif invalid == "duplicate_row":
        return pd.concat([frame.iloc[[0]], frame])
    elif invalid == "missing_row":
        return frame.iloc[1:]
    elif invalid == "missing_month":
        return (
            frame.drop(index=12, level="month") if isinstance(frame.index, pd.MultiIndex) else frame.drop(12)
        )
    elif invalid == "index_names":
        frame.index = frame.index.set_names(["wrong"] * frame.index.nlevels)
    elif invalid == "index_order":
        frame = frame.reorder_levels(list(reversed(range(frame.index.nlevels))))
    elif invalid == "flat_index":
        frame.index = pd.RangeIndex(len(frame))
    elif invalid in {"float_counts", "bool_counts", "signed_counts", "negative_counts", "nan_counts"}:
        dtype = (
            bool
            if invalid == "bool_counts"
            else float if invalid in {"float_counts", "nan_counts"} else np.int64
        )
        frame = frame.astype(dtype)
        if invalid == "negative_counts":
            frame.iloc[0, 0] = -1
        elif invalid == "nan_counts":
            frame.iloc[0, 0] = np.nan
    elif invalid in {"eighths_dtype", "cloud_dtype", "time_dtype"}:
        column, dtype = {
            "eighths_dtype": ("eighths", np.int64),
            "cloud_dtype": ("cloud", np.float32),
            "time_dtype": ("c_date", np.int64),
        }[invalid]
        frame[column] = frame[column].astype(dtype)
    elif invalid == "state_range":
        frame.iloc[0, frame.columns.get_loc("eighths")] = 10
    elif invalid in {"cloud_inconsistent", "cloud_nan", "cloud_infinite", "missing_inconsistent"}:
        value = {
            "cloud_inconsistent": 0.5,
            "cloud_nan": np.nan,
            "cloud_infinite": np.inf,
            "missing_inconsistent": 0.0,
        }[invalid]
        frame.iloc[0, frame.columns.get_loc("cloud")] = value
        if invalid == "missing_inconsistent":
            frame.iloc[0, frame.columns.get_loc("eighths")] = 9
    elif invalid in {"time_duplicate", "time_decreasing"}:
        frame.iloc[1, frame.columns.get_loc("c_date")] = frame.c_date.iloc[0] - (invalid == "time_decreasing")
    else:
        arrays = (
            [frame.index.get_level_values(i).to_numpy(copy=True) for i in range(frame.index.nlevels)]
            if isinstance(frame.index, pd.MultiIndex)
            else [frame.index.to_numpy(copy=True)]
        )
        names = frame.index.names
        if invalid.endswith("_dtype"):
            level = names.index(invalid.removesuffix("_dtype"))
            arrays[level] = arrays[level].astype(np.int64)
        elif invalid == "month_float":
            arrays[names.index("month")] = arrays[names.index("month")].astype(float)
        else:
            field, value = {
                "year_range": ("year", 1974),
                "month_zero": ("month", 0),
                "month_range": ("month", 13),
                "day_zero": ("sday", 0),
                "day_range": ("sday", 32),
                "invalid_date": ("sday", 30),
                "quarter_zero": ("quarter", 0),
                "quarter_range": ("quarter", 5),
                "origin_range": ("origin", 9),
            }[invalid]
            if invalid == "invalid_date":
                arrays[names.index("month")][0] = 2
            arrays[names.index(field)][0] = value
        frame.index = (
            pd.MultiIndex.from_arrays(arrays, names=names)
            if len(arrays) > 1
            else pd.Index(arrays[0], name=names[0])
        )
    return frame


@pytest.mark.parametrize("consumer", ["counts", "transitions", "save"])
@pytest.mark.parametrize(
    "invalid",
    [
        "not_dataframe",
        "column_order",
        "missing_column",
        "extra_column",
        "duplicate_column",
        "column_names",
        "column_missing_label",
        "row_order",
        "duplicate_row",
        "index_names",
        "index_order",
        "flat_index",
        "year_dtype",
        "month_dtype",
        "sday_dtype",
        "quarter_dtype",
        "eighths_dtype",
        "cloud_dtype",
        "time_dtype",
        "year_range",
        "month_zero",
        "month_range",
        "day_zero",
        "day_range",
        "invalid_date",
        "quarter_zero",
        "quarter_range",
        "state_range",
        "cloud_inconsistent",
        "cloud_nan",
        "cloud_infinite",
        "missing_inconsistent",
        "time_duplicate",
        "time_decreasing",
    ],
)
def test_clouds_schema_invalid_for_all_consumers(historical, tmp_path, consumer, invalid):
    clouds = invalid_schema(historical.iloc[:8], invalid)
    database = tmp_path / "invalid.db"
    with pytest.raises(ValueError):
        if consumer == "counts":
            count_cloud_states(clouds)
        elif consumer == "transitions":
            compute_transition_matrix(clouds)
        else:
            save_clouds(database, clouds)
    assert not database.exists()


@pytest.mark.parametrize("consumer", ["stochastic", "generate", "plot_historical", "plot_simulated"])
@pytest.mark.parametrize(
    "invalid",
    [
        "not_dataframe",
        "column_order",
        "missing_column",
        "extra_column",
        "duplicate_column",
        "column_names",
        "column_missing_label",
        "destination_name",
        "column_float_labels",
        "row_order",
        "duplicate_row",
        "missing_row",
        "missing_month",
        "index_names",
        "index_order",
        "flat_index",
        "month_dtype",
        "origin_dtype",
        "month_zero",
        "month_range",
        "origin_range",
        "float_counts",
        "bool_counts",
        "signed_counts",
        "negative_counts",
        "nan_counts",
    ],
)
def test_transition_schema_invalid_for_all_consumers(clear_statistics, monkeypatch, consumer, invalid):
    matrix, distribution = clear_statistics
    invalid_matrix = invalid_schema(matrix, invalid)
    figures = plt.get_fignums()
    monkeypatch.setattr(
        "rubin_sim.clouds.generate._generate_month", lambda *args: pytest.fail("Sampled invalid schema")
    )
    with pytest.raises(ValueError):
        if consumer == "stochastic":
            compute_stochastic_matrix(invalid_matrix)
        elif consumer == "generate":
            generate_average_clouds(invalid_matrix, distribution, 1)
        elif consumer == "plot_historical":
            plot_transition_histograms(invalid_matrix, matrix)
        else:
            plot_transition_histograms(matrix, invalid_matrix)
    assert plt.get_fignums() == figures


@pytest.mark.parametrize("consumer", ["generate", "plot_historical", "plot_simulated", "plot_dictionary"])
@pytest.mark.parametrize(
    "invalid",
    [
        "not_dataframe",
        "column_order",
        "missing_column",
        "extra_column",
        "duplicate_column",
        "column_names",
        "column_missing_label",
        "column_float_labels",
        "duplicate_row",
        "index_names",
        "month_float",
        "month_zero",
        "month_range",
        "float_counts",
        "bool_counts",
        "negative_counts",
        "nan_counts",
        "nullable_counts",
        "nullable_month",
    ],
)
def test_distribution_schema_invalid_for_all_consumers(clear_statistics, monkeypatch, consumer, invalid):
    matrix, distribution = clear_statistics
    invalid_distribution = invalid_schema(distribution, invalid)
    figures = plt.get_fignums()
    monkeypatch.setattr(
        "rubin_sim.clouds.generate._generate_month", lambda *args: pytest.fail("Sampled invalid schema")
    )
    with pytest.raises(ValueError):
        if consumer == "generate":
            generate_average_clouds(matrix, invalid_distribution, 1)
        elif consumer == "plot_historical":
            plot_cloud_histogram(invalid_distribution, distribution)
        elif consumer == "plot_simulated":
            plot_cloud_histogram(distribution, invalid_distribution)
        else:
            plot_cloud_histogram(distribution, {1975: distribution, 1976: invalid_distribution})
    assert plt.get_fignums() == figures


def test_generation_rejects_incomplete_distribution_before_sampling(clear_statistics, monkeypatch):
    matrix, distribution = clear_statistics
    monkeypatch.setattr(
        "rubin_sim.clouds.generate._generate_month", lambda *args: pytest.fail("Sampled incomplete months")
    )
    with pytest.raises(ValueError):
        generate_average_clouds(matrix, distribution.drop(12), 1)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint16, np.uintp])
def test_distribution_accepts_integer_dtypes(clear_statistics, dtype):
    matrix, distribution = clear_statistics
    distribution = distribution.astype(dtype)
    distribution.index = distribution.index.astype(dtype)
    clouds = generate_average_clouds(matrix, distribution, 1, max_retries=1)
    assert_monthly_requirements(clouds, matrix, distribution)
    figure = plot_cloud_histogram(distribution, {1975: distribution})
    assert len(figure.axes) == 12


def test_transition_adjacency_destination_month_and_missing_quarters():
    indices = [
        (1975, 1, 31, 3),
        (1975, 1, 31, 4),
        (1975, 2, 1, 1),
        (1975, 2, 1, 2),
        (1975, 2, 1, 3),
        (1975, 2, 1, 4),
        (1975, 2, 2, 1),
        (1975, 2, 2, 3),
        (1975, 2, 2, 4),
        (1975, 2, 4, 1),
        (1975, 2, 4, 2),
    ]
    clouds = make_cloud_frame(
        [date(1975, 1, 31), date(1975, 2, 1), date(1975, 2, 2), date(1975, 2, 4)],
        [0, 0, 1, 2, 3, 9, 4, 5, 6, 0, 7, 8, 0, 1, 0, 0],
    )
    clouds = clouds.loc[indices]
    assert_cloud_schema(clouds)
    matrix = compute_transition_matrix(clouds)
    expected = matrix * 0
    for month, origin, destination in [(1, 1, 2), (2, 2, 3), (2, 4, 5), (2, 5, 6), (2, 7, 8), (2, 0, 1)]:
        expected.loc[(month, origin), destination] = 1
    pd.testing.assert_frame_equal(matrix, expected)
    assert matrix.index.names == ["month", "origin"]
    assert all(level.dtype == np.dtype(np.uint8) for level in matrix.index.levels)
    assert matrix.columns.name == "destination"
    assert list(matrix.columns) == list(range(9))
    assert all(dtype == np.dtype(np.uintp) for dtype in matrix.dtypes)
    pd.testing.assert_frame_equal(compute_transition_matrix(clouds, prior_state=9), matrix)
    with_prior = compute_transition_matrix(clouds, prior_state=8)
    expected.loc[(1, 8), 1] = 1
    pd.testing.assert_frame_equal(with_prior, expected)


def test_historical_counts_pool_years_without_bridging_gaps():
    clouds = make_cloud_frame([date(1975, 1, 1), date(1976, 1, 1)], [0, 0, 0, 0, 0, 8, 9, 9])
    counts = count_cloud_states(clouds)
    assert counts.loc[1, 0] == 5
    assert counts.loc[1, 8] == 1
    matrix = compute_transition_matrix(clouds)
    assert matrix.loc[(1, 0), 0] == 3
    assert matrix.loc[(1, 0), 8] == 1
    assert matrix.to_numpy().sum() == 4
    # Other months need empirical transitions even when testing January.
    full_matrix = matrix.copy()
    for month in range(2, 13):
        full_matrix.loc[(month, 0), 0] = 1
    stochastic = compute_stochastic_matrix(full_matrix)
    np.testing.assert_allclose(stochastic.loc[(1, 0), [0, 8]], [0.75, 0.25])


FALLBACKS = [
    [1, 2, 3, 4, 5, 6, 7, 8],
    [2, 0, 3, 4, 5, 6, 7, 8],
    [1, 0, 3, 4, 5, 6, 7, 8],
    [4, 5, 6, 7, 8, 2, 1, 0],
    [3, 5, 6, 7, 8, 2, 1, 0],
    [6, 7, 4, 8, 3, 2, 1, 0],
    [5, 7, 4, 8, 3, 2, 1, 0],
    [6, 5, 8, 4, 3, 2, 1, 0],
    [7, 6, 5, 4, 3, 2, 1, 0],
]


@pytest.mark.parametrize("origin", range(9))
def test_stochastic_fallback_order(origin, clear_statistics):
    template, _ = clear_statistics
    # Progressively remove preferred rows to test the entire priority list.
    for position, preferred in enumerate(FALLBACKS[origin]):
        matrix = template.copy()
        matrix.loc[1] = 0
        for source in FALLBACKS[origin][position:]:
            matrix.loc[(1, source), source] = 3
            matrix.loc[(1, source), (source + 1) % 9] = 1
        stochastic = compute_stochastic_matrix(matrix)
        expected = np.zeros(9)
        expected[preferred] = 0.75
        expected[(preferred + 1) % 9] = 0.25
        np.testing.assert_allclose(stochastic.loc[(1, origin)], expected)
        np.testing.assert_allclose(stochastic.sum(axis=1), 1)
        assert all(dtype == np.dtype(np.float64) for dtype in stochastic.dtypes)
        pd.testing.assert_index_equal(stochastic.index, matrix.index)
        pd.testing.assert_index_equal(stochastic.columns, matrix.columns)
        np.testing.assert_allclose(stochastic.loc[(1, preferred)], expected)


def test_stochastic_empty_month_rejected(clear_statistics):
    matrix, _ = clear_statistics
    matrix = matrix.copy()
    matrix.loc[2] = 0
    with pytest.raises(ValueError):
        compute_stochastic_matrix(matrix)


def test_generate_clear_months_first_transition_leap_and_duration(generated, clear_statistics):
    matrix, distribution = clear_statistics
    assert_cloud_schema(generated)
    assert_monthly_requirements(generated, matrix, distribution)
    assert len(generated.loc[(1976, 2)]) == 29 * 4
    assert (1976, 2, 29, 4) in generated.index
    assert int(generated.c_date.iloc[-1]) - int(generated.c_date.iloc[0]) >= 2 * 31557600
    first_month = generated.loc[[(1975, 1, day, quarter) for day in range(1, 32) for quarter in range(1, 5)]]
    assert compute_transition_matrix(first_month).to_numpy().sum() == 123
    assert compute_transition_matrix(generated).to_numpy().sum() == len(generated) - 1


def test_generate_nontrivial_monthly_requirements_and_seed(clear_statistics):
    transitions, distribution = (frame.copy() * 0 for frame in clear_statistics)
    for month in range(1, 13):
        weights = np.array([1, 3]) if month % 2 else np.array([2, 2])
        distribution.loc[month, [0, 8]] = weights * 100
        transitions.loc[(month, 0), [0, 8]] = weights[0] * weights * 100
        transitions.loc[(month, 8), [0, 8]] = weights[1] * weights * 100
    clouds = generate_average_clouds(transitions, distribution, 1, seed=6563)
    assert set(clouds.eighths) == {0, 8}
    assert_monthly_requirements(clouds, transitions, distribution)
    repeated = generate_average_clouds(transitions, distribution, 1, seed=6563)
    pd.testing.assert_frame_equal(clouds, repeated)
    other = generate_average_clouds(transitions, distribution, 1, seed=42)
    assert_monthly_requirements(other, transitions, distribution)
    assert not clouds.eighths.equals(other.eighths)


@pytest.mark.parametrize("parameter", ["num_years", "max_retries", "max_steps"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_generation_invalid_limits(clear_statistics, parameter, value):
    transitions, distribution = clear_statistics
    kwargs = {"num_years": 1, parameter: value}
    with pytest.raises(ValueError):
        generate_average_clouds(transitions, distribution, **kwargs)


def test_generation_impossible_counts_fail_with_bounded_retries(clear_statistics):
    transitions, distribution = (frame.copy() for frame in clear_statistics)
    distribution.loc[:, :] = 0
    distribution.loc[:, 8] = 100
    # All observed transitions are 0->0, while all output states must be 8.
    with pytest.raises(RuntimeError):
        generate_average_clouds(transitions, distribution, 1, max_retries=2, max_steps=2)


def test_generation_required_rare_state_with_one_count_crossings(clear_statistics):
    transitions, distribution = (frame.copy() * 0 for frame in clear_statistics)
    for month in range(1, 13):
        distribution.loc[month, [0, 8]] = [12252, 148]
        transitions.loc[(month, 0), 0] = 99075
        transitions.loc[(month, 0), 8] = 100
        transitions.loc[(month, 8), 0] = 100
        transitions.loc[(month, 8), 8] = 725
    # January requires state 8. Revised R-4 allows one crossing even
    # though the historical expected crossing count is only .123.
    expected_eight = 124 * 148 / 12400
    expected_crossing = 123 * 100 / 100000
    assert expected_eight - 1 > 0
    assert expected_crossing + np.sqrt(expected_crossing) < 1
    clouds = generate_average_clouds(transitions, distribution, 1, max_steps=2)
    assert_monthly_requirements(clouds, transitions, distribution)
    assert 8 in clouds.loc[(1975, 1), "eighths"].to_numpy()


@pytest.mark.parametrize(
    "num_quarters, prior_state, accepted",
    [(2, None, True), (3, None, True), (4, None, False), (2, 8, True), (3, 8, False)],
)
def test_generate_month_zero_expected_transition_tolerance(num_quarters, prior_state, accepted):
    # Only unused origins have historical evidence. Fallback sampling can
    # produce one zero-expectation crossing, but not two of the same pair.
    transitions = np.zeros((9, 9))
    transitions[1:8, [0, 8]] = 1
    distribution = np.zeros(9)
    distribution[[0, 8]] = [(num_quarters + 1) // 2, num_quarters // 2]
    draws = iter([0.0, 0.99, 0.0, 0.99])
    rng = SimpleNamespace(random=lambda: next(draws))
    if accepted:
        sequence = _generate_month(transitions, distribution, num_quarters, prior_state, rng, 2)
        np.testing.assert_array_equal(sequence, [0, 8, 0][:num_quarters])
    else:
        with pytest.raises(RuntimeError, match="R-4"):
            _generate_month(transitions, distribution, num_quarters, prior_state, rng, 2)


@pytest.mark.parametrize("num_quarters, accepted", [(2, True), (3, False)])
def test_generate_month_positive_subunit_transition_tolerance(num_quarters, accepted):
    transitions = np.zeros((9, 9))
    transitions[0, 0] = 0.1
    transitions[1:, 0] = 1
    distribution = np.zeros(9)
    distribution[0] = 1
    expected = (num_quarters - 1) * transitions / transitions.sum()
    assert 0 < expected[0, 0] < 1
    assert ((num_quarters - 1) - expected[0, 0] <= 1) == accepted
    rng = np.random.default_rng(6563)
    if accepted:
        sequence = _generate_month(transitions, distribution, num_quarters, None, rng, 2)
        np.testing.assert_array_equal(sequence, np.zeros(num_quarters, dtype=np.uint8))
    else:
        with pytest.raises(RuntimeError, match="R-4"):
            _generate_month(transitions, distribution, num_quarters, None, rng, 2)


@pytest.mark.parametrize("num_quarters, accepted", [(7, True), (8, False)])
def test_generate_month_square_root_transition_tolerance(num_quarters, accepted):
    # E(0->0)=4: six transitions are exactly at sqrt(E), seven exceed it.
    transitions = np.eye(9) * ((num_quarters - 5) / 8)
    transitions[0, 0] = 4
    distribution = np.zeros(9)
    distribution[0] = 1
    rng = np.random.default_rng(6563)
    if accepted:
        sequence = _generate_month(transitions, distribution, num_quarters, None, rng, 2)
        np.testing.assert_array_equal(sequence, np.zeros(num_quarters, dtype=np.uint8))
    else:
        with pytest.raises(RuntimeError, match="R-4"):
            _generate_month(transitions, distribution, num_quarters, None, rng, 2)


def test_quarter_center_astronomy_and_tai_epoch(generated):
    location = EarthLocation.from_geodetic(-70.74941 * u.deg, -30.244628 * u.deg, 2650 * u.m)
    epoch = Time("1975-01-01T00:00:00", scale="tai")
    for night in [date(1975, 1, 1), date(1975, 6, 21), date(1975, 12, 21), date(1976, 2, 29)]:
        centers = generated.loc[(night.year, night.month, night.day), "c_date"].to_numpy(dtype=float)
        spacings = np.diff(centers)
        np.testing.assert_allclose(spacings, spacings.mean(), atol=1)
        spacing = (centers[-1] - centers[0]) / 3
        boundaries = epoch + np.array([centers[0] - spacing / 2, centers[-1] + spacing / 2]) * u.s
        altitude = (
            get_sun(boundaries)
            .transform_to(AltAz(obstime=boundaries, location=location, pressure=0 * u.hPa))
            .alt.deg
        )
        np.testing.assert_allclose(altitude, [-12, -12], atol=0.02)
        assert 6 * 3600 < 4 * spacing < 14 * 3600
        # The UTC midnight anchor belongs to the day after the local evening.
        midpoint = epoch + centers.mean() * u.s
        assert midpoint.utc.datetime.date() == night + timedelta(days=1)


def test_save_sql_schema_values_and_clouddata_wrap(tmp_path, generated):
    database = tmp_path / "clouds.db"
    varying = generated.copy()
    varying["eighths"] = (np.arange(len(varying)) % 9).astype(np.uint8)
    varying["cloud"] = varying.eighths / 8
    save_clouds(database, varying)
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
    assert len(saved) == len(varying)
    np.testing.assert_array_equal([row[0] for row in saved], np.arange(1, len(varying) + 1))
    np.testing.assert_array_equal([row[1] for row in saved], varying.c_date)
    np.testing.assert_array_equal([row[2] for row in saved], varying.cloud)
    assert {row[3] for row in saved} == {"simulation"}
    reader = CloudData(Time("2025-01-01", scale="tai"), cloud_db=str(database))
    assert reader.time_range >= 2 * 31557600
    rows = varying.iloc[[10, 100, 500]]
    lookup_times = reader.start_time + rows.c_date.to_numpy(dtype=float) * u.s
    np.testing.assert_allclose(reader(lookup_times), rows.cloud)
    np.testing.assert_allclose(reader(lookup_times + reader.time_range * u.s), rows.cloud)
    assert reader(lookup_times[0] + 1 * u.s) == rows.cloud.iloc[0]


def test_save_rejects_missing_states(tmp_path, historical):
    missing = historical.copy()
    missing.iloc[0, missing.columns.get_loc("eighths")] = 9
    missing.iloc[0, missing.columns.get_loc("cloud")] = np.nan
    with pytest.raises(ValueError):
        save_clouds(tmp_path / "missing.db", missing)


def test_diagnostic_figures_and_year_dictionary(clear_statistics, generated):
    transitions, distribution = clear_statistics
    by_year = {int(year): count_cloud_states(group) for year, group in generated.groupby(level="year")}
    for simulated in [count_cloud_states(generated), by_year]:
        figure = plot_cloud_histogram(distribution, simulated, linewidth=2, figsize=(12, 8))
        assert isinstance(figure, Figure)
        assert len(figure.axes) == 12
        np.testing.assert_allclose(figure.get_size_inches(), [12, 8])
        expected_labels = {str(year) for year in by_year} if isinstance(simulated, dict) else {"Simulated"}
        assert expected_labels <= {line.get_label() for line in figure.axes[0].lines}
        for axis in figure.axes:
            for line in axis.lines:
                assert np.isfinite(line.get_ydata()).all()
                assert line.get_linewidth() == 2
        plt.close(figure)
    selected = generated.loc[generated.index.get_level_values("year") == 1976]
    simulated_matrix = compute_transition_matrix(selected)
    figure = plot_transition_histograms(transitions, simulated_matrix, cmap="magma")
    assert isinstance(figure, Figure)
    image_axes = [axis for axis in figure.axes if axis.images]
    assert len(image_axes) == 24
    for month in range(1, 13):
        for offset, matrix in enumerate([transitions, simulated_matrix]):
            image = image_axes[2 * (month - 1) + offset].images[0]
            np.testing.assert_array_equal(image.get_array(), matrix.loc[month].to_numpy())
            assert image.get_cmap().name == "magma"


def test_workflow_writes_database_and_diagnostics(tmp_path, historical_path, clear_statistics):
    database, histogram, transitions_plot = [
        tmp_path / name for name in ["clouds.db", "clouds.png", "pairs.png"]
    ]
    simulated = cloud_generation_workflow(
        historical_path,
        database,
        1,
        cloud_histogram=histogram,
        transition_histograms=transitions_plot,
        max_retries=1,
    )
    assert_monthly_requirements(simulated, *clear_statistics)
    for path in [database, histogram, transitions_plot]:
        assert path.stat().st_size > 0
    assert not plt.get_fignums()


def test_cli_success_and_errors(tmp_path, historical_path):
    runner = CliRunner()
    database, histogram, transition_plot = [
        tmp_path / name for name in ["cli.db", "cli.png", "cli-pairs.png"]
    ]
    result = runner.invoke(
        generate_average_clouds_cli,
        [
            str(historical_path),
            "-o",
            str(database),
            "--num-years",
            "1",
            "--seed",
            "123",
            "--max-retries",
            "1",
            "--max-steps",
            "2",
            "--cloud-histogram",
            str(histogram),
            "--transition-histograms",
            str(transition_plot),
        ],
    )
    assert result.exit_code == 0, result.output
    assert all(path.stat().st_size > 0 for path in [database, histogram, transition_plot])
    result = runner.invoke(
        generate_average_clouds_cli, [str(historical_path), "-o", str(database), "--num-years", "0"]
    )
    assert result.exit_code != 0
    invalid = tmp_path / "invalid.txt"
    pd.DataFrame({"wrong": [1]}).to_csv(invalid, index=False)
    result = runner.invoke(generate_average_clouds_cli, [str(invalid), "-o", str(database)])
    assert result.exit_code != 0
    assert "Error" in result.output


@pytest.mark.skipif(
    not os.environ.get("LSST64_HISTORICAL_DATA"),
    reason="Set LSST64_HISTORICAL_DATA for real 20-year acceptance test",
)
def test_real_data_workflow(tmp_path):
    """Opt-in R-3/R-4 acceptance run; suitable for the issue's Slurm job."""
    path = Path(os.environ["LSST64_HISTORICAL_DATA"])
    historical = read_historical_clouds(path)
    transitions = compute_transition_matrix(historical)
    distribution = count_cloud_states(historical)
    database = tmp_path / "average-clouds.db"
    histogram = tmp_path / "average-clouds.png"
    transition_plot = tmp_path / "average-transitions.png"
    clouds = cloud_generation_workflow(
        path,
        database,
        20,
        cloud_histogram=histogram,
        transition_histograms=transition_plot,
    )
    assert_cloud_schema(clouds)
    assert_monthly_requirements(clouds, transitions, distribution)
    assert int(clouds.c_date.iloc[-1]) - int(clouds.c_date.iloc[0]) >= 20 * 31557600
    assert all(output.stat().st_size > 0 for output in [database, histogram, transition_plot])
    reader = CloudData(Time("2025-01-01", scale="tai"), cloud_db=str(database))
    assert reader.time_range >= 20 * 31557600
