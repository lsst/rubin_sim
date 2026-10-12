"""LSST-64 section 6 design contracts, independent of test_clouds fixtures.

Algorithm requirements (including candidate selection and bounded retries) are
already exercised in test_clouds.py; these tests target the data dictionary
and public interfaces instead of repeating those algorithm tests.
"""

import inspect
from datetime import date
from typing import TypeAlias, get_type_hints

import matplotlib
import numpy as np
import pandas as pd
import pytest
from astropy.utils import iers
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

import rubin_sim.clouds as clouds_api
import rubin_sim.clouds.data as cloud_data

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def offline_figures():
    with iers.conf.set_temp("auto_download", False):
        yield
    plt.close("all")


@pytest.fixture
def clouds():
    """One varying night per month, built without production schema helpers."""
    index = pd.MultiIndex.from_arrays(
        [
            np.full(48, 1975, dtype=np.uint16),
            np.repeat(np.arange(1, 13, dtype=np.uint8), 4),
            np.ones(48, dtype=np.uint8),
            np.tile(np.arange(1, 5, dtype=np.uint8), 12),
        ],
        names=["year", "month", "sday", "quarter"],
    )
    states = np.tile(np.array([0, 2, 8, 0], dtype=np.uint8), 12)
    return pd.DataFrame(
        {
            "eighths": states,
            "cloud": np.where(states == 9, np.nan, states / 8),
            "c_date": np.arange(1, 49, dtype=np.uintp),
        },
        index=index,
    )


@pytest.fixture
def statistics():
    """All-clear evidence provides feasible, inexpensive generation input."""
    months = pd.Index(np.arange(1, 13, dtype=np.uint8), name="month")
    states = np.arange(9, dtype=np.uint8)
    matrix = pd.DataFrame(
        np.zeros((108, 9), dtype=np.uintp),
        index=pd.MultiIndex.from_product([months, states], names=["month", "origin"]),
        columns=pd.Index(states, name="destination"),
    )
    distribution = pd.DataFrame(np.zeros((12, 9), dtype=np.uintp), index=months, columns=states)
    matrix.loc[(slice(None), 0), 0] = 100
    distribution[0] = np.uintp(100)
    return matrix, distribution


def assert_clouds_schema(frame):
    assert isinstance(frame, pd.DataFrame)
    assert isinstance(frame.index, pd.MultiIndex)
    assert frame.index.names == ["year", "month", "sday", "quarter"]
    assert frame.index.is_unique and frame.index.is_monotonic_increasing
    for name, dtype in zip(frame.index.names, [np.uint16, np.uint8, np.uint8, np.uint8]):
        assert frame.index.get_level_values(name).dtype == np.dtype(dtype)
    for year, month, day, quarter in frame.index:
        assert date(int(year), int(month), int(day)) >= date(1975, 1, 1)
        assert 1 <= quarter <= 4
    assert list(frame.columns) == ["eighths", "cloud", "c_date"]
    assert list(frame.dtypes) == [np.dtype(np.uint8), np.dtype(np.float64), np.dtype(np.uintp)]
    assert frame.eighths.between(0, 9).all()
    np.testing.assert_array_equal(frame.cloud, np.where(frame.eighths == 9, np.nan, frame.eighths / 8))
    seconds = frame.c_date.to_numpy()
    assert np.all(seconds[1:] > seconds[:-1])


def assert_matrix_schema(frame, dtype):
    assert isinstance(frame, pd.DataFrame)
    assert isinstance(frame.index, pd.MultiIndex)
    assert frame.index.names == ["month", "origin"]
    assert list(frame.index) == [(month, origin) for month in range(1, 13) for origin in range(9)]
    assert frame.index.is_unique and frame.index.is_monotonic_increasing
    assert all(frame.index.get_level_values(name).dtype == np.dtype(np.uint8) for name in frame.index.names)
    assert frame.columns.name == "destination"
    assert list(frame.columns) == list(range(9))
    assert pd.api.types.is_integer_dtype(frame.columns.dtype)
    assert all(column_dtype == np.dtype(dtype) for column_dtype in frame.dtypes)
    assert np.isfinite(frame.to_numpy()).all()
    assert (frame.to_numpy() >= 0).all()


def assert_distribution_schema(frame):
    assert isinstance(frame, pd.DataFrame)
    assert frame.index.name == "month"
    assert pd.api.types.is_integer_dtype(frame.index.dtype)
    assert frame.index.is_unique and set(frame.index) <= set(range(1, 13))
    assert list(frame.columns) == list(range(9))
    assert pd.api.types.is_integer_dtype(frame.columns.dtype)
    assert all(pd.api.types.is_integer_dtype(dtype) for dtype in frame.dtypes)
    assert not frame.isna().to_numpy().any()
    assert (frame.to_numpy() >= 0).all()


@pytest.mark.parametrize("name", ["Clouds", "TransitionMatrix", "StochasticMatrix", "CloudDistribution"])
def test_data_dictionary_type_aliases(name):
    assert getattr(clouds_api, name) is pd.DataFrame
    assert get_type_hints(cloud_data)[name] is TypeAlias


def test_produced_data_dictionary_schemas(tmp_path, statistics):
    records = pd.DataFrame(
        {"sday": 1, "eday": 2, "month": range(1, 13), "year": 1975, "q1": 0, "q2": 2, "q3": 8, "q4": -1}
    )
    path = tmp_path / "historical.txt"
    records.to_csv(path, sep="\t", index=False)
    historical = clouds_api.read_historical_clouds(path)
    generated = clouds_api.generate_average_clouds(*statistics, 1, max_retries=1)
    for frame in (historical, generated):
        assert_clouds_schema(frame)
        assert_distribution_schema(clouds_api.count_cloud_states(frame))
        assert_matrix_schema(clouds_api.compute_transition_matrix(frame), np.uintp)
    assert historical.eighths.eq(9).any()
    assert generated.eighths.le(8).all()
    stochastic = clouds_api.compute_stochastic_matrix(clouds_api.compute_transition_matrix(historical))
    assert_matrix_schema(stochastic, np.float64)
    assert (stochastic.to_numpy() <= 1).all()
    np.testing.assert_allclose(stochastic.sum(axis=1), 1)


@pytest.mark.parametrize("case", ["header", "month", "missing", "date", "early_year", "negative", "state"])
def test_historical_input_validation(tmp_path, case):
    records = pd.DataFrame(
        {"sday": 1, "eday": 2, "month": range(1, 13), "year": 1975, "q1": 0, "q2": 0, "q3": 0, "q4": 0}
    )
    if case == "header":
        records = records.drop(columns="q4")
    elif case == "month":
        records = records.loc[records.month != 12]
    elif case == "missing":
        records.loc[records.month == 12, ["q1", "q2", "q3", "q4"]] = 9
    else:
        column, value = {
            "date": ("sday", 32),
            "early_year": ("year", 1974),
            "negative": ("q1", -2),
            "state": ("q1", 10),
        }[case]
        records.loc[0, column] = value
    path = tmp_path / "invalid.txt"
    records.to_csv(path, sep=" ", index=False)
    with pytest.raises(ValueError):
        clouds_api.read_historical_clouds(path)


def test_historical_input_requires_a_path():
    with pytest.raises(ValueError):
        clouds_api.read_historical_clouds([])


def test_valid_clouds_are_accepted_by_consumers(clouds, tmp_path):
    assert_distribution_schema(clouds_api.count_cloud_states(clouds))
    assert_matrix_schema(clouds_api.compute_transition_matrix(clouds), np.uintp)
    path = tmp_path / "valid.db"
    clouds_api.save_clouds(path, clouds)
    assert path.exists()
    pytest.importorskip("statsmodels.tsa.stattools")
    assert isinstance(clouds_api.observability_acf(0.5, clouds, clouds, 1, nlags=2), Figure)


def test_missing_clouds_are_valid_except_for_simulation_output(clouds, tmp_path):
    clouds.iloc[1, 0] = 9
    clouds.iloc[1, 1] = np.nan
    assert_distribution_schema(clouds_api.count_cloud_states(clouds))
    assert_matrix_schema(clouds_api.compute_transition_matrix(clouds), np.uintp)
    path = tmp_path / "missing.db"
    with pytest.raises(ValueError):
        clouds_api.save_clouds(path, clouds)
    assert not path.exists()


def test_transition_diagnostics_accept_both_valid_matrices(statistics):
    matrix, _ = statistics
    assert isinstance(clouds_api.plot_transition_histograms(matrix, matrix, cmap="magma"), Figure)


def test_stochastic_matrix_requires_transition_evidence(statistics):
    matrix, _ = statistics
    matrix.loc[12] = 0
    with pytest.raises(ValueError):
        clouds_api.compute_stochastic_matrix(matrix)


# Each corruption changes one contract; validators are exercised only via
# public consumers, never by importing private validation functions.
COMMON_INVALID = (
    "not_dataframe",
    "column_order",
    "missing_column",
    "extra_column",
    "duplicate_column",
    "column_label",
    "missing_label",
    "row_duplicate",
    "index_names",
    "flat_index",
)
CLOUD_INVALID = COMMON_INVALID + (
    "row_order",
    "index_order",
    "year_dtype",
    "month_dtype",
    "sday_dtype",
    "quarter_dtype",
    "eighths_dtype",
    "cloud_dtype",
    "c_date_dtype",
    "year_range",
    "month_zero",
    "month_range",
    "day_zero",
    "day_range",
    "invalid_date",
    "quarter_zero",
    "quarter_range",
    "state_range",
    "fraction",
    "unexpected_nan",
    "infinite",
    "missing_fraction",
    "time_duplicate",
    "time_reverse",
)
MATRIX_INVALID = COMMON_INVALID + (
    "row_order",
    "index_order",
    "month_dtype",
    "origin_dtype",
    "missing_row",
    "missing_month",
    "destination_name",
    "float_labels",
    "origin_range",
    "month_zero",
    "month_range",
    "float_counts",
    "signed_counts",
    "bool_counts",
    "negative_counts",
    "nan_counts",
    "nullable_counts",
)
DISTRIBUTION_INVALID = COMMON_INVALID + (
    "month_dtype",
    "month_zero",
    "month_range",
    "float_labels",
    "float_counts",
    "bool_counts",
    "negative_counts",
    "nan_counts",
    "nullable_counts",
    "nullable_month",
)


def corrupt(frame, case):
    frame = frame.copy(deep=True)
    if case == "not_dataframe":
        return frame.to_numpy()
    if case == "column_order":
        return frame.iloc[:, ::-1]
    if case == "missing_column":
        return frame.iloc[:, :-1]
    if case == "extra_column":
        frame["extra"] = 0
    elif case == "duplicate_column":
        return pd.concat([frame, frame.iloc[:, [0]]], axis=1)
    elif case == "column_label":
        frame.columns = pd.Index(["wrong", *frame.columns[1:]], name=frame.columns.name)
    elif case == "missing_label":
        frame.columns = pd.Index([*frame.columns[:-1], pd.NA], name=frame.columns.name)
    elif case == "row_duplicate":
        return pd.concat([frame.iloc[:1], frame])
    elif case == "row_order":
        return frame.iloc[::-1]
    elif case == "missing_row":
        return frame.iloc[1:]
    elif case == "missing_month":
        return frame.drop(12, level="month")
    elif case == "index_names":
        frame.index = frame.index.set_names(["wrong"] * frame.index.nlevels)
    elif case == "flat_index":
        frame.index = pd.RangeIndex(len(frame))
    elif case == "index_order":
        return frame.reorder_levels(list(reversed(range(frame.index.nlevels))))
    elif case == "destination_name":
        frame.columns.name = "wrong"
    elif case == "float_labels":
        frame.columns = frame.columns.astype(float)
    elif case in {"float_counts", "signed_counts", "bool_counts", "negative_counts", "nan_counts"}:
        dtype = (
            bool
            if case == "bool_counts"
            else np.int64 if case in {"signed_counts", "negative_counts"} else float
        )
        frame = frame.astype(dtype)
        if case == "negative_counts":
            frame.iloc[0, 0] = -1
        elif case == "nan_counts":
            frame.iloc[0, 0] = np.nan
    elif case == "nullable_counts":
        frame = frame.astype("Int64")
        frame.iloc[0, 0] = pd.NA
    elif case == "nullable_month":
        frame.index = pd.Index([pd.NA, *frame.index[1:]], dtype="Int64", name="month")
    elif case in {"eighths_dtype", "cloud_dtype", "c_date_dtype"}:
        column = case.removesuffix("_dtype")
        frame[column] = frame[column].astype(np.float32 if column == "cloud" else np.int64)
    elif case in {"state_range", "fraction", "unexpected_nan", "infinite", "missing_fraction"}:
        if case in {"state_range", "missing_fraction"}:
            frame.iloc[0, 0] = 10 if case == "state_range" else 9
        else:
            frame.iloc[0, 1] = {"fraction": 0.3, "unexpected_nan": np.nan, "infinite": np.inf}[case]
    elif case in {"time_duplicate", "time_reverse"}:
        frame.iloc[1, 2] = frame.iloc[0, 2] - (case == "time_reverse")
    else:
        arrays = [frame.index.get_level_values(i).to_numpy(copy=True) for i in range(frame.index.nlevels)]
        names = list(frame.index.names)
        if case.endswith("_dtype"):
            level = names.index(case.removesuffix("_dtype"))
            arrays[level] = arrays[level].astype(float if frame.index.nlevels == 1 else np.int64)
        else:
            name, value = {
                "year_range": ("year", 1974),
                "month_zero": ("month", 0),
                "month_range": ("month", 13),
                "day_zero": ("sday", 0),
                "day_range": ("sday", 32),
                "invalid_date": ("sday", 30),
                "quarter_zero": ("quarter", 0),
                "quarter_range": ("quarter", 5),
                "origin_range": ("origin", 9),
            }[case]
            if case == "invalid_date":
                arrays[names.index("month")][0] = 2
            arrays[names.index(name)][0] = value
        frame.index = (
            pd.Index(arrays[0], name=names[0])
            if len(arrays) == 1
            else pd.MultiIndex.from_arrays(arrays, names=names)
        )
    return frame


@pytest.fixture
def no_sampling(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Invalid inputs reached the sampler")

    monkeypatch.setattr("rubin_sim.clouds.generate._generate_month", fail)


@pytest.mark.parametrize("case", CLOUD_INVALID)
@pytest.mark.parametrize("consumer", ["counts", "transitions", "save", "acf_historical", "acf_simulated"])
def test_clouds_consumers_reject_invalid_schema(clouds, tmp_path, case, consumer):
    broken = corrupt(clouds, case)
    path = tmp_path / "invalid.db"
    figures = plt.get_fignums()
    if consumer.startswith("acf"):
        pytest.importorskip("statsmodels.tsa.stattools")
    with pytest.raises(ValueError):
        if consumer == "counts":
            clouds_api.count_cloud_states(broken)
        elif consumer == "transitions":
            clouds_api.compute_transition_matrix(broken)
        elif consumer == "save":
            clouds_api.save_clouds(path, broken)
        else:
            historical, simulated = (broken, clouds) if consumer == "acf_historical" else (clouds, broken)
            clouds_api.observability_acf(0.5, historical, simulated, 1, nlags=2)
    assert not path.exists()
    assert plt.get_fignums() == figures


@pytest.mark.parametrize("case", MATRIX_INVALID)
@pytest.mark.parametrize("consumer", ["stochastic", "generate", "plot_historical", "plot_simulated"])
def test_transition_consumers_reject_invalid_schema(statistics, no_sampling, case, consumer):
    matrix, distribution = statistics
    broken = corrupt(matrix, case)
    figures = plt.get_fignums()
    with pytest.raises(ValueError):
        if consumer == "stochastic":
            clouds_api.compute_stochastic_matrix(broken)
        elif consumer == "generate":
            clouds_api.generate_average_clouds(broken, distribution, 1)
        elif consumer == "plot_historical":
            clouds_api.plot_transition_histograms(broken, matrix)
        else:
            clouds_api.plot_transition_histograms(matrix, broken)
    assert plt.get_fignums() == figures


@pytest.mark.parametrize("case", DISTRIBUTION_INVALID)
@pytest.mark.parametrize("consumer", ["generate", "plot_historical", "plot_simulated", "plot_dictionary"])
def test_distribution_consumers_reject_invalid_schema(statistics, no_sampling, case, consumer):
    matrix, distribution = statistics
    broken = corrupt(distribution, case)
    figures = plt.get_fignums()
    with pytest.raises(ValueError):
        if consumer == "generate":
            clouds_api.generate_average_clouds(matrix, broken, 1)
        elif consumer == "plot_historical":
            clouds_api.plot_cloud_histogram(broken, distribution)
        else:
            simulated = broken if consumer == "plot_simulated" else {1975: distribution, 1976: broken}
            clouds_api.plot_cloud_histogram(distribution, simulated)
    assert plt.get_fignums() == figures


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint16, np.uintp])
def test_distribution_does_not_require_unsigned_or_sorted_rows(statistics, dtype):
    # The dictionary specifies integer counts and months, but no exact dtype
    # or row ordering. Diagnostics can also consume only selected months.
    matrix, distribution = statistics
    distribution = distribution.astype(dtype)
    distribution.index = distribution.index.astype(dtype)
    distribution = distribution.iloc[::-1]
    assert_clouds_schema(clouds_api.generate_average_clouds(matrix, distribution, 1, max_retries=1))
    seasonal = distribution.loc[[12, 1]]
    assert isinstance(clouds_api.plot_cloud_histogram(seasonal, {1975: seasonal}), Figure)


@pytest.mark.parametrize("case", ["missing_month", "empty_counts", "empty_transitions"])
def test_generation_requires_evidence_for_every_month(statistics, no_sampling, case):
    matrix, distribution = statistics
    if case == "missing_month":
        distribution = distribution.drop(12)
    elif case == "empty_counts":
        distribution.loc[12] = 0
    else:
        matrix.loc[12] = 0
    with pytest.raises(ValueError):
        clouds_api.generate_average_clouds(matrix, distribution, 1)


@pytest.mark.parametrize("prior_state", [-1, 10, 0.5, True, "0", np.nan])
def test_transition_prior_state_validation(clouds, prior_state):
    with pytest.raises(ValueError):
        clouds_api.compute_transition_matrix(clouds, prior_state)


def test_transition_extra_keywords_preserve_counts(clouds):
    expected = clouds_api.compute_transition_matrix(clouds, prior_state=8)
    actual = clouds_api.compute_transition_matrix(clouds, prior_state=8, design_option=True)
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("parameter", ["num_years", "max_retries", "max_steps"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True, None])
def test_generation_limits_validation(statistics, no_sampling, parameter, value):
    with pytest.raises(ValueError):
        clouds_api.generate_average_clouds(*statistics, **{"num_years": 1, parameter: value})


@pytest.mark.parametrize(
    "options",
    [
        {"month": 0},
        {"month": 13},
        {"month": True},
        {"month": 1.5},
        {"year": 1974},
        {"year": 65536},
        {"year": True},
        {"year": 1975.5},
        {"cloud_threshold": np.nan},
        {"cloud_threshold": np.inf},
        {"cloud_threshold": True},
        {"cloud_threshold": "0.5"},
    ],
)
def test_acf_selection_validation(clouds, options):
    pytest.importorskip("statsmodels.tsa.stattools")
    with pytest.raises(ValueError):
        clouds_api.observability_acf(
            historical_clouds=clouds,
            simulated_clouds=clouds,
            **{"cloud_threshold": 0.5, "month": 1, **options},
        )


# Binding checks the documented calls without forbidding additional optional
# parameters. Objects stand in for frames: these tests do not execute calls.
FRAME = object()
CALLS = [
    ("make_cloud_figure", (FRAME,), {}),
    ("make_cloud_figure", (FRAME, FRAME), {}),
    ("make_cloud_figure", (), {"eighths": FRAME, "origin": None}),
    ("read_historical_clouds", ("historical.txt",), {}),
    ("read_historical_clouds", (["first.txt", "second.txt"],), {}),
    ("count_cloud_states", (FRAME,), {}),
    ("compute_transition_matrix", (FRAME, None), {}),
    ("compute_transition_matrix", (), {"clouds": FRAME, "prior_state": 9}),
    ("compute_transition_matrix", (FRAME, None), {"design_option": True}),
    ("compute_stochastic_matrix", (FRAME,), {}),
    ("generate_average_clouds", (FRAME, FRAME, 2), {}),
    ("save_clouds", ("clouds.db", FRAME), {}),
    ("plot_cloud_histogram", (FRAME, FRAME), {"linewidth": 2}),
    ("plot_cloud_histogram", (), {"historical_distribution": FRAME, "simulated_distribution": {1975: FRAME}}),
    ("plot_transition_histograms", (FRAME, FRAME), {"cmap": "magma"}),
    ("observability_acf", (0.5, FRAME, FRAME, 1), {"nlags": 6, "missing": "conservative"}),
    ("observability_acf", (0.5, FRAME, FRAME, 1, 1975), {"fft": False}),
    (
        "observability_acf",
        (),
        {
            "cloud_threshold": 0.5,
            "historical_clouds": FRAME,
            "simulated_clouds": FRAME,
            "month": 1,
            "year": None,
        },
    ),
    ("cloud_generation_workflow", (["historical.txt"], "clouds.db"), {}),
    (
        "cloud_generation_workflow",
        ("historical.txt", "clouds.db", 2),
        {
            "cloud_histogram": "clouds.png",
            "transition_histograms": "transitions.png",
            "cloud_figure": "sequence.png",
        },
    ),
]


@pytest.mark.parametrize(
    "name,args,kwargs", CALLS, ids=[f"{name}-{i}" for i, (name, _, _) in enumerate(CALLS)]
)
def test_documented_call_shapes(name, args, kwargs):
    inspect.signature(getattr(clouds_api, name)).bind(*args, **kwargs)


def test_acf_year_defaults_to_all_years():
    assert inspect.signature(clouds_api.observability_acf).parameters["year"].default is None


def test_cloud_figure_origin_defaults_to_none():
    assert inspect.signature(clouds_api.make_cloud_figure).parameters["origin"].default is None


@pytest.mark.parametrize(
    "case",
    [
        "frame",
        "empty",
        "names",
        "duplicate",
        "order",
        "dtype",
        "date",
        "quarter",
        "state",
        "text",
        "origin_index",
        "origin_text",
    ],
)
def test_cloud_figure_input_validation(clouds, case):
    eighths = clouds.eighths.copy()
    origin = None
    if case == "frame":
        eighths = clouds
    elif case == "empty":
        eighths = eighths.iloc[:0]
    elif case == "names":
        eighths.index = eighths.index.set_names(["year", "month", "day", "quarter"])
    elif case == "duplicate":
        eighths = pd.concat([eighths.iloc[:1], eighths])
    elif case == "order":
        eighths = eighths.iloc[::-1]
    elif case in ("dtype", "date", "quarter"):
        fields = eighths.index.to_frame(index=False)
        if case == "dtype":
            fields["year"] = fields.year.astype(np.int64)
        elif case == "date":
            fields.loc[0, "sday"] = 32
        else:
            fields.loc[0, "quarter"] = 0
        eighths.index = pd.MultiIndex.from_frame(fields)
    elif case == "state":
        eighths = eighths.astype(float)
        eighths.iloc[0] = 1.5
    elif case == "text":
        eighths = eighths.astype(str)
    elif case == "origin_index":
        origin = pd.Series("ctio", index=eighths.index[::-1])
    else:
        origin = pd.Series(1, index=eighths.index)
    with pytest.raises(ValueError):
        clouds_api.make_cloud_figure(eighths, origin)
