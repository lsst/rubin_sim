"""Tools for use of project-generated opsim simulations and analysis.
"""

__all__ = (
    "get_runs",
    "get_family_runs",
    "download_runs",
    "get_metric_sets",
    "get_metric_summaries",
    "get_family_descriptions",
    "describe_families",
    "create_metric_set_df",
    "write_metric_sets",
)


import copy
import os
import sys
import urllib
import warnings

import numpy as np
import pandas as pd

try:
    import IPython
except ModuleNotFoundError:
    pass

from .summary_plots import plot_run_metric

FAMILY_SOURCE = os.environ.get(
    "RUBIN_SIM_FAMILY_SOURCE",
    "https://raw.githubusercontent.com/lsst-pst/survey_strategy/main/fbs_2.0/runs_v2.2.json",
)

METRIC_SET_SOURCE = os.environ.get(
    "RUBIN_SIM_METRIC_SET_SOURCE",
    "https://raw.githubusercontent.com/lsst-pst/survey_strategy/main/fbs_2.0/metric_sets.json",
)

SUMMARY_SOURCE = os.environ.get(
    "RUBIN_SIM_SUMMARY_SOURCE",
    "https://raw.githubusercontent.com/lsst-pst/survey_strategy/main/fbs_2.0/summary_2023_01_01.csv",
)

if os.uname().nodename.endswith(".datalab.noao.edu"):
    DEFAULT_OPSIM_DB_DIR = "/sims_maf"
else:
    DEFAULT_OPSIM_DB_DIR = os.getcwd()
OPSIM_DB_DIR = os.environ.get("OPSIM_DB_DIR", DEFAULT_OPSIM_DB_DIR)

BY_RUN_COLS = ["run", "brief", "filepath", "url"]


def get_family_runs(run_source=None):
    """Load a data frame that supplies run names for each run family

    Parameters
    ----------
    run_source : `None` or `str`
        File name or URL for the json file from which to load the metadata.
        If it is set to `None`, the data is loaded from the URL specified
        by the `archive.RUNS_SOURCE` constant.

    Returns
    -------
    families : `pandas.DataFrame`
        ``families``
            The index is the run family. (`str`)
        ``run``
            the project-standard name for the run (`str`)
        ``OpsimGroup``
            The name for the group to which the runs belong (`str`)
        ``OpsimComment``
            Short description of the run (`str`)
        ``OpsimVersion``
            Opsim version name (`str`)
        ``OpsimDate``
            Date for the version of opsim (TODO: ?)
        ``brief``
            A list of descriptions for the run. Runs may have a
            different description for each family it belongs to, so it
            a list of the same length as the families column (`list`
            [`str`])
        ``url``
            The URL from which the opsim output database for this run can be
            downloaded.

    Notes
    -----
    Because runs can be members of multiple families, more than one row may
    provide metadata on the same run.

    The same content (in a different form) can be obtained using
    ``get_runs``. ``get_runs`` is more convenient when indexing by
    run; ``get_family_runs`` when indexing by family.

    """

    run_source = FAMILY_SOURCE if run_source is None else run_source
    if isinstance(run_source, pd.DataFrame):
        runs = run_source
    else:
        families = pd.read_json(run_source, orient="index")
        families.index.name = "family"
        runs = families.explode(BY_RUN_COLS)

    return runs


def get_runs(run_source=None):
    """Load metadata on opsim runs into a `pandas.DataFrame`.

    Parameters
    ----------
    run_source : `None` or `str`
        File name or URL for the json file from which to load the metadata.
        If it is set to `None`, the data is loaded from the URL specified
        by the `archive.RUNS_SOURCE` constant.

    Returns
    -------
    runs : `pandas.DataFrame`
        ``run``
            The index of the DataFrame is the project-standard name for the run
            (`str`)
        ``family``
            A list of run families to which this run belongs (`list` [`str`])
        ``version``
            The simulation version
        ``brief``
            A list of descriptions for the run. Runs may have a
            different description for each family it belongs to, so it
            a list of the same length as the families column (`list`
            [`str`])
        ``filepath``
            The file path, relative to a base opsim output directory.
        ``url``
            The URL from which the opsim output database for this run can be
            downloaded.

    Notes
    -----
    The same content (in a different form) can be obtained using
    ``get_family_runs``. ``get_runs`` is more convenient when indexing by
    run; ``get_family_runs`` when indexing by family.

    """

    family_runs = get_family_runs(run_source)

    runs = (
        family_runs.reset_index()
        .groupby(BY_RUN_COLS)
        .agg({c: list for c in family_runs.reset_index().columns if c not in BY_RUN_COLS})
        .reset_index()
        .set_index("run")
        .loc[:, ["family", "version", "brief", "filepath", "url"]]
    )

    return runs


def download_runs(runs, dest_dir=None, runs_source=None, clobber=False):
    """Download opsim visit databases for specified runs to a local directory.

    Parameters
    ----------
    runs : `pandas.DataFrame` or iterable [`str`]
        If a `pandas.DataFrame` is provided, the `OpsimRun` column will be used
        to get run names, and data will be read from the url specified in the
        `url` column.
        If a collection of `str` is provided, these will be interpreted as
        run names supplied by data originating in the run metadata provided
        by the ``runs_source`` parameter.
    dest_dir : `str`
        The local directory into which to write downloaded visit databases.
    runs_source : `str`
        File name or URL for the json file from which to load the metadata.
        If it is set to `None`, the data is loaded from the URL specified
        by the `archive.RUNS_SOURCE` constant. This parameter is ignored
        if the ``runs`` parameter is set to a `pandas.DataFrame`.
    clobber : `bool`
        If ``False``, runs that would clobber an existing file will be skipped.
        If ``True``, existing files will be overwritten.


    Returns
    -------
    runs : `pandas.DataFrame`
        Metadata on runs downloaded (in the same structure as the return of
        ``archive.get_runs``).
    """

    if isinstance(runs, str):
        runs = [runs]

    if not isinstance(runs, pd.DataFrame):
        all_runs = get_runs(runs_source)
        runs = all_runs.loc[runs, :]

    if dest_dir is None:
        dest_dir = OPSIM_DB_DIR

    if not os.path.exists(dest_dir):
        raise FileNotFoundError(dest_dir)

    dest_fnames = pd.Series(name="fname", index=pd.Index([], name="OpsimRun"), dtype=object)

    for run_name, run in runs.iterrows():
        dest_fnames[run_name] = os.path.join(dest_dir, run.filepath)

        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(dest_fnames[run_name]), exist_ok=True)

        if clobber or not os.path.exists(dest_fnames[run_name]):
            urllib.request.urlretrieve(run.url, dest_fnames[run_name])
        else:
            warnings.warn(f"{dest_fnames[run_name]} already exists; not downloading")

    return dest_fnames


def get_metric_sets(metric_set_source=METRIC_SET_SOURCE):
    """Get metadata on named sets of related metrics.

    Parameters
    ----------
    metric_set_source : `str`
        File name or URL for the json file from which to load the data.
        If it is set to `None`, the data is loaded from the URL specified
        by the `archive.METRIC_SET_SOURCE` constant.

    Returns
    -------
    metric_sets : `pandas.DataFrame`
        ``metric_set``
            The 1st level of the index is the name of a set of metrics (`str`).
        ``metric``
            The 2nd level of the index is the full name of the metric (`str`).
        ``metric``
            The full name of the metric (`str`).
        ``short_name``
            An abbreviated name for the metric (`str`)..
        ``style``
            The ``matplotlib`` linestyle suggested for plots of the
            metric (`str`).
        ``invert``
            When normalizing, invert the metric value first? (`bool`)
        ``mag``
            Is the value an (astronomical) magnitude? (`bool`)
    """
    metric_set_source = METRIC_SET_SOURCE if metric_set_source is None else metric_set_source
    if isinstance(metric_set_source, pd.DataFrame):
        metric_sets = metric_set_source
    else:
        metric_sets = (
            pd.read_json(metric_set_source)
            .set_index("metric set")
            .set_index("metric", append=True, drop=False)
        )
    return metric_sets


def write_metric_sets(metric_set_file, metric_sets):
    """Write an updated metric_set dataframe to disk

    Parameters
    ----------
    metric_set_file : `str`
        Output file name.
    metric_sets : `pandas.DataFrame`
        Metric_set dataframe, as defined in get_metric_sets
    """
    tmp = metric_sets.reset_index("metric set")
    tmp.to_json(metric_set_file, orient="records", indent=2)


def get_metric_summaries(
    run_families=tuple(),
    metric_sets=tuple(),
    runs=tuple(),
    metrics=tuple(),
    summary_source=None,
    runs_source=None,
    metric_set_source=None,
    run_order="family",
    metric_order="summary",
):
    """Get summary metric values for a set of runs and metrics.

    Parameters
    ----------
    run_families : iterable [`str`]
        Families of runs to include in the summary.
    metric_sets : iterable [`str`]
        Sets of metrics to include in the summary.
    runs : iterable [`str`]
        Runs to include in the summary (in addition to any that are part
        of families included in ``run_families``).
    metrics : iterable [`str`]
        Metrics to include in the summary (in addition to any that are
        part of sets included in ``metric_sets``).
    summary_source : `str` or `pandas.DataFrame`
        File name or URL for the file from which to load the data.
        If it is set to `None`, the data is loaded from the URL specified
        by the `archive.METRIC_SET_SOURCE` constant.
        If the supplied value is a `pandas.DataFrame`, it the table
        returned will be a subset of this supplied table.
    run_source : `pandas.DataFrame` or `str`
        Either a `pandas.DataFrame` of runs metadata (as returned by
        `archive.get_runs`), or a file name or URL for the json file
        from which to load the run metadata.
        If it is set to `None`, the data is loaded from the URL specified
        by the `archive.RUNS_SOURCE` constant.
    metric_set_source : `pandas.DataFrame` or `str`
        Either a `pandas.DataFrame` of metric set specifications
        (as returned by `archive.get_metric_sets`) or a
        file name or URL for the json file from which to load the data.
        If it is set to `None`, the data is loaded from the URL specified
        by the `archive.SUMMARY_SOURCE` constant.
    run_order : `str`
        Sort runs according to family definition ("family") or summary file
        ("summary") order.
    metric_order : `str`
        Sort metrics according to set definition ("set") or summary file
        ("summary") order.

    Returns
    -------
    summaries : `pandas.DataFrame`
        Metric summary values are returned in a `pandas.DataFrame`, with
        each column providing the metrics for one run, and each row the
        values for one metric. The metric names constitute the index, and
        the column names are the canonical run names.

    Note
    ----
    The entire summary statistic values for all of the runs and metrics can
    be downloaded from the default sources first, by simply calling
    .. code-block:: python
        summary = get_metric_summaries()

    Then, you can use `get_metric_summaries` to get a subset without
    redownloading the whole set by passing `summary_source=summary`. If you are
    selecting multiple subsets of the summary, this avoids needing
    to download the summary data multiple times.
    """
    summary_source = SUMMARY_SOURCE if summary_source is None else summary_source

    runs = list(runs)
    metrics = list(metrics)

    if isinstance(run_families, str):
        run_families = [run_families]

    if isinstance(metric_sets, str):
        metric_sets = [metric_sets]

    if isinstance(summary_source, pd.DataFrame):
        all_summaries = summary_source
    else:
        try:
            all_summaries = pd.read_csv(summary_source, index_col=0, low_memory=False)
        except UnicodeDecodeError:
            # then this was probably the h5 file instead
            all_summaries = pd.read_hdf(summary_source)
        all_summaries.index.name = "OpsimRun"

    if len(run_families) > 0:
        families = get_family_runs(runs_source)
        for run_family in run_families:
            runs.extend(pd.Series(families.loc[run_family, "run"]).tolist())

    if len(metric_sets) > 0:
        metric_set_df = get_metric_sets(metric_set_source)
        for metric_set in metric_sets:
            metrics.extend(list(metric_set_df.loc[metric_set, "metric"]))

    if len(runs) == 0:
        runs = slice(None)
    else:
        if run_order == "summary":
            runs = [r for r in all_summaries.index if r in runs]

    if len(metrics) == 0:
        metrics = slice(None)
    else:
        requested_metrics = copy.copy(metrics)
        for metric in requested_metrics:
            if metric not in all_summaries.columns:
                warnings.warn(f'Metric "{metric}" not in summary, skipping')
                metrics.remove(metric)

        if metric_order == "summary":
            metrics = [m for m in all_summaries.columns if m in metrics]

    summaries = all_summaries.loc[runs, metrics]
    summaries.columns.name = "metric"
    summaries.index.name = "run"
    return summaries


def get_family_descriptions(family_source=None):
    """Get description of families or funs.

    Parameters
    ----------
    family_source : `str
        File name or URL for the json file from which to load the
        family descriptinos.  If it is set to `None`, the data is
        loaded from the URL specified by the
        `archive.FAMILY_SOURCE` constant.


    Returns
    -------
    families : `pandas.DataFrame`
        Family descriptions, with comments.

    """
    family_source = FAMILY_SOURCE if family_source is None else family_source
    if isinstance(family_source, pd.DataFrame):
        families = family_source
    else:
        families = pd.read_json(family_source, orient="index")
        families.index.name = "family"
        by_family_cols = [c for c in families.columns if c not in BY_RUN_COLS]
        families = families.loc[:, by_family_cols + BY_RUN_COLS]
    return families


def describe_families(
    families,
    summary=None,
    table_metric_set=None,
    plot_metric_set=None,
    baseline_run=None,
    round_table=2,
):
    """Display (in a jupyter on IPython notebook) family descirptions

    Parameters
    ----------
    families : `pandas.DataFrame`
        Data family descriptions as returned by get_family_descriptions.
    summary : `pandas.DataFrame`
        Summary metrics for each run, as returned by get_metric_summaries.
    table_metric_set : `pandas.DataFrame`
        Metadata on metrics to be included in the table, with columns and
        index as returned by get_metric_sets. None if no metrics should be
        included in the table.
    plot_metric_set : `pandas.DataFrame`
        Metadata on metrics to be included in the plot, with columns and
        index as returned by get_metric_sets. None if no plot should be
        made.
    baseline_run : `str`
        The name of the run to use to normalize metrics in the plot.
        None if normalization should be skipped.
    round_table : `int`, opt
        Decimal places to which to round the table_metrics. Default 2.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        The plot figure.
    ax : `matplotilb.axes.Axes`
        The plot axes.
    """

    family_runs = families.explode(["run", "brief", "filepath"]).loc[:, ["run", "brief", "filepath"]]

    # If there is just one run in the family, we might
    # get a pd.Series back rather than a pd.DataFrame.
    # Make sure we have a DataFrame
    if isinstance(family_runs, pd.Series):
        family_runs = pd.DataFrame([family_runs])

    for family_name, family in families.iterrows():
        # Use awkward appending of each line to string rather
        # than a tripple quote to keep flake8 from complaining
        # about blanks at the end of lines, which are
        # meaningful in markdown (and desired here).
        description = "---\n"
        description += f"{family.description}  \n"
        description += f"**version**: {family.version}  \n"
        description += "**runs**:  \n"

        these_runs = family_runs.loc[[family_name], :]
        if summary is not None:
            if table_metric_set is not None:
                table_metric_summary = summary.loc[these_runs["run"], table_metric_set["metric"]]
                table_metric_summary.rename(table_metric_set["short_name"], axis=1, inplace=True)
                if round_table is not None:
                    table_metric_summary = table_metric_summary.round(round_table)
            else:
                table_metric_summary = summary.loc[these_runs["run"]]

            these_runs = these_runs.join(table_metric_summary, on="run", how="left")

            num_columns = len(these_runs.columns)
            if num_columns > 5 and "filepath" in these_runs.columns:
                these_runs = these_runs.drop(columns=["filepath"])

        with pd.option_context("display.max_colwidth", 0):
            if "IPython" in sys.modules:
                IPython.display.display_markdown(description, raw=True)
                IPython.display.display(
                    IPython.display.HTML(these_runs.set_index("run").to_html().replace("\\n", "<br>"))
                )
            else:
                print(description)
                print(these_runs.set_index("run"))

    if plot_metric_set is not None:
        these_runs = family_runs["run"].values
        if baseline_run is not None and baseline_run not in these_runs:
            these_runs = np.concatenate([[baseline_run], these_runs])
        these_metrics = [m for m in plot_metric_set["metric"] if m in summary.columns]
        fig, ax = plot_run_metric(  # pylint: disable=invalid-name
            summary.loc[these_runs, these_metrics],
            metric_set=plot_metric_set,
            metric_label_map=plot_metric_set["short_name"],
            baseline_run=baseline_run,
            vertical_quantity="value",
            horizontal_quantity="run",
        )
    else:
        fig, ax = None, None  # pylint: disable=invalid-name

    return fig, ax


def create_metric_set_df(
    metric_set,
    metrics,
    short_name=None,
    style="-",
    invert=False,
    mag=False,
):
    """Create a DataFrame that defines a metric set.

    Parameters
    ----------
    metric_set : `str`
        The name of a metric set.
    metrics : `list` [`str`]
        A list of metric names in the set.
    short_name : `list` [`str`], optional
        A list of shorter metric names, by default None
    style : `list` [`str`], optional
        The matplotlib line style symbol for lines representing the metric,
        by default "-"
    invert : `list` [`bool`], optional
        Are smaller values of the metric better, such as for errors?, by default False
    mag : `list` [`bool`], optional
        Is the metric an astronomical magnitude?, by default False

    Returns
    -------
    metric_set : `pandas.DataFrame`
        A table of metrics and normalization and plotting flags defining the
        content of a metric set.
    """
    if short_name is None:
        short_name = metrics

    metric_set = (
        pd.DataFrame(
            {
                "metric set": metric_set,
                "metric": metrics,
                "short_name": short_name,
                "style": style,
                "invert": invert,
                "mag": mag,
            }
        )
        .set_index("metric set")
        .set_index("metric", append=True, drop=False)
    )

    return metric_set
