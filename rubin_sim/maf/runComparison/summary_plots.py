"""Summary metric plotting functions
"""

# imports
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import colorcet
import cycler

# constants

RUN_COL_NAMES = ["run", "OpsimRun", "run_name"]
METRIC_COL_NAMES = ["metric"]

# exception classes

# interface functions


def normalize_metric_summaries(
    baseline_run,
    summary,
    metric_sets=None,
):
    """Create a normalized `pandas.DataFrame` of metric summary values.

    Parameters
    ----------
    baseline_run : `str`
        The name of the run that defines a normalized value of 1.
    summary : `pandas.DataFrame`
        The summary metrics to normalize (as returned by `get_metric_summaries`)
    metric_sets : `pandas.DataFrame`
        Metric metadata as returned by `archive.get_metric_sets`

    Returns
    -------
    norm_summary : `pandas.DataFrame`
        Metric summary values are returned in a `pandas.DataFrame`, with each
        column providing the metrics for one run, and each row the values for
        one metric. The metric names constitute the columns, and the
        index names are the canonical run names.
        Values of 1 indicate metric values that match that of the baseline,
        differences with with indicate fractional improvement (if > 1) or
        degradation (if < 1).
    """

    # If the DataFrame is transposed but the axes are properly
    # labeled, just fix it.
    cols_are_runs = summary.columns.name in RUN_COL_NAMES
    rows_are_metrics = summary.index.name in METRIC_COL_NAMES
    if cols_are_runs or rows_are_metrics:
        summary = summary.T

    runs = summary.index.values

    # Use only those metrics present both in the
    # summary and metrics sets dataframe
    if metric_sets is None:
        summary = summary.copy()
        used_metrics = summary.columns.values
    else:
        used_metrics = [
            s for s in summary.columns.values if s in metric_sets.metric.values
        ]
        summary = summary.loc[:, used_metrics].copy()

    if summary.columns.name is None:
        summary.rename_axis(columns="metric", inplace=True)

    if summary.index.name in [None] + RUN_COL_NAMES:
        summary.rename_axis("run", inplace=True)

    # Get rid of duplicate metrics and runs
    summary = summary.T.groupby("metric").first().T.groupby("run").first()

    if metric_sets is None:
        norm_summary = 1 + (
            summary.loc[:, :].sub(summary.loc[baseline_run, :], axis="columns")
        ).div(summary.loc[baseline_run, :], axis="columns")
    else:
        metric_names = [n for n in metric_sets.index.names if not n == "metric"]
        metric_sets = (
            metric_sets.reset_index(metric_names)
            .groupby(level="metric")
            .first()
            .loc[used_metrics, :]
            .copy()
        )

        norm_summary = pd.DataFrame(
            columns=summary.columns,
            index=summary.index,
            dtype="float",
        )

        assert not np.any(np.logical_and(metric_sets["invert"], metric_sets["mag"]))

        # Direct metrics are those that are neither inverted, nor compared as magnitudes
        direct = ~np.logical_or(metric_sets["invert"], metric_sets["mag"])
        norm_summary.loc[:, direct] = summary.loc[:, direct]

        norm_summary.loc[:, metric_sets["invert"]] = (
            1.0 / summary.loc[:, metric_sets["invert"]]
        )

        norm_summary.loc[:, metric_sets["mag"]] = (
            1.0
            + summary.loc[
                :,
                metric_sets["mag"],
            ].subtract(summary.loc[baseline_run, metric_sets["mag"]], axis="columns")
        )

        # Look a the fractional difference compared with the baseline
        norm_summary.loc[:, :] = 1 + (
            norm_summary.loc[:, :].sub(
                norm_summary.loc[baseline_run, :], axis="columns"
            )
        ).div(norm_summary.loc[baseline_run, :], axis="columns")

    # Set the index name
    norm_summary.columns.name = "metric"
    norm_summary.index.name = "run"

    # Make sure we return the rows and metrics in the original order
    norm_summary = norm_summary.loc[runs, used_metrics]

    return norm_summary


def plot_run_metric(
    summary,
    baseline_run=None,
    vertical_quantity="run",
    horizontal_quantity="value",
    vwidth=1,
    run_label_map=None,
    metric_label_map=None,
    metric_set=None,
    ax=None,
    cmap=colorcet.glasbey_hv,
    linestyles=None,
    markers=["o"],
):
    """Plot normalized metric values as colored points on a cartesian plane.

    Parameters
    ----------
    summary : `pandas.DataFrame`
        Values to be plotted. Should only include runs and metrics that
        should actually appear on the plot.
    baseline_run : `str`
        Name of the run to use as the baseline for normalization (see
        (archive.normalize_metric_summaries).
    vertical_quantity : {'run', 'metric', 'value'}
        Should the run, metric name, or metric value be mapped onto the y axis?
    horizontal_quantity : {'run', 'metric', 'value'}
        Should the run, metric name, or metric value be mapped onto the x axis?
    vwidth : `float`
        The width of the plot, in normalized metrics summary units. (The limits
        of the x axis will be 1-vwidth/2 and 1+width/2).
    run_label_map : mapping
        A python `mapping` between canonical run names and run labels as they
        should appear on plot labels. Use of this option is discouraged, because
        it makes it harder to match plots to data.
    metric_label_map : mapping
        A python `mapping` between canonical metric names and metric labels
        as they should appear on plot labels. Use of this option is discouraged, because
        it makes it harder to match plots to metric calculation code..
    metric_set : `pandas.DataFrame`
        Metric metadata as returned by `archive.get_metric_sets`
    ax : `matplotlib.axes.Axes`
        The axes on which to plot the data.
    cmap : `matplotlib.colors.ColorMap`
        The color map to use for point colors.
    linestyles : `list`
        A list of matplotlib linestyles to use to connect the lines
    markers : `list`
        A list of matplotlib markers to use to represent the points

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        The plot figure.
    ax : `matplotilb.axes.Axes`
        The plot axes.

    """

    # If the metric sets we are passed has a multilevel index,
    # get rid of the levels we do not need.
    if metric_set is not None and metric_set.index.nlevels > 1:
        extra_levels = list(set(metric_set.index.names) - set(["metric"]))
        metric_set = metric_set.droplevel(extra_levels).groupby(level="metric").first()

    # Mark whether we have a default, or whether
    # one was specified
    has_linestyle_arg = linestyles is not None
    if not has_linestyle_arg:
        linestyles = [""]

    quantities = set(["run", "metric", "value"])
    assert horizontal_quantity in quantities
    assert vertical_quantity in quantities
    assigned_quantities = set([horizontal_quantity, vertical_quantity])
    color_quantity = (quantities - assigned_quantities).pop()

    # Normalize the summary values, if a baseline was specified
    norm_summary = (
        (
            summary.rename_axis(index="run", columns="metric").copy()
            if baseline_run is None
            else normalize_metric_summaries(
                baseline_run, summary, metric_sets=metric_set
            )
        )
        .stack()
        .rename("value")
        .reset_index()
        .rename(columns={"OpsimRun": "run"})
    )
    norm_summary = norm_summary.loc[np.isfinite(norm_summary.value), :]

    if run_label_map is not None:
        norm_summary["run"] = norm_summary["run"].map(run_label_map)

    if metric_label_map is not None:
        norm_summary["metric"] = norm_summary["metric"].map(metric_label_map)
        if metric_set is not None:
            this_metric_set = (
                metric_set.drop(columns=["metric"])
                .assign(metric=metric_set["metric"].map(metric_label_map))
                .set_index("metric", drop=False)
            )
        else:
            this_metric_set = None
    else:
        this_metric_set = metric_set

    plot_df = pd.DataFrame(
        {
            "x": norm_summary[horizontal_quantity],
            "y": norm_summary[vertical_quantity],
        }
    )

    if color_quantity == "value":
        warnings.warn("plot_run_metric_mesh is probably a better choice")

        # turn the continuous metric value into a categorical one
        if baseline_run is None:
            # Not normalized, so we have no idea how to scale values.
            # Just let pandas create 7 bins.
            bins = 8
        else:
            bins = (-np.inf, 0, 0.8, 0.95, 1.05, 1.2, 2, np.inf)

        plot_df["color"] = pd.cut(norm_summary[color_quantity], bins)
    else:
        plot_df["color"] = norm_summary[color_quantity]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = plt.Figure(figsize=(6, 10)) if ax is None else ax.get_figure()

    # make the linestyles and symbols list the same length as cmap
    try:
        num_colors = len(cmap)
        colors = cmap
    except TypeError:
        num_colors = len(cmap.colors)
        colors = cmap.colors

    ls_grow = int(np.ceil(num_colors / len(linestyles)))
    linestyles = (list(linestyles) * ls_grow)[:num_colors]
    marker_grow = int(np.ceil(num_colors / len(markers)))
    markers = (list(markers) * marker_grow)[:num_colors]
    ax.set_prop_cycle(
        cycler.cycler(color=colors)
        + cycler.cycler(linestyle=linestyles)
        + cycler.cycler(marker=markers)
    )

    plot_df.set_index("color", inplace=True)
    for idx in plot_df.index.unique():
        # good_points = np.isfinite(plot_df.loc[idx, "x"])

        # Due to wierdness with matplotlib arg handling,
        # make sure we get to pass the style argument
        # as a positional argument, whether or not it is
        # specified.
        plot_args = [plot_df.loc[idx, "x"], plot_df.loc[idx, "y"]]
        if (
            this_metric_set is not None
            and "style" in this_metric_set.columns
            and idx in this_metric_set.index
        ):
            metric_style = this_metric_set.loc[idx, "style"]
            if metric_style is not None:
                plot_args.append(metric_style)

        ax.plot(*plot_args, label=str(idx).strip())

    if horizontal_quantity in ["run", "metric"]:
        ax.tick_params("x", labelrotation=90)

    ax.legend(bbox_to_anchor=(1.0, 1.0))
    ax.grid()

    return fig, ax


def plot_run_metric_mesh(
    summary,
    metric_set=None,
    baseline_run=None,
    color_range=1,
    run_label_map=None,
    metric_label_map=None,
    ax=None,
    cmap=colorcet.cm.coolwarm_r,
):
    """Plot normalized metric values as colored points on a cartesian plane.

    Parameters
    ----------
    summary : `pandas.DataFrame`
        Values to be plotted. Should only include runs and metrics that
        should actually appear on the plot.
    baseline_run : `str`
        Name of the run to use as the baseline for normalization (see
        (archive.normalize_metric_summaries).
    color_range : `float`
        The color range of the plot, in normalized metrics summary
        units. (The color range will be 1-color_range/2 and
        1+color_range/2).
    run_label_map : mapping
        A python `mapping` between canonical run names and run labels as they
        should appear on plot labels.
    metric_label_map : mapping
        A python `mapping` between canonical metric names and metric labels
        as they should appear on plot labels.
    ax : `matplotlib.axes.Axes`
        The axes on which to plot the data.
    cmap : `matplotlib.colors.ColorMap`
        The color map to use for point colors.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        The plot figure..
    ax : `matplotilb.axes.Axes`
        The plot axes.

    """
    cols_are_runs = summary.columns.name in RUN_COL_NAMES
    rows_are_metrics = summary.index.name in METRIC_COL_NAMES
    if cols_are_runs or rows_are_metrics:
        summary = summary.T

    # Normalize the summary values, if a baseline was specified
    if baseline_run is not None:
        norm_summary = normalize_metric_summaries(baseline_run, summary, metric_set)
    else:
        norm_summary = summary.rename_axis(index="run", columns="metric").copy()

    vmin = 1 - color_range / 2
    vmax = vmin + color_range
    norm_values = norm_summary.T.values

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = plt.Figure(figsize=(6, 10)) if ax is None else ax.get_figure()

    y_nums, x_nums = np.arange(norm_summary.shape[1] + 1), np.arange(
        norm_summary.shape[0] + 1
    )
    im = ax.pcolormesh(
        x_nums,
        y_nums,
        norm_values,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    ax.set_yticks(np.arange(0.5, norm_summary.shape[1] + 0.5))
    metrics = norm_summary.columns.tolist()
    if metric_label_map is None:
        metric_labels = metrics
    else:
        metric_labels = [metric_label_map[m] for m in metrics]
    ax.set_yticklabels(metric_labels)

    ax.set_xticks(np.arange(0.5, norm_summary.shape[0] + 0.5))
    runs = norm_summary.index.tolist()
    if run_label_map is None:
        run_labels = runs
    else:
        run_labels = [run_label_map[r] for r in runs]
    ax.set_xticklabels(run_labels, rotation="vertical")

    fig.colorbar(im, ax=ax, label="Fractional difference")

    return fig, ax


# classes

# internal functions & classes
