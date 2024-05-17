"""Summary metric plotting functions.
"""

__all__ = (
    "normalize_metric_summaries",
    "plot_run_metric",
    "plot_run_metric_mesh",
    "plot_run_metric_uncert",
    "find_family_lines",
)

# imports
import warnings

import colorcet
import cycler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# constants

RUN_COL_NAMES = ["run", "OpsimRun", "run_name"]
METRIC_COL_NAMES = ["metric"]

# exception classes

# interface functions


def normalize_metric_summaries(
    baseline_run,
    summary,
    metric_subsets=None,
):
    """Create a normalized `pandas.DataFrame` of metric summary values.

    Parameters
    ----------
    baseline_run : `str` or `list` of `str
        The name of the run that defines a normalized value of 1.
        If a list is provided, the median value of each metric across that
        list is used as the reference.
    summary : `pandas.DataFrame`
        The summary metrics to normalize (as returned by
        `get_metric_summaries`)
    metric_subsets : `pandas.DataFrame`
        Metric metadata as returned by `archive.get_metric_subsets`

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
    if metric_subsets is None:
        summary = summary.copy()
        used_metrics = summary.columns.values
    else:
        used_metrics = [s for s in summary.columns.values if s in metric_subsets.metric.values]
        summary = summary.loc[:, used_metrics].copy()

    if summary.columns.name is None:
        summary.rename_axis(columns="metric", inplace=True)

    if summary.index.name in [None] + RUN_COL_NAMES:
        summary.rename_axis("run", inplace=True)

    # Get rid of duplicate metrics and runs
    summary = summary.T.groupby("metric").first().T.groupby("run").first()

    # And now create a line just for "baseline" --
    # if baseline_run is >1, this is created from the median values
    # per metric of those runs
    # Make up a nonsense name for the reference, that is not currently
    # in the summary dataframe
    baseline_comparison = "bbb"
    while baseline_comparison in summary.index:
        baseline_comparison += "b"

    if isinstance(summary.loc[baseline_run], pd.DataFrame):
        summary.loc[baseline_comparison] = summary.loc[baseline_run].median(axis="rows")
    else:
        summary.loc[baseline_comparison] = summary.loc[baseline_run]

    if metric_subsets is None:
        # If no invert/mag - just do simple normalization (1 + (x-0)/x0)
        norm_summary = 1 + (summary.loc[:, :].sub(summary.loc[baseline_comparison, :], axis="columns")).div(
            summary.loc[baseline_comparison, :], axis="columns"
        )
    else:
        # Reindex metric set and remove duplicates or non-available metrics
        metric_names = [n for n in metric_subsets.index.names if not n == "metric"]
        metric_subsets = (
            metric_subsets.reset_index(metric_names)
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

        # Direct metrics are those that are neither inverted,
        # nor compared as magnitudes
        # direct = 1 + (value - norm) / norm == value / norm
        direct = ~np.logical_or(metric_subsets["invert"], metric_subsets["mag"])
        norm_summary.loc[:, direct] = summary.loc[:, direct]

        # invert = 1 + (1/value - 1/norm) / (1/norm) == norm / value
        norm_summary.loc[:, metric_subsets["invert"]] = 1.0 / summary.loc[:, metric_subsets["invert"]]

        # mag = 1 + (1+value-norm - (1+norm-norm)) / (1+norm-norm)
        # == 1 + (value - norm)
        norm_summary.loc[:, metric_subsets["mag"]] = 1.0 + summary.loc[
            :,
            metric_subsets["mag"],
        ].subtract(summary.loc[baseline_comparison, metric_subsets["mag"]], axis="columns")

        # Some metrics can be both inverted and magnitudes (eg rms mag values)
        both = np.logical_and(metric_subsets["invert"], metric_subsets["mag"])
        # both = 1 + (1-(value-norm) - (1-(norm-norm))) / (1-(norm-norm))
        # == norm - value
        norm_summary.loc[:, both] = 1.0 - summary.loc[:, both].subtract(
            summary.loc[baseline_comparison, both], axis="columns"
        )

        # Turn the values above into the fractional difference
        # compared with the baseline
        norm_summary.loc[:, :] = 1 + (
            norm_summary.loc[:, :].sub(norm_summary.loc[baseline_comparison, :], axis="columns")
        ).div(norm_summary.loc[baseline_comparison, :], axis="columns")

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
    run_label_map=None,
    metric_label_map=None,
    metric_subset=None,
    ax=None,
    cmap=colorcet.glasbey_hv,
    linestyles=None,
    markers=["o"],
    shade_fraction=0.05,
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
        The width of the plot, in normalized metrics summary units.
        (The limits of the x axis will be 1-vwidth/2 and 1+width/2).
    run_label_map : mapping
        A python `mapping` between canonical run names and run labels as they
        should appear on plot labels. Use of this option is discouraged,
        because it makes it harder to match plots to data.
        run_label_map could be created by
        archive.get_runs().loc[these_runs]['brief']
    metric_label_map : mapping
        A python `mapping` between canonical metric names and metric labels
        as they should appear on plot labels. Use this option carefully,
        because it makes it harder to match plots to metric calculation code..
        metric_label_map could be equivalent to metric_subset['short_name']
    metric_subset : `pandas.DataFrame`
        Metric metadata as returned by `archive.get_metric_subsets`
    ax : `matplotlib.axes.Axes`
        The axes on which to plot the data.
    cmap : `matplotlib.colors.ColorMap`
        The color map to use for point colors.
    linestyles : `list`
        A list of matplotlib linestyles to use to connect the lines
    markers : `list`, opt
        A list of matplotlib markers to use to represent the points
    shade_fraction : `float`, opt
        Add a red/green shading to the plot, starting at 1 +/- shade_fraction.
        Set to 0 or None for no shading.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        The plot figure.
    ax : `matplotilb.axes.Axes`
        The plot axes.


    The run order and metric order (imposed into the summary dataframe
    passed here as `summary`) are important and preserved in the plot.
    These should be set in the (subset) `summary` dataframe
    passed here; the metric_subset is available, but used for
    normalization and plot styling.
    """

    # If the metric sets we are passed has a multilevel index,
    # get rid of the levels we do not need.
    if metric_subset is not None and metric_subset.index.nlevels > 1:
        extra_levels = list(set(metric_subset.index.names) - set(["metric"]))
        metric_subset = metric_subset.droplevel(extra_levels).groupby(level="metric").first()

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
            else normalize_metric_summaries(baseline_run, summary, metric_subsets=metric_subset)
        )
        .stack(future_stack=True)
        .rename("value")
        .reset_index()
        .rename(columns={"OpsimRun": "run"})
    )
    # Pull original order for metric & runs from summary
    run_order = summary.index.values
    metric_order = summary.columns.values
    if run_label_map is not None:
        run_order = [run_label_map[r] for r in run_order]
        norm_summary["run"] = norm_summary["run"].map(run_label_map)
    if metric_label_map is not None:
        metric_order = [metric_label_map[m] for m in metric_order]
        norm_summary["metric"] = norm_summary["metric"].map(metric_label_map)
        # Create this_metric_subset - equivalent to metric_subset
        # but with updated names.
        if metric_subset is not None:
            this_metric_subset = (
                metric_subset.drop(columns=["metric"])
                .assign(metric=metric_subset["metric"].map(metric_label_map))
                .set_index("metric", drop=False)
            )
        else:
            this_metric_subset = None
    else:
        this_metric_subset = metric_subset

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
        # At this point, color has to be metric or run
        plot_df["color"] = norm_summary[color_quantity]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
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
        cycler.cycler(color=colors) + cycler.cycler(linestyle=linestyles) + cycler.cycler(marker=markers)
    )

    plot_df.set_index("color", inplace=True)

    # 'color' or 'plot_df.index' is now either 'run' or 'metric'
    if color_quantity == "run":
        idx_order = run_order
    if color_quantity == "metric":
        idx_order = metric_order
    for idx in idx_order:  # plot_df.index.unique():
        # Due to weirdness with matplotlib arg handling,
        # make sure we get to pass the style argument
        # as a positional argument, whether or not it is
        # specified.
        # Let the user know why some of their plot values might be
        # disappearing
        # (tends to happen if baseline value is 0 or Nan and plot_df
        # being normalized)
        if vertical_quantity == "value" and np.isinf(plot_df.loc[idx, "y"]).any():
            warnings.warn(f"There are infinite values in the plot of {idx}.")
        if horizontal_quantity == "value" and np.isinf(plot_df.loc[idx, "x"]).any():
            warnings.warn(f"There are infinite values in the plot of {idx}.")
        plot_args = [plot_df.loc[idx, "x"], plot_df.loc[idx, "y"]]
        idx_label = f"{str(idx).strip()}"
        if this_metric_subset is not None and idx in this_metric_subset.index:
            # Set the style from the metric_subset if available
            if "style" in this_metric_subset.columns:
                metric_style = this_metric_subset.loc[idx, "style"]
                if metric_style is not None:
                    plot_args.append(metric_style)
            # Update the plot label if we inverted the column during
            # normalization
            if "invert" in this_metric_subset.columns and baseline_run is not None:
                inv = this_metric_subset.loc[idx, "invert"]
                if inv:
                    idx_label = f"1 / {idx_label}"
        ax.plot(*plot_args, label=idx_label)

    if vertical_quantity == "value":
        # Set xlim to be exact length of number of runs
        xlim_new = [0, len(summary) - 1]
        ax.set_xlim(xlim_new)

    if vertical_quantity == "run":
        ylim_new = [0, len(summary) - 1]
        ax.set_ylim(ylim_new)

    if shade_fraction is not None and shade_fraction > 0:
        if vertical_quantity == "value":
            xlim = ax.get_xlim()
            high_shade_bottom = 1 + shade_fraction
            high_shade_top = ax.get_ylim()[1]
            if high_shade_top > high_shade_bottom:
                ax.fill_between(
                    xlim,
                    high_shade_bottom,
                    high_shade_top,
                    color="g",
                    alpha=0.1,
                )

            low_shade_top = 1 - shade_fraction
            low_shade_bottom = ax.get_ylim()[0]
            if low_shade_top > low_shade_bottom:
                ax.fill_between(xlim, low_shade_bottom, low_shade_top, color="r", alpha=0.1)

        elif horizontal_quantity == "value":
            ylim = ax.get_ylim()
            high_shade_left = 1 + shade_fraction
            high_shade_right = ax.get_xlim()[1]
            if high_shade_right > high_shade_left:
                ax.fill_betweenx(
                    ylim,
                    high_shade_left,
                    high_shade_right,
                    color="g",
                    alpha=0.1,
                )

            low_shade_right = 1 - shade_fraction
            low_shade_left = ax.get_xlim()[0]
            if low_shade_right > low_shade_left:
                ax.fill_betweenx(ylim, low_shade_left, low_shade_right, color="r", alpha=0.1)

    if horizontal_quantity in ["run", "metric"]:
        ax.tick_params("x", labelrotation=90)

    ax.legend(bbox_to_anchor=(1.0, 1.0))
    ax.grid()

    return fig, ax


def plot_run_metric_mesh(
    summary,
    metric_subset=None,
    baseline_run=None,
    color_range=1,
    run_label_map=None,
    metric_label_map=None,
    ax=None,
    cmap=colorcet.cm["CET_D1A_r"],
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
        A red/blue diverging color map - CET_D1A_r or CET_D1_r
        A rainbow diverging color map - CET_R3_r

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
        norm_summary = normalize_metric_summaries(baseline_run, summary, metric_subset)
    else:
        norm_summary = summary.rename_axis(index="run", columns="metric").copy()

    if color_range is not None and (isinstance(color_range, float) or isinstance(color_range, int)):
        vmin = 1 - color_range / 2
        vmax = vmin + color_range
    elif isinstance(color_range, list):
        vmin = color_range[0]
        vmax = color_range[1]
    else:
        vmin = norm_summary.min()
        vmax = norm_summary.max()

    norm_values = norm_summary.T.values

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = plt.Figure(figsize=(6, 10)) if ax is None else ax.get_figure()

    y_nums, x_nums = (
        np.arange(norm_summary.shape[1] + 1),
        np.arange(norm_summary.shape[0] + 1),
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
        try:
            # When there are multiple metric families specified,
            # there might be duplicate elements in the
            # metric_label_map. Remove the duplicates.
            metric_label_map = metric_label_map[~metric_label_map.index.duplicated(keep="first")]
        except AttributeError:
            # if metric_label_map is a dict, it won't have
            # an index attrubute, but there isn't any danger
            # if duplicates either
            pass

        # Figure out which metrics get inverted
        if baseline_run is not None and metric_subset is not None:
            inverted_metrics = set(metric_subset.query("invert").metric.values)
        else:
            inverted_metrics = set()

        metric_labels = [
            f"1/{metric_label_map[m]}" if m in inverted_metrics else metric_label_map[m] for m in metrics
        ]

    ax.set_yticklabels(metric_labels)

    ax.set_xticks(np.arange(0.5, norm_summary.shape[0] + 0.5))
    runs = norm_summary.index.tolist()
    if run_label_map is None:
        run_labels = runs
    else:
        run_labels = [run_label_map[r] for r in runs]
    ax.set_xticklabels(run_labels, rotation="vertical")

    if baseline_run is None:
        fig.colorbar(im, ax=ax, label=None)
    else:
        fig.colorbar(im, ax=ax, label="Fractional difference")

    return fig, ax


def plot_run_metric_uncert(
    summary,
    uncertainty,
    run_label_map=None,
    metric_label_map=None,
    metric_subset=None,
    cmap=None,
    linestyles=["-"],
    markers=["."],
    sep_plots=True,
    ax=None,
):
    """Plot normalized metric values as colored points on a cartesian plane.

    Parameters
    ----------
    summary : `pandas.DataFrame`
        Values to be plotted. Should only include runs and metrics that
        should actually appear on the plot.
    uncertainty : `pandas.DataFrame`
        Uncertainty values to plot on each data point.
        Should match summary metric columns.
    run_label_map : mapping
        A python `mapping` between canonical run names and run labels as they
        should appear on plot labels. Use of this option is discouraged,
        because it makes it harder to match plots to data.
        run_label_map could be created by
        archive.get_runs().loc[these_runs]['brief']
    metric_label_map : mapping
        A python `mapping` between canonical metric names and metric labels
        as they should appear on plot labels. Use this option carefully,
        because it makes it harder to match plots to metric calculation code..
        metric_label_map could be equivalent to metric_subset['short_name']
    metric_subset : `pandas.DataFrame`
        Metric metadata as returned by `archive.get_metric_subsets`
    ax : `matplotlib.axes.Axes`
        The axes on which to plot the data.
    cmap : `matplotlib.colors.ColorMap`
        The color map to use for point colors.
    linestyles : `list`
        A list of matplotlib linestyles to use to connect the lines
    markers : `list`, opt
        A list of matplotlib markers to use to represent the points

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        The plot figure.
    ax : `matplotilb.axes.Axes`
        The plot axes.


    The run order and metric order (imposed into the summary
    dataframe passed here as `summary`) are important and preserved in the
    plot. These should be set in the (subset) `summary` dataframe
    passed here; the metric_subset is available, but used for 'invert'
    and plot styling and alternate labels.
    """

    # If the metric sets we are passed has a multilevel index,
    # get rid of the levels we do not need.
    if metric_subset is not None and metric_subset.index.nlevels > 1:
        extra_levels = list(set(metric_subset.index.names) - set(["metric"]))
        metric_subset = metric_subset.droplevel(extra_levels).groupby(level="metric").first()

    # Pull original order for metric & runs from summary
    run_order = summary.index.values
    metric_order = summary.columns.values
    if run_label_map is not None:
        run_order = [run_label_map[r] for r in run_order]
    if metric_label_map is not None:
        metric_order = [metric_label_map[m] for m in metric_order]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    if cmap is None:
        cmap = colorcet.glasbey_hv
        cmap_default = True
    else:
        cmap_default = False
    # make the linestyles and symbols list the same length as cmap, for cycler
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

    # But use styles from metric_subset if available
    if metric_subset is not None:
        for i, m in enumerate(summary):
            if m in metric_subset.index:
                style = metric_subset.loc[m, "style"]
                if len(style) > 1:
                    ci = style[0]
                    # Let user specify color map for override
                    if cmap_default:
                        colors[i] = ci
                    li = style[1]
                    linestyles[i] = li
                else:
                    li = style[0:]
                    linestyles[i] = li

    ax.set_prop_cycle(
        cycler.cycler(color=colors) + cycler.cycler(linestyle=linestyles) + cycler.cycler(marker=markers)
    )

    for i, m in enumerate(summary):
        # new plots for each metric?
        if sep_plots and i > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            cc = [colors[i]]
            ax.set_prop_cycle(
                cycler.cycler(color=cc)
                + cycler.cycler(linestyle=linestyles[i])
                + cycler.cycler(marker=markers[i])
            )
        if metric_label_map is not None:
            label = metric_label_map[m]
        else:
            label = m
        ax.errorbar(run_order, summary[m], yerr=uncertainty[m], label=label)
        ax.set_ylabel(label, fontsize="large")
        if sep_plots:
            if metric_subset is not None:
                if m in metric_subset.index:
                    if metric_subset.loc[m, "invert"]:
                        ax.invert_yaxis()
            ax.tick_params(axis="x", labelrotation=90, labelsize="large")
            ax.grid(True, alpha=0.5)
            ax.legend()

    if not sep_plots:
        ax.tick_params(axis="x", labelrotation=90, labelsize="large")
        ax.grid(True, alpha=0.5)
        ax.legend(bbox_to_anchor=(1.0, 1.0))

    return fig, ax


def find_family_lines(families, family_list):
    lines = []
    for f in family_list:
        lines += [len(families.loc[f]["run"])]
    lines = np.array(lines).cumsum()
    return lines


# classes

# internal functions & classes
