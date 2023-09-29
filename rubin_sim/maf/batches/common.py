__all__ = (
    "combine_info_labels",
    "filter_list",
    "standard_summary",
    "extended_summary",
    "standard_metrics",
    "extended_metrics",
    "lightcurve_summary",
    "standard_angle_metrics",
    "summary_completeness_at_time",
    "summary_completeness_over_h",
    "fraction_population_at_threshold",
    "microlensing_summary",
)


import rubin_sim.maf.metrics as metrics


def combine_info_labels(info1, info2):
    if info1 is not None and info2 is not None:
        info = info1 + " " + info2
    elif info1 is not None:
        info = info1
    elif info2 is not None:
        info = info2
    else:
        info = None
    return info


def filter_list(all=True, extra_sql=None, extra_info_label=None):
    """Return a list of filters, plot colors and orders.

    Parameters
    ----------
    all : `bool`, optional
        Include 'all' in the list of filters and as part of the
        colors/orders dictionaries.
    extra_sql : `str`, optional
        Additional sql constraint to add to constraints returned per filter.
    extra_info_label : `str`, optional
        Substitute info_label to add to info_label strings composed per band.

    Returns
    -------
    filterlist : `list` [`str]
    colors : `dict` {`str`: `str`}
    orders : `dict` {`str`: int}
    sqls : `dict` {`str`: `str`}
    info_labels : `dict` {`str`: `str}
    """
    if all:
        filterlist = ("all", "u", "g", "r", "i", "z", "y")
    else:
        filterlist = ("u", "g", "r", "i", "z", "y")
    colors = {"u": "cyan", "g": "g", "r": "orange", "i": "r", "z": "m", "y": "b"}
    orders = {"u": 1, "g": 2, "r": 3, "i": 4, "z": 5, "y": 6}
    if all:
        colors["all"] = "k"
        orders["all"] = 0
    sqls = {}
    info_labels = {}
    if extra_info_label is None:
        if extra_sql is None or len(extra_sql) == 0:
            md = ""
        else:
            md = "%s " % extra_sql
    else:
        md = "%s " % extra_info_label
    for f in filterlist:
        if f == "all":
            sqls[f] = ""
            info_labels[f] = md + "all bands"
        else:
            sqls[f] = 'filter = "%s"' % f
            info_labels[f] = md + "%s band" % f
    if extra_sql is not None and len(extra_sql) > 0:
        for s in sqls:
            if s == "all":
                sqls[s] = extra_sql
            else:
                sqls[s] = "(%s) and (%s)" % (extra_sql, sqls[s])
    return filterlist, colors, orders, sqls, info_labels


def standard_summary(with_count=True):
    """A set of standard summary metrics, to calculate
    Mean, RMS, Median, #, Max/Min, and # 3-sigma outliers.

    Parameters
    ----------
    with_count : `bool`, optional
        Include the "Count" metric in the set of summary metrics or not.

    Returns
    -------
    standardSummary : `list` [`maf.BaseMetric`]
        List of metrics appropriate to use to summarize the results from
        another metric.
    """
    standardSummary = [
        metrics.MeanMetric(),
        metrics.RmsMetric(),
        metrics.MedianMetric(),
        metrics.MaxMetric(),
        metrics.MinMetric(),
        metrics.NoutliersNsigmaMetric(metric_name="N(+3Sigma)", n_sigma=3),
        metrics.NoutliersNsigmaMetric(metric_name="N(-3Sigma)", n_sigma=-3.0),
    ]
    if with_count:
        standardSummary += [metrics.CountMetric()]
    return standardSummary


def extended_summary():
    """An extended set of summary metrics, to calculate all that is in
    the standard summary stats, plus 25/75 percentiles.

    Returns
    --------
    extendedSummary : `list` [`maf.BaseMetric`]
        List of metrics appropriate to use to summarize the results
        from another metric.
    """
    extendedStats = standard_summary()
    extendedStats += [
        metrics.PercentileMetric(metric_name="25th%ile", percentile=25),
        metrics.PercentileMetric(metric_name="75th%ile", percentile=75),
    ]
    return extendedStats


def lightcurve_summary():
    lightcurveSummary = [
        metrics.SumMetric(metric_name="Total detected"),
        metrics.CountMetric(metric_name="Total lightcurves in footprint"),
        metrics.CountMetric(metric_name="Total lightcurves on sky", mask_val=0),
        metrics.MeanMetric(metric_name="Fraction detected in footprint (mean)"),
        metrics.MeanMetric(mask_val=0, metric_name="Fraction detected of total (mean)"),
    ]
    return lightcurveSummary


def standard_metrics(colname, replace_colname=None):
    """A set of standard simple metrics for some quantity.
    Typically would be applied with unislicer.

    Parameters
    ----------
    colname : `str`
        The column name to apply the metrics to.
    replace_colname: `str` or None, optional
        Value to replace colname with in the metric_name.
        i.e. if replace_colname='' then metric name is Mean,
        instead of Mean Airmass, or
        if replace_colname='seeingGeom', then metric name is
        Mean seeingGeom instead of Mean seeingFwhmGeom.

    Returns
    -------
    standardMetrics : `list` [`maf.BaseMetric`]
        List of appropriate MAF metrics to evaluate a distribution.
    """
    standardMetrics = [
        metrics.MeanMetric(colname),
        metrics.MedianMetric(colname),
        metrics.MinMetric(colname),
        metrics.MaxMetric(colname),
    ]
    if replace_colname is not None:
        for m in standardMetrics:
            if len(replace_colname) > 0:
                m.name = m.name.replace("%s" % colname, "%s" % replace_colname)
            else:
                m.name = m.name.rstrip(" %s" % colname)
    return standardMetrics


def extended_metrics(colname, replace_colname=None):
    """An extended set of simple metrics for some quantity.
    Typically applied with unislicer.

    Parameters
    ----------
    colname : `str`
        The column name to apply the metrics to.
    replace_colname: `str` or None, optional
        Value to replace colname with in the metric_name.
        i.e. if replace_colname='' then metric name is Mean,
        instead of Mean Airmass, or
        if replace_colname='seeingGeom', then metric name is
        Mean seeingGeom instead of Mean seeingFwhmGeom.

    Returns
    -------
    extendedMetrics : `list` [`maf.BaseMetric`]
        List of appropriate MAF metrics to evaluate a distribution.
    """
    extendedMetrics = standard_metrics(colname, replace_colname=None)
    extendedMetrics += [
        metrics.RmsMetric(colname),
        metrics.NoutliersNsigmaMetric(colname, metric_name="N(+3Sigma) " + colname, n_sigma=3),
        metrics.NoutliersNsigmaMetric(colname, metric_name="N(-3Sigma) " + colname, n_sigma=-3),
        metrics.PercentileMetric(colname, percentile=25),
        metrics.PercentileMetric(colname, percentile=75),
        metrics.CountMetric(colname),
    ]
    if replace_colname is not None:
        for m in extendedMetrics:
            if len(replace_colname) > 0:
                m.name = m.name.replace("%s" % colname, "%s" % replace_colname)
            else:
                m.name = m.name.rstrip(" %s" % colname)
    return extendedMetrics


def standard_angle_metrics(colname, replace_colname=None):
    """A set of standard simple metrics for a wrap-around angle quantity.

    Parameters
    ----------
    colname : `str`
        The column name to apply the metrics to.
    replace_colname: `str` or None, optional
        Value to replace colname with in the metric_name.
        i.e. if replace_colname='' then metric name is Mean,
        instead of Mean Airmass, or
        if replace_colname='seeingGeom', then metric name is
        Mean seeingGeom instead of Mean seeingFwhmGeom.

    Returns
    -------
    standardAngleMetrics : `list` [`maf.BaseMetric`]
        List of appropriate MAF metrics for angle distributions.
    """
    standardAngleMetrics = [
        metrics.MeanAngleMetric(colname),
        metrics.RmsAngleMetric(colname),
        metrics.FullRangeAngleMetric(colname),
        metrics.MinMetric(colname),
        metrics.MaxMetric(colname),
    ]
    if replace_colname is not None:
        for m in standardAngleMetrics:
            if len(replace_colname) > 0:
                m.name = m.name.replace("%s" % colname, "%s" % replace_colname)
            else:
                m.name = m.name.rstrip(" %s" % colname)
    return standardAngleMetrics


def summary_completeness_at_time(times, h_val, h_index=0.33):
    """A simple list of summary metrics to be applied to the Discovery_Time
    or PreviouslyKnown metrics.
    (can be used with any moving object metric which returns the
    time of discovery).

    Parameters
    ----------
    times : `np.ndarray `or list` [`float`]
        The times at which to evaluate the completeness @ Hval.
    h_val : `float`
        The H value at which to evaluate the completeness
        (cumulative and differential).
    h_index : `float`, optional
        The index of the power law to integrate H over
        (for cumulative completeness).

    Returns
    -------
    summaryMetrics : `list` [`maf.MoCompletenessAtTimeMetric`]
        List of completeness metrics to be evaluated at the specified times.
    """
    summaryMetrics = [
        metrics.MoCompletenessAtTimeMetric(times=times, hval=h_val, hindex=h_index, cumulative=False),
        metrics.MoCompletenessAtTimeMetric(times=times, hval=h_val, hindex=h_index, cumulative=True),
    ]
    return summaryMetrics


def summary_completeness_over_h(requiredChances=1, Hindex=0.33):
    """A simple list of summary metrics to be applied to the
    Discovery_N_Chances metric.

    Parameters
    ----------
    requiredChances : `int`, optional
        Number of discovery opportunities required to consider an
        object 'discovered'.
    Hindex : `float`, optional
        The index of the power law to integrate H over
        (for cumulative completeness).

    Returns
    -------
    summaryMetrics : `list` [`maf.MoCompletenessMetric`]
        List of moving object MoCompleteness metrics
        (cumulative and differential)
    """
    summaryMetrics = [
        metrics.MoCompletenessMetric(threshold=requiredChances, cumulative=False, hindex=Hindex),
        metrics.MoCompletenessMetric(threshold=requiredChances, cumulative=True, hindex=Hindex),
    ]
    return summaryMetrics


def fraction_population_at_threshold(thresholds, optnames=None):
    """Creates a list of summary metrics to be applied to any moving object
    metric which reports a float value, calculating the fraction of the
    population above X.

    Parameters
    ----------
    thresholds : `list` [`float`]
        The thresholds at which to calculate what fraction of the population
        exceeds these values.
    optnames : `list` [`str`], optional
        If provided, these names will be used instead of the threshold values
        when constructing the metric names.
        This allows more descriptive summary statistic names.

    Returns
    -------
    fracMetrics : `list` [`maf.MoCompletenessMetric`]
        List of moving object MoCompleteness metrics
        (differential fractions of the population).
    """
    fracMetrics = []
    for i, threshold in enumerate(thresholds):
        if optnames is not None:
            o = optnames[i]
        else:
            o = threshold
        m = metrics.MoCompletenessMetric(
            threshold=threshold, cumulative=False, metric_name=f"FractionPop {o}"
        )
        fracMetrics.append(m)
    return fracMetrics


def microlensing_summary(metric_type, npts_required=10, Fisher_sigmatE_tE_cutoff=0.1):
    """Calculate summary metrics for the microlensing population metrics.

    Parameters
    -----------
    metric_type : `str`
        Identify whether the metric is "Npts" or Fisher"
    npts_required : `int`, optional
        Count the fraction of microlensing events with more than npts_required
        observations
    Fisher_sigmatE_tE_cutoff : `float`, optional
        Count the fraction of microlensing events with characterization
        uncertainty less than this

    Returns
    -------
    microlensingSummary : `list` [`maf.BaseMetric`]
        List of appropriate MAF metrics for this type of microlensing
        metric with the specified threshold values.
    """
    if metric_type != "Npts" and metric_type != "Fisher":
        raise Exception('metric_type must be "Npts" or "Fisher"')
    if metric_type == "Npts":
        microlensingSummary = [
            metrics.FracAboveMetric(
                cutoff=npts_required,
                metric_name=f"Fraction w/ at least {npts_required} points",
            ),
            metrics.CountMetric(metric_name="Total lightcurves in footprint"),
            metrics.CountMetric(metric_name="Total lightcurves on sky", mask_val=0),
            metrics.MeanMetric(metric_name="Mean number of points per lightcurves in footprint"),
            metrics.MeanMetric(mask_val=0, metric_name="Mean number of points per lightcurves in total"),
        ]
    elif metric_type == "Fisher":
        microlensingSummary = [
            metrics.FracBelowMetric(
                cutoff=Fisher_sigmatE_tE_cutoff,
                metric_name=f"Fraction w/ sigma_tE/tE < {Fisher_sigmatE_tE_cutoff}",
            ),
            metrics.CountMetric(metric_name="Total lightcurves in footprint"),
            metrics.CountMetric(metric_name="Total lightcurves on sky", mask_val=0),
            metrics.RealMeanMetric(metric_name="Mean sigma_tE/tE in footprint (mean)"),
            metrics.RealMeanMetric(mask_val=0, metric_name="Mean sigma_tE/tE of total (mean)"),
        ]
    return microlensingSummary
