from __future__ import print_function
import inspect

import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.stackers as stackers

__all__ = [
    "combineInfoLabels",
    "filterList",
    "radecCols",
    "standardSummary",
    "extendedSummary",
    "standardMetrics",
    "extendedMetrics",
    "lightcurveSummary",
    "standardAngleMetrics",
    "summaryCompletenessAtTime",
    "summaryCompletenessOverH",
    "fractionPopulationAtThreshold",
    "microlensingSummary",
]


def combineInfoLabels(info1, info2):
    if info1 is not None and info2 is not None:
        info = info1 + " " + info2
    elif info1 is not None:
        info = info1
    elif info2 is not None:
        info = info2
    else:
        info = None
    return info


def filterList(all=True, extraSql=None, extraInfoLabel=None):
    """Return a list of filters, plot colors and orders.

    Parameters
    ----------
    all : `bool`, optional
        Include 'all' in the list of filters and as part of the colors/orders dictionaries.
        Default True.
    extraSql : str, optional
        Additional sql constraint to add to sqlconstraints returned per filter.
        Default None.
    extraInfoLabel : str, optional
        Substitute info_label to add to info_label strings composed per band.
        Default None.

    Returns
    -------
    list, dict, dict
        List of filter names, dictionary of colors (for plots), dictionary of orders (for display)
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
    if extraInfoLabel is None:
        if extraSql is None or len(extraSql) == 0:
            md = ""
        else:
            md = "%s " % extraSql
    else:
        md = "%s " % extraInfoLabel
    for f in filterlist:
        if f == "all":
            sqls[f] = ""
            info_labels[f] = md + "all bands"
        else:
            sqls[f] = 'filter = "%s"' % f
            info_labels[f] = md + "%s band" % f
    if extraSql is not None and len(extraSql) > 0:
        for s in sqls:
            if s == "all":
                sqls[s] = extraSql
            else:
                sqls[s] = "(%s) and (%s)" % (extraSql, sqls[s])
    return filterlist, colors, orders, sqls, info_labels


def radecCols(ditherStacker, colmap, ditherkwargs=None):
    degrees = colmap["raDecDeg"]
    if ditherStacker is None:
        raCol = colmap["ra"]
        decCol = colmap["dec"]
        stacker = None
        ditherInfoLabel = None
    else:
        if isinstance(ditherStacker, stackers.BaseDitherStacker):
            stacker = ditherStacker
        else:
            s = stackers.BaseStacker().registry[ditherStacker]
            args = [f for f in inspect.getfullargspec(s).args if f.endswith("Col")]
            # Set up default dither kwargs.
            kwargs = {}
            for a in args:
                colmapCol = a.replace("Col", "")
                if colmapCol in colmap:
                    kwargs[a] = colmap[colmapCol]
            # Update with passed values, if any.
            if ditherkwargs is not None:
                kwargs.update(ditherkwargs)
            stacker = s(degrees=degrees, **kwargs)
        raCol = stacker.colsAdded[0]
        decCol = stacker.colsAdded[1]
        # Send back some info_label information.
        ditherInfoLabel = stacker.__class__.__name__.replace("Stacker", "")
        if ditherkwargs is not None:
            for k, v in ditherkwargs.items():
                ditherInfoLabel += " " + "%s:%s" % (k, v)
    return raCol, decCol, degrees, stacker, ditherInfoLabel


def standardSummary(withCount=True):
    """A set of standard summary metrics, to calculate Mean, RMS, Median, #, Max/Min, and # 3-sigma outliers."""
    standardSummary = [
        metrics.MeanMetric(),
        metrics.RmsMetric(),
        metrics.MedianMetric(),
        metrics.MaxMetric(),
        metrics.MinMetric(),
        metrics.NoutliersNsigmaMetric(metric_name="N(+3Sigma)", n_sigma=3),
        metrics.NoutliersNsigmaMetric(metric_name="N(-3Sigma)", n_sigma=-3.0),
    ]
    if withCount:
        standardSummary += [metrics.CountMetric()]
    return standardSummary


def extendedSummary():
    """An extended set of summary metrics, to calculate all that is in the standard summary stats,
    plus 25/75 percentiles."""

    extendedStats = standardSummary()
    extendedStats += [
        metrics.PercentileMetric(metric_name="25th%ile", percentile=25),
        metrics.PercentileMetric(metric_name="75th%ile", percentile=75),
    ]
    return extendedStats


def lightcurveSummary():
    lightcurveSummary = [
        metrics.SumMetric(metric_name="Total detected"),
        metrics.CountMetric(metric_name="Total lightcurves in footprint"),
        metrics.CountMetric(metric_name="Total lightcurves on sky", maskVal=0),
        metrics.MeanMetric(metric_name="Fraction detected in footprint (mean)"),
        metrics.MeanMetric(mask_val=0, metric_name="Fraction detected of total (mean)"),
    ]
    return lightcurveSummary


def standardMetrics(colname, replace_colname=None):
    """A set of standard simple metrics for some quantity. Typically would be applied with unislicer.

    Parameters
    ----------
    colname : str
        The column name to apply the metrics to.
    replace_colname: str or None, optional
        Value to replace colname with in the metric_name.
        i.e. if replace_colname='' then metric name is Mean, instead of Mean Airmass, or
        if replace_colname='seeingGeom', then metric name is Mean seeingGeom instead of Mean seeingFwhmGeom.
        Default is None, which does not alter the metric name.

    Returns
    -------
    List of configured metrics.
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


def extendedMetrics(colname, replace_colname=None):
    """An extended set of simple metrics for some quantity. Typically applied with unislicer.

    Parameters
    ----------
    colname : str
        The column name to apply the metrics to.
    replace_colname: str or None, optional
        Value to replace colname with in the metric_name.
        i.e. if replace_colname='' then metric name is Mean, instead of Mean Airmass, or
        if replace_colname='seeingGeom', then metric name is Mean seeingGeom instead of Mean seeingFwhmGeom.
        Default is None, which does not alter the metric name.

    Returns
    -------
    List of configured metrics.
    """
    extendedMetrics = standardMetrics(colname, replace_colname=None)
    extendedMetrics += [
        metrics.RmsMetric(colname),
        metrics.NoutliersNsigmaMetric(
            colname, metric_name="N(+3Sigma) " + colname, n_sigma=3
        ),
        metrics.NoutliersNsigmaMetric(
            colname, metric_name="N(-3Sigma) " + colname, n_sigma=-3
        ),
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


def standardAngleMetrics(colname, replace_colname=None):
    """A set of standard simple metrics for some quantity which is a wrap-around angle.

    Parameters
    ----------
    colname : str
        The column name to apply the metrics to.
    replace_colname: str or None, optional
        Value to replace colname with in the metric_name.
        i.e. if replace_colname='' then metric name is Mean, instead of Mean Airmass, or
        if replace_colname='seeingGeom', then metric name is Mean seeingGeom instead of Mean seeingFwhmGeom.
        Default is None, which does not alter the metric name.

    Returns
    -------
    List of configured metrics.
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


def summaryCompletenessAtTime(times, Hval, Hindex=0.33):
    """A simple list of summary metrics to be applied to the Discovery_Time or PreviouslyKnown metrics.
    (can be used with any moving object metric which returns the time of discovery).

    Parameters
    ----------
    times : np.ndarray or list
        The times at which to evaluate the completeness @ Hval.
    Hval : float
        The H value at which to evaluate the completeness (cumulative and differential).
    Hindex : float, optional
        The index of the power law to integrate H over (for cumulative completeness).
        Default is 0.33.

    Returns
    -------
    List of moving object MoCompletenessAtTime metrics (cumulative and differential)
    """
    summaryMetrics = [
        metrics.MoCompletenessAtTimeMetric(
            times=times, Hval=Hval, Hindex=Hindex, cumulative=False
        ),
        metrics.MoCompletenessAtTimeMetric(
            times=times, Hval=Hval, Hindex=Hindex, cumulative=True
        ),
    ]
    return summaryMetrics


def summaryCompletenessOverH(requiredChances=1, Hindex=0.33):
    """A simple list of summary metrics to be applied to the Discovery_N_Chances metric.

    Parameters
    ----------
    requiredChances : int, optional
        Number of discovery opportunities required to consider an object 'discovered'.
    Hindex : float, optional
        The index of the power law to integrate H over (for cumulative completeness).
        Default is 0.33.

    Returns
    -------
    List of moving object MoCompleteness metrics (cumulative and differential)
    """
    summaryMetrics = [
        metrics.MoCompletenessMetric(
            threshold=requiredChances, cumulative=False, Hindex=Hindex
        ),
        metrics.MoCompletenessMetric(
            threshold=requiredChances, cumulative=True, Hindex=Hindex
        ),
    ]
    return summaryMetrics


def fractionPopulationAtThreshold(thresholds, optnames=None):
    """Creates a list of summary metrics to be applied to any moving object metric
    which reports a float value, calculating the fraction of the population above X.

    Parameters
    ----------
    thresholds : list of float
        The thresholds at which to calculate what fraction of the population exceeds these values.
    optnames : list of str, optional
        If provided, these names will be used instead of the threshold values when constructing
        the metric names. This allows more descriptive summary statistic names.
    Returns
    -------
    List of moving object MoCompleteness metrics (differential fractions of the population).
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


def microlensingSummary(metricType, npts_required=10, Fisher_sigmatE_tE_cutoff=0.1):
    if metricType != "Npts" and metricType != "Fisher":
        raise Exception('metricType must be "Npts" or "Fisher"')
    if metricType == "Npts":
        microlensingSummary = [
            metrics.FracAboveMetric(
                cutoff=npts_required,
                metric_name=f"Fraction w/ at least {npts_required} points",
            ),
            metrics.CountMetric(metric_name="Total lightcurves in footprint"),
            metrics.CountMetric(metric_name="Total lightcurves on sky", maskVal=0),
            metrics.MeanMetric(
                metric_name="Mean number of points per lightcurves in footprint"
            ),
            metrics.MeanMetric(
                mask_val=0, metric_name="Mean number of points per lightcurves in total"
            ),
        ]
    elif metricType == "Fisher":
        microlensingSummary = [
            metrics.FracBelowMetric(
                cutoff=Fisher_sigmatE_tE_cutoff,
                metric_name=f"Fraction w/ sigma_tE/tE < {Fisher_sigmatE_tE_cutoff}",
            ),
            metrics.CountMetric(metric_name="Total lightcurves in footprint"),
            metrics.CountMetric(metric_name="Total lightcurves on sky", mask_val=0),
            metrics.RealMeanMetric(metric_name="Mean sigma_tE/tE in footprint (mean)"),
            metrics.RealMeanMetric(
                mask_val=0, metric_name="Mean sigma_tE/tE of total (mean)"
            ),
        ]
    return microlensingSummary
