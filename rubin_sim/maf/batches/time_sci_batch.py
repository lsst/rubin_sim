"""Sets of metrics to look at impact of cadence on science
"""
import numpy as np

import rubin_sim.maf.metric_bundles as mb
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.plots as plots
import rubin_sim.maf.slicers as slicers

from .col_map_dict import col_map_dict
from .common import combine_info_labels, filter_list, radec_cols, standard_summary

__all__ = ["phaseGap"]


def phaseGap(
    colmap=None,
    runName="opsim",
    nside=64,
    extraSql=None,
    extraInfoLabel=None,
    ditherStacker=None,
    ditherkwargs=None,
):
    """Generate a set of statistics about the pair/triplet/etc. rate within a night.

    Parameters
    ----------
    colmap : dict or None, optional
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    runName : str, optional
        The name of the simulated survey. Default is "opsim".
    nside : int, optional
        Nside for the healpix slicer. Default 64.
    extraSql : str or None, optional
        Additional sql constraint to apply to all metrics.
    extraInfoLabel : str or None, optional
        Additional info_label to apply to all results.
    ditherStacker: str or rubin_sim.maf.stackers.BaseDitherStacker
        Optional dither stacker to use to define ra/dec columns.
    ditherkwargs: dict or None, optional
        Optional dictionary of kwargs for the dither stacker.

    Returns
    -------
    metric_bundleDict
    """

    if colmap is None:
        colmap = col_map_dict("opsimV4")

    info_label = extraInfoLabel
    if extraSql is not None and len(extraSql) > 0:
        if info_label is None:
            info_label = extraSql

    raCol, decCol, degrees, ditherStacker, ditherMeta = radec_cols(ditherStacker, colmap, ditherkwargs)
    info_label = combine_info_labels(info_label, ditherMeta)

    bundleList = []
    standardStats = standard_summary()
    subsetPlots = [plots.HealpixSkyMap(), plots.HealpixHistogram()]

    slicer = slicers.HealpixSlicer(nside=nside, lat_col=decCol, lon_col=raCol, lat_lon_deg=degrees)

    # largest phase gap for periods
    periods = [0.1, 1.0, 10.0, 100.0]
    sqls = [
        'filter = "u"',
        'filter="r"',
        'filter="g" or filter="r" or filter="i" or filter="z"',
        "",
    ]
    filter_names = ["u", "r", "griz", "all"]
    info_labels = filter_names
    if info_label is not None:
        info_labels = [combine_info_labels(m, info_label) for m in info_labels]
    if extraSql is not None and len(extraSql) > 0:
        for sql in sqls:
            sqls[sql] = "(%s) and (%s)" % (sqls[sql], extraSql)

    for sql, md, f in zip(sqls, info_labels, filter_names):
        for period in periods:
            displayDict = {
                "group": "PhaseGap",
                "subgroup": "Filter %s: Period %.2f days" % (f, period),
                "caption": "Maximum phase gap, given a period of %.2f days." % period,
            }
            metric = metrics.PhaseGapMetric(
                nPeriods=1,
                periodMin=period,
                periodMax=period,
                nVisitsMin=5,
                metric_name="PhaseGap %.1f day" % period,
            )
            metric.reduceFuncs = {metric.reduceFuncs["reduceLargestGap"]}
            metric.reduceOrder = {0}
            bundle = mb.MetricBundle(
                metric,
                slicer,
                constraint=sql,
                info_label=md,
                display_dict=displayDict,
                summary_metrics=standardStats,
                plot_funcs=subsetPlots,
            )
            bundleList.append(bundle)

    # Set the run_name for all bundles and return the bundleDict.
    for b in bundleList:
        b.set_run_name(runName)
    return mb.make_bundles_dict_from_list(bundleList)
