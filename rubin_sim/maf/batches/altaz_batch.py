__all__ = ("altazHealpix", "altazLambert")

import rubin_sim.maf.metric_bundles as mb
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.plots as plots
import rubin_sim.maf.slicers as slicers

from .col_map_dict import col_map_dict
from .common import filter_list


def basicSetup(metric_name, colmap=None, nside=64):
    if colmap is None:
        colmap = col_map_dict()

    slicer = slicers.HealpixSlicer(
        nside=nside,
        lat_col=colmap["alt"],
        lon_col=colmap["az"],
        lat_lon_deg=colmap["raDecDeg"],
        use_cache=False,
    )
    metric = metrics.CountMetric(colmap["mjd"], metric_name=metric_name)

    return colmap, slicer, metric


def altazHealpix(
    colmap=None,
    run_name="run name",
    extraSql=None,
    extraInfoLabel=None,
    metric_name="NVisits Alt/Az",
):
    """
    Generate a set of metrics measuring the number visits as a function
    of alt/az plotted on a HealpixSkyMap.

    Parameters
    ----------
    colmap : `dict`, optional
        A dictionary with a mapping of column names.
    run_name : `str`, optional
        The name of the simulated survey.
    extraSql : `str`, optional
        Additional constraint to add to any sql constraints.
    extraInfoLabel : `str`, optional
        Additional info_label to add before any below (i.e. "WFD").
    metric_name : `str`, optional
        Unique name to assign to metric

    Returns
    -------
    metric_bundleDict : `dict` of {`str`: `maf.MetricBundle`}
    """

    colmap, slicer, metric = basicSetup(metric_name=metric_name, colmap=colmap)

    # Set up basic all and per filter sql constraints.
    filterlist, colors, orders, sqls, info_label = filter_list(
        all=True, extra_sql=extraSql, extra_info_label=extraInfoLabel
    )

    bundleList = []

    plotDict = {"rot": (90, 90, 90), "flip": "geo"}
    plotFunc = plots.HealpixSkyMap()

    for f in filterlist:
        if f == "all":
            subgroup = "All Observations"
        else:
            subgroup = "Per filter"
        displayDict = {
            "group": "Alt/Az",
            "order": orders[f],
            "subgroup": subgroup,
            "caption": "Pointing History on the alt-az sky (zenith center) for filter %s" % f,
        }
        bundle = mb.MetricBundle(
            metric,
            slicer,
            sqls[f],
            plot_dict=plotDict,
            run_name=run_name,
            info_label=info_label[f],
            plot_funcs=[plotFunc],
            display_dict=displayDict,
        )
        bundleList.append(bundle)

    for b in bundleList:
        b.set_run_name(run_name)
    return mb.make_bundles_dict_from_list(bundleList)


def altazLambert(
    colmap=None,
    runName="opsim",
    extraSql=None,
    extraInfoLabel=None,
    metric_name="Nvisits as function of Alt/Az",
):
    """
    Generate a set of metrics measuring the number visits as a
    function of alt/az plotted on a LambertSkyMap.

    Parameters
    ----------
    colmap : `dict`, optional
        A dictionary with a mapping of column names.
    runName : `str`, optional
        The name of the simulated survey.
    extraSql : `str`, optional
        Additional constraint to add to any sql constraints.
    extraInfoLabel : `str`, optional
        Additional info_label to add before any below (i.e. "WFD").
    metric_name : `str`, optional
        Unique name to assign to metric

    Returns
    -------
    metric_bundleDict : `dict` of {`str`: `maf.MetricBundle`}
    """

    colmap, slicer, metric = basicSetup(metric_name=metric_name, colmap=colmap)

    # Set up basic all and per filter sql constraints.
    filterlist, colors, orders, sqls, info_label = filter_list(
        all=True, extra_sql=extraSql, extra_info_label=extraInfoLabel
    )

    bundleList = []

    plotFunc = plots.LambertSkyMap()

    for f in filterlist:
        if f == "all":
            subgroup = "All Observations"
        else:
            subgroup = "Per filter"
        displayDict = {
            "group": "Alt/Az",
            "order": orders[f],
            "subgroup": subgroup,
            "caption": "Alt/Az pointing distribution for filter %s" % f,
        }
        bundle = mb.MetricBundle(
            metric,
            slicer,
            sqls[f],
            run_name=runName,
            info_label=info_label[f],
            plot_funcs=[plotFunc],
            display_dict=displayDict,
        )
        bundleList.append(bundle)

    for b in bundleList:
        b.set_run_name(runName)
    return mb.make_bundles_dict_from_list(bundleList)
