"""Run the hourglass metric.
"""
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.metric_bundles as mb
from .col_map_dict import col_map_dict

__all__ = ["hourglassPlots"]


def hourglassPlots(
    colmap=None, runName="opsim", nyears=10, extraSql=None, extraInfoLabel=None
):
    """Run the hourglass metric, for each individual year.

    Parameters
    ----------
    colmap : dict, optional
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    run_name : str, optional
        The name of the simulated survey. Default is "opsim".
    nyears : int (10), optional
        How many years to attempt to make hourglass plots for. Default is 10.
    extraSql : str, optional
        Add an extra sql constraint before running metrics. Default None.
    extraInfoLabel : str, optional
        Add an extra piece of info_label before running metrics. Default None.
    """
    if colmap is None:
        colmap = col_map_dict("opsimV4")
    bundleList = []

    sql = ""
    info_label = ""
    # Add additional sql constraint (such as wfdWhere) and info_label, if provided.
    if (extraSql is not None) and (len(extraSql) > 0):
        sql = extraSql
        if extraInfoLabel is None:
            info_label = extraSql.replace("filter =", "").replace("filter=", "")
            info_label = info_label.replace('"', "").replace("'", "")
    if extraInfoLabel is not None:
        info_label = extraInfoLabel

    years = list(range(nyears + 1))
    displayDict = {"group": "Hourglass"}
    for year in years[1:]:
        displayDict["subgroup"] = "Year %d" % year
        displayDict["caption"] = (
            "Visualization of the filter usage of the telescope. "
            "The black wavy line indicates lunar phase; the red and blue "
            "solid lines indicate nautical and civil twilight."
        )
        sqlconstraint = "night > %i and night <= %i" % (
            365.25 * (year - 1),
            365.25 * year,
        )
        if len(sql) > 0:
            sqlconstraint = "(%s) and (%s)" % (sqlconstraint, sql)
        md = info_label + " year %i-%i" % (year - 1, year)
        slicer = slicers.HourglassSlicer()
        metric = metrics.HourglassMetric(
            night_col=colmap["night"], mjd_col=colmap["mjd"], metric_name="Hourglass"
        )
        bundle = mb.MetricBundle(
            metric,
            slicer,
            constraint=sqlconstraint,
            info_label=md,
            display_dict=displayDict,
        )
        bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.set_run_name(runName)
    return mb.make_bundles_dict_from_list(bundleList)
