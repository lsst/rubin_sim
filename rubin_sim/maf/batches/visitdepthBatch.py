"""Sets of metrics to look at general sky coverage - nvisits/coadded depth/Teff.
"""
import numpy as np
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.stackers as stackers
import rubin_sim.maf.plots as plots
import rubin_sim.maf.metricBundles as mb
import rubin_sim.maf.utils as mafUtils
from .colMapDict import ColMapDict, getColMap
from .common import standardSummary, filterList, radecCols, combineInfoLabels

__all__ = ["nvisitsM5Maps", "tEffMetrics", "nvisitsPerNight", "nvisitsPerSubset"]


def nvisitsM5Maps(
    colmap=None,
    runName="opsim",
    extraSql=None,
    extraInfoLabel=None,
    slicer=None,
    runLength=10.0,
):
    """Generate maps of the number of visits and coadded depth (with and without dust extinction)
    in all bands and per filter.

    Parameters
    ----------
    colmap : `dict`, optional
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    runName : `str`, optional
        The name of the simulated survey. Default is "opsim".
    extraSql : `str`, optional
        Additional constraint to add to any sql constraints (e.g. 'propId=1' or 'fieldID=522').
        Default None, for no additional constraints.
    extraInfoLabel : `str`, optional
        Additional info_label to add before any below (i.e. "WFD").  Default is None.
    slicer :  `rubin_sim.maf.slicer` or None, optional
        Optionally, use something other than an nside=64 healpix slicer
    runLength : `float`, optional
        Length of the simulated survey, for scaling values for the plot limits.
        Default 10.

    Returns
    -------
    metricBundleDict
    """
    if colmap is None:
        colmap = ColMapDict("opsimV4")
    bundleList = []

    subgroup = extraInfoLabel
    if subgroup is None:
        subgroup = "All visits"

    raCol, decCol, degrees, ditherStacker, ditherMeta = radecCols(None, colmap, None)
    extraInfoLabel = combineInfoLabels(extraInfoLabel, ditherMeta)
    # Set up basic all and per filter sql constraints.
    filterlist, colors, orders, sqls, info_label = filterList(
        all=True, extraSql=extraSql, extraInfoLabel=extraInfoLabel
    )
    # Set up some values to make nicer looking plots.
    benchmarkVals = mafUtils.scaleBenchmarks(runLength, benchmark="design")
    # Check that nvisits is not set to zero (for very short run length).
    for f in benchmarkVals["nvisits"]:
        if benchmarkVals["nvisits"][f] == 0:
            print("Updating benchmark nvisits value in %s to be nonzero" % (f))
            benchmarkVals["nvisits"][f] = 1
    benchmarkVals["coaddedDepth"] = mafUtils.calcCoaddedDepth(
        benchmarkVals["nvisits"], benchmarkVals["singleVisitDepth"]
    )
    # Scale the nvisit ranges for the runLength.
    nvisitsRange = {
        "u": [20, 80],
        "g": [50, 150],
        "r": [100, 250],
        "i": [100, 250],
        "z": [100, 300],
        "y": [100, 300],
        "all": [700, 1200],
    }
    scale = runLength / 10.0
    for f in nvisitsRange:
        for i in [0, 1]:
            nvisitsRange[f][i] = int(np.floor(nvisitsRange[f][i] * scale))

    # Generate Nvisit maps in all and per filters
    displayDict = {"group": "Nvisits Maps", "subgroup": subgroup}
    metric = metrics.CountMetric(colmap["mjd"], metricName="NVisits", units="")

    if slicer is None:
        slicer = slicers.HealpixSlicer(
            nside=64, latCol=decCol, lonCol=raCol, latLonDeg=degrees
        )
        slicerDust = slicers.HealpixSlicer(
            nside=64, latCol=decCol, lonCol=raCol, latLonDeg=degrees, usecache=False
        )

    for f in filterlist:
        sql = sqls[f]
        displayDict["caption"] = "Number of visits per healpix in %s." % info_label[f]
        displayDict["order"] = orders[f]
        binsize = 2
        if f == "all":
            binsize = 5
        plotDict = {
            "xMin": nvisitsRange[f][0],
            "xMax": nvisitsRange[f][1],
            "colorMin": nvisitsRange[f][0],
            "colorMax": nvisitsRange[f][1],
            "binsize": binsize,
            "color": colors[f],
        }
        bundle = mb.MetricBundle(
            metric,
            slicer,
            sql,
            info_label=info_label[f],
            stackerList=ditherStacker,
            displayDict=displayDict,
            plotDict=plotDict,
            summaryMetrics=standardSummary(),
        )
        bundleList.append(bundle)

    # Generate Coadded depth maps per filter
    displayDict = {"group": "Coadded M5 Maps", "subgroup": subgroup}
    metric = metrics.Coaddm5Metric(m5Col=colmap["fiveSigmaDepth"], metricName="CoaddM5")

    for f in filterlist:
        # Skip "all" for coadded depth.
        if f == "all":
            continue
        mag_zp = benchmarkVals["coaddedDepth"][f]
        sql = sqls[f]
        displayDict["caption"] = (
            "Coadded depth per healpix, with %s benchmark value subtracted (%.1f) "
            "in %s." % (f, mag_zp, info_label[f])
        )
        displayDict[
            "caption"
        ] += " More positive numbers indicate fainter limiting magnitudes."
        displayDict["order"] = orders[f]
        plotDict = {
            "zp": mag_zp,
            "xMin": -0.6,
            "xMax": 0.6,
            "xlabel": "coadded m5 - %.1f" % mag_zp,
            "colorMin": -0.6,
            "colorMax": 0.6,
            "color": colors[f],
        }
        bundle = mb.MetricBundle(
            metric,
            slicer,
            sql,
            info_label=info_label[f],
            stackerList=ditherStacker,
            displayDict=displayDict,
            plotDict=plotDict,
            summaryMetrics=standardSummary(),
        )
        bundleList.append(bundle)

    # Add Coadded depth maps per filter WITH extragalactic extinction added
    displayDict = {"group": "Extragalactic Coadded M5 Maps", "subgroup": subgroup}
    metric = metrics.ExgalM5(m5Col=colmap["fiveSigmaDepth"], metricName="Exgal_CoaddM5")

    for f in filterlist:
        # Skip "all" for coadded depth.
        if f == "all":
            continue
        mag_zp = benchmarkVals["coaddedDepth"][f]
        sql = sqls[f]
        displayDict["caption"] = (
            "Coadded depth per healpix for extragalactic purposes "
            "(i.e. combined with dust extinction maps), "
            "with %s benchmark value subtracted (%.1f) "
            "in %s." % (f, mag_zp, info_label[f])
        )
        displayDict[
            "caption"
        ] += " More positive numbers indicate fainter limiting magnitudes."
        displayDict["order"] = orders[f]
        plotDict = {
            "zp": mag_zp,
            "xMin": -0.6,
            "xMax": 0.6,
            "xlabel": "coadded m5 - %.1f" % mag_zp,
            "colorMin": -0.6,
            "colorMax": 0.6,
            "color": colors[f],
        }
        bundle = mb.MetricBundle(
            metric,
            slicerDust,
            sql,
            info_label=info_label[f],
            stackerList=ditherStacker,
            displayDict=displayDict,
            plotDict=plotDict,
            summaryMetrics=standardSummary(),
        )
        bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)


def tEffMetrics(
    colmap=None,
    runName="opsim",
    extraSql=None,
    extraInfoLabel=None,
    slicer=None,
):
    """Generate a series of Teff metrics. Teff total, per night, and sky maps (all and per filter).

    Parameters
    ----------
    colmap : `dict`, optional
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    runName : `str`, optional
        The name of the simulated survey. Default is "opsim".
    extraSql : `str`, optional
        Additional constraint to add to any sql constraints (e.g. 'propId=1' or 'fieldID=522').
        Default None, for no additional constraints.
    extraInfoLabel : `str`, optional
        Additional info_label to add before any below (i.e. "WFD").  Default is None.
    slicer : `rubin_sim.maf.slicer` or None, optional
        Optionally, use something other than an nside=64 healpix slicer

    Returns
    -------
    metricBundleDict
    """
    if colmap is None:
        colmap = ColMapDict("opsimV4")
    bundleList = []

    subgroup = extraInfoLabel
    if subgroup is None:
        subgroup = "All visits"

    raCol, decCol, degrees, ditherStacker, ditherMeta = radecCols(None, colmap, None)
    extraInfoLabel = combineInfoLabels(extraInfoLabel, ditherMeta)

    if slicer is not None:
        skyslicer = slicer
    else:
        skyslicer = slicers.HealpixSlicer(
            nside=64, latCol=decCol, lonCol=raCol, latLonDeg=degrees
        )

    # Set up basic all and per filter sql constraints.
    filterlist, colors, orders, sqls, info_label = filterList(
        all=True, extraSql=extraSql, extraInfoLabel=extraInfoLabel
    )
    if info_label["all"] is None:
        info_label["all"] = "All visits"

    # Total Teff and normalized Teff.
    displayDict = {"group": "T_eff Summary", "subgroup": subgroup}
    displayDict["caption"] = "Total effective time of the survey (see Teff metric)."
    displayDict["order"] = 0
    metric = metrics.TeffMetric(
        m5Col=colmap["fiveSigmaDepth"],
        filterCol=colmap["filter"],
        normed=False,
        metricName="Total Teff",
    )
    slicer = slicers.UniSlicer()
    bundle = mb.MetricBundle(
        metric,
        slicer,
        constraint=sqls["all"],
        displayDict=displayDict,
        info_label=info_label["all"],
    )
    bundleList.append(bundle)

    displayDict[
        "caption"
    ] = "Normalized total effective time of the survey (see Teff metric)."
    displayDict["order"] = 1
    metric = metrics.TeffMetric(
        m5Col=colmap["fiveSigmaDepth"],
        filterCol=colmap["filter"],
        normed=True,
        metricName="Normalized Teff",
    )
    slicer = slicers.UniSlicer()
    bundle = mb.MetricBundle(
        metric,
        slicer,
        constraint=sqls["all"],
        displayDict=displayDict,
        info_label=info_label["all"],
    )
    bundleList.append(bundle)

    # Generate Teff maps in all and per filters
    displayDict = {"group": "T_eff Maps", "subgroup": subgroup}
    if ditherMeta is not None:
        for m in info_label:
            info_label[m] = combineInfoLabels(info_label[m], ditherMeta)

    metric = metrics.TeffMetric(
        m5Col=colmap["fiveSigmaDepth"],
        filterCol=colmap["filter"],
        normed=True,
        metricName="Normalized Teff",
    )
    for f in filterlist:
        displayDict["caption"] = (
            "Normalized effective time of the survey, for %s" % info_label[f]
        )
        displayDict["order"] = orders[f]
        plotDict = {"color": colors[f]}
        bundle = mb.MetricBundle(
            metric,
            skyslicer,
            sqls[f],
            info_label=info_label[f],
            stackerList=ditherStacker,
            displayDict=displayDict,
            plotDict=plotDict,
            summaryMetrics=standardSummary(),
        )
        bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)


def nvisitsPerNight(
    colmap=None,
    runName="opsim",
    binNights=1,
    extraSql=None,
    extraInfoLabel=None,
    subgroup=None,
):
    """Count the number of visits per night through the survey.

    Parameters
    ----------
    colmap : `dict` or None, optional
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    runName : `str`, optional
        The name of the simulated survey. Default is "opsim".
    binNights : `int`, optional
        Number of nights to count in each bin. Default = 1, count number of visits in each night.
    extraSql : `str` or None, optional
        Additional constraint to add to any sql constraints (e.g. 'propId=1' or 'fieldID=522').
        Default None, for no additional constraints.
    extraInfoLabel : `str` or None, optional
        Additional info_label to add before any below (i.e. "WFD").  Default is None.
    subgroup : `str` or None, optional
        Use this for the 'subgroup' in the displayDict, instead of info_label. Default is None.

    Returns
    -------
    metricBundleDict
    """
    if colmap is None:
        colmap = ColMapDict("opsimV4")

    subgroup = subgroup
    if subgroup is None:
        subgroup = extraInfoLabel
        if subgroup is None:
            subgroup = "All visits"

    infoCaption = extraInfoLabel
    if extraInfoLabel is None:
        if extraSql is not None:
            infoCaption = extraSql
        else:
            infoCaption = "all visits"

    bundleList = []

    displayDict = {"group": "Nvisits Per Night", "subgroup": subgroup}
    displayDict["caption"] = "Number of visits per night for %s." % (infoCaption)
    displayDict["order"] = 0
    metric = metrics.CountMetric(colmap["mjd"], metricName="Nvisits")
    slicer = slicers.OneDSlicer(sliceColName=colmap["night"], binsize=binNights)
    bundle = mb.MetricBundle(
        metric,
        slicer,
        extraSql,
        info_label=infoCaption,
        displayDict=displayDict,
        summaryMetrics=standardSummary(),
    )
    bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)


def nvisitsPerSubset(
    opsdb,
    colmap=None,
    runName="opsim",
    binNights=1,
    constraint=None,
    footprintConstraint=None,
    extraInfoLabel=None,
):
    """Look at the distribution of a given sql constraint or footprint constraint's visits,
    total number and distribution over time (# per night), if possible.

    Parameters
    ----------
    opsdb : `str` or database connection
        Name of the opsim sqlite database.
    colmap : `dict` or None, optional
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    runName : `str`, optional
        The name of the simulated survey. Default is "opsim".
    binNights : `int`, optional
        Number of nights to count in each bin. Default = 1, count number of visits in each night.
    constraint : `str` or None, optional
        SQL constraint to add to all metrics.
        This would be the way to select only a given "Note" out of a simulation.
    footprintConstraint : `np.ndarray` or None, optional
        Footprint to look for visits within (and then identify via WFDlabelStacker).
        The footprint = a full length heapix array, filled with 0/1 values.
    extraInfoLabel : `str` or None, optional
        Additional info_label to add before any below (i.e. "WFD").  Default is None.

    Returns
    -------
    metricBundleDict : `dict` of `rubin_sim.maf.MetricBundle`
    """
    if colmap is None:
        colmap = getColMap(opsdb)

    bdict = {}
    bundleList = []

    if constraint is not None and len(constraint) > 0:
        if extraInfoLabel is None:
            extraInfoLabel += " %s" % constraint
        # Nvisits per night, this constraint.
        bdict.update(
            nvisitsPerNight(
                colmap=colmap,
                runName=runName,
                binNights=binNights,
                extraSql=constraint,
                extraInfoLabel=extraInfoLabel,
            )
        )
        # Nvisits total, this constraint.
        metric = metrics.CountMetric(colmap["mjd"], metricName="Nvisits")
        slicer = slicers.UniSlicer()
        displayDict = {
            "group": "Nvisit Summary",
            "subgroup": extraInfoLabel,
        }
        displayDict["caption"] = f"Total number of visits for {extraInfoLabel}."
        bundle = mb.MetricBundle(
            metric,
            slicer,
            constraint,
            info_label=extraInfoLabel,
            displayDict=displayDict,
        )
        bundleList.append(bundle)

    # Or count the total number of visits that contribute towards a given footprint
    if footprintConstraint is not None:
        # Set up a stacker to use this footprint to label visits
        if extraInfoLabel is None:
            extraInfoLabel = "Footprint"
        footprintStacker = stackers.WFDlabelStacker(
            footprint=footprintConstraint, fp_threshold=0.4, area_id_name=extraInfoLabel
        )
        metric = metrics.CountSubsetMetric(col="areaId", subset=extraInfoLabel)
        slicer = slicers.UniSlicer()
        displayDict = {
            "group": "Nvisit Summary",
            "subgroup": extraInfoLabel,
            "caption": f"Visits within footprint {extraInfoLabel}.",
        }
        bundle = mb.MetricBundle(
            metric,
            slicer,
            constraint,
            stackerList=[footprintStacker],
            info_label=extraInfoLabel,
            displayDict=displayDict,
        )
        bundleList.append(bundle)

    for b in bundleList:
        b.setRunName(runName)
    bdict.update(mb.makeBundlesDictFromList(bundleList))
    return bdict
