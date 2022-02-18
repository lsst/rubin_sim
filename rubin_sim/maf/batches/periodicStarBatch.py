import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.plots as plots
import rubin_sim.maf.metricBundles as mb
from .colMapDict import ColMapDict
from .common import filterList, radecCols, combineMetadata, standardSummary

__all__ = ["periodicStarsBatch"]


def periodicStarsBatch(
    colmap=None,
    runName="opsim",
    nside=64,
    extraSql=None,
    extraMetadata=None,
    slicer=None,
    display_group="Variables/Transients",
):
    """Generate a set of statistics about the spacing between nights with observations.

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
    extraMetadata : str or None, optional
        Additional metadata to use for all outputs.
    slicer : slicer object (None)
        Optionally use something other than a HealpixSlicer

    Returns
    -------
    metricBundleDict
    """

    if colmap is None:
        colmap = ColMapDict("fbs")

    bundleList = []

    # Set up basic per filter sql constraints.
    raCol, decCol, degrees, ditherStacker, ditherMeta = radecCols(None, colmap, None)
    filterlist, colors, orders, sqls, metadata = filterList(
        all=False, extraSql=extraSql, extraMetadata=extraMetadata
    )

    if slicer is None:
        slicer = slicers.HealpixSlicer(
            nside=nside, latCol=decCol, lonCol=raCol, latLonDeg=degrees, useCache=False
        )

    standardStats = standardSummary(withCount=False)
    healslicer = slicers.HealpixSlicer(nside=nside)
    subsetPlots = [plots.HealpixSkyMap(), plots.HealpixHistogram()]

    displayDict = {"group": display_group, "order": 0}

    # PeriodicStarModulation metric (Nina Hernischek)
    # colors for c type RRLyrae
    displayDict["subgroup"] = "Periodic Star Modulation"
    I_rrc_lmc = 18.9
    V_rrc_lmc = 19.2
    Vi = V_rrc_lmc - (2.742 * 0.08) - 18.5
    Ii = I_rrc_lmc - (1.505 * 0.08) - 18.5
    ii_rrc = Ii + 0.386 * 0.013 + 0.397  # 0.013 = (i-z)_0
    gi_rrc = ii_rrc + 1.481 * (Vi - Ii) - 0.536
    ri_rrc = (1 / 0.565) * (Vi - 0.435 * gi_rrc + 0.016)
    ui_rrc = gi_rrc + 0.575
    zi_rrc = ii_rrc - 0.013
    yi_rrc = zi_rrc
    rrc = np.array([ui_rrc, gi_rrc, ri_rrc, ii_rrc, zi_rrc, yi_rrc])
    time_intervals = (15, 30)
    distMod = (18, 19, 20, 21)
    summaryStats = [metrics.MeanMetric(), metrics.MedianMetric(), metrics.MaxMetric()]
    s = slicers.HealpixSlicer(nside=8)
    sql = "night < 365*2"
    for time_interval in time_intervals:
        for dM in distMod:
            displayDict["caption"] = (
                "Periodic star modulation metric, evaluates the likelihood of "
                "measuring variation in an RRLyrae periodic variable. "
                "Evaluated based on the first two years of the LSST survey data only. "
                f"Searching time interval of {time_interval} and distance modulus {dM}."
            )
            m = metrics.PeriodicStarModulationMetric(
                period=0.3,
                amplitude=0.3,
                random_phase=True,
                time_interval=time_interval,
                nMonte=100,
                periodTol=0.002,
                ampTol=0.01,
                means=rrc + dM,
                magTol=0.01,
                nBands=3,
            )
            bundle = mb.MetricBundle(
                m,
                s,
                sql,
                displayDict=displayDict,
                runName=runName,
                summaryMetrics=summaryStats,
                metadata=f"dm {dM} interval {time_interval} RRc",
            )
            bundleList.append(bundle)

    # PulsatingStarRecovery metric (to be added; Marcella)

    # our periodic star metrics
    displayDict["subgroup"] = "Periodic Stars"
    for period in [0.5, 1, 2]:
        for magnitude in [21.0, 24.0]:
            amplitudes = [0.05, 0.1, 1.0]
            periods = [period] * len(amplitudes)
            starMags = [magnitude] * len(amplitudes)

            plotDict = {"nTicks": 3, "colorMin": 0, "colorMax": 3, "xMin": 0, "xMax": 3}
            metadata = combineMetadata(
                "P_%.1f_Mag_%.0f_Amp_0.05-0.1-1" % (period, magnitude), extraMetadata
            )
            sql = extraSql
            displayDict["caption"] = (
                "Metric evaluates if a periodic signal of period %.1f days could "
                "be detected for an r=%i star. A variety of amplitudes of periodicity "
                "are tested: [1, 0.1, and 0.05] mag amplitudes, which correspond to "
                "metric values of [1, 2, or 3]. " % (period, magnitude)
            )
            metric = metrics.PeriodicDetectMetric(
                periods=periods,
                starMags=starMags,
                amplitudes=amplitudes,
                metricName="PeriodDetection",
            )
            bundle = mb.MetricBundle(
                metric,
                healslicer,
                sql,
                metadata=metadata,
                displayDict=displayDict,
                plotDict=plotDict,
                plotFuncs=subsetPlots,
                summaryMetrics=standardStats,
            )
            bundleList.append(bundle)
            displayDict["order"] += 1

    # Add the kuiper metric here too

    return mb.makeBundlesDictFromList(bundleList)
