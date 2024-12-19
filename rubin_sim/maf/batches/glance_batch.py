__all__ = ("glanceBatch",)

import warnings

from rubin_scheduler.scheduler.utils import EuclidOverlapFootprint
from rubin_scheduler.utils import ddf_locations

import rubin_sim.maf.metric_bundles as metric_bundles
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.plots as plots
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.stackers as stackers

from .col_map_dict import col_map_dict
from .common import standard_summary
from .hourglass_batch import hourglassPlots
from .slew_batch import slewBasics


def glanceBatch(
    colmap=None,
    run_name="run_name",
    nside=64,
    filternames=("u", "g", "r", "i", "z", "y"),
    nyears=10,
    pairnside=32,
    sql_constraint=None,
    slicer_camera="LSST",
):
    """Generate a handy set of metrics that give a quick overview
    of how well a survey performed.
    This is a meta-set of other batches, to some extent.

    Parameters
    ----------
    colmap : `dict`, optional
        A dictionary with a mapping of column names.
    run_name : `str`, optional
        The name of the simulated survey.
    nside : `int`, optional
        The nside for the healpix slicers.
    filternames : `list` of `str`, optional
        The list of individual filters to use when running metrics.
        There is always an all-visits version of the metrics run as well.
    nyears : `int`, optional
        How many years to attempt to make hourglass plots for
    pairnside : `int`, optional
        nside to use for the pair fraction metric
        (it's slow, so nice to use lower resolution)
    sql_constraint : `str` or None, optional
        Additional SQL constraint to apply to all metrics.
    slicer_camera : `str`
        Sets which spatial slicer to use. options are 'LSST' and 'ComCam'

    Returns
    -------
    metric_bundleDict : `dict` of `maf.MetricBundle`
    """
    if isinstance(colmap, str):
        raise ValueError("colmap must be a dictionary, not a string")

    if colmap is None:
        colmap = col_map_dict()

    bundle_list = []

    if sql_constraint is None:
        sqlC = ""
    else:
        sqlC = "(%s) and" % sql_constraint

    if slicer_camera == "LSST":
        spatial_slicer = slicers.HealpixSlicer
    elif slicer_camera == "ComCam":
        spatial_slicer = slicers.HealpixComCamSlicer
    else:
        raise ValueError("Camera must be LSST or Comcam")

    sql_per_filt = ['%s %s="%s"' % (sqlC, colmap["filter"], filtername) for filtername in filternames]
    sql_per_and_all_filters = [sql_constraint] + sql_per_filt

    standardStats = standard_summary()
    subsetPlots = [plots.HealpixSkyMap(), plots.HealpixHistogram()]

    # Super basic things
    displayDict = {"group": "Basic Stats", "order": 1}
    sql = sql_constraint
    slicer = slicers.UniSlicer()
    # Length of Survey
    metric = metrics.FullRangeMetric(col=colmap["mjd"], metric_name="Length of Survey (days)")
    bundle = metric_bundles.MetricBundle(metric, slicer, sql, display_dict=displayDict)
    bundle_list.append(bundle)

    # Total number of filter changes
    metric = metrics.NChangesMetric(col=colmap["filter"], order_by=colmap["mjd"])
    bundle = metric_bundles.MetricBundle(metric, slicer, sql, display_dict=displayDict)
    bundle_list.append(bundle)

    # Total open shutter fraction
    metric = metrics.OpenShutterFractionMetric(
        slew_time_col=colmap["slewtime"],
        exp_time_col=colmap["exptime"],
        visit_time_col=colmap["visittime"],
    )
    bundle = metric_bundles.MetricBundle(metric, slicer, sql, display_dict=displayDict)
    bundle_list.append(bundle)

    # Total effective exposure time
    metric = metrics.MeanMetric(col="t_eff")
    teff_stacker = stackers.TeffStacker(normed=True)
    for sql in sql_per_and_all_filters:
        bundle = metric_bundles.MetricBundle(
            metric, slicer, sql, stacker_list=[teff_stacker], display_dict=displayDict
        )
        bundle_list.append(bundle)

    # Number of observations, all and each filter
    metric = metrics.CountMetric(col=colmap["mjd"], metric_name="Number of Exposures")
    for sql in sql_per_and_all_filters:
        bundle = metric_bundles.MetricBundle(metric, slicer, sql, display_dict=displayDict)
        bundle_list.append(bundle)

    # The alt/az plots of all the pointings
    slicer = spatial_slicer(
        nside=nside,
        lat_col=colmap["alt"],
        lon_col=colmap["az"],
        lat_lon_deg=colmap["raDecDeg"],
        use_cache=False,
    )
    metric = metrics.CountMetric(colmap["mjd"], metric_name="Nvisits as function of Alt/Az")
    plotFuncs = [plots.LambertSkyMap()]

    plotDict = {"norm": "log"}
    for sql in sql_per_and_all_filters:
        bundle = metric_bundles.MetricBundle(
            metric,
            slicer,
            sql,
            plot_funcs=plotFuncs,
            display_dict=displayDict,
            plot_dict=plotDict,
        )
        bundle_list.append(bundle)

    # alt az of long gaps
    sql = "scheduler_note = 'long'"
    metric = metrics.CountMetric(colmap["mjd"], metric_name="Nvisits long")
    bundle = metric_bundles.MetricBundle(
        metric,
        slicer,
        sql,
        plot_funcs=plotFuncs,
        display_dict=displayDict,
        plot_dict=plotDict,
    )
    bundle_list.append(bundle)

    sql = "scheduler_note like 'blob_long%'"
    metric = metrics.CountMetric(colmap["mjd"], metric_name="Nvisits blob long")
    bundle = metric_bundles.MetricBundle(
        metric,
        slicer,
        sql,
        plot_funcs=plotFuncs,
        display_dict=displayDict,
        plot_dict=plotDict,
    )
    bundle_list.append(bundle)

    sql = "scheduler_note like '%neo%' or scheduler_note like '%near_sun%'"
    metric = metrics.CountMetric(colmap["mjd"], metric_name="Nvisits twilight near sun")
    bundle = metric_bundles.MetricBundle(
        metric,
        slicer,
        sql,
        plot_funcs=plotFuncs,
        display_dict=displayDict,
        plot_dict=plotDict,
    )
    bundle_list.append(bundle)

    # alt,az pf ToO

    sql = "scheduler_note like 'ToO%'"
    metric = metrics.CountMetric(colmap["mjd"], metric_name="Nvisits long")
    bundle = metric_bundles.MetricBundle(
        metric,
        slicer,
        sql,
        plot_funcs=plotFuncs,
        display_dict=displayDict,
        plot_dict=plotDict,
    )
    bundle_list.append(bundle)

    # Things to check per night
    # Open Shutter per night
    displayDict = {"group": "Pointing Efficency", "order": 2}
    slicer = slicers.OneDSlicer(slice_col_name=colmap["night"], bin_size=1, bin_min=-0.5)
    metric = metrics.OpenShutterFractionMetric(
        slew_time_col=colmap["slewtime"],
        exp_time_col=colmap["exptime"],
        visit_time_col=colmap["visittime"],
    )
    sql = sql_constraint
    bundle = metric_bundles.MetricBundle(
        metric, slicer, sql, summary_metrics=standardStats, display_dict=displayDict
    )
    bundle_list.append(bundle)

    # Number of filter changes per night
    slicer = slicers.OneDSlicer(slice_col_name=colmap["night"], bin_size=1, bin_min=-0.5)
    metric = metrics.NChangesMetric(
        col=colmap["filter"], order_by=colmap["mjd"], metric_name="Filter Changes"
    )
    bundle = metric_bundles.MetricBundle(
        metric, slicer, sql, summary_metrics=standardStats, display_dict=displayDict
    )
    bundle_list.append(bundle)

    # A few basic maps
    # Number of observations, coadded depths
    extended_stats = standardStats.copy()
    extended_stats.append(metrics.AreaSummaryMetric(decreasing=True, metric_name="top18k"))
    extended_stats.append(metrics.PercentileMetric(col="metricdata", percentile=10))
    displayDict = {"group": "Basic Maps", "order": 3}
    slicer = spatial_slicer(
        nside=nside,
        lat_col=colmap["dec"],
        lon_col=colmap["ra"],
        lat_lon_deg=colmap["raDecDeg"],
    )
    metric = metrics.CountMetric(col=colmap["mjd"])
    plotDict = {"percentile_clip": 95.0}
    for sql in sql_per_and_all_filters:
        bundle = metric_bundles.MetricBundle(
            metric,
            slicer,
            sql,
            summary_metrics=extended_stats,
            display_dict=displayDict,
            plot_dict=plotDict,
        )
        bundle_list.append(bundle)

    metric = metrics.Coaddm5Metric(m5_col=colmap["fiveSigmaDepth"])
    for sql in sql_per_and_all_filters:
        bundle = metric_bundles.MetricBundle(
            metric,
            slicer,
            sql,
            summary_metrics=extended_stats,
            display_dict=displayDict,
        )
        bundle_list.append(bundle)

    # Let's look at two years
    displayDict = {"group": "Roll Check", "order": 1}
    rolling_metrics = []
    rolling_metrics.append(metrics.CountMetric(col=colmap["mjd"], metric_name="Year1.0Count"))
    rolling_metrics.append(metrics.CountMetric(col=colmap["mjd"], metric_name="Year2.5Count"))
    rolling_metrics.append(metrics.CountMetric(col=colmap["mjd"], metric_name="Year3.5Count"))
    rolling_sqls = []
    rolling_sqls.append("night < 365.25")
    rolling_sqls.append("night > %f and night < %f" % (365.25 * 2.5, 365.25 * 3.5))
    rolling_sqls.append("night > %f and night < %f" % (365.25 * 3.5, 365.25 * 4.5))
    for metric, sql in zip(rolling_metrics, rolling_sqls):
        bundle = metric_bundles.MetricBundle(
            metric,
            slicer,
            sql,
            summary_metrics=extended_stats,
            plot_dict=plotDict,
            display_dict=displayDict,
        )
        bundle_list.append(bundle)

    # Make a cumulative plot of a WFD spot
    sql = "scheduler_note not like '%NEO%' and scheduler_note not like '%near_sun%'"
    uslicer = slicers.UserPointsSlicer(ra=0, dec=-20)
    metric = metrics.CumulativeMetric()
    metricb = metric_bundles.MetricBundle(
        metric,
        uslicer,
        sql,
        plot_funcs=[plots.XyPlotter()],
        run_name=run_name,
        display_dict=displayDict,
    )
    metricb.summary_metrics = []
    bundle_list.append(metricb)

    # Checking a few basic science things
    # Maybe check astrometry, observation pairs, SN
    plotDict = {"percentile_clip": 95.0}
    displayDict = {"group": "Science", "subgroup": "Astrometry", "order": 4}

    stackerList = []
    stacker = stackers.ParallaxFactorStacker(
        ra_col=colmap["ra"],
        dec_col=colmap["dec"],
        degrees=colmap["raDecDeg"],
        date_col=colmap["mjd"],
    )
    stackerList.append(stacker)

    astrom_stats = [
        metrics.AreaSummaryMetric(decreasing=False, metric_name="best18k"),
        metrics.PercentileMetric(col="metricdata", percentile=90),
    ]
    displayDict["caption"] = r"Parallax precision of an $r=20$ flat SED star"
    metric = metrics.ParallaxMetric(
        m5_col=colmap["fiveSigmaDepth"],
        filter_col=colmap["filter"],
        seeing_col=colmap["seeingGeom"],
    )
    sql = sql_constraint
    bundle = metric_bundles.MetricBundle(
        metric,
        slicer,
        sql,
        plot_funcs=subsetPlots,
        display_dict=displayDict,
        stacker_list=stackerList,
        plot_dict=plotDict,
        summary_metrics=astrom_stats,
    )
    bundle_list.append(bundle)
    displayDict["caption"] = r"Proper motion precision of an $r=20$ flat SED star"
    metric = metrics.ProperMotionMetric(
        m5_col=colmap["fiveSigmaDepth"],
        mjd_col=colmap["mjd"],
        filter_col=colmap["filter"],
        seeing_col=colmap["seeingGeom"],
    )
    bundle = metric_bundles.MetricBundle(
        metric,
        slicer,
        sql,
        plot_funcs=subsetPlots,
        display_dict=displayDict,
        plot_dict=plotDict,
        summary_metrics=astrom_stats,
    )
    bundle_list.append(bundle)

    # Solar system stuff
    displayDict["caption"] = "Fraction of observations that are in pairs"
    displayDict["subgroup"] = "Solar System"

    sql = '%s (filter="g" or filter="r" or filter="i")' % sqlC
    pairSlicer = slicers.HealpixSlicer(
        nside=pairnside,
        lat_col=colmap["dec"],
        lon_col=colmap["ra"],
        lat_lon_deg=colmap["raDecDeg"],
    )
    metric = metrics.PairFractionMetric(mjd_col=colmap["mjd"])
    bundle = metric_bundles.MetricBundle(
        metric, pairSlicer, sql, plot_funcs=subsetPlots, display_dict=displayDict
    )
    bundle_list.append(bundle)

    # stats from the scheduler_note column
    if "scheduler_note" in colmap.keys():
        displayDict = {"group": "Basic Stats", "subgroup": "Percent root stats"}
        metric = metrics.StringCountMetric(
            col=colmap["scheduler_note_root"], percent=True, metric_name="Percents", clip_end=False
        )
        sql = ""
        slicer = slicers.UniSlicer()
        bundle = metric_bundles.MetricBundle(metric, slicer, sql, display_dict=displayDict)
        bundle_list.append(bundle)
        displayDict["subgroup"] = "Count root Stats"
        metric = metrics.StringCountMetric(
            col=colmap["scheduler_note_root"], metric_name="Counts", clip_end=False
        )
        bundle = metric_bundles.MetricBundle(metric, slicer, sql, display_dict=displayDict)
        bundle_list.append(bundle)

    # For pairs and twilights
    if "scheduler_note" in colmap.keys():
        displayDict = {"group": "Basic Stats", "subgroup": "Percent stats"}
        metric = metrics.StringCountMetric(
            col=colmap["scheduler_note"], percent=True, metric_name="Percents", clip_end=False
        )
        sql = (
            "scheduler_note like 'pair%%' or scheduler_note like 'twilight%%' or scheduler_note like 'blob%%'"
        )
        slicer = slicers.UniSlicer()
        bundle = metric_bundles.MetricBundle(metric, slicer, sql, display_dict=displayDict)
        bundle_list.append(bundle)
        displayDict["subgroup"] = "Count Stats"
        metric = metrics.StringCountMetric(col=colmap["scheduler_note"], metric_name="Counts", clip_end=False)
        bundle = metric_bundles.MetricBundle(metric, slicer, sql, display_dict=displayDict)
        bundle_list.append(bundle)

    # DDF progress
    ddf_surveys = ddf_locations()
    displayDict["group"] = "DDF"
    displayDict["subgroup"] = ""
    for ddf in ddf_surveys:
        label = ddf.replace("DD:", "")
        sql = 'scheduler_note like "%s%%"' % ("DD:" + label)
        slicer = slicers.UniSlicer()
        metric = metrics.CumulativeMetric()
        metricb = metric_bundles.MetricBundle(
            metric,
            slicer,
            sql,
            plot_funcs=[plots.XyPlotter()],
            run_name=run_name,
            display_dict=displayDict,
        )
        metricb.summary_metrics = []
        bundle_list.append(metricb)

    # Add a sky saturation check
    displayDict = {}
    displayDict["group"] = "Basic Stats"
    displayDict["subgroup"] = "Saturation"
    sql = ""
    metric = metrics.SkySaturationMetric()
    summary = metrics.SumMetric()
    slicer = slicers.UniSlicer()
    bundle_list.append(
        metric_bundles.MetricBundle(metric, slicer, sql, summary_metrics=summary, display_dict=displayDict)
    )

    benchmarkArea = 18000
    benchmarkNvisits = 825
    minNvisits = 750
    displayDict = {"group": "SRD", "subgroup": "FO metrics", "order": 0}

    # Configure the count metric which is what is used for f0 slicer.
    metric = metrics.CountExplimMetric(metric_name="fO")
    plotDict = {
        "xlabel": "Number of Visits",
        "asky": benchmarkArea,
        "n_visits": minNvisits,
        "x_min": 0,
        "x_max": 1500,
    }
    summaryMetrics = [
        metrics.FOArea(
            nside=nside,
            norm=False,
            metric_name="fOArea",
            asky=benchmarkArea,
            n_visit=benchmarkNvisits,
        ),
        metrics.FOArea(
            nside=nside,
            norm=True,
            metric_name="fOArea/benchmark",
            asky=benchmarkArea,
            n_visit=benchmarkNvisits,
        ),
        metrics.FONv(
            nside=nside,
            norm=False,
            metric_name="fONv",
            asky=benchmarkArea,
            n_visit=benchmarkNvisits,
        ),
        metrics.FONv(
            nside=nside,
            norm=True,
            metric_name="fONv/benchmark",
            asky=benchmarkArea,
            n_visit=benchmarkNvisits,
        ),
        metrics.FOArea(
            nside=nside,
            norm=False,
            metric_name=f"fOArea_{minNvisits}",
            asky=benchmarkArea,
            n_visit=minNvisits,
        ),
    ]
    caption = "The FO metric evaluates the overall efficiency of observing. "
    caption += (
        "foNv: out of %.2f sq degrees, the area receives at least X and a median of Y visits "
        "(out of %d, if compared to benchmark). " % (benchmarkArea, benchmarkNvisits)
    )
    caption += (
        "fOArea: this many sq deg (out of %.2f sq deg if compared "
        "to benchmark) receives at least %d visits. " % (benchmarkArea, benchmarkNvisits)
    )
    displayDict["caption"] = caption
    slicer = slicers.HealpixSlicer(nside=nside)
    bundle = metric_bundles.MetricBundle(
        metric,
        slicer,
        "",
        plot_dict=plotDict,
        display_dict=displayDict,
        summary_metrics=summaryMetrics,
        plot_funcs=[plots.FOPlot()],
    )
    bundle_list.append(bundle)

    # check that we have coverage for the first year
    displayDict = {}
    displayDict["group"] = "Year 1"
    displayDict["subgroup"] = "Coverage"
    nside_foot = 32
    slicer = slicers.HealpixSlicer(nside=nside_foot, badval=0)
    sky = EuclidOverlapFootprint(nside=nside_foot, smc_radius=4, lmc_radius=6)
    footprints_hp_array, labels = sky.return_maps()
    for filtername in filternames:
        sql = "filter='%s' and night < 365" % filtername
        metric = metrics.CountMetric(col="night", metric_name="N year 1")
        summary_stat = metrics.FootprintFractionMetric(
            footprint=footprints_hp_array[filtername],
            n_min=3,
        )
        bundle = metric_bundles.MetricBundle(
            metric,
            slicer,
            sql,
            display_dict=displayDict,
            summary_metrics=summary_stat,
            plot_funcs=subsetPlots,
            plot_dict={"color_max": 10},
        )
        bundle_list.append(bundle)

    # Some ToO stats
    displayDict = {"group": "ToO", "order": 1}
    slicer = spatial_slicer(
        nside=nside,
        lat_col=colmap["dec"],
        lon_col=colmap["ra"],
        lat_lon_deg=colmap["raDecDeg"],
    )
    for filtername in filternames:
        sql = "filter='%s' and scheduler_note like 'ToO%%'" % filtername
        metric = metrics.CountMetric(col=colmap["mjd"], metric_name="N ToO")
        bundle = metric_bundles.MetricBundle(
            metric,
            slicer,
            sql,
            display_dict=displayDict,
            summary_metrics=extended_stats,
            plot_funcs=subsetPlots,
            plot_dict={},
        )
        bundle_list.append(bundle)

    too_sqls = ["scheduler_note like 'ToO, %" + "t%i'" % hour for hour in [0, 1, 2, 4, 24, 48]] + [
        "scheduler_note like 'ToO, %'"
    ]
    slicer = slicers.UniSlicer()
    for sql in too_sqls:
        metric = metrics.CountMetric(col="night")
        bundle = metric_bundles.MetricBundle(
            metric,
            slicer,
            sql,
            display_dict=displayDict,
        )
        bundle_list.append(bundle)

        metric = metrics.CountUniqueMetric(col="night")
        bundle = metric_bundles.MetricBundle(
            metric,
            slicer,
            sql,
            display_dict=displayDict,
        )
        bundle_list.append(bundle)

    for b in bundle_list:
        b.set_run_name(run_name)

    bd = metric_bundles.make_bundles_dict_from_list(bundle_list)

    # Add hourglass plots.
    hrDict = hourglassPlots(colmap=colmap, runName=run_name, nyears=nyears, extraSql=sql_constraint)
    bd.update(hrDict)

    # Add basic slew stats.
    try:
        slewDict = slewBasics(colmap=colmap, run_name=run_name)
        bd.update(slewDict)
    except KeyError as e:
        warnings.warn("Could not add slew stats: missing required key %s from colmap" % (e))

    return bd
