import numpy as np
import healpy as hp
from rubin_sim.scheduler.surveys import generate_dd_surveys
from rubin_sim.utils import _hpid2RaDec, _angularSeparation
import rubin_sim.maf as maf

__all__ = ["ddfBatch"]


def ddfBatch(runName="opsim", nside=512, radius=4.0):

    radius = np.radians(radius)
    bundle_list = []
    sql = ""
    ra, dec = _hpid2RaDec(nside, np.arange(hp.nside2npix(nside)))

    ddf_surveys = generate_dd_surveys()

    summary_stats = [maf.MeanMetric(), maf.MedianMetric(), maf.SumMetric()]
    standardSummary = [
        maf.MeanMetric(),
        maf.RmsMetric(),
        maf.MedianMetric(),
        maf.MaxMetric(),
        maf.MinMetric(),
        maf.NoutliersNsigmaMetric(metricName="N(+3Sigma)", nSigma=3),
        maf.NoutliersNsigmaMetric(metricName="N(-3Sigma)", nSigma=-3.0),
    ]

    filternames = 'ugrizy'
    filter_all_sql = ['filter="%s"' % filtername for filtername in filternames]
    filter_all_sql.append('')

    depth_stats = [maf.MedianMetric()]
    plotFuncs = [maf.HealpixSkyMap()]

    for ddf in ddf_surveys:
        label = ddf.survey_name.replace("DD:", "")
        plotDict = {
            "visufunc": hp.gnomview,
            "rot": (np.degrees(np.mean(ddf.ra)), np.degrees(np.mean(ddf.dec)), 0),
            "xsize": 500,
        }
        dist = _angularSeparation(ra, dec, np.mean(ddf.ra), np.mean(ddf.dec))
        good = np.where(dist <= radius)[0]

        # Number of SNe
        slicer = maf.HealpixSubsetSlicer(nside, good)
        metric = maf.metrics.SNNSNMetric(verbose=False, metricName="%s, SNe" % label)
        bundle_list.append(
            maf.MetricBundle(
                metric,
                slicer,
                sql,
                plotDict=plotDict,
                plotFuncs=plotFuncs,
                summaryMetrics=summary_stats,
            )
        )

        # Strong lensed SNe
        metric = maf.SNSLMetric(metricName="SnL_%s" % label)
        bundle_list.append(
            maf.MetricBundle(
                metric,
                slicer,
                sql,
                plotDict=plotDict,
                plotFuncs=plotFuncs,
                summaryMetrics=summary_stats,
            )
        )

        # Number of QSOs in each band
        zmin = 0.3
        extinction_cut = 1.0
        for f in "ugrizy":
            summaryMetrics = [maf.SumMetric(metricName="Total QSO")]
            metric = maf.QSONumberCountsMetric(
                f,
                units="mag",
                extinction_cut=extinction_cut,
                qlf_module="Shen20",
                qlf_model="A",
                SED_model="Richards06",
                zmin=zmin,
                zmax=None,
                metricName="QSO_N_%s_%s" % (f, label),
            )
            bundle_list.append(
                maf.MetricBundle(
                    metric,
                    slicer,
                    sql,
                    plotDict=plotDict,
                    plotFuncs=plotFuncs,
                    summaryMetrics=summaryMetrics,
                )
            )

        # Coadded depth per filter, and count per filter
        for filtername in "ugrizy":
            metric = maf.Coaddm5Metric(
                metricName="%s, 5-sigma %s" % (label, filtername)
            )
            sql = 'filter="%s"' % filtername
            bundle_list.append(
                maf.MetricBundle(
                    metric,
                    slicer,
                    sql,
                    plotDict=plotDict,
                    runName=runName,
                    plotFuncs=plotFuncs,
                    summaryMetrics=depth_stats,
                )
            )

            metric = maf.CountMetric(
                col="night", units="#", metricName="%s, Count %s" % (label, filtername)
            )
            sql = 'filter="%s"' % filtername
            bundle_list.append(
                maf.MetricBundle(
                    metric,
                    slicer,
                    sql,
                    plotDict=plotDict,
                    runName=runName,
                    plotFuncs=plotFuncs,
                    summaryMetrics=depth_stats,
                )
            )
        # Count over all filter
        metric = maf.CountMetric(
            col="night", units="#", metricName="%s, Count all" % (label)
        )
        sql = ""
        bundle_list.append(
            maf.MetricBundle(
                metric,
                slicer,
                sql,
                plotDict=plotDict,
                runName=runName,
                plotFuncs=plotFuncs,
                summaryMetrics=depth_stats,
            )
        )

        # Now to compute some things at just the center of the DDF
        slicer = maf.UserPointSlicer(np.degrees(np.mean(ddf.ra)), np.degrees(np.mean(ddf.dec)))

        # Median inter-night gap (each and all filters)
        for filtername in filternames:
            sql = 'filter="%s"' % filtername
            metric = maf.InterNightGapsMetric(metricName="Median Inter-Night Gap, %s" % filtername,
                                              reduceFunc=np.median)
            bundle_list.append(
                maf.MetricBundle(
                    metric,
                    slicer,
                    sql,
                    runName=runName,
                    summaryMetrics=standardSummary
                )
            )
        sql = ''
        metric = maf.InterNightGapsMetric(metricName="Median Inter-Night Gap",
                                          reduceFunc=np.median)
        bundle_list.append(
            maf.MetricBundle(
                metric,
                slicer,
                sql,
                runName=runName,
                summaryMetrics=standardSummary))

    for b in bundle_list:
        b.setRunName(runName)
    bundleDict = maf.makeBundlesDictFromList(bundle_list)

    return bundleDict
