import numpy as np
import healpy as hp
from rubin_sim.scheduler.surveys import generate_dd_surveys
from rubin_sim.utils import _hpid2RaDec, _angularSeparation
import rubin_sim.maf as maf

__all__ = ["ddfBatch"]


def ddfBatch(runName="opsim", nside=512, radius=2.5, nside_sne=128):

    radius = np.radians(radius)
    bundle_list = []

    ra, dec = _hpid2RaDec(nside, np.arange(hp.nside2npix(nside)))
    ra_sne, dec_sne = _hpid2RaDec(nside_sne, np.arange(hp.nside2npix(nside_sne)))

    ddf_surveys = generate_dd_surveys()

    summary_stats = [maf.MeanMetric(), maf.MedianMetric(), maf.SumMetric()]

    filternames = "ugrizy"
    filter_all_sql = ['filter="%s"' % filtername for filtername in filternames]
    filter_all_sql.append("")

    depth_stats = [maf.MedianMetric()]
    plotFuncs = [maf.HealpixSkyMap()]

    displayDict = {"group": "", "subgroup": "", "order": 0}

    for ddf in ddf_surveys:
        sql = ""
        label = ddf.survey_name.replace("DD:", "")
        plotDict = {
            "visufunc": hp.gnomview,
            "rot": (np.degrees(np.mean(ddf.ra)), np.degrees(np.mean(ddf.dec)), 0),
            "xsize": 500,
        }
        if np.size(ddf.ra) > 1:
            goods = []
            goods_sne = []
            for ddf_ra, ddf_dec in zip(ddf.ra, ddf.dec):
                dist = _angularSeparation(ra, dec, ddf_ra, ddf_dec)
                goods.append(np.where(dist <= radius)[0])
                dist = _angularSeparation(ra_sne, dec_sne, ddf_ra, ddf_dec)
                goods_sne.append(np.where(dist <= radius)[0])
            good = np.unique(np.concatenate(goods))
            good_sne = np.unique(np.concatenate(goods_sne))
        else:
            dist = _angularSeparation(ra, dec, np.mean(ddf.ra), np.mean(ddf.dec))
            good = np.where(dist <= radius)[0]
            dist = _angularSeparation(
                ra_sne, dec_sne, np.mean(ddf.ra), np.mean(ddf.dec)
            )
            good_sne = np.where(dist <= radius)[0]

        # Number of SNe
        displayDict["order"] = 1
        displayDict["group"] = "SNe"
        displayDict["subgroup"] = "N SNe"
        slicer = maf.HealpixSubsetSlicer(nside_sne, good_sne, useCache=False)
        metric = maf.metrics.SNNSNMetric(verbose=False, metricName="%s, SNe" % label)
        bundle_list.append(
            maf.MetricBundle(
                metric,
                slicer,
                sql,
                plotDict=plotDict,
                plotFuncs=plotFuncs,
                summaryMetrics=summary_stats,
                displayDict=displayDict,
            )
        )
        # Strong lensed SNe
        displayDict["subgroup"] = "SL SNe"
        slicer = maf.HealpixSubsetSlicer(nside, good, useCache=False)
        metric = maf.SNSLMetric(metricName="SnL_%s" % label)
        bundle_list.append(
            maf.MetricBundle(
                metric,
                slicer,
                sql,
                plotDict=plotDict,
                plotFuncs=plotFuncs,
                summaryMetrics=summary_stats,
                displayDict=displayDict,
            )
        )

        # Number of QSOs in each band
        displayDict["group"] = "QSO"
        displayDict["order"] = 2
        displayDict["subgroup"] = "Number"
        zmin = 0.3
        extinction_cut = 1.0
        for f in "ugrizy":
            sql = 'filter="%s"' % f
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
                    displayDict=displayDict,
                )
            )

        # AGN structure function
        displayDict["group"] = "QSO"
        displayDict["order"] = 2
        displayDict["subgroup"] = "Structure Function"
        agn_mag = 24.0
        for f in "ugrizy":
            sql = 'filter="%s"' % f
            summaryMetrics = [
                maf.MedianMetric(metricName="Median AGN SF Uncert, %s" % (label))
            ]
            metric = maf.SFUncertMetric(mag=agn_mag, metricName="SFU, %s" % label)
            bundle_list.append(
                maf.MetricBundle(
                    metric,
                    slicer,
                    sql,
                    plotDict=plotDict,
                    plotFuncs=plotFuncs,
                    summaryMetrics=summaryMetrics,
                    displayDict=displayDict,
                )
            )

        # Coadded depth per filter, and count per filter
        displayDict["group"] = "Basics"
        displayDict["subgroup"] = "Depth"
        displayDict["order"] = 3
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
                    displayDict=displayDict,
                )
            )

            displayDict["subgroup"] = "Count"
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
                    displayDict=displayDict,
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
                displayDict=displayDict,
            )
        )

        # Now to compute some things at just the center of the DDF
        slicer = maf.UserPointsSlicer(
            np.degrees(np.mean(ddf.ra)), np.degrees(np.mean(ddf.dec))
        )

        displayDict["order"] = 4
        displayDict["group"] = "Gaps"
        displayDict["subgroup"] = "Internight Gap"
        # Median inter-night gap (each and all filters)
        # I think need to use the note label so that the regular WFD
        # observations don't skew the results. The griy filters should all have about the same
        # median internight gap.
        for filtername in filternames:
            sql = 'note like "%s%%" and filter="%s"' % ("DD:" + label, filtername)
            metric = maf.InterNightGapsMetric(
                metricName="Median Inter-Night Gap, %s %s" % (label, filtername),
                reduceFunc=np.median,
            )
            bundle_list.append(
                maf.MetricBundle(
                    metric,
                    slicer,
                    sql,
                    runName=runName,
                    summaryMetrics=[maf.MeanMetric()],
                    plotFuncs=[],
                    displayDict=displayDict,
                )
            )
        sql = 'note like "%s%%"' % ("DD:" + label)
        metric = maf.InterNightGapsMetric(
            metricName="Median Inter-Night Gap, %s" % label, reduceFunc=np.median
        )
        bundle_list.append(
            maf.MetricBundle(
                metric,
                slicer,
                sql,
                runName=runName,
                summaryMetrics=[maf.MeanMetric()],
                plotFuncs=[],
                displayDict=displayDict,
            )
        )

        # median season Length
        displayDict["subgroup"] = "Season Length"
        metric = maf.SeasonLengthMetric(metricName="Median Season Length, %s" % label)
        bundle_list.append(
            maf.MetricBundle(
                metric,
                slicer,
                sql,
                runName=runName,
                summaryMetrics=[maf.MeanMetric()],
                plotFuncs=[],
                displayDict=displayDict,
            )
        )

        # Cumulative distribution
        displayDict["group"] = "Progress"
        displayDict["subgroup"] = ""
        slicer = maf.UniSlicer()
        metric = maf.CumulativeMetric()
        metricb = maf.MetricBundle(
            metric,
            slicer,
            sql,
            plotFuncs=[maf.XyPlotter()],
            runName=runName,
            displayDict=displayDict,
        )
        metricb.summaryMetrics = []
        bundle_list.append(metricb)

    for b in bundle_list:
        b.setRunName(runName)
    bundleDict = maf.makeBundlesDictFromList(bundle_list)

    return bundleDict
