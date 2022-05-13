import numpy as np
import healpy as hp
from rubin_sim.utils import hpid2RaDec, angularSeparation, ddf_locations
import rubin_sim.maf as maf

__all__ = ["ddfBatch"]


def ddfBatch(runName="opsim", nside=512, radius=2.5, nside_sne=128):
    """
    Parameters
    ----------
    nside : int (512)
        The HEALpix nside to run most of the metrics on. default 512.
    radius : float (2.5)
        The radius to select around each ddf (degrees). Default 2.5. Note that
        Going too large will result in more background being selected, which
        can throw off things like the median number of visits. But going too
        small risks missing some DDF area on the double Euclid field, or a regular
        field with large dithers.
    nside_sne : int (128)
        The HEALpix nside to use with the SNe metric.
    """

    bundle_list = []

    ra, dec = hpid2RaDec(nside, np.arange(hp.nside2npix(nside)))
    ra_sne, dec_sne = hpid2RaDec(nside_sne, np.arange(hp.nside2npix(nside_sne)))

    ddfs_rough = ddf_locations()

    # Reformat as a dict for later
    ddfs = {}
    for ddf in ddfs_rough:
        ddfs[ddf] = {"ra": ddfs_rough[ddf][0], "dec": ddfs_rough[ddf][1]}
    # Combine the Euclid double-field into one
    ddfs["EDFS"] = {
        "ra": [ddfs["EDFS_a"]["ra"], ddfs["EDFS_b"]["ra"]],
        "dec": [ddfs["EDFS_a"]["dec"], ddfs["EDFS_b"]["dec"]],
    }
    del ddfs["EDFS_a"]
    del ddfs["EDFS_b"]

    # Let's include an arbitrary point that should be in the WFD for comparision
    ddfs["WFD"] = {"ra": 0, "dec": -20.0}

    summary_stats = [maf.MeanMetric(), maf.MedianMetric(), maf.SumMetric()]

    filternames = "ugrizy"
    filter_all_sql = ['filter="%s"' % filtername for filtername in filternames]
    filter_all_sql.append("")

    depth_stats = [maf.MedianMetric()]
    plotFuncs = [maf.HealpixSkyMap()]

    displayDict = {"group": "", "subgroup": "", "order": 0}

    for ddf in ddfs:
        sql = ""
        label = ddf.replace("DD:", "")
        plotDict = {
            "visufunc": hp.gnomview,
            "rot": (np.mean(ddfs[ddf]["ra"]), np.mean(ddfs[ddf]["dec"]), 0),
            "xsize": 500,
        }
        if np.size(ddfs[ddf]["ra"]) > 1:
            goods = []
            goods_sne = []
            for ddf_ra, ddf_dec in zip(ddfs[ddf]["ra"], ddfs[ddf]["dec"]):
                dist = angularSeparation(ra, dec, ddf_ra, ddf_dec)
                goods.append(np.where(dist <= radius)[0])
                dist = angularSeparation(ra_sne, dec_sne, ddf_ra, ddf_dec)
                goods_sne.append(np.where(dist <= radius)[0])
            good = np.unique(np.concatenate(goods))
            good_sne = np.unique(np.concatenate(goods_sne))
        else:
            dist = angularSeparation(
                ra, dec, np.mean(ddfs[ddf]["ra"]), np.mean(ddfs[ddf]["dec"])
            )
            good = np.where(dist <= radius)[0]
            dist = angularSeparation(
                ra_sne, dec_sne, np.mean(ddfs[ddf]["ra"]), np.mean(ddfs[ddf]["dec"])
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
        agn_mags = {"u": 22.0, "g": 24, "r": 24, "i": 24, "z": 22, "y": 22}
        for f in "ugrizy":
            sql = 'filter="%s"' % f
            summaryMetrics = [
                maf.MedianMetric(metricName="Median AGN SF Uncert, %s" % (label))
            ]
            metric = maf.SFUncertMetric(mag=agn_mags[f], metricName="SFU, %s" % label)
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
            np.mean(ddfs[ddf]["ra"]), np.mean(ddfs[ddf]["dec"])
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

        # Median season Length
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
