import numpy as np
import healpy as hp
from rubin_sim.utils import hpid2_ra_dec, angular_separation, ddf_locations
import rubin_sim.maf as maf

__all__ = ["ddfBatch"]


def ddfBatch(
    runName="opsim",
    nside=512,
    radius=2.5,
    nside_sne=128,
    extraSql=None,
    extra_info_label=None,
):
    """
    Parameters
    ----------
    nside : `int` (512)
        The HEALpix nside to run most of the metrics on. default 512.
    radius : `float` (2.5)
        The radius to select around each ddf (degrees). Default 2.5. Note that
        Going too large will result in more background being selected, which
        can throw off things like the median number of visits. But going too
        small risks missing some DDF area on the double Euclid field, or a regular
        field with large dithers.
    nside_sne : `int` (128)
        The HEALpix nside to use with the SNe metric.
    extraSql : `str` (None)
        Additional sql constraint (such as night<=365) to add to the necessary sql constraints below
    extra_info_label : `str` (None)
        Additional description information to add (alongside the extraSql)
    """

    bundle_list = []

    # Define the slicer to use for each DDF
    # Get standard DDF locations and reformat information as a dictionary
    ddfs = {}
    ddfs_rough = ddf_locations()
    for ddf in ddfs_rough:
        ddfs[ddf] = {"ra": ddfs_rough[ddf][0], "dec": ddfs_rough[ddf][1]}
    # Combine the Euclid double-field into one - but with two ra/dec values
    ddfs["EDFS"] = {
        "ra": [ddfs["EDFS_a"]["ra"], ddfs["EDFS_b"]["ra"]],
        "dec": [ddfs["EDFS_a"]["dec"], ddfs["EDFS_b"]["dec"]],
    }
    del ddfs["EDFS_a"]
    del ddfs["EDFS_b"]
    # Let's include an arbitrary point that should be in the WFD for comparision
    ddfs["WFD"] = {"ra": 0, "dec": -20.0}

    ra, dec = hpid2_ra_dec(nside, np.arange(hp.nside2npix(nside)))
    ra_sne, dec_sne = hpid2_ra_dec(nside_sne, np.arange(hp.nside2npix(nside_sne)))

    ddf_slicers = {}
    ddf_slicers_sne = {}
    for ddf in ddfs:
        # Define the healpixels to use for this DDF
        if np.size(ddfs[ddf]["ra"]) > 1:
            goods = []
            goods_sne = []
            for ddf_ra, ddf_dec in zip(ddfs[ddf]["ra"], ddfs[ddf]["dec"]):
                dist = angular_separation(ra, dec, ddf_ra, ddf_dec)
                goods.append(np.where(dist <= radius)[0])
                dist = angular_separation(ra_sne, dec_sne, ddf_ra, ddf_dec)
                goods_sne.append(np.where(dist <= radius)[0])
            good = np.unique(np.concatenate(goods))
            good_sne = np.unique(np.concatenate(goods_sne))
        else:
            dist = angular_separation(
                ra, dec, np.mean(ddfs[ddf]["ra"]), np.mean(ddfs[ddf]["dec"])
            )
            good = np.where(dist <= radius)[0]
            dist = angular_separation(
                ra_sne, dec_sne, np.mean(ddfs[ddf]["ra"]), np.mean(ddfs[ddf]["dec"])
            )
            good_sne = np.where(dist <= radius)[0]
        ddf_slicers_sne[ddf] = maf.HealpixSubsetSlicer(
            nside_sne, good_sne, useCache=False
        )
        ddf_slicers[ddf] = maf.HealpixSubsetSlicer(nside, good, useCache=False)

    # Now define metrics

    # Set up basic all and per filter sql constraints.
    filterlist, colors, orders, sqls, info_labels = maf.filterList(
        all=True, extraSql=extraSql, extraInfoLabel=extra_info_label
    )

    summary_stats = [maf.MeanMetric(), maf.MedianMetric(), maf.SumMetric()]
    depth_stats = [maf.MedianMetric()]

    plotFuncs = [maf.HealpixSkyMap()]

    displayDict = {"group": "", "subgroup": "", "order": 0}

    order = 0
    for ddf in ddfs:
        fieldname = ddf
        if not (fieldname.startswith("DD")):
            fieldname = f"DD:{fieldname}"

        plotDict = {
            "visufunc": hp.gnomview,
            "rot": (np.mean(ddfs[ddf]["ra"]), np.mean(ddfs[ddf]["dec"]), 0),
            "xsize": 500,
        }
        order += 1

        # Number of SNe
        displayDict["group"] = "SNe"
        displayDict["subgroup"] = "N SNe"
        displayDict["caption"] = f"SNIa in the {fieldname} DDF."
        displayDict["order"] = order

        metric = maf.metrics.SNNSNMetric(
            verbose=False,
            n_bef=4,
            n_aft=10,
            zmin=0.1,
            zmax=1.1,
            zStep=0.03,
            daymaxStep=3,
            coadd_night=True,
            gammaName="gamma_DDF.hdf5",
            metricName=f"SNNSNMetric {fieldname}",  # have to add here, as must be in reduceDict key
        )
        bundle_list.append(
            maf.MetricBundle(
                metric,
                ddf_slicers_sne[ddf],
                constraint="note like '%DD%'",
                info_label=" ".join([fieldname, "all bands, only DDF observations"]),
                plot_dict=plotDict,
                plot_funcs=plotFuncs,
                summary_metrics=summary_stats,
                display_dict=displayDict,
            )
        )
        # Strong lensed SNe
        displayDict["group"] = "SNe"
        displayDict["subgroup"] = "SL SNe"
        displayDict["caption"] = f"Strongly Lensed SN metric in the {fieldname} DDF."
        displayDict["order"] = order
        metric = maf.SNSLMetric()
        bundle_list.append(
            maf.MetricBundle(
                metric,
                ddf_slicers[ddf],
                constraint=sqls["all"],
                info_label=" ".join([fieldname, info_labels["all"]]),
                plot_dict=plotDict,
                plot_funcs=plotFuncs,
                summary_metrics=summary_stats,
                display_dict=displayDict,
            )
        )

        # Number of QSOs in each band
        displayDict["group"] = "QSO"
        displayDict["subgroup"] = "Number"
        displayDict["caption"] = f"Number of QSO in the {fieldname} DDF."
        zmin = 0.3
        extinction_cut = 1.0
        for f in "ugrizy":
            displayDict["order"] = orders[f]
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
            )
            bundle_list.append(
                maf.MetricBundle(
                    metric,
                    ddf_slicers[ddf],
                    sqls[f],
                    info_label=" ".join([fieldname, info_labels[f]]),
                    plot_dict=plotDict,
                    plot_funcs=plotFuncs,
                    summary_metrics=summaryMetrics,
                    display_dict=displayDict,
                )
            )

        # Run the TimeLag for each filter *and* all filters
        displayDict["group"] = "QSO"
        displayDict["subgroup"] = "TimeLags"
        nquist_threshold = 2.2
        lag = 100
        summaryMetrics = [maf.MeanMetric(), maf.MedianMetric(), maf.RmsMetric()]
        m = maf.AGN_TimeLagMetric(threshold=nquist_threshold, lag=lag)
        for f in filterlist:
            displayDict["order"] = orders[f]
            displayDict["caption"] = (
                f"Comparion of the time between visits compared to a defined sampling gap ({lag} days) in "
                f"{f} band."
            )
            bundle_list.append(
                maf.MetricBundle(
                    m,
                    ddf_slicers[ddf],
                    constraint=sqls[f],
                    info_label=" ".join([fieldname, info_labels[f]]),
                    run_name=runName,
                    plot_dict=plotDict,
                    plot_funcs=plotFuncs,
                    summary_metrics=summaryMetrics,
                    display_dict=displayDict,
                )
            )

        # Run the TimeLag for each filter *and* all filters
        displayDict["group"] = "QSO"
        displayDict["subgroup"] = "TimeLags"
        nquist_threshold = 2.2
        lag = 5
        summaryMetrics = [maf.MeanMetric(), maf.MedianMetric(), maf.RmsMetric()]
        m = maf.AGN_TimeLagMetric(threshold=nquist_threshold, lag=lag)
        for f in filterlist:
            displayDict["order"] = orders[f]
            displayDict["caption"] = (
                f"Comparion of the time between visits compared to a defined sampling gap ({lag} days) in "
                f"{f} band."
            )
            bundle_list.append(
                maf.MetricBundle(
                    m,
                    ddf_slicers[ddf],
                    constraint=sqls[f],
                    info_label=" ".join([fieldname, info_labels[f]]),
                    run_name=runName,
                    plot_dict=plotDict,
                    plot_funcs=plotFuncs,
                    summary_metrics=summaryMetrics,
                    display_dict=displayDict,
                )
            )

        # AGN structure function
        displayDict["group"] = "QSO"
        displayDict["subgroup"] = "Structure Function"
        agn_mags = {"u": 22.0, "g": 24, "r": 24, "i": 24, "z": 22, "y": 22}
        for f in "ugrizy":
            displayDict["order"] = orders[f]
            displayDict[
                "caption"
            ] = f"AGN Structure Function Error in {f} band in the {fieldname} DDF."
            summaryMetrics = [maf.MedianMetric(), maf.RmsMetric()]
            metric = maf.SFUncertMetric(
                mag=agn_mags[f],
                bins=np.logspace(0, np.log10(3650), 21),
            )
            bundle_list.append(
                maf.MetricBundle(
                    metric,
                    ddf_slicers[ddf],
                    sqls[f],
                    info_label=" ".join([fieldname, info_labels[f]]),
                    plot_dict=plotDict,
                    plot_funcs=plotFuncs,
                    summary_metrics=summaryMetrics,
                    display_dict=displayDict,
                )
            )

        # Coadded depth per filter, and count per filter
        displayDict["group"] = "Basics"
        for f in "ugrizy":
            displayDict["subgroup"] = "Coadd M5"
            displayDict["order"] = orders[f]
            displayDict["caption"] = f"Coadded m5 in {f} band in the {fieldname} DDF."
            metric = maf.Coaddm5Metric(metricName=f"{fieldname} CoaddM5")
            bundle_list.append(
                maf.MetricBundle(
                    metric,
                    ddf_slicers[ddf],
                    sqls[f],
                    info_label=info_labels[f],
                    plot_dict=plotDict,
                    run_name=runName,
                    plot_funcs=plotFuncs,
                    summary_metrics=depth_stats,
                    display_dict=displayDict,
                )
            )
            displayDict["subgroup"] = "N Visits"
            displayDict[
                "caption"
            ] = f"Number of visits in the {f} band in the {fieldname} DDF."
            metric = maf.CountMetric(
                col="observationStartMJD", units="#", metricName=f"{fieldname} NVisits"
            )
            bundle_list.append(
                maf.MetricBundle(
                    metric,
                    ddf_slicers[ddf],
                    sqls[f],
                    info_label=info_labels[f],
                    plot_dict=plotDict,
                    run_name=runName,
                    plot_funcs=plotFuncs,
                    summary_metrics=depth_stats,
                    display_dict=displayDict,
                )
            )
        # Count over all filter
        displayDict["subgroup"] = "N Visits"
        displayDict["order"] = orders["all"]
        displayDict[
            "caption"
        ] = f"Number of visits in all bands in the {fieldname} DDF."
        metric = maf.CountMetric(
            col="observationStartMJD", units="#", metricName=f"{fieldname} NVisits"
        )
        bundle_list.append(
            maf.MetricBundle(
                metric,
                ddf_slicers[ddf],
                constraint=sqls["all"],
                info_label=info_labels["all"],
                plot_dict=plotDict,
                run_name=runName,
                plot_funcs=plotFuncs,
                summary_metrics=depth_stats,
                display_dict=displayDict,
            )
        )
        # Count number of unique nights with visits
        displayDict["group"] = "Cadence"
        displayDict["subgroup"] = "N Nights"
        displayDict["order"] = orders["all"]
        displayDict["caption"] = f"Number of nights with visits in the {fieldname} DDF."
        metric = maf.CountUniqueMetric(
            col="night", units="#", metricName=f"{fieldname} N Unique Nights"
        )
        bundle_list.append(
            maf.MetricBundle(
                metric,
                ddf_slicers[ddf],
                constraint=sqls["all"],
                info_label=info_labels["all"],
                plot_dict=plotDict,
                run_name=runName,
                plot_funcs=plotFuncs,
                summary_metrics=depth_stats,
                display_dict=displayDict,
            )
        )

        # Now to compute some things at just the center of the DDF
        # For these metrics, add a requirement that the 'note' label match the DDF,
        # to avoid WFD visits skewing the results (we want to exclude these)
        ptslicer = maf.UserPointsSlicer(
            np.mean(ddfs[ddf]["ra"]), np.mean(ddfs[ddf]["dec"])
        )

        displayDict["group"] = "Cadence"
        displayDict["order"] = order

        fieldsqls = {}
        if ddf == "WFD":
            for f in filterlist:
                fieldsqls[f] = sqls[f]
        else:
            fieldsql = f"note like '%{fieldname}%'"
            for f in filterlist:
                if len(sqls[f]) > 0:
                    fieldsqls[f] = fieldsql + " and " + sqls[f]
                else:
                    fieldsqls[f] = fieldsql

        displayDict["subgroup"] = "Sequence length"

        # Number of observations per night, any filter (sequence length)
        # Histogram the number of visits per night at the center of the DDF
        countbins = np.arange(0, 200, 5)
        metric = maf.NVisitsPerNightMetric(
            nightCol="night", bins=countbins, metricName=f"{fieldname} NVisitsPerNight"
        )
        plotDict = {"bins": countbins, "xlabel": "Number of visits per night"}
        displayDict[
            "caption"
        ] = f"Histogram of the number of visits in each night, at the center of {fieldname}."
        plotFunc = maf.SummaryHistogram()
        bundle = maf.MetricBundle(
            metric,
            ptslicer,
            fieldsqls["all"],
            info_label=info_labels["all"],
            plot_dict=plotDict,
            display_dict=displayDict,
            plot_funcs=[plotFunc],
        )
        bundle_list.append(bundle)

        if fieldname.endswith("WFD"):
            pass
        else:
            displayDict["caption"] = f"Number of visits per night for {fieldname}."
            metric = maf.CountMetric(
                "observationStartMJD", metricName=f"{fieldname} Nvisits Per Night"
            )
            slicer = maf.OneDSlicer(sliceColName="night", binsize=1)
            bundle = maf.MetricBundle(
                metric,
                slicer,
                fieldsqls["all"],
                info_label=info_labels["all"],
                display_dict=displayDict,
                summary_metrics=[
                    maf.MedianMetric(),
                    maf.PercentileMetric(percentile=80, metricName="80thPercentile"),
                    maf.MinMetric(),
                    maf.MaxMetric(),
                ],
            )
            bundle_list.append(bundle)

        displayDict["subgroup"] = "Sequence gaps"

        # Histogram of the number of nights between visits, all filters
        bins = np.arange(1, 40, 1)
        metric = maf.NightgapsMetric(
            bins=bins,
            nightCol="night",
            metricName=f"{fieldname} Delta Nights Histogram",
        )
        displayDict[
            "caption"
        ] = f"Histogram of intervals between nights with visits, in the {fieldname} DDF."
        plotDict = {"bins": bins, "xlabel": "dT (nights)"}
        plotFunc = maf.SummaryHistogram()
        bundle = maf.MetricBundle(
            metric,
            ptslicer,
            constraint=fieldsqls["all"],
            info_label=info_labels["all"],
            plot_dict=plotDict,
            display_dict=displayDict,
            plot_funcs=[plotFunc],
        )
        bundle_list.append(bundle)

        # Median inter-night gap in each and all filters
        for f in filterlist:
            metric = maf.InterNightGapsMetric(
                metricName=f"{fieldname} Median Inter-Night Gap", reduceFunc=np.median
            )
            displayDict["order"] = orders[f]
            displayDict[
                "caption"
            ] = f"Median internight gap in {f} band in the {fieldname} DDF."
            bundle_list.append(
                maf.MetricBundle(
                    metric,
                    ptslicer,
                    fieldsqls[f],
                    info_label=info_labels[f],
                    run_name=runName,
                    summary_metrics=[maf.MeanMetric()],
                    plot_funcs=[],
                    display_dict=displayDict,
                )
            )

        displayDict["subgroup"] = "Season length"
        # Histogram of the season lengths, all filters
        def rfunc(simdata):
            # Sometimes number of seasons is 10, sometimes 11 (depending on where survey starts/end)
            # so normalize it so there's always 11 values
            if len(simdata) < 11:
                simdata = np.concatenate([simdata, np.array([0], float)])
            return simdata

        metric = maf.SeasonLengthMetric(reduceFunc=rfunc, metricDtype="object")
        plotDict = {"bins": np.arange(0, 12), "xlabel": "Season length (days)"}
        plotFunc = maf.SummaryHistogram()
        displayDict[
            "caption"
        ] = f"Plot of the season length per season in the {fieldname} DDF."
        displayDict["order"] = order
        bundle = maf.MetricBundle(
            metric,
            ptslicer,
            fieldsqls["all"],
            info_label=" ".join([fieldname, info_labels["all"]]),
            plot_dict=plotDict,
            display_dict=displayDict,
            plot_funcs=[plotFunc],
        )
        bundle_list.append(bundle)

        # Median season Length
        metric = maf.SeasonLengthMetric(
            metricName=f"{fieldname} Median Season Length", reduceFunc=np.median
        )
        displayDict["caption"] = f"Median season length in the {fieldname} DDF."
        bundle_list.append(
            maf.MetricBundle(
                metric,
                ptslicer,
                fieldsqls[f],
                info_label=info_labels["all"],
                run_name=runName,
                summary_metrics=[maf.MeanMetric()],
                plot_funcs=[],
                display_dict=displayDict,
            )
        )

        # Cumulative distribution - only for DDF fields
        if fieldname.endswith("WFD"):
            pass
        else:
            displayDict["group"] = "Progress"
            displayDict["subgroup"] = ""
            displayDict[
                "caption"
            ] = f"Cumulative number of visits for the {fieldname.replace('DD:', '')} field."
            slicer = maf.UniSlicer()
            metric = maf.CumulativeMetric(metricName=f"{fieldname} Cumulative NVisits")
            metricb = maf.MetricBundle(
                metric,
                slicer,
                fieldsqls["all"],
                info_label=info_labels["all"],
                plot_funcs=[maf.XyPlotter()],
                run_name=runName,
                display_dict=displayDict,
            )
            metricb.summary_metrics = []
            bundle_list.append(metricb)

    for b in bundle_list:
        b.set_run_name(runName)
    bundleDict = maf.make_bundles_dict_from_list(bundle_list)

    return bundleDict
