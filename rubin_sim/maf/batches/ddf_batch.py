__all__ = ("ddfBatch",)

import healpy as hp
import numpy as np
from rubin_scheduler.utils import (
    angular_separation,
    ddf_locations,
    ddf_locations_pre3_5,
    hpid2_ra_dec,
    sample_patch_on_sphere,
)

import rubin_sim.maf as maf

from .common import lightcurve_summary


def ddfBatch(
    run_name="run_name",
    nside=512,
    radius=2.5,
    nside_sne=128,
    extra_sql=None,
    extra_info_label=None,
    old_coords=False,
):
    """
    A set of metrics to evaluate DDF fields.

    Parameters
    ----------
    run_name : `str`, optional
        The name of the simulation (for plot titles and file outputs).
    nside : `int`, optional
        The HEALpix nside to run most of the metrics on.
    radius : `float`
        The radius to select around each ddf (degrees).
        The default value of 2.5 degrees has been chosen to balance
        selecting a large enough area to ensure gathering all of the double
        Euclid field or a run with large dithers,
        while not including too much background area (which can
        skew metrics of the median number of visits, etc.).
    nside_sne : `int`, optional
        The HEALpix nside to use with the SNe metric. The default is lower
        than the default nside for other metrics, as the SNe metric is
        more computationally expensive.
    extra_sql : `str`, optional
        Additional sql constraint (such as night<=365) to add to the
        necessary sql constraints for each metric.
    extra_info_label : `str`, optional
        Additional description information to add (alongside the extra_sql)
    old_coords : `bool`
        Use the default locations for the DDFs from pre-July 2024.
        Default False.

    Returns
    -------
    metric_bundleDict : `dict` of `maf.MetricBundle`
    """

    bundle_list = []

    # Define the slicer to use for each DDF
    # Get standard DDF locations and reformat information as a dictionary
    ddfs = {}
    if old_coords:
        ddfs_rough = ddf_locations_pre3_5()
    else:
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
    # Let's include an arbitrary point that should be in the WFD for comparison
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
            dist = angular_separation(ra, dec, np.mean(ddfs[ddf]["ra"]), np.mean(ddfs[ddf]["dec"]))
            good = np.where(dist <= radius)[0]
            dist = angular_separation(ra_sne, dec_sne, np.mean(ddfs[ddf]["ra"]), np.mean(ddfs[ddf]["dec"]))
            good_sne = np.where(dist <= radius)[0]
        ddf_slicers_sne[ddf] = maf.HealpixSubsetSlicer(nside_sne, good_sne, use_cache=False)
        ddf_slicers[ddf] = maf.HealpixSubsetSlicer(nside, good, use_cache=False)

    # Now define metrics

    # Set up basic all and per filter sql constraints.
    filterlist, colors, orders, sqls, info_labels = maf.filter_list(
        all=True, extra_sql=extra_sql, extra_info_label=extra_info_label
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
            z_step=0.03,
            daymax_step=3,
            coadd_night=True,
            gamma_name="gamma_DDF.hdf5",
            # have to add field name here, to avoid reduceDict key collissions
            metric_name=f"SNNSNMetric {fieldname}",
        )
        bundle_list.append(
            maf.MetricBundle(
                metric,
                ddf_slicers_sne[ddf],
                constraint="",
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

        # KNe

        delta = 5.0  # degrees
        n_kne = 5000
        displayDict["group"] = "KNe"
        displayDict["subgroup"] = ""
        displayDict["caption"] = f"Number of KNe in the {fieldname} DDF from %i injected." % n_kne

        ra, dec = sample_patch_on_sphere(
            np.mean(ddfs[ddf]["ra"]), np.mean(ddfs[ddf]["dec"]), delta, n_kne, seed=1
        )

        metric = maf.KNePopMetric(metric_name="KNePopMetric_%s" % fieldname)
        slicer = maf.generate_kn_pop_slicer(n_events=n_kne, ra=ra, dec=dec)

        bundle_list.append(
            maf.MetricBundle(
                metric,
                slicer,
                "",
                info_label=" ".join([fieldname]),
                summary_metrics=lightcurve_summary(),
                display_dict=displayDict,
                plot_funcs=[],
            )
        )

        # kuiper metric
        displayDict["group"] = "Kuiper"
        displayDict["subgroup"] = ""
        displayDict["caption"] = f"Kuiper metric in the {fieldname} DDF."

        sqls_gri = {
            "gri": "filter='g' or filter='r' or filter='i'",
            "riz": "filter='r' or filter='i' or filter='z'",
        }

        for sql in sqls_gri:
            metrics = [
                maf.KuiperMetric(
                    "rotSkyPos",
                    metric_name=f"Kuiper statistic (0 is uniform, 1 is delta function),rotSkyPos,{fieldname},"
                    + sql,
                ),
                maf.KuiperMetric(
                    "rotTelPos",
                    metric_name=f"Kuiper statistic (0 is uniform, 1 is delta function),rotTelPos,{fieldname},"
                    + sql,
                ),
            ]
            for metric in metrics:
                bundle_list.append(
                    maf.MetricBundle(
                        metric,
                        ddf_slicers[ddf],
                        sqls_gri[sql],
                        info_label=" ".join([fieldname]),
                        plot_dict=plotDict,
                        plot_funcs=plotFuncs,
                        summary_metrics=summary_stats,
                        display_dict=displayDict,
                    )
                )

        # Weak lensing visits
        lim_ebv = 0.2
        mag_cuts = 26.0
        displayDict["group"] = "Weak Lensing"
        displayDict["subgroup"] = ""
        displayDict["caption"] = f"Weak lensing metric in the {fieldname} DDF."

        sqls_gri = {
            "gri": "filter='g' or filter='r' or filter='i'",
            "riz": "filter='r' or filter='i' or filter='z'",
        }

        for sql in sqls_gri:
            metric = maf.WeakLensingNvisits(
                lsst_filter="i",
                depth_cut=mag_cuts,
                ebvlim=lim_ebv,
                min_exp_time=20.0,
                metric_name="WeakLensingNvisits_" + sql,
            )
            bundle_list.append(
                maf.MetricBundle(
                    metric,
                    ddf_slicers[ddf],
                    sqls_gri[sql],
                    info_label=" ".join([fieldname, sql]),
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
            summaryMetrics = [maf.SumMetric(metric_name="Total QSO")]
            metric = maf.QSONumberCountsMetric(
                f,
                units="mag",
                extinction_cut=extinction_cut,
                qlf_module="Shen20",
                qlf_model="A",
                sed_model="Richards06",
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
        m = maf.AgnTimeLagMetric(threshold=nquist_threshold, lag=lag)
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
                    run_name=run_name,
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
        m = maf.AgnTimeLagMetric(threshold=nquist_threshold, lag=lag)
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
                    run_name=run_name,
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
            displayDict["caption"] = f"AGN Structure Function Error in {f} band in the {fieldname} DDF."
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
        #######
        # Coadded depth per filter, and count per filter
        displayDict["group"] = "Basics"
        for f in "ugrizy":
            displayDict["subgroup"] = "Coadd M5"
            displayDict["order"] = orders[f]
            displayDict["caption"] = f"Coadded m5 in {f} band in the {fieldname} DDF."
            metric = maf.Coaddm5Metric(metric_name=f"{fieldname} CoaddM5")
            bundle_list.append(
                maf.MetricBundle(
                    metric,
                    ddf_slicers[ddf],
                    sqls[f],
                    info_label=info_labels[f],
                    plot_dict=plotDict,
                    run_name=run_name,
                    plot_funcs=plotFuncs,
                    summary_metrics=depth_stats,
                    display_dict=displayDict,
                )
            )
            displayDict["subgroup"] = "N Visits"
            displayDict["caption"] = f"Number of visits in the {f} band in the {fieldname} DDF."
            metric = maf.CountMetric(col="observationStartMJD", units="#", metric_name=f"{fieldname} NVisits")
            bundle_list.append(
                maf.MetricBundle(
                    metric,
                    ddf_slicers[ddf],
                    sqls[f],
                    info_label=info_labels[f],
                    plot_dict=plotDict,
                    run_name=run_name,
                    plot_funcs=plotFuncs,
                    summary_metrics=depth_stats,
                    display_dict=displayDict,
                )
            )
        # Count over all filter
        displayDict["subgroup"] = "N Visits"
        displayDict["order"] = orders["all"]
        displayDict["caption"] = f"Number of visits in all bands in the {fieldname} DDF."
        metric = maf.CountMetric(col="observationStartMJD", units="#", metric_name=f"{fieldname} NVisits")
        bundle_list.append(
            maf.MetricBundle(
                metric,
                ddf_slicers[ddf],
                constraint=sqls["all"],
                info_label=info_labels["all"],
                plot_dict=plotDict,
                run_name=run_name,
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
        metric = maf.CountUniqueMetric(col="night", units="#", metric_name=f"{fieldname} N Unique Nights")
        bundle_list.append(
            maf.MetricBundle(
                metric,
                ddf_slicers[ddf],
                constraint=sqls["all"],
                info_label=info_labels["all"],
                plot_dict=plotDict,
                run_name=run_name,
                plot_funcs=plotFuncs,
                summary_metrics=depth_stats,
                display_dict=displayDict,
            )
        )

        # Now to compute some things ~~at just the center of the DDF~~ NOPE
        # (will compute these "per DDF" not just at the center, since
        # the dithering pattern is not yet set and that will influence the
        # result -- once dithering is better determined, could add ptslicer).
        # For these metrics, add a requirement that the 'note' label
        # match the DDF, to avoid WFD visits skewing the results
        # (we want to exclude non-DD visits),

        if fieldname == "WFD":
            ptslicer = maf.UserPointsSlicer(np.mean(ddfs[ddf]["ra"]), np.mean(ddfs[ddf]["dec"]))
        else:
            ptslicer = maf.UniSlicer()  # rely on query to remove non-DD visits
            # Add RA and Dec to slice_point data (for season calculations)
            # slice_points store ra/dec internally in radians.
            ptslicer.slice_points["ra"] = np.radians(np.mean(ddfs[ddf]["ra"]))
            ptslicer.slice_points["dec"] = np.radians(np.mean(ddfs[ddf]["dec"]))

        displayDict["group"] = "Cadence"
        displayDict["order"] = order

        fieldsqls = {}
        if ddf == "WFD":
            for f in filterlist:
                fieldsqls[f] = sqls[f]
        else:
            fieldsql = f"scheduler_note like '%{fieldname}%'"
            for f in filterlist:
                if len(sqls[f]) > 0:
                    fieldsqls[f] = fieldsql + " and " + sqls[f]
                else:
                    fieldsqls[f] = fieldsql

        displayDict["subgroup"] = "Sequence length"

        # Number of observations per night, any filter (sequence length)
        # Histogram the number of visits per night
        countbins = np.arange(0, 200, 5)
        metric = maf.NVisitsPerNightMetric(
            night_col="night",
            bins=countbins,
            metric_name=f"{fieldname} NVisitsPerNight",
        )
        plotDict = {"bins": countbins, "xlabel": "Number of visits per night"}
        displayDict["caption"] = "Histogram of the number of visits in each night per DDF."
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

        # Coadded depth of observations per night, each filter
        # "magic numbers" to fill plot come from baseline v3.4
        min_coadds = {"u": 22.3, "g": 22.3, "r": 22.9, "i": 23.1, "z": 21.7, "y": 21.5}
        max_coadds = {"u": 26, "g": 27.2, "r": 27, "i": 26.5, "z": 26.5, "y": 25.1}
        # Histogram the coadded depth per night, per filter
        for f in "ugrizy":
            magbins = np.arange(min_coadds[f], max_coadds[f], 0.05)
            metric = maf.CoaddM5PerNightMetric(
                night_col="night",
                m5_col="fiveSigmaDepth",
                bins=magbins,
                metric_name=f"{fieldname} CoaddM5PerNight",
            )
            plotDict = {"bins": magbins, "xlabel": "Coadded Depth Per Night"}
            displayDict["caption"] = f"Histogram of the coadded depth in {f} in each night per DDF."
            plotFunc = maf.SummaryHistogram()
            bundle = maf.MetricBundle(
                metric,
                ptslicer,
                fieldsqls[f],
                info_label=info_labels[f],
                plot_dict=plotDict,
                display_dict=displayDict,
                plot_funcs=[plotFunc],
            )
            bundle_list.append(bundle)

        # Plot of number of visits per night over time
        if fieldname.endswith("WFD"):
            pass
        else:
            displayDict["caption"] = f"Number of visits per night for {fieldname}."
            metric = maf.CountMetric("observationStartMJD", metric_name=f"{fieldname} Nvisits Per Night")
            slicer = maf.OneDSlicer(slice_col_name="night", bin_size=1, badval=0)
            plot_dict = {"filled_data": True}
            bundle = maf.MetricBundle(
                metric,
                slicer,
                fieldsqls["all"],
                info_label=info_labels["all"],
                plot_dict=plot_dict,
                display_dict=displayDict,
                summary_metrics=[
                    maf.MedianMetric(),
                    maf.PercentileMetric(percentile=80, metric_name="80thPercentile"),
                    maf.MinMetric(),
                    maf.MaxMetric(),
                ],
            )
            bundle_list.append(bundle)

        # Likewise, but coadded depth per filter
        if fieldname.endswith("WFD"):
            pass
        else:
            for f in "ugrizy":
                displayDict["caption"] = f"Coadded depth per night for {fieldname} in band {f}."
                metric = maf.Coaddm5Metric(metric_name=f"{fieldname} CoaddedM5 Per Night")
                slicer = maf.OneDSlicer(slice_col_name="night", bin_size=1, badval=min_coadds[f])
                plot_dict = {"filled_data": True}
                bundle = maf.MetricBundle(
                    metric,
                    slicer,
                    fieldsqls[f],
                    info_label=info_labels[f],
                    plot_dict=plot_dict,
                    display_dict=displayDict,
                    summary_metrics=[
                        maf.MedianMetric(),
                        maf.PercentileMetric(percentile=80, metric_name="80thPercentile"),
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
            night_col="night",
            metric_name=f"{fieldname} Delta Nights Histogram",
        )
        displayDict["caption"] = f"Histogram of intervals between nights with visits, in the {fieldname} DDF."
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
                metric_name=f"{fieldname} Median Inter-Night Gap", reduce_func=np.median
            )
            displayDict["order"] = orders[f]
            displayDict["caption"] = f"Median internight gap in {f} band in the {fieldname} DDF."
            bundle_list.append(
                maf.MetricBundle(
                    metric,
                    ptslicer,
                    fieldsqls[f],
                    info_label=info_labels[f],
                    run_name=run_name,
                    summary_metrics=[maf.MeanMetric()],
                    plot_funcs=[],
                    display_dict=displayDict,
                )
            )

        displayDict["subgroup"] = "Season length"

        # Histogram of the season lengths, all filters
        def rfunc(simdata):
            # Sometimes number of seasons is 10, sometimes 11
            # (depending on where survey starts/end)
            # so normalize it so there's always 11 values
            # by adding 0 at the end.
            if len(simdata) < 11:
                simdata = np.concatenate([simdata, np.array([0], float)])
            return simdata

        metric = maf.SeasonLengthMetric(reduce_func=rfunc, metric_dtype="object")
        plotDict = {"bins": np.arange(0, 12), "ylabel": "Season length (days)", "xlabel": "Season"}
        plotFunc = maf.SummaryHistogram()
        displayDict["caption"] = f"Plot of the season length per season in the {fieldname} DDF."
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
            metric_name=f"{fieldname} Median Season Length", reduce_func=np.median
        )
        displayDict["caption"] = f"Median season length in the {fieldname} DDF."
        bundle_list.append(
            maf.MetricBundle(
                metric,
                ptslicer,
                fieldsqls[f],
                info_label=info_labels["all"],
                run_name=run_name,
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
            displayDict["caption"] = (
                f"Cumulative number of visits for the {fieldname.replace('DD:', '')} field."
            )
            slicer = maf.UniSlicer()
            metric = maf.CumulativeMetric(metric_name=f"{fieldname} Cumulative NVisits")
            metricb = maf.MetricBundle(
                metric,
                slicer,
                fieldsqls["all"],
                info_label=info_labels["all"],
                plot_funcs=[maf.XyPlotter()],
                run_name=run_name,
                display_dict=displayDict,
            )
            metricb.summary_metrics = []
            bundle_list.append(metricb)

    for b in bundle_list:
        b.set_run_name(run_name)
    bundleDict = maf.make_bundles_dict_from_list(bundle_list)

    return bundleDict
