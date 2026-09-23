__all__ = ("progressBatch",)

from collections import namedtuple
from typing import Sequence

from rubin_sim import maf
import rubin_sim.maf.metric_bundles as metric_bundles
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.plots as plots
import rubin_sim.maf.slicers as slicers

from .col_map_dict import col_map_dict
from .common import standard_summary


LabelConstraint = namedtuple("LabelConstraint", ["label", "pdconstraint"])
MetricSlicerSummaryStackers = namedtuple(
    "MetricSlicerSummaryStackers", ["metric", "slicer", "summary", "stackers"]
)

# Benchmark values used for the fO metrics.
BENCHMARK_AREA = 18000
BENCHMARK_NVISITS = 825
MIN_NVISITS = 750


def progressBatch(
    colmap=None,
    run_name="run_name",
    nside=32,
    bands=("u", "g", "r", "i", "z", "y"),
):
    """Generate metrics for progress tracking.

    Parameters
    ----------
    colmap : `dict`, optional
        A dictionary with a mapping of column names.
    run_name : `str`, optional
        The name of the simulated survey.
    nside : `int`, optional
        The nside for the HEALPix slicers.
    bands : `list` of `str`, optional
        The list of individual filters to use when running metrics.
        There is always an all-visits version of the metrics run as well.

    Returns
    -------
    metric_bundleDict : `dict` of `maf.MetricBundle`
    """
    if isinstance(colmap, str):
        raise ValueError("colmap must be a dictionary, not a string")

    if colmap is None:
        colmap = col_map_dict()

    bundle_list = []

    band_constraints = []
    for band in bands:
        band_constraints.append(
            LabelConstraint(f"chimera_{band}", f"band == '{band}'")
        )
        band_constraints.append(
            LabelConstraint(f"snapshot_{band}", f"(not simulated) and (band == '{band}')")
        )

    all_constraints = list(band_constraints)
    all_constraints.append(LabelConstraint("chimera_all", "simulated or not simulated"))
    all_constraints.append(LabelConstraint("snapshot_all", "not simulated"))

    spatial_stats = standard_summary()
    spatial_stats.append(metrics.AreaSummaryMetric(decreasing=True, metric_name="top18k"))
    spatial_stats.append(metrics.PercentileMetric(col="metricdata", percentile=10))

    unislicer = slicers.UniSlicer()
    spatial_slicer = slicers.HealpixSlicer(
        nside=nside,
        lat_col=colmap["dec"],
        lon_col=colmap["ra"],
        lat_lon_deg=colmap["raDecDeg"],
    )

    metric_slicer_summary_stackers = (
        MetricSlicerSummaryStackers(
            metric=metrics.SumMetric(col="t_eff"),
            slicer=unislicer,
            summary=None,
            stackers=[maf.stackers.TeffStacker(normed=False)],
        ),
        MetricSlicerSummaryStackers(
            metric=metrics.CountMetric(
                col=colmap["mjd"], metric_name="Numbers of exposures"
            ),
            slicer=unislicer,
            summary=None,
            stackers=None,
        ),
        MetricSlicerSummaryStackers(
            metric=metrics.CountMetric(
                col=colmap["mjd"], metric_name="Number of exposure area stats"
            ),
            slicer=spatial_slicer,
            summary=spatial_stats,
            stackers=None,
        ),
        MetricSlicerSummaryStackers(
            metric=metrics.Coaddm5Metric(
                m5_col=colmap["fiveSigmaDepth"], metric_name="Depth area stats"
            ),
            slicer=spatial_slicer,
            summary=spatial_stats,
            stackers=None,
        ),
    )

    for metric, slicer, summary, stackers in metric_slicer_summary_stackers:
        for label, pdconstraint in all_constraints:
            kwargs = {
                "info_label": label,
                "pdconstraint": pdconstraint,
                "plot_funcs": [],
            }
            if stackers is not None:
                kwargs["stacker_list"] = stackers
            if summary is not None:
                kwargs["summary_metrics"] = summary

            bundle = metric_bundles.MetricBundle(
                metric,
                slicer,
                **kwargs,
            )
            bundle_list.append(bundle)

    # Configure the count metric used for the fO slicer.
    metric = metrics.CountExplimMetric(metric_name="fO")
    summary_metrics = [
        metrics.FOArea(
            nside=nside,
            norm=False,
            metric_name="fOArea",
            asky=BENCHMARK_AREA,
            n_visit=BENCHMARK_NVISITS,
        ),
        metrics.FOArea(
            nside=nside,
            norm=True,
            metric_name="fOArea/benchmark",
            asky=BENCHMARK_AREA,
            n_visit=BENCHMARK_NVISITS,
        ),
        metrics.FONv(
            nside=nside,
            norm=False,
            metric_name="fONv",
            asky=BENCHMARK_AREA,
            n_visit=BENCHMARK_NVISITS,
        ),
        metrics.FONv(
            nside=nside,
            norm=True,
            metric_name="fONv/benchmark",
            asky=BENCHMARK_AREA,
            n_visit=BENCHMARK_NVISITS,
        ),
        metrics.FOArea(
            nside=nside,
            norm=False,
            metric_name=f"fOArea_{MIN_NVISITS}",
            asky=BENCHMARK_AREA,
            n_visit=MIN_NVISITS,
        ),
    ]
    slicer = slicers.HealpixSlicer(nside=nside)
    bundle = metric_bundles.MetricBundle(
        metric,
        slicer,
        constraint="",
        summary_metrics=summary_metrics,
        plot_funcs=[],
    )
    bundle_list.append(bundle)

    for bundle in bundle_list:
        bundle.set_run_name(run_name)

    return metric_bundles.make_bundles_dict_from_list(bundle_list)
