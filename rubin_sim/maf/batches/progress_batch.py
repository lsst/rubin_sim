"""Metric bundle batches for progress tracking and chimera/snapshot subsets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

from astropy.time import Time

from rubin_sim import maf

from .col_map_dict import col_map_dict
from .common import standard_summary


__all__ = ("snapshot_batch", "chimera_batch")

# Benchmark values used for the fO metrics.
BENCHMARK_AREA = 18000
BENCHMARK_NVISITS = 825
MIN_NVISITS = 750


class MetricSlicerSummaryStackers(NamedTuple):
    """A grouping of a metric, slicer, summary metrics, and stackers.

    Parameters
    ----------
    metric : `rubin_sim.maf.metrics.BaseMetric`
        The metric to evaluate.
    slicer : `rubin_sim.maf.slicers.BaseSlicer`
        The slicer to use with the metric.
    summary : `list` [`rubin_sim.maf.metrics.BaseMetric`] or `None`
        Summary metrics to apply to the sliced metric data.
    stackers : `list` [`rubin_sim.maf.stackers.BaseStacker`] or `None`
        Stackers to apply before computing the metric.
    """

    metric: maf.metrics.BaseMetric
    slicer: maf.slicers.BaseSlicer
    summary: list[maf.metrics.BaseMetric] | None
    stackers: list[maf.stackers.BaseStacker] | None


def _make_colmap(colmap: dict[str, str] | None = None) -> dict[str, str]:
    if colmap is None:
        colmap = col_map_dict()

    if not isinstance(colmap, dict):
        raise ValueError(f"colmap must be a dictionary, not a {type(colmap)}")

    return colmap


def _make_base_progress_bundle_list(
    pdconstraints: dict[str, str],
    colmap: dict[str, str] | None = None,
    nside: int = 32,
) -> list[maf.metric_bundles.MetricBundle]:
    """Create the base list of progress-tracking metric bundles.

    Parameters
    ----------
    pdconstraints : `dict` [`str`, `str`]
        Mapping from bundle label strings to pandas query strings.
    colmap : `dict` [`str`, `str`], optional
        A dictionary mapping column name aliases to actual column names.
    nside : `int`, optional
        HEALPix nside parameter for spatial slicers.

    Returns
    -------
    bundle_list : `list` [`rubin_sim.maf.metric_bundles.MetricBundle`]
        The list of metric bundles.
    """
    colmap = _make_colmap(colmap)

    bundle_list: list[maf.metric_bundles.MetricBundle] = []

    spatial_stats: list[maf.metrics.BaseMetric] = list(standard_summary())
    spatial_stats.append(maf.metrics.AreaSummaryMetric(decreasing=True, metric_name="top18k"))
    spatial_stats.append(maf.metrics.PercentileMetric(col="metricdata", percentile=10))

    unislicer = maf.slicers.UniSlicer()
    spatial_slicer = maf.slicers.HealpixSlicer(
        nside=nside,
        lat_col=colmap["dec"],
        lon_col=colmap["ra"],
        lat_lon_deg=colmap["raDecDeg"],
    )

    metric_slicer_summary_stackers: tuple[MetricSlicerSummaryStackers, ...] = (
        MetricSlicerSummaryStackers(
            metric=maf.metrics.SumMetric(col="t_eff"),
            slicer=unislicer,
            summary=None,
            stackers=[maf.stackers.TeffStacker(normed=False)],
        ),
        MetricSlicerSummaryStackers(
            metric=maf.metrics.CountMetric(
                col=colmap["mjd"], metric_name="Numbers of exposures"
            ),
            slicer=unislicer,
            summary=None,
            stackers=None,
        ),
        MetricSlicerSummaryStackers(
            metric=maf.metrics.CountMetric(
                col=colmap["mjd"], metric_name="Number of exposure area stats"
            ),
            slicer=spatial_slicer,
            summary=spatial_stats,
            stackers=None,
        ),
        MetricSlicerSummaryStackers(
            metric=maf.metrics.Coaddm5Metric(
                m5_col=colmap["fiveSigmaDepth"], metric_name="Depth area stats"
            ),
            slicer=spatial_slicer,
            summary=spatial_stats,
            stackers=None,
        ),
    )

    for metric, slicer, summary, stackers in metric_slicer_summary_stackers:
        for label, pdconstraint in pdconstraints.items():
            kwargs: dict[str, Any] = {
                "info_label": label,
                "pdconstraint": pdconstraint,
                "plot_funcs": [],
            }
            if stackers is not None:
                kwargs["stacker_list"] = stackers
            if summary is not None:
                kwargs["summary_metrics"] = summary

            bundle = maf.metric_bundles.MetricBundle(
                metric,
                slicer,
                **kwargs,
            )
            bundle_list.append(bundle)

    return bundle_list


def _make_fO_bundle(
    nside: int = 32,
) -> maf.metric_bundles.MetricBundle:
    """Create the fO metric bundle.

    Parameters
    ----------
    nside : `int`, optional
        HEALPix nside parameter for the spatial slicer.

    Returns
    -------
    bundle : `rubin_sim.maf.metric_bundles.MetricBundle`
        The fO metric bundle.
    """
    # Configure the count metric used for the fO slicer.
    metric = maf.metrics.CountExplimMetric(metric_name="fO")
    summary_metrics: list[maf.metrics.BaseMetric] = [
        maf.metrics.FOArea(
            nside=nside,
            norm=False,
            metric_name="fOArea",
            asky=BENCHMARK_AREA,
            n_visit=BENCHMARK_NVISITS,
        ),
        maf.metrics.FOArea(
            nside=nside,
            norm=True,
            metric_name="fOArea/benchmark",
            asky=BENCHMARK_AREA,
            n_visit=BENCHMARK_NVISITS,
        ),
        maf.metrics.FONv(
            nside=nside,
            norm=False,
            metric_name="fONv",
            asky=BENCHMARK_AREA,
            n_visit=BENCHMARK_NVISITS,
        ),
        maf.metrics.FONv(
            nside=nside,
            norm=True,
            metric_name="fONv/benchmark",
            asky=BENCHMARK_AREA,
            n_visit=BENCHMARK_NVISITS,
        ),
        maf.metrics.FOArea(
            nside=nside,
            norm=False,
            metric_name=f"fOArea_{MIN_NVISITS}",
            asky=BENCHMARK_AREA,
            n_visit=MIN_NVISITS,
        ),
    ]
    slicer = maf.slicers.HealpixSlicer(nside=nside)
    bundle = maf.metric_bundles.MetricBundle(
        metric,
        slicer,
        constraint="",
        summary_metrics=summary_metrics,
        plot_funcs=[],
    )
    return bundle


def chimera_batch(
    colmap: dict[str, str] | None = None,
    run_name: str = "run_name",
    nside: int = 32,
    bands: Sequence[str] = ("u", "g", "r", "i", "z", "y"),
    label_prefix: str = "chimera",
) -> dict[str, maf.metric_bundles.MetricBundle]:
    """Generate progress-tracking metrics for the chimera subsets.

    Parameters
    ----------
    colmap : `dict` [`str`, `str`], optional
        A dictionary with a mapping of column names.
    run_name : `str`, optional
        The name of the simulated survey.
    nside : `int`, optional
        The nside for the HEALPix slicers.
    bands : `collections.abc.Sequence` [`str`], optional
        The list of individual filters to use when running metrics.
        There is always an all-visits version of the metrics run as well.
    label_prefix : `str`, optional
        Prefix for metric info labels.

    Returns
    -------
    metric_bundleDict : `dict` [`str`, `MetricBundle`]
        A dictionary of metric bundles keyed by their file names.
    """
    colmap = _make_colmap(colmap)

    pdconstraints: dict[str, str] = {}
    for band in bands:
        pdconstraints[f"{label_prefix}_{band}"] = f"{colmap['band']} == '{band}'"

    pdconstraints[f"{label_prefix}_all"] = ""

    bundle_list = _make_base_progress_bundle_list(pdconstraints, colmap, nside)
    fO_bundle = _make_fO_bundle()

    bundle_list.append(fO_bundle)

    for bundle in bundle_list:
        bundle.set_run_name(run_name)

    return maf.metric_bundles.make_bundles_dict_from_list(bundle_list)


def snapshot_batch(
    colmap: dict[str, str] | None = None,
    run_name: str = "run_name",
    nside: int = 32,
    bands: Sequence[str] = ("u", "g", "r", "i", "z", "y"),
    label_prefix: str = "snapshot",
    end_dayobs: int | None = None,
) -> dict[str, maf.metric_bundles.MetricBundle]:
    """Generate progress-tracking metrics for the snapshot subsets.

    Note: the visits database must include a boolean ``simulated`` column.

    Parameters
    ----------
    colmap : `dict` [`str`, `str`], optional
        A dictionary with a mapping of column names.
    run_name : `str`, optional
        The name of the simulated survey.
    nside : `int`, optional
        The nside for the HEALPix slicers.
    bands : `collections.abc.Sequence` [`str`], optional
        The list of individual filters to use when running metrics.
        There is always an all-visits version of the metrics run as well.
    label_prefix : `str`, optional
        Prefix for metric info labels.
    end_dayobs : `int`, optional
        If provided, include visits before the end of this observing day
        (YYYYMMDD integer format, UTC-12).

    Returns
    -------
    metric_bundleDict : `dict` [`str`, `MetricBundle`]
        A dictionary of metric bundles keyed by their file names.
    """
    colmap = _make_colmap(colmap)

    mjd_filter = ""
    if end_dayobs is not None:
        end_mjd = Time.strptime(str(end_dayobs), "%Y%m%d").mjd + 1.5
        mjd_filter = f"{colmap['mjd']} < {end_mjd}"

    pdconstraints: dict[str, str] = {}
    for band in bands:
        pdconstraints[f"{label_prefix}_{band}"] = (
            f"{mjd_filter} and {colmap['band']} == '{band}'"
        )

    pdconstraints[f"{label_prefix}_all"] = mjd_filter

    bundle_list = _make_base_progress_bundle_list(pdconstraints, colmap, nside)

    for bundle in bundle_list:
        bundle.set_run_name(run_name)

    return maf.metric_bundles.make_bundles_dict_from_list(bundle_list)
