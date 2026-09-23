__all__ = ("progressBatch",)

import warnings

from functools import partial

from rubin_scheduler.scheduler.utils import EuclidOverlapFootprint
from rubin_scheduler.utils import ddf_locations

import rubin_sim.maf.metric_bundles as metric_bundles
from collections import namedtuple
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.plots as plots
import rubin_sim.maf.slicers as slicers
from rubin_sim import maf

from .col_map_dict import col_map_dict
from .common import standard_summary
from .hourglass_batch import hourglassPlots
from .slew_batch import slewBasics

LabelConstraint = namedtuple('LabelConstraint', ['label', 'dbconstraint'])
MetricSlicerSummaryStackers = namedtuple('MetricSlicerSummaryStackers', ['metric', 'slicer', 'summary', 'stackers'])


def progressBatch(
    colmap=None,
    run_name="run_name",
    nside=32,
    # bands=("u", "g", "r", "i", "z", "y"),
    bands=("g", "i"),
):
    """Generate metrics for progress tracking

    Parameters
    ----------
    colmap : `dict`, optional
        A dictionary with a mapping of column names.
    run_name : `str`, optional
        The name of the simulated survey.
    nside : `int`, optional
        The nside for the healpix slicers.
    bands : `list` of `str`, optional
        The list of individual filters to use when running metrics.
        There is always an all-visits version of the metrics run as well.
    pairnside : `int`, optional
        nside to use for the pair fraction metric
        (it's slow, so nice to use lower resolution)

    Returns
    -------
    metric_bundleDict : `dict` of `maf.MetricBundle`
    """
    if isinstance(colmap, str):
        raise ValueError("colmap must be a dictionary, not a string")

    if colmap is None:
        colmap = col_map_dict()

    bundle_list = []

    label_pdconstraints = {}
    band_pdconstraints = []
    for band in bands:
        label = f"chimera_{band}"
        constraints = f"band == '{band}'"
        band_pdconstraints.append(LabelConstraint(label, constraints))
        
        label = f"snapshot_{band}"
        constraints = f"(not simulated) and (band == '{band}')"
        band_pdconstraints.append(LabelConstraint(label, constraints))

    allband_pdconstraints = [lc for lc in band_pdconstraints]
    allband_pdconstraints.append(LabelConstraint("chimera_all", "simulated or not simulated"))
    allband_pdconstraints.append(LabelConstraint("snapshot_all", "not simulated"))

    spatial_stats = standard_summary()
    spatial_stats.append(metrics.AreaSummaryMetric(decreasing=True, metric_name="top18k"))
    spatial_stats.append(metrics.PercentileMetric(col="metricdata", percentile=10))

    unislicer = slicers.UniSlicer()
    spatial_slicer = slicers.HealpixSlicer(
        nside=nside,
        lat_col=colmap["dec"],
        lon_col=colmap["ra"],
        lat_lon_deg=colmap["raDecDeg"]
        )
    
    metric_slicer_stackers = (
        MetricSlicerSummaryStackers(
            metrics.SumMetric(col="t_eff"),
            unislicer,
            None,
            [maf.stackers.TeffStacker(normed=False)]),
        MetricSlicerSummaryStackers(
            metrics.CountMetric(col=colmap["mjd"], metric_name="Numbers of exposures"),
            unislicer,
            None,
            None),
        MetricSlicerSummaryStackers(
            metrics.CountMetric(col=colmap["mjd"], metric_name="Number of exposure area stats"),
            spatial_slicer,
            spatial_stats,
            None),
        MetricSlicerSummaryStackers(
            metrics.Coaddm5Metric(m5_col=colmap["fiveSigmaDepth"], metric_name="Depth area stats"),
            spatial_slicer,
            spatial_stats,
            None),
    )

    
    for metric, slicer, summary, stackers in metric_slicer_stackers:
        for info_label, pdconstraints in allband_pdconstraints:
            metric_bundle_kwargs = {
                'info_label': info_label,
                'pdconstraint': pdconstraints,
                'plot_funcs': []
            }
            if stackers is not None:
                metric_bundle_kwargs['stacker_list'] = stackers
            if summary is not None:
                metric_bundle_kwargs['summary_metrics'] = summary
                
            bundle = metric_bundles.MetricBundle(
                metric,
                slicer,
                "",
                **metric_bundle_kwargs
            )
            bundle_list.append(bundle)        


    benchmarkArea = 18000
    benchmarkNvisits = 825
    minNvisits = 750

    # Configure the count metric which is what is used for f0 slicer.
    metric = metrics.CountExplimMetric(metric_name="fO")
    summary_metrics = [
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
    slicer = slicers.HealpixSlicer(nside=nside)
    bundle = metric_bundles.MetricBundle(
        metric,
        slicer,
        "",
        summary_metrics=summary_metrics,
        plot_funcs=[],
    )
    bundle_list.append(bundle)

    for b in bundle_list:
        b.set_run_name(run_name)

    bd = metric_bundles.make_bundles_dict_from_list(bundle_list)


    return bd

