import numpy as np
import healpy as hp
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.plots as plots
import rubin_sim.maf.maps as maps
import rubin_sim.maf.metricBundles as mb
from .common import standardSummary, filterList
from .colMapDict import ColMapDict
from rubin_sim.scheduler.surveys import generate_dd_surveys
from rubin_sim.utils import hpid2RaDec, angularSeparation
from copy import deepcopy


def ddfBatch(colmap=None, runName='opsim', nside=256, radius=3.):
    if colmap is None:
        colmap = ColMapDict('fbs')

    radius = radius
    bundleList = []

    dd_surveys = generate_dd_surveys()

    hpid = np.arange(hp.nside2npix(nside))
    hp_ra, hp_dec = hpid2RaDec(nside, hpid)
    sql = ''

    # XXX--add some captions noting that the chip gaps are on. I should make a spatial 
    # plotter that does a gnomic projection rotated to mean position.

    summary_stats = [metrics.SumMetric(), metrics.MedianMetric()]

    # put the metrics here so we can copy rather than load from disk each time
    num_metric = metrics.SNNSNMetric(verbose=False)
    lens_metric = metrics.SNSLMetric(night_collapse=True)

    for ddf in dd_surveys:
        # If it's the euclid double field
        if np.size(ddf.ra) > 1:
            dist = angularSeparation(np.degrees(np.mean(ddf.ra)), np.degrees(np.mean(ddf.dec)), hp_ra, hp_dec)
            good_pix = np.where(dist <= (radius*2))[0]
        else:
            dist = angularSeparation(np.degrees(ddf.ra), np.degrees(ddf.dec), hp_ra, hp_dec)
            good_pix = np.where(dist <= radius)[0]
        slicer = slicers.UserPointsSlicer(ra=hp_ra[good_pix],
                                          dec=hp_dec[good_pix],
                                          useCamera=True, radius=1.75*2**0.5)
        # trick the metrics into thinking they are using healpix slicer
        slicer.slicePoints['nside'] = nside

        name = ddf.survey_name.replace('DD:', '')
        metric = deepcopy(num_metric)
        metric.name = 'SnN_%s' % name
        bundleList.append(mb.MetricBundle(metric, slicer, sql, summaryMetrics=summary_stats,
                                          plotFuncs=[plots.HealpixSkyMap()]))

        metric = deepcopy(lens_metric)
        metric.name = 'SnL_%s' % name
        bundleList.append(mb.MetricBundle(metric, slicer, sql, summaryMetrics=summary_stats,
                                          plotFuncs=[plots.HealpixSkyMap()]))

    for b in bundleList:
        b.setRunName(runName)
    bundleDict = mb.makeBundlesDictFromList(bundleList)

    return bundleDict
