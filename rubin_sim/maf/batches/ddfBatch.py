import numpy as np
import healpy as hp
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.plots as plots
import rubin_sim.maf.maps as maps
import rubin_sim.maf.metricBundles as mb
from .common import standardSummary, filterList
from .colMapDict import ColMapDict
from rubin_sim.utils import hpid2RaDec, angularSeparation, ddf_locations
from copy import deepcopy


def ddfBatch(colmap=None, runName='opsim', nside=256, radius=3.):
    if colmap is None:
        colmap = ColMapDict('fbs')

    radius = radius
    bundleList = []

    dd_surveys = ddf_locations()

    hpid = np.arange(hp.nside2npix(nside))
    hp_ra, hp_dec = hpid2RaDec(nside, hpid)
    sql = ''

    # XXX--add some captions noting that the chip gaps are on. I should make a spatial 
    # plotter that does a gnomic projection rotated to mean position.

    summary_stats = [metrics.SumMetric(), metrics.MedianMetric()]

    # put the metrics here so we can copy rather than load from disk each time
    num_metric = metrics.SNNSNMetric(verbose=False)
    lens_metric = metrics.SNSLMetric(night_collapse=True)

    displayDict = {'group': 'DDFs', 'subgroup': ''}

    for ddf in dd_surveys:
        if ddf != 'EDFS_a':
            if 'EDFS_' not in ddf:
                dist = angularSeparation(dd_surveys[ddf][0], dd_surveys[ddf][1], hp_ra, hp_dec)
                good_pix = np.where(dist <= radius)[0]
            elif ddf == 'EDFS_b':
                # Combine the Euclid fields into 1
                d1 = angularSeparation(dd_surveys['EDFS_a'][0], dd_surveys['EDFS_a'][1], hp_ra, hp_dec)
                good_pix1 = np.where(d1 <= radius)[0]
                d2 = angularSeparation(dd_surveys['EDFS_b'][0], dd_surveys['EDFS_b'][1], hp_ra, hp_dec)
                good_pix2 = np.where(d2 <= radius)[0]
                good_pix = np.unique(np.concatenate((good_pix1, good_pix2)))

            slicer = slicers.UserPointsSlicer(ra=hp_ra[good_pix],
                                              dec=hp_dec[good_pix],
                                              useCamera=True, radius=1.75*2**0.5)
            # trick the metrics into thinking they are using healpix slicer
            slicer.slicePoints['nside'] = nside
            slicer.slicePoints['sid'] = good_pix

            name = ddf.replace('DD:', '').replace('_b', '')
            metric = deepcopy(num_metric)
            metric.name = 'SnN_%s' % name
            displayDict['subgroup'] = name
            displayDict['caption'] = 'SNe Ia, with chip gaps on'
            bundleList.append(mb.MetricBundle(metric, slicer, sql, summaryMetrics=summary_stats,
                                              plotFuncs=[plots.HealpixSkyMap()], displayDict=displayDict))

            metric = deepcopy(lens_metric)
            displayDict['caption'] = 'Strongly lensed SNe, with chip gaps on'
            metric.name = 'SnL_%s' % name
            bundleList.append(mb.MetricBundle(metric, slicer, sql, summaryMetrics=summary_stats,
                                              plotFuncs=[plots.HealpixSkyMap()]))

    for b in bundleList:
        b.setRunName(runName)
    bundleDict = mb.makeBundlesDictFromList(bundleList)

    return bundleDict
