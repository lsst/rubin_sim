import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.metricBundles as mb
from .colMapDict import ColMapDict
from .common import extendedSummary, filterList, radecCols

__all__ = ['agnBatch']


def agnBatch(colmap=None, runName='opsim', nside=64,
        extraSql=None, extraMetadata=None, slicer=None,
        display_group='AGN'):
    """Generate a set of statistics about the spacing between nights with observations.

     Parameters
     ----------
     colmap : dict or None, optional
         A dictionary with a mapping of column names. Default will use OpsimV4 column names.
     runName : str, optional
         The name of the simulated survey. Default is "opsim".
     nside : int, optional
         Nside for the healpix slicer. Default 64.
     extraSql : str or None, optional
         Additional sql constraint to apply to all metrics.
     extraMetadata : str or None, optional
         Additional metadata to use for all outputs.
     slicer : slicer object (None)
         Optionally use something other than a HealpixSlicer

     Returns
     -------
     metricBundleDict
     """

    if colmap is None:
        colmap = ColMapDict('fbs')

    bundleList = []

    # Set up basic per filter sql constraints.
    raCol, decCol, degrees, ditherStacker, ditherMeta = radecCols(None, colmap, None)
    filterlist, colors, orders, sqls, metadata = filterList(all=False,
                                                            extraSql=extraSql,
                                                            extraMetadata=extraMetadata)

    if slicer is None:
        slicer = slicers.HealpixSlicer(nside=nside, latCol=decCol, lonCol=raCol, latLonDeg=degrees,
                                       useCache=False)

    displayDict = {'group': display_group,  'order': 0}

    # These agn test magnitude values are determined by looking at the baseline median m5 depths
    # For v1.7.1 these values are:
    agn_m5 = {'u': 22.89, 'g': 23.94, 'r': 23.5, 'i': 22.93, 'z': 22.28, 'y': 21.5}
    # And the expected median SF error at those values is about 0.04
    threshold = 0.04

    summaryMetrics = extendedSummary()
    summaryMetrics += [metrics.AreaThresholdMetric(upper_threshold = threshold)]
    for f in filterlist:
        m = metrics.SFErrorMetric(mag=agn_m5[f], metricName='AGN SF_error',
                                  timesCol=colmap['mjd'], m5Col=colmap['fiveSigmaDepth'],
                                  filterCol=colmap['filter'])
        plotDict = {'color': colors[f]}
        displayDict['order'] = orders[f]
        displayDict['subgroup'] = 'SFError'
        displayDict['caption'] = 'Expected AGN structure function errors, based on observations in ' \
                                 f'{f} band, for an AGN of magnitude {agn_m5[f]:.2f}'
        bundleList.append(mb.MetricBundle(m, slicer, constraint=sqls[f], metadata=metadata[f],
                                          runName=runName, plotDict=plotDict,
                                          summaryMetrics=summaryMetrics,
                                          displayDict=displayDict))
    plotBundles = None
    return mb.makeBundlesDictFromList(bundleList), plotBundles
