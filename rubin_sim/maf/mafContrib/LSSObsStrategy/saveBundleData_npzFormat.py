#####################################################################################################
# Purpose: save data from a metricBundle object as .npz files.

# Humna Awan: humna.awan@rutgers.edu
#####################################################################################################

__all__ = ['saveBundleData_npzFormat']

def saveBundleData_npzFormat(path, bundle, baseFilename, filterBand):
    """

     Save data in the metricBundle. For each key, a new .npz file is created to
     save the content of the metricBundle object.
     
     Required Parameters
     -------------------
       * path: str: path to the directory where the files should be saved
       * bundle: metricBundle object whose contents are to be saved.
       * baseFilename: str: basic filename wanted. Final filename would be
                            <baseFilename>_<filterBand>_<dither>.npz
       * filterBand: str: filter of the data in the bundle, e.g. 'r'

    """
    # run over keys in the bundle and save the data
    for dither in bundle:
        outfile = '%s_%s_%s.npz'%(baseFilename, filterBand, dither)
        bundle[dither].slicer.writeData('%s/%s'%(path, outfile),
                                        bundle[dither].metricValues,
                                        metricName = bundle[dither].metric.name,
                                        simDataName = bundle[dither].runName,
                                        constraint = bundle[dither].constraint,
                                        metadata = bundle[dither].metadata,
                                        displayDict = bundle[dither].displayDict,
                                        plotDict = bundle[dither].plotDict)
