import os
import warnings
import glob
import numpy as np
import pandas as pd
from rubin_sim.maf.db import ResultsDb
import rubin_sim.maf.metricBundles as mb
import rubin_sim.maf.plots as plots

__all__ = ['RunComparison']


class RunComparison():
    """
    Class to read multiple results databases, find requested summary metric comparisons,
    and stores results in DataFrames in class.

    This class can operate either as:
    * define a single root directory, automatically (recursively) find all subdirectories that contain
    resultsDbs (in which case, leave runDirs as None)
    * define the directories in which to search for resultsDb (no search for further subdirectories, and
    limits the search to the directories listed). In this case, the root directory can be specified
    (and then further directory paths are relative to this root directory) or defined as None, in which case
    the full path names must be specified for each directory).

    The runNames (simulation names) are fetched from the resultsDB directly. This relies on the user
    specifying the simulation name when the metrics are run.

    This class can also pull information from the resultsDb about where files for the metric data
    are located; this is helpful to re-read data from disk and plot of multiple runs in the same image.

    Parameters
    ----------
    baseDir : `str` or None
        The root directory containing all of the underlying runs and their subdirectories.
        If this is "None", then the full pathnames must be specified in runDirs. If
        not None, then runDirs (if specified) is assumed to contain relative pathnames.
    runDirs : `list` or None
        A list of directories containing MAF outputs and resultsDB_sqlite.db files.
        If this is None, the baseDir is searched (recursively) for directories containing resultsDB files.
        The contents of runDirs can be relative paths (in which case baseDir must be specified) or
        absolute paths (in which case baseDir must be None).
    defaultResultsDb : `str`, opt
        This should be the expected name for the resultsDB files in each directory.
        Default is resultsDb_sqlite.db, which is also the default for the resultsDB class.
    """
    def __init__(self, baseDir=None, runDirs=None,
                 defaultResultsDb='resultsDb_sqlite.db'):
        self.defaultResultsDb = defaultResultsDb
        self.baseDir = baseDir
        if runDirs is not None:
            if baseDir is not None:
                self.runDirs = [os.path.join(self.baseDir, r) for r in runDirs]
            else:
                self.runDirs = runDirs
            # Check if each of these specified run directories contain a resultsDb file
            runDirs = []
            for r in self.runDirs:
                if not (os.path.isfile(os.path.join(r, self.defaultResultsDb))):
                    warnings.warn(f'Could not find resultsDb file {self.defaultResultsDb} in {r}')
                else:
                    runDirs.append(r)
            self.runDirs = runDirs
        else:
            if self.baseDir is None:
                raise Exception('Both baseDir and runDirs cannot be None - please specify '
                                'baseDir to search recursively for resultsDb files, or '
                                'runDirs to search specific directories for resultsDb files.')
            # Find subdirectories with resultsDb files
            self.runDirs = [r.replace(f'/{self.defaultResultsDb}', '') for r in
                            glob.glob(self.baseDir + "/**/" + self.defaultResultsDb, recursive=True)]
        self._connect_to_results()
        # Class attributes to store the stats data:
        self.summaryStats = None      # summary stats
        self.normalizedStats = None   # normalized (to baselineRun) version of the summary stats
        self.baselineRun = None       # name of the baseline run

    def _connect_to_results(self):
        """
        Open access to all the results database files.
        Sets up dictionary of connections.
        """
        # Open access to all results database files in self.runDirs
        self.runresults = {}
        # Make a look-up table for simulation runName - runDir.
        # This is really only used in case the user wants to double-check which runs are represented.
        self.runNames = {}
        for rdir in self.runDirs:
            # Connect to resultsDB
            self.runresults[rdir] = ResultsDb(outDir=rdir)
            # Get simulation names
            self.runNames[rdir] = self.runresults[rdir].getSimDataName()

    def close(self):
        """
        Close all connections to the results database files.
        """
        self.__del__()

    def __del__(self):
        for r in self.runresults:
            self.runresults[r].close()

    def buildMetricDict(self, metricNameLike=None, metricMetadataLike=None,
                        slicerNameLike=None):
        """Return a metric dictionary based on finding all metrics which match 'like' the various kwargs.
        Note that metrics may not be present in all runDirs, and may not all have summary statistics.

        Parameters
        ----------
        metricNameLike: `str`, optional
            Metric name like this -- i.e. will look for metrics which match metricName like "value".
        metricMetadataLike: `str`, optional
            Metric Metadata like this.
        slicerNameLike: `str`, optional
            Slicer name like this.

        Returns
        -------
        mDict : `dict`
            Dictionary of union of metric bundle information across all directories.
            Key = self-created metric 'name', value = Dict{metricName, metricMetadata, slicerName}
        """
        if metricNameLike is None and metricMetadataLike is None and slicerNameLike is None:
            getAll = True
        else:
            getAll = False
        mDict = {}

        # Go through each results database and gather up all of the available metric bundles
        for r in self.runDirs:
            if getAll:
                mIds = self.runresults[r].getAllMetricIds()
            else:
                mIds = self.runresults[r].getMetricIdLike(metricNameLike=metricNameLike,
                                                          metricMetadataLike=metricMetadataLike,
                                                          slicerNameLike=slicerNameLike)
            for mId in mIds:
                info = self.runresults[r].getMetricInfo(mId)
                metricName = info['metricName'][0]
                metricMetadata = info['metricMetadata'][0]
                slicerName = info['slicerName'][0]
                # Build a hash from the metric Name, metadata, slicer --
                # this will automatically remove duplicates
                hash = ResultsDb.buildSummaryName(metricName, metricMetadata, slicerName, None)
                mDict[hash] = {'metricName': metricName,
                               'metricMetadata': metricMetadata,
                               'slicerName': slicerName}
        return mDict

    def _findSummaryStats(self, metricName, metricMetadata=None, slicerName=None, summaryName=None,
                          verbose=False):
        """
        Look for summary metric values matching metricName (and optionally metricMetadata, slicerName
        and summaryName) among the results databases.
        Note that some metrics may not be present in some runDirs.

        Parameters
        ----------
        metricName : `str`
            The name of the original metric.
        metricMetadata : `str`, optional
            The metric metadata specifying the metric desired (optional).
        slicerName : `str`, optional
            The slicer name specifying the metric desired (optional).
        summaryName : `str`, optional
            The name of the summary statistic desired (optional).
        verbose : `bool`, optional
            Issue warnings resulting from not finding the summary stat information
            (such as if it was never calculated) will not be issued.   Default False.

        Returns
        -------
        summaryStats: `pd.DataFrame`
            <index>   <metricName>  (possibly additional metricNames - multiple summary stats or metadata..)
             runName    value
        """
        summaryValues = {}
        for r in self.runDirs:
            # Look for this metric/metadata/slicer/summary stat name combo in this resultsDb.
            mId = self.runresults[r].getMetricId(metricName=metricName,
                                                 metricMetadata=metricMetadata,
                                                 slicerName=slicerName)
            # Note that we may have more than one matching summary metric value per resultsDb.
            stats = self.runresults[r].getSummaryStats(mId, summaryName=summaryName, withSimName=True)
            for i in range(len(stats['summaryName'])):
                name = stats['summaryName'][i]
                runName = stats['simDataName'][i]
                if runName not in summaryValues:
                    summaryValues[runName] = {}
                summaryValues[runName][name] = stats['summaryValue'][i]
            if len(stats) == 0 and verbose:
                warnings.warn("Warning: Found no metric results for %s %s %s %s in run %s"
                              % (metricName, metricMetadata, slicerName, summaryName, r))
        # Make DataFrame for stat values
        stats = pd.DataFrame(summaryValues).T
        return stats

    def addSummaryStats(self, metricDict=None, verbose=False):
        """
        Combine the summary statistics of a set of metrics into a pandas
        dataframe that is indexed by the opsim run name.

        Parameters
        ----------
        metricDict: `dict`, optional
            A dictionary of metrics with all of the information needed to query
            a results database.  The metric/metadata/slicer/summary values referred to
            by a metricDict value could be unique but don't have to be.
            If None (default), then fetches all metric results.
            (This can be slow if there are a lot of metrics.)
        verbose : `bool`, optional
            Issue warnings resulting from not finding the summary stat information
            (such as if it was never calculated) will not be issued.   Default False.


        Sets self.summaryStats
        """
        if metricDict is None:
            metricDict = self.buildMetricDict()
        for mName, metric in metricDict.items():
            # In general this will not be present (if only auto-built metric dictionary)
            # But the summaryMetric could be specified (if only 'Medians' were desired, etc.)
            if 'summaryMetric' not in metric:
                metric['summaryMetric'] = None
            tempStats = self._findSummaryStats(metricName=metric['metricName'],
                                               metricMetadata=metric['metricMetadata'],
                                               slicerName=metric['slicerName'],
                                               summaryName=metric['summaryMetric'],
                                               verbose=verbose)
            if self.summaryStats is None:
                self.summaryStats = tempStats
            else:
                self.summaryStats = self.summaryStats.join(tempStats, how='outer', lsuffix='_x')

        self.summaryStats.index.name = 'run_name'
        self.summaryStats.columns.name = 'metric'

    def getFileNames(self, metricName, metricMetadata=None, slicerName=None):
        """Find the locations of a given metric in all available directories.

        Parameters
        ----------
        metricName : `str`
            The name of the original metric.
        metricMetadata : `str`, optional
            The metric metadata specifying the metric desired (optional).
        slicerName : `str`, optional
            The slicer name specifying the metric desired (optional).

        Returns
        -------
        filepaths: `dict`
            Keys: runName, Value: path to file
        """
        filepaths = {}
        for r in self.runDirs:
            mId = self.runresults[r].getMetricId(metricName=metricName,
                                                 metricMetadata=metricMetadata,
                                                 slicerName=slicerName)
            if len(mId) > 0:
                if len(mId) > 1:
                    warnings.warn("Found more than one metric data file matching " +
                                  "metricName %s metricMetadata %s and slicerName %s"
                                  % (metricName, metricMetadata, slicerName) +
                                  " Skipping this combination.")
                else:
                    filename = self.runresults[r].getMetricDataFiles(metricId=mId)
                    filepaths[r] = os.path.join(r, filename[0])
        return filepaths

    # Plot actual metric values (skymaps or histograms or power spectra) (values not stored in class).
    def readMetricData(self, metricName, metricMetadata, slicerName):
        # Get the names of the individual files for all runs.
        # Dictionary, keyed by run name.
        filenames = self.getFileNames(metricName, metricMetadata, slicerName)
        mname = ResultsDb.buildSummaryName(metricName, metricMetadata, slicerName, None)
        bundleDict = {}
        for r in filenames:
            b = mb.createEmptyMetricBundle()
            b.read(filenames[r])
            hash = b.runName + ' ' + mname
            bundleDict[hash] = b
        return bundleDict, mname

    def plotMetricData(self, bundleDict, plotFunc, userPlotDict=None,
                       layout=None, outDir=None, savefig=False):
        if userPlotDict is None:
            userPlotDict = {}

        ph = plots.PlotHandler(outDir=outDir, savefig=savefig)
        ph.setMetricBundles(bundleDict)

        plotDicts = [{} for b in bundleDict]
        # Depending on plotFunc, overplot or make many subplots.
        if plotFunc.plotType == 'SkyMap':
            # Note that we can only handle 9 subplots currently due
            # to how subplot identification (with string) is handled.
            if len(bundleDict) > 9:
                raise ValueError('Please try again with < 9 subplots for skymap.')
            # Many subplots.
            if 'colorMin' not in userPlotDict:
                colorMin = 100000000
                for b in bundleDict:
                    if 'zp' not in bundleDict[b].plotDict:
                        tmp = bundleDict[b].metricValues.compressed().min()
                        colorMin = min(tmp, colorMin)
                    else:
                        colorMin = bundleDict[b].plotDict['colorMin']
                userPlotDict['colorMin'] = colorMin
            if 'colorMax' not in userPlotDict:
                colorMax = -100000000
                for b in bundleDict:
                    if 'zp' not in bundleDict[b].plotDict:
                        tmp = bundleDict[b].metricValues.compressed().max()
                        colorMax = max(tmp, colorMax)
                    else:
                        colorMax = bundleDict[b].plotDict['colorMax']
                userPlotDict['colorMax'] = colorMax
            for i, (pdict, bundle) in enumerate(zip(plotDicts, bundleDict.values())):
                # Add user provided dictionary.
                pdict.update(userPlotDict)
                # Set subplot information.
                if layout is None:
                    ncols = int(np.ceil(np.sqrt(len(bundleDict))))
                    nrows = int(np.ceil(len(bundleDict) / float(ncols)))
                else:
                    ncols = layout[0]
                    nrows = layout[1]
                pdict['subplot'] = int(str(nrows) + str(ncols) + str(i + 1))
                pdict['title'] = bundle.runName
                # For the subplots we do not need the label
                pdict['label'] = ''
                pdict['legendloc'] = None
                if 'suptitle' not in userPlotDict:
                    pdict['suptitle'] = ph._buildTitle()
        elif plotFunc.plotType == 'Histogram':
            # Put everything on one plot.
            if 'xMin' not in userPlotDict:
                xMin = 100000000
                for b in bundleDict:
                    if 'zp' not in bundleDict[b].plotDict:
                        tmp = bundleDict[b].metricValues.compressed().min()
                        xMin = min(tmp, xMin)
                    else:
                        xMin = bundleDict[b].plotDict['xMin']
                userPlotDict['xMin'] = xMin
            if 'xMax' not in userPlotDict:
                xMax = -100000000
                for b in bundleDict:
                    if 'zp' not in bundleDict[b].plotDict:
                        tmp = bundleDict[b].metricValues.compressed().max()
                        xMax = max(tmp, xMax)
                    else:
                        xMax = bundleDict[b].plotDict['xMax']
                userPlotDict['xMax'] = xMax
            for i, pdict in enumerate(plotDicts):
                pdict.update(userPlotDict)
                # Legend and title will automatically be ok, I think.
        elif plotFunc.plotType == 'BinnedData':
            # Put everything on one plot.
            if 'yMin' not in userPlotDict:
                yMin = 100000000
                for b in bundleDict:
                    tmp = bundleDict[b].metricValues.compressed().min()
                    yMin = min(tmp, yMin)
                userPlotDict['yMin'] = yMin
            if 'yMax' not in userPlotDict:
                yMax = -100000000
                for b in bundleDict:
                    tmp = bundleDict[b].metricValues.compressed().max()
                    yMax = max(tmp, yMax)
                userPlotDict['yMax'] = yMax
            if 'xMin' not in userPlotDict:
                xMin = 100000000
                for b in bundleDict:
                    tmp = bundleDict[b].slicer.slicePoints['bins'].min()
                    xMin = min(tmp, xMin)
                userPlotDict['xMin'] = xMin
            if 'xMax' not in userPlotDict:
                xMax = -100000000
                for b in bundleDict:
                    tmp = bundleDict[b].slicer.slicePoints['bins'].max()
                    xMax = max(tmp, xMax)
                userPlotDict['xMax'] = xMax
            for i, pdict in enumerate(plotDicts):
                pdict.update(userPlotDict)
                # Legend and title will automatically be ok, I think.
        ph.plot(plotFunc, plotDicts=plotDicts)
