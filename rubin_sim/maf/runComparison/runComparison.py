from builtins import zip
from builtins import range
from builtins import object
import os
import warnings
import numpy as np
import pandas as pd
from rubin_sim.maf.db import ResultsDb
import rubin_sim.maf.metricBundles as mb
import rubin_sim.maf.plots as plots

_BOKEH_HERE = True
try:
    from bokeh.models import CustomJS, ColumnDataSource
    from bokeh.io import output_file, output_notebook
    from bokeh.layouts import widgetbox, layout, row, column
    from bokeh.models.widgets import DataTable, DateFormatter, TableColumn, NumberFormatter, Select
    from bokeh.plotting import Figure, output_file, show
    output_notebook()
except ImportError:
    _BOKEH_HERE = False


__all__ = ['RunComparison']


class RunComparison(object):
    """
    Class to read multiple results databases, find requested summary metric comparisons,
    and stores results in DataFrames in class.

    Set up the runs to compare and opens connections to all resultsDb_sqlite directories under
    baseDir/runNames[1-N] and their subdirectories.
    There are two ways to approach the storage and access to the MAF outputs:
    EITHER the outputs can be stored directly in the runNames directories or subdirectories of these:
    baseDir -> run1  -> subdirectory1 (e.g. 'scheduler', containing a resultsDb_sqlite.db file)
    ................ -> subdirectoryN
    ....... -> runN -> subdirectoryX
    OR the outputs can be stored in a variety of different locations, and the names/locations
    then would be provided by [runNames][rundirs] -- having a one-to-one correlation. In this case, you might
    expect the runNames to contain duplicates if there is more than one MAF output directory per run.

    Parameters
    ----------
    baseDir : `str`
        The root directory containing all of the underlying runs and their subdirectories.
    runNames : `list` [`str`]
        The names to label different runs. Can contain duplicate entries.
    rundirs : `list`
        A list of directories (relative to baseDir) where the MAF outputs in runNames reside.
        Optional - if not provided, assumes directories are simply the names in runNames.
        Must have same length as runNames (note that runNames can contain duplicate entries).
    """
    def __init__(self, baseDir, runNames, rundirs=None,
                 defaultResultsDb='resultsDb_sqlite.db', verbose=False):
        self.baseDir = baseDir
        self.runlist = runNames
        self.verbose = verbose
        self.defaultResultsDb = defaultResultsDb
        if rundirs is not None:
            if len(rundirs) != len(self.runlist):
                raise ValueError('runNames and rundirs must be the same length')
            self.rundirs = rundirs
        else:
            self.rundirs = self.runlist
        self._connect_to_results()
        # Class attributes to store the stats data:
        self.headerStats = None       # Save information on the summary stat values
        self.summaryStats = None      # summary stats
        self.normalizedStats = None   # normalized (to baselineRun) version of the summary stats
        self.baselineRun = None       # name of the baseline run

    def _connect_to_results(self):
        """
        Open access to all the results database files.
        Sets nested dictionary of results databases:
        .. dictionary[run1][subdirectory1] = resultsDb
        .. dictionary[run1][subdirectoryN] = resultsDb ...
        """
        # Open access to all results database files in any subdirectories under 'runs'.
        self.runresults = {}
        for r, rdir in zip(self.runlist, self.rundirs):
            checkdir = os.path.join(self.baseDir, rdir)
            if not os.path.isdir(checkdir):
                warnings.warn('Warning: could not find a directory at %s' % checkdir)
            else:
                # Add a dictionary to runresults to store resultsDB connections.
                if r not in self.runresults:
                    self.runresults[r] = {}
                # Check for a resultsDB in the current checkdir
                if os.path.isfile(os.path.join(checkdir, self.defaultResultsDb)):
                    s = os.path.split(rdir)[-1]
                    self.runresults[r][s] = ResultsDb(outDir=checkdir)
                # And look for resultsDb files in subdirectories.
                sublist = os.listdir(checkdir)
                for s in sublist:
                    if os.path.isfile(os.path.join(checkdir, s, 'resultsDb_sqlite.db')):
                        self.runresults[r][s] = ResultsDb(outDir=os.path.join(checkdir, s))
        # Remove any runs from runlist which we could not find results databases for.
        for r in self.runlist:
            if len(self.runresults[r]) == 0:
                warnings.warn('Warning: could not find any results databases for run %s'
                              % (os.path.join(self.baseDir, r)))
        # Now de-duplicate the runlist (we don't need to loop over extra items).
        self.runlist = list(self.runresults.keys())

    def close(self):
        """
        Close all connections to the results database files.
        """
        self.__del__()

    def __del__(self):
        for r in self.runresults:
            for s in self.runresults[r]:
                self.runresults[r][s].close()

    def buildMetricDict(self, metricNameLike=None, metricMetadataLike=None,
                        slicerNameLike=None, subdir=None):
        """Return a metric dictionary based on finding all metrics which match 'like' the various kwargs.

        Parameters
        ----------
        metricNameLike: str, optional
            Metric name like this -- i.e. will look for metrics which match metricName like "value".
        metricMetadataLike: str, optional
            Metric Metadata like this.
        slicerNameLike: str, optional
            Slicer name like this.
        subdir: str, optional
            Find metrics from this subdir only.
            If other parameters are not specified, this returns all metrics within this subdir.

        Returns
        -------
        Dict
            Key = self-created metric 'name', value = Dict{metricName, metricMetadata, slicerName}
        """
        if metricNameLike is None and metricMetadataLike is None and slicerNameLike is None:
            getAll = True
        else:
            getAll = False
        mDict = {}

        # Track this here, so that if directories have different subdirectories, this will still work.
        insubdir = subdir
        for r in self.runlist:
            if insubdir is not None:
                subdirs = [insubdir]
            else:
                subdirs = list(self.runresults[r].keys())
            for subdir in subdirs:
                if getAll:
                    mIds = self.runresults[r][subdir].getAllMetricIds()
                else:
                    mIds = self.runresults[r][subdir].getMetricIdLike(metricNameLike=metricNameLike,
                                                                      metricMetadataLike=metricMetadataLike,
                                                                      slicerNameLike=slicerNameLike)
                for mId in mIds:
                    info = self.runresults[r][subdir].getMetricInfo(mId)
                    metricName = info['metricName'][0]
                    metricMetadata = info['metricMetadata'][0]
                    slicerName = info['slicerName'][0]
                    name = self._buildSummaryName(metricName, metricMetadata, slicerName, None)
                    mDict[name] = {'metricName': metricName,
                                   'metricMetadata': metricMetadata,
                                   'slicerName': slicerName}
        return mDict

    def _buildSummaryName(self, metricName, metricMetadata, slicerName, summaryStatName):
        if metricMetadata is None:
            metricMetadata = ''
        if slicerName is None:
            slicerName = ''
        sName = summaryStatName
        if sName == 'Identity' or sName == 'Id' or sName == 'Count' or sName is None:
            sName = ''
        slName = slicerName
        if slName == 'UniSlicer':
            slName = ''
        name = ' '.join([sName, metricName, metricMetadata, slName]).rstrip(' ').lstrip(' ')
        name.replace(',', '')
        return name

    def _findSummaryStats(self, metricName, metricMetadata=None, slicerName=None, summaryName=None,
                          colName=None, verbose=False):
        """
        Look for summary metric values matching metricName (and optionally metricMetadata, slicerName
        and summaryName) among the results databases for each run.

        Parameters
        ----------
        metricName : str
            The name of the original metric.
        metricMetadata : str, optional
            The metric metadata specifying the metric desired (optional).
        slicerName : str, optional
            The slicer name specifying the metric desired (optional).
        summaryName : str, optional
            The name of the summary statistic desired (optional).
        colName : str, optional
            Name of the column header for the dataframe. If more than one summary stat is
            returned from the database, then this will be ignored.
        verbose : bool, optional
            Issue warnings resulting from not finding the summary stat information
            (such as if it was never calculated) will not be issued.   Default False.

        Returns
        -------
        summaryStats: `pd.DataFrame`
            <index>   <metricName>  (possibly additional metricNames - multiple summary stats or metadata..)
             runName    value
        """
        summaryValues = {}
        summaryNames = {}
        for r in self.runlist:
            summaryValues[r] = {}
            summaryNames[r] = {}
            # Check if this metric/metadata/slicer/summary stat name combo is in
            # this resultsDb .. or potentially in another subdirectory's resultsDb.
            for subdir in self.runresults[r]:
                mId = self.runresults[r][subdir].getMetricId(metricName=metricName,
                                                             metricMetadata=metricMetadata,
                                                             slicerName=slicerName)
                # Note that we may have more than one matching summary metric value per run.
                if len(mId) > 0:
                    # And we may have more than one summary metric value per resultsDb
                    stats = self.runresults[r][subdir].getSummaryStats(mId, summaryName=summaryName)
                    if len(stats['summaryName']) == 1 and colName is not None:
                        name = colName
                        summaryValues[r][name] = stats['summaryValue'][0]
                        summaryNames[r][name] = stats['summaryName'][0]
                    else:
                        for i in range(len(stats['summaryName'])):
                            name = self._buildSummaryName(metricName, metricMetadata, slicerName,
                                                          stats['summaryName'][i])
                            summaryValues[r][name] = stats['summaryValue'][i]
                            summaryNames[r][name] = stats['summaryName'][i]
            if len(summaryValues[r]) == 0 and verbose:
                warnings.warn("Warning: Found no metric results for %s %s %s %s in run %s"
                              % (metricName, metricMetadata, slicerName, summaryName, r))
        # Make DataFrame.
        # First determine list of all summary stats pulled from all databases.
        unique_stats = set()
        for r in self.runlist:
            for name in summaryNames[r]:
                unique_stats.add(name)
        # Make sure every runName (key) in summaryValues dictionary has a value for each stat.
        # And build summaryname properly
        suNames = {}
        for s in unique_stats:
            for r in self.runlist:
                try:
                    summaryValues[r][s]
                    suNames[s] = summaryNames[r][s]
                except KeyError:
                    summaryValues[r][s] = np.nan
        # Create data frames for each run. This is the simplest way to handle it in pandas.
        summaryBase = {}
        mName = {}
        mData = {}
        sName = {}
        basemetricname = self._buildSummaryName(metricName, metricMetadata, slicerName, None)
        for s in unique_stats:
            summaryBase[s] = basemetricname
            mName[s] = metricName
            mData[s] = metricMetadata
            sName[s] = slicerName
        header = pd.DataFrame([summaryBase, mName, mData, sName, suNames],
                              index=['BaseName', 'MetricName', 'MetricMetadata',
                                     'SlicerName', 'SummaryName'])
        tempDFList = []
        for r in self.runlist:
            tempDFList.append(pd.DataFrame(summaryValues[r], index=[r]))
        # Concatenate dataframes for each run.
        stats = pd.concat(tempDFList)
        return header, stats

    def addSummaryStats(self, metricDict=None, verbose=False):
        """
        Combine the summary statistics of a set of metrics into a pandas
        dataframe that is indexed by the opsim run name.and

        Parameters
        ----------
        metricDict: dict, optional
            A dictionary of metrics with all of the information needed to query
            a results database.  The metric/metadata/slicer/summary values referred to
            by a metricDict value could be unique but don't have to be.
            If None (default), then fetches all metric results.
            (This can be slow if there are a lot of metrics.)
        verbose : bool, optional
            Issue warnings resulting from not finding the summary stat information
            (such as if it was never calculated) will not be issued.   Default False.

        Returns
        -------
        pandas DataFrame
            A pandas dataframe containing a column for each of the
            dictionary keys and related summary stats in the metricDict.
            The resulting dataframe is indexed by runNames.
            index      metric1         metric2
            <run_123>    <metricValue1>  <metricValue2>
            <run_124>    <metricValue1>  <metricValue2>
        """
        if metricDict is None:
            metricDict = self.buildMetricDict()
        for mName, metric in metricDict.items():
            if 'summaryName' not in metric:
                metric['summaryName'] = None
            tempHeader, tempStats = self._findSummaryStats(metricName=metric['metricName'],
                                                           metricMetadata=metric['metricMetadata'],
                                                           slicerName=metric['slicerName'],
                                                           summaryName=metric['summaryName'],
                                                           colName=mName, verbose=verbose)
            if self.summaryStats is None:
                self.summaryStats = tempStats
                self.headerStats = tempHeader
            else:
                self.summaryStats = self.summaryStats.join(tempStats, lsuffix='_x')
                self.headerStats = self.headerStats.join(tempHeader, lsuffix='_x')

    def normalizeStats(self, baselineRun):
        """
        Normalize the summary metric values in the dataframe
        resulting from combineSummaryStats based on the values of a single
        baseline run.

        Parameters
        ----------
        baselineRun : str
            The name of the opsim run that will serve as baseline.

        Returns
        -------
        pandas DataFrame
            A pandas dataframe containing a column for each of the configuration
            parameters given in paramNamelike and a column for each of the
            dictionary keys in the metricDict. The resulting dataframe is
            indexed the name of the opsim runs.
            index        metric1               metric2
            <run_123>    <norm_metricValue1>  <norm_metricValue2>
            <run_124>    <norm_metricValue1>  <norm_metricValue2>


        The metric values are normalized in the following way:
        norm_metric_value(run) = metric_value(run) - metric_value(baselineRun) / metric_value(baselineRun)
        """
        self.normalizedStats = self.summaryStats.copy(deep=True)
        self.normalizedStats = self.normalizedStats - self.summaryStats.loc[baselineRun]
        self.normalizedStats /= self.summaryStats.loc[baselineRun]
        self.baselineRun = baselineRun

    def sortCols(self, baseName=True, summaryName=True):
        """Return the columns (in order) to display a sorted version of the stats dataframe.

        Parameters
        ----------
        baseName : bool, optional
            Sort by the baseName. Default True.
            If True, this takes priority in the sorted results.
        summaryName : bool, optional
            Sort by the summary stat name (summaryName). Default True.

        Returns
        -------
        list
        """
        sortby = []
        if baseName:
            sortby.append('BaseName')
        if summaryName:
            sortby.append('SummaryName')
        o = self.headerStats.sort_values(by=sortby, axis=1)
        return o.columns

    def filterCols(self, summaryName):
        """Return a dataframe containing only stats which match summaryName.

        Parameters
        ----------
        summaryName : str
            The type of summary stat to match. (i.e. Max, Mean)

        Returns
        -------
        pd.DataFrame
        """
        o = self.headerStats.loc['SummaryName'] == summaryName
        return self.summaryStats.loc[:, o]

    def findChanges(self, threshold=0.05):
        """Return a dataframe containing only values which changed by threshhold.

        Parameters
        ----------
        threshold : float, optional
            Identify values which change by more than threshold (%) in the normalized values.
            Default 5% (0.05).

        Returns
        -------
        pd.DataFrame
        """
        o = abs(self.normalizedStats) > 0.05
        o = o.any(axis=0)
        return self.summaryStats.loc[:, o]

    def getFileNames(self, metricName, metricMetadata=None, slicerName=None):
        """For each of the runs in runlist, get the paths to the datafiles for a given metric.

        Parameters
        ----------
        metricName : str
            The name of the original metric.
        metricMetadata : str, optional
            The metric metadata specifying the metric desired (optional).
        slicerName : str, optional
            The slicer name specifying the metric desired (optional).

        Returns
        -------
        Dict
            Keys: runName, Value: path to file
        """
        filepaths = {}
        for r in self.runlist:
            for s in self.runresults[r]:
                mId = self.runresults[r][s].getMetricId(metricName=metricName,
                                                        metricMetadata=metricMetadata,
                                                        slicerName=slicerName)
                if len(mId) > 0 :
                    if len(mId) > 1:
                        warnings.warn("Found more than one metric data file matching " +
                                      "metricName %s metricMetadata %s and slicerName %s"
                                      % (metricName, metricMetadata, slicerName) +
                                      " Skipping this combination.")
                    else:
                        filename = self.runresults[r][s].getMetricDataFiles(metricId=mId)
                        filepaths[r] = os.path.join(r, s, filename[0])
        return filepaths

    # Plot actual metric values (skymaps or histograms or power spectra) (values not stored in class).
    def readMetricData(self, metricName, metricMetadata, slicerName):
        # Get the names of the individual files for all runs.
        # Dictionary, keyed by run name.
        filenames = self.getFileNames(metricName, metricMetadata, slicerName)
        mname = self._buildSummaryName(metricName, metricMetadata, slicerName, None)
        bundleDict = {}
        for r in filenames:
            bundleDict[r] = mb.createEmptyMetricBundle()
            bundleDict[r].read(filenames[r])
        return bundleDict, mname

    def plotMetricData(self, bundleDict, plotFunc, runlist=None, userPlotDict=None,
                       layout=None, outDir=None, savefig=False):
        if runlist is None:
            runlist = self.runlist
        if userPlotDict is None:
            userPlotDict = {}

        ph = plots.PlotHandler(outDir=outDir, savefig=savefig)
        bundleList = []
        for r in runlist:
            bundleList.append(bundleDict[r])
        ph.setMetricBundles(bundleList)

        plotDicts = [{} for r in runlist]
        # Depending on plotFunc, overplot or make many subplots.
        if plotFunc.plotType == 'SkyMap':
            # Note that we can only handle 9 subplots currently due
            # to how subplot identification (with string) is handled.
            if len(runlist) > 9:
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
            for i, pdict in enumerate(plotDicts):
                # Add user provided dictionary.
                pdict.update(userPlotDict)
                # Set subplot information.
                if layout is None:
                    ncols = int(np.ceil(np.sqrt(len(runlist))))
                    nrows = int(np.ceil(len(runlist) / float(ncols)))
                else:
                    ncols = layout[0]
                    nrows = layout[1]
                pdict['subplot'] = int(str(nrows) + str(ncols) + str(i + 1))
                pdict['title'] = runlist[i]
                # For the subplots we do not need the label
                pdict['label'] = ''
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
                pdict['subplot'] = '111'
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
                pdict['subplot'] = '111'
                # Legend and title will automatically be ok, I think.
        if self.verbose:
            print(plotDicts)
        ph.plot(plotFunc, plotDicts=plotDicts)

    def generateDiffHtml(self, normalized = False, html_out = None, show_page = False,
                         combined = False, fullStats = False):
        """
        Use `bokeh` to convert a summaryStats dataframe to interactive html
        table.

        Parameters
        ----------
        normalized : bool, optional
            If True generate html table with normalizedStats
        html_out : str, optional
            Name of the html that will be output and saved. If no string
            is provided then the html table will not be saved.
        show_page : bool, optional
            If True the html page generate by this function will automatically open
            in your browser
        combined : bool, optional
            If True the html produce will have columns for the original
            summaryStats values, as well as their normalized values. The baselineRun
            used to calculate the normalized values will be dropped from the table.
        fullStats : bool, optional
            If False the final html table will not include summaryStats that
            contain '3Sigma','Rms','Min','Max','RobustRms', or '%ile' in their
            names.
        """

        if not _BOKEH_HERE:
            raise ImportError('This method requires bokeh to be installed.'+ '\n'
                              'Run: pip install bokeh'+'\n' +
                              'Then restart your jupyter notebook kernel.')

        if html_out is not None:
            output_file(html_out, title = html_out.strip('.html'))

        if normalized is False:
            # HTML table based on summary stat values
            dataframe = self.headerStats.T.merge(self.summaryStats.T,
                                                        left_index=True, right_index=True)
        else:
            # HTML table based on normalized summary stats
            dataframe = self.headerStats.T.merge(self.normalizedStats.T,
                                                        left_index=True, right_index=True)

        if combined is True:
            # HTML table of combined stat values and normalized values into single table.
            # The baseline run is removed from the final table.
            # The normalized values are given a suffix of '_norm'
            combo = self.summaryStats.T.merge(self.normalizedStats.T, left_index=True, right_index=True,
                                              suffixes=('','_norm')).drop([self.baselineRun+'_norm'],axis=1)

            dataframe = self.headerStats.T.merge(combo, left_index=True, right_index=True)

        dataframe.reset_index(level=0, inplace=True)
        dataframe.columns.values[0]='FullName'

        if fullStats is False:
            # For a more manageable table do no include the summaryStats that
            # have names included in the avoid_summarys list.
            avoid_summarys = ['3Sigma','Rms','Min','Max','RobustRms','%ile']
            summary_pattern = '|'.join(avoid_summarys)

            dataframe = dataframe[((dataframe['SummaryName'].str.contains(summary_pattern))==False) &
                                ((dataframe['MetricName'].str.contains(summary_pattern))==False)]

        columns = []

        for col in dataframe.columns:

            if col not in ['FullName', 'BaseName','MetricName',
                           'MetricMetadata', 'SlicerName', 'SummaryName']:
                columns.append(TableColumn(field=col, title=col,
                                           formatter=NumberFormatter(format="0.0000")))
            else:
                columns.append(TableColumn(field=col, title=col))

        source = ColumnDataSource(dataframe)
        original_source = ColumnDataSource(dataframe)
        data_table = DataTable(source=source, columns=columns, width=1900, height=900)

        js_code = """
        var data = source.data;
        var original_data = original_source.data;
        var FullName= FullName_select_obj.value;
        var BaseName = BaseName_select_obj.value;
        var SummaryName = SummaryName_select_obj.value;
        var MetricName = MetricName_select_obj.value;
        var MetricMetadata = MetricMetadata_select_obj.value;
         for (var key in original_data) {
             data[key] = [];
             for (var i = 0; i < original_data['FullName'].length; ++i) {
                 if ((FullName === "ALL" || original_data['FullName'][i] === FullName) &&
                     (BaseName === "ALL" || original_data['BaseName'][i] === BaseName) &&
                     (MetricMetadata === "ALL" || original_data['MetricMetadata'][i] === MetricMetadata) &&
                     (MetricName === "ALL" || original_data['MetricName'][i] === MetricName) &&
                     (SummaryName === "ALL" || original_data['SummaryName'][i] === SummaryName)) {
                     data[key].push(original_data[key][i]);
                 }
             }
         }
        source.change.emit();
        target_obj.change.emit();
        """

        FullName_list = dataframe['FullName'].unique().tolist()
        FullName_list.sort()
        FullName_list.insert(0,'ALL')
        FullName_select = Select(title="FullName:", value=FullName_list[0], options=FullName_list)

        BaseName_list = dataframe['BaseName'].unique().tolist()
        BaseName_list.sort()
        BaseName_list.insert(0,'ALL')
        BaseName_select = Select(title="BaseName:",
                                 value=BaseName_list[0],
                                 options=BaseName_list)

        dataframe['SummaryName'].fillna('None', inplace = True)
        SummaryName_list = dataframe['SummaryName'].unique().tolist()
        SummaryName_list.sort()
        SummaryName_list.insert(0,'ALL')
        SummaryName_select = Select(title="SummaryName:",
                                    value=SummaryName_list[0],
                                    options=SummaryName_list)

        MetricName_list = dataframe['MetricName'].unique().tolist()
        MetricName_list.sort()
        MetricName_list.insert(0,'ALL')
        MetricName_select = Select(title="MetricName:",
                                    value=MetricName_list[0],
                                    options=MetricName_list)

        MetricMetadata_list = dataframe['MetricMetadata'].unique().tolist()
        MetricMetadata_list.sort()
        MetricMetadata_list.insert(0,'ALL')
        MetricMetadata_select = Select(title="MetricMetadata:",
                                    value=MetricMetadata_list[0],
                                    options=MetricMetadata_list)

        generic_callback = CustomJS(args=dict(source=source,
                                              original_source=original_source,
                                              FullName_select_obj=FullName_select,
                                              BaseName_select_obj=BaseName_select,
                                              SummaryName_select_obj=SummaryName_select,
                                              MetricName_select_obj=MetricName_select,
                                              MetricMetadata_select_obj=MetricMetadata_select,
                                              target_obj=data_table),
                                    code=js_code)

        FullName_select.callback = generic_callback
        BaseName_select.callback = generic_callback
        SummaryName_select.callback = generic_callback
        MetricName_select.callback = generic_callback
        MetricMetadata_select.callback = generic_callback

        dropdownMenus = column([SummaryName_select, MetricName_select,
                                MetricMetadata_select, FullName_select, BaseName_select])
        page_layout = layout([dropdownMenus,data_table])
        show(page_layout)
