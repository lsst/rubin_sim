from builtins import zip
from builtins import object
import os
import re
from collections import OrderedDict
import numpy as np
import rubin_sim.maf.db as db
import rubin_sim.maf.metricBundles as metricBundles

__all__ = ['MafRunResults']

class MafRunResults(object):
    """
    Class to read MAF's resultsDb_sqlite.db and organize the output for display on web pages.

    Deals with a single MAF run (one output directory, one resultsDb) only.
    """
    def __init__(self, outDir, runName=None, resultsDb=None):
        """
        Instantiate the (individual run) layout visualization class.

        This class provides methods used by our jinja2 templates to help interact
        with the outputs of MAF.
        """
        self.outDir = os.path.relpath(outDir, '.')
        self.runName = runName
        # Set the config summary filename, if available.
        self.configSummary = os.path.join(self.outDir, 'configSummary.txt')
        if not os.path.isfile(self.configSummary):
            self.configSummary = 'Config Summary Not Available'
        # if the config summary existed and we don't know the runName, find it.
        elif self.runName is None:
            # Read the config file to get the runName.
            with open(self.configSummary, "r") as myfile:
                config = myfile.read()
            spot = config.find('RunName')
            # If we found the runName, use that.
            if spot != -1:
                self.runName = config[spot:].split('\n')[0][8:]
            # Otherwise, set it to be not available.
            else:
                self.runName = 'RunName not available'

        self.configDetails = os.path.join(self.outDir, 'configDetails.txt')
        if not os.path.isfile(self.configDetails):
            self.configDetails = 'Config Details Not Available.'

        # Read in the results database.
        if resultsDb is None:
            resultsDb = os.path.join(self.outDir, 'resultsDb_sqlite.db')
        database = db.ResultsDb(database=resultsDb)

        # Get the metric and display info (1-1 match)
        self.metrics = database.getMetricDisplayInfo()
        self.metrics = self.sortMetrics(self.metrics)

        # Get the plot and stats info (many-1 metric match)
        skip_stats =  ['Completeness@Time', 'Completeness H', 'FractionPop ']
        self.stats = database.getSummaryStats(summaryNameNotLike=skip_stats)
        self.plots = database.getPlotFiles()

        # Pull up the names of the groups and subgroups.
        groups = sorted(np.unique(self.metrics['displayGroup']))
        self.groups = OrderedDict()
        for g in groups:
            groupMetrics = self.metrics[np.where(self.metrics['displayGroup'] == g)]
            self.groups[g] = sorted(np.unique(groupMetrics['displaySubgroup']))

        self.summaryStatOrder = ['Id', 'Identity', 'Median', 'Mean', 'Rms', 'RobustRms',
                                 'N(-3Sigma)', 'N(+3Sigma)', 'Count',
                                 '25th%ile', '75th%ile', 'Min', 'Max']
        # Add in the table fraction sorting to summary stat ordering.
        tableFractions = [x for x in list(np.unique(self.stats['summaryName']))
                          if x.startswith('TableFraction')]
        if len(tableFractions) > 0:
            for x in ('TableFraction 0 == P', 'TableFraction 1 == P',
                      'TableFraction 1 < P'):
                if x in tableFractions:
                    tableFractions.remove(x)
            tableFractions = sorted(tableFractions)
            self.summaryStatOrder.append('TableFraction 0 == P')
            for tableFrac in tableFractions:
                self.summaryStatOrder.append(tableFrac)
            self.summaryStatOrder.append('TableFraction 1 == P')
            self.summaryStatOrder.append('TableFraction 1 < P')

        self.plotOrder = ['SkyMap', 'Histogram', 'PowerSpectrum', 'Combo']

    # Methods to deal with metricIds

    def convertSelectToMetrics(self, groupList, metricIdList):
        """
        Convert the lists of values returned by 'select metrics' template page
        into an appropriate dataframe of metrics (in sorted order).
        """
        metricIds = set()
        for group_subgroup in groupList:
            group = group_subgroup.split('_')[0]
            subgroup = group_subgroup.split('_')[-1].replace('+', ' ')
            mIds = self.metricIdsInSubgroup(group, subgroup)
            for mId in mIds:
                metricIds.add(mId)
        for mId in metricIdList:
            mId = int(mId)
            metricIds.add(mId)
        metricIds = list(metricIds)
        metrics = self.metricIdsToMetrics(metricIds)
        metrics = self.sortMetrics(metrics)
        return metrics

    def getJson(self, metric):
        """
        Return the JSON string containing the data for a particular metric.
        """
        if len(metric) > 1:
            return None
        metric = metric[0]
        filename = metric['metricDataFile']
        if filename.upper() == 'NULL':
            return None
        datafile = os.path.join(self.outDir, filename)
        # Read data back into a  bundle.
        mB = metricBundles.createEmptyMetricBundle()
        mB.read(datafile)
        io = mB.outputJSON()
        if io is None:
            return None
        return io.getvalue()

    def getNpz(self, metric):
        """
        Return the npz data.
        """
        if len(metric) > 1:
            return None
        metric = metric[0]
        filename = metric['metricDataFile']
        if filename.upper() == 'NULL':
            return None
        else:
            datafile = os.path.join(self.outDir, filename)
            return datafile

    def getResultsDb(self):
        """
        Return the summary results sqlite filename.

        Note that this assumes the resultsDB is stored in 'resultsDB_sqlite.db'.
        """
        return os.path.join(self.outDir, 'resultsDb_sqlite.db')

    def metricIdsInSubgroup(self, group, subgroup):
        """
        Return the metricIds within a given group/subgroup.
        """
        metrics = self.metricsInSubgroup(group, subgroup)
        metricIds = list(metrics['metricId'])
        return metricIds

    def metricIdsToMetrics(self, metricIds, metrics=None):
        """
        Return an ordered numpy array of metrics matching metricIds.
        """
        if metrics is None:
            metrics = self.metrics
        # this should be faster with pandas (and self.metrics.query('metricId in @metricIds'))
        metrics = metrics[np.in1d(metrics['metricId'], metricIds)]
        return metrics

    def metricsToMetricIds(self, metrics):
        """
        Return a list of the metric Ids corresponding to a subset of metrics.
        """
        return list(metrics['metricId'])

    # Methods to deal with metrics in numpy recarray.

    def sortMetrics(self, metrics, order=('displayGroup', 'displaySubgroup',
                                          'baseMetricNames', 'slicerName', 'displayOrder',
                                          'metricMetadata')):
        """
        Sort the metrics by order specified by 'order'.

        Default is to sort by group, subgroup, metric name, slicer, display order, then metadata.
        Returns sorted numpy array.
        """
        if len(metrics) > 0:
            metrics = np.sort(metrics, order=order)
        return metrics

    def metricsInGroup(self, group, metrics=None, sort=True):
        """
        Given a group, return the metrics belonging to this group, in display order.
        """
        if metrics is None:
            metrics = self.metrics
        metrics = metrics[np.where(metrics['displayGroup'] == group)]
        if sort:
            metrics = self.sortMetrics(metrics)
        return metrics

    def metricsInSubgroup(self, group, subgroup, metrics=None):
        """
        Given a group and subgroup, return a dataframe of the metrics belonging to these
        group/subgroups, in display order.

        If 'metrics' is provided, then only consider this subset of metrics.
        """
        metrics = self.metricsInGroup(group, metrics, sort=False)
        if len(metrics) > 0:
            metrics = metrics[np.where(metrics['displaySubgroup'] == subgroup)]
            metrics = self.sortMetrics(metrics)
        return metrics

    def metricsToSubgroups(self, metrics):
        """
        Given an array of metrics, return an ordered dict of their group/subgroups.
        """
        groupList = sorted(np.unique(metrics['displayGroup']))
        groups = OrderedDict()
        for group in groupList:
            groupmetrics = self.metricsInGroup(group, metrics, sort=False)
            groups[group] = sorted(np.unique(groupmetrics['displaySubgroup']))
        return groups

    def metricsWithPlotType(self, plotType='SkyMap', metrics=None):
        """
        Return an array of metrics with plot=plotType (optional, metric subset).
        """
        # Allow some variation in plotType names for backward compatibility,
        #  even if plotType is  a list.
        if not isinstance(plotType, list):
            plotType = [plotType]
        plotTypes = []
        for pT in plotType:
            plotTypes.append(pT)
            if pT.endswith('lot'):
                plotTypes.append(pT[:-4])
            else:
                plotTypes.append(pT.lower() + 'Plot')
        if metrics is None:
            metrics = self.metrics
        # Identify the plots with the right plotType, get their IDs.
        plotMatch = self.plots[np.in1d(self.plots['plotType'], plotTypes)]
        # Convert those potentially matching metricIds to metrics, using the subset info.
        metrics = self.metricIdsToMetrics(plotMatch['metricId'], metrics)
        return metrics

    def uniqueMetricNames(self, metrics=None, baseonly=True):
        """
        Return a list of the unique metric names, preserving the order of 'metrics'.
        """
        if metrics is None:
            metrics = self.metrics
        if baseonly:
            sortName = 'baseMetricNames'
        else:
            sortName = 'metricName'
        metricNames = list(np.unique(metrics[sortName]))
        return metricNames

    def metricsWithSummaryStat(self, summaryStatName='Identity', metrics=None):
        """
        Return metrics with summary stat matching 'summaryStatName' (optional, metric subset).
        """
        if metrics is None:
            metrics = self.metrics
        # Identify the potentially matching stats.
        stats = self.stats[np.in1d(self.stats['summaryName'], summaryStatName)]
        # Identify the subset of relevant metrics.
        metrics = self.metricIdsToMetrics(stats['metricId'], metrics)
        # Re-sort metrics because at this point, probably want displayOrder + metadata before metric name.
        metrics = self.sortMetrics(metrics, order=['displayGroup', 'displaySubgroup', 'slicerName',
                                                   'displayOrder', 'metricMetadata', 'baseMetricNames'])
        return metrics

    def metricsWithStats(self, metrics=None):
        """
        Return metrics that have any summary stat.
        """
        if metrics is None:
            metrics = self.metrics
        # Identify metricIds which are also in stats.
        metrics = metrics[np.in1d(metrics['metricId'], self.stats['metricId'])]
        metrics = self.sortMetrics(metrics, order=['displayGroup', 'displaySubgroup', 'slicerName',
                                                   'displayOrder', 'metricMetadata', 'baseMetricNames'])
        return metrics

    def uniqueSlicerNames(self, metrics=None):
        """
        For an array of metrics, return the unique slicer names.
        """
        if metrics is None:
            metrics = self.metrics
        return list(np.unique(metrics['slicerName']))

    def metricsWithSlicer(self, slicer, metrics=None):
        """
        For an array of metrics, return the subset which match a particular 'slicername' value.
        """
        if metrics is None:
            metrics = self.metrics
        metrics = metrics[np.where(metrics['slicerName'] == slicer)]
        return metrics

    def uniqueMetricNameAndMetadata(self, metrics=None):
        """
        For an array of metrics, return the unique metric names + metadata combo in same order.
        """
        if metrics is None:
            metrics = self.metrics
        metricmetadata = []
        for metricName, metadata in zip(metrics['metricName'], metrics['metricMetadata']):
            metricmeta = ' '.join([metricName, metadata])
            if metricmeta not in metricmetadata:
                metricmetadata.append(metricmeta)
        return metricmetadata

    def uniqueMetricMetadata(self, metrics=None):
        """
        For an array of metrics, return a list of the unique metadata.
        """
        if metrics is None:
            metrics = self.metrics
        return list(np.unique(metrics['metricMetadata']))

    def metricsWithMetadata(self, metadata, metrics=None):
        """
        For an array of metrics, return the subset which match a particular 'metadata' value.
        """
        if metrics is None:
            metrics = self.metrics
        metrics = metrics[np.where(metrics['metricMetadata'] == metadata)]
        return metrics

    def metricsWithMetricName(self, metricName, metrics=None, baseonly=True):
        """
        Return all metrics which match metricName (default, only the 'base' metric name).
        """
        if metrics is None:
            metrics = self.metrics
        if baseonly:
            metrics = metrics[np.where(metrics['baseMetricNames'] == metricName)]
        else:
            metrics = metrics[np.where(metrics['metricName'] == metricName)]
        return metrics

    def metricInfo(self, metric=None, withDataLink=True, withSlicerName=True):
        """
        Return a dict with the metric info we want to show on the webpages.

        Currently : MetricName / Slicer/ Metadata / datafile (for download)
        Used to build a lot of tables in showMaf.
        """
        metricInfo = OrderedDict()
        if metric is None:
            metricInfo['MetricName'] = ''
            if withSlicerName:
                metricInfo['Slicer'] = ''
            metricInfo['Metadata'] = ''
            if withDataLink:
                metricInfo['Data'] = []
                metricInfo['Data'].append([None, None])
            return metricInfo
        # Otherwise, do this for real (not a blank).
        metricInfo['MetricName'] = metric['metricName']
        if withSlicerName:
            metricInfo['Slicer'] = metric['slicerName']
        metricInfo['Metadata'] = metric['metricMetadata']
        if withDataLink:
            metricInfo['Data'] = []
            metricInfo['Data'].append(metric['metricDataFile'])
            metricInfo['Data'].append(os.path.join(self.outDir, metric['metricDataFile']))
        return metricInfo

    def captionForMetric(self, metric):
        """
        Return the caption for a given metric.
        """
        caption = metric['displayCaption']
        if caption == 'NULL':
            return ''
        else:
            return caption

    # Methods for plots.

    def plotsForMetric(self, metric):
        """
        Return a numpy array of the plots which match a given metric.
        """
        return self.plots[np.where(self.plots['metricId'] == metric['metricId'])]

    def plotDict(self, plots=None):
        """
        Given an array of plots (for a single metric usually).
        Returns an ordered dict with 'plotType' for interfacing with jinja2 templates.
        plotDict == {'SkyMap': {'plotFile': [], 'thumbFile', []}, 'Histogram': {}..}

        If no plot of a particular type, the plotFile and thumbFile are empty lists.
        Calling with plots=None returns a blank plotDict.
        """
        plotDict = OrderedDict()
        # Go through plots in 'plotOrder'.
        if plots is None:
            for p in self.plotOrder:
                plotDict[p] = {}
                plotDict[p]['plotFile'] = ''
                plotDict[p]['thumbFile'] = ''
        else:
            plotTypes = list(np.unique(plots['plotType']))
            for p in self.plotOrder:
                if p in plotTypes:
                    plotDict[p] = {}
                    plotmatch = plots[np.where(plots['plotType'] == p)]
                    plotDict[p]['plotFile'] = []
                    plotDict[p]['thumbFile'] = []
                    for pm in plotmatch:
                        plotDict[p]['plotFile'].append(self.getPlotfile(pm))
                        plotDict[p]['thumbFile'].append(self.getThumbfile(pm))
                    plotTypes.remove(p)
            # Round up remaining plots.
            for p in plotTypes:
                plotDict[p] = {}
                plotmatch = plots[np.where(plots['plotType'] == p)]
                plotDict[p]['plotFile'] = []
                plotDict[p]['thumbFile'] = []
                for pm in plotmatch:
                    plotDict[p]['plotFile'].append(self.getPlotfile(pm))
                    plotDict[p]['thumbFile'].append(self.getThumbfile(pm))
        return plotDict

    def getThumbfile(self, plot):
        """
        Return the thumbnail file name for a given plot.
        """
        thumbfile = os.path.join(self.outDir, plot['thumbFile'])
        return thumbfile

    def getPlotfile(self, plot):
        """
        Return the filename for a given plot.
        """
        plotFile = os.path.join(self.outDir, plot['plotFile'])
        return plotFile

    def orderPlots(self, skyPlots):
        """
        skyPlots = numpy array of skymap plots.

        Returns an ordered list of plotDicts.

        The goal is to lay out the skymaps in a 3x2 grid on the MultiColor page, in ugrizy order.
        If a plot for a filter is missing, add a gap. (i.e. if there is no u, keep a blank spot).
        If there are other plots, with multiple filters or no filter info, they are added to the end.
        If skyPlots includes multiple plots in the same filter, just goes back to displayOrder.
        """
        orderedSkyPlots = []
        if len(skyPlots) == 0:
            return orderedSkyPlots

        orderList = ['u', 'g', 'r', 'i', 'z', 'y']
        blankPlotDict = self.plotDict(None)

        # Look for filter names in the plot filenames.
        tooManyPlots = False
        for f in orderList:
            pattern = '_' + f + '_'
            matches = np.array([bool(re.search(pattern, x)) for x in skyPlots['plotFile']])
            matchSkyPlot = skyPlots[matches]
            # in pandas: matchSkyPlot = skyPlots[skyPlots.plotFile.str.contains(pattern)]
            if len(matchSkyPlot) == 1:
                orderedSkyPlots.append(self.plotDict(matchSkyPlot))
            elif len(matchSkyPlot) == 0:
                orderedSkyPlots.append(blankPlotDict)
            else:
                # If we found more than one plot in the same filter, we just go back to displayOrder.
                tooManyPlots = True
                break

        if tooManyPlots is False:
            # Add on any additional non-filter plots (e.g. joint completeness)
            #  that do NOT match original _*_ pattern.
            pattern = '_[ugrizy]_'
            nonmatches = np.array([bool(re.search(pattern, x)) for x in skyPlots['plotFile']])
            nonmatchSkyPlots = skyPlots[nonmatches == False]
            if len(nonmatchSkyPlots) > 0:
                for skyPlot in nonmatchSkyPlots:
                    orderedSkyPlots.append(self.plotDict(np.array([skyPlot])))

        elif tooManyPlots:
            metrics = self.metrics[np.in1d(self.metrics['metricId'], skyPlots['metricId'])]
            metrics = self.sortMetrics(metrics, order=['displayOrder'])
            orderedSkyPlots = []
            for m in metrics:
                skyPlot = skyPlots[np.where(skyPlots['metricId'] == m['metricId'])]
                orderedSkyPlots.append(self.plotDict(skyPlot))

        # Pad out to make sure there are rows of 3
        while len(orderedSkyPlots) % 3 != 0:
            orderedSkyPlots.append(blankPlotDict)

        return orderedSkyPlots

    def getSkyMaps(self, metrics=None, plotType='SkyMap'):
        """
        Return a numpy array of the plots with plotType=plotType, optionally for subset of metrics.
        """
        if metrics is None:
            metrics = self.metrics
        # Match the plots to the metrics required.
        plotMetricMatch = self.plots[np.in1d(self.plots['metricId'], metrics['metricId'])]
        # Match the plot type (which could be a list)
        plotMatch = plotMetricMatch[np.in1d(plotMetricMatch['plotType'], plotType)]
        return plotMatch

    # Set of methods to deal with summary stats.

    def statsForMetric(self, metric, statName=None):
        """
        Return a numpy array of summary statistics which match a given metric(s).

        Optionally specify a particular statName that you want to match.
        """
        stats = self.stats[np.where(self.stats['metricId'] == metric['metricId'])]
        if statName is not None:
            stats = stats[np.where(stats['summaryName'] == statName)]
        return stats

    def statDict(self, stats):
        """
        Returns an ordered dictionary with statName:statValue for an array of stats.

        Note that if you pass 'stats' from multiple metrics with the same summary names, they
        will be overwritten in the resulting dictionary!
        So just use stats from one metric, with unique summaryNames.
        """
        # Result = dict with key == summary stat name, value = summary stat value.
        sdict = OrderedDict()
        statnames = self.orderStatNames(stats)
        for n in statnames:
            match = stats[np.where(stats['summaryName'] == n)]
            # We're only going to look at the first value; and this should be a float.
            sdict[n] = match['summaryValue'][0]
        return sdict

    def orderStatNames(self, stats):
        """
        Given an array of stats, return a list containing all the unique 'summaryNames'
        in a default ordering (identity-count-mean-median-rms..).
        """
        names = list(np.unique(stats['summaryName']))
        # Add some default sorting:
        namelist = []
        for nord in self.summaryStatOrder:
            if nord in names:
                namelist.append(nord)
                names.remove(nord)
        for remaining in names:
            namelist.append(remaining)
        return namelist

    def allStatNames(self, metrics):
        """
        Given an array of metrics, return a list containing all the unique 'summaryNames'
        in a default ordering.
        """
        names = np.unique(self.stats['summaryName'][np.in1d(self.stats['metricId'], metrics['metricId'])])
        names = list(names)
        # Add some default sorting.
        namelist = []
        for nord in self.summaryStatOrder:
            if nord in names:
                namelist.append(nord)
                names.remove(nord)
        for remaining in names:
            namelist.append(remaining)
        return namelist
