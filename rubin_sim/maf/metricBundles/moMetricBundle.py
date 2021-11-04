from __future__ import print_function
from builtins import object
import os
import warnings
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from rubin_sim.maf.metrics import BaseMoMetric
from rubin_sim.maf.metrics import MoCompletenessMetric, ValueAtHMetric
from rubin_sim.maf.slicers import MoObjSlicer
from rubin_sim.maf.stackers import BaseMoStacker, MoMagStacker
from rubin_sim.maf.plots import PlotHandler
from rubin_sim.maf.plots import MetricVsH

from .metricBundle import MetricBundle

__all__ = ['MoMetricBundle', 'MoMetricBundleGroup', 'createEmptyMoMetricBundle', 'makeCompletenessBundle']


def createEmptyMoMetricBundle():
    """Create an empty metric bundle.

    Returns
    -------
    ~rubin_sim.maf.metricBundles.MoMetricBundle
        An empty metric bundle, configured with just the :class:`BaseMetric` and :class:`BaseSlicer`.
    """
    return MoMetricBundle(BaseMoMetric(), MoObjSlicer(), None)


def makeCompletenessBundle(bundle, completenessMetric, Hmark=None, resultsDb=None):
    """
    Make a mock metric bundle from a bundle which had MoCompleteness or MoCumulativeCompleteness summary
    metrics run. This lets us use the plotHandler + plots.MetricVsH to generate plots.
    Will also work with completeness metric run in order to calculate fraction of the population,
    or with MoCompletenessAtTime metric.

    Parameters
    ----------
    bundle : ~rubin_sim.maf.metricBundles.MetricBundle
        The metric bundle with a completeness summary statistic.
    completenessMetric : ~rubin_sim.maf.metric
        The summary (completeness) metric to run on the bundle.
    Hmark : float, optional
        The Hmark value to add to the plotting dictionary of the new mock bundle. Default None.
    resultsDb : ~rubin_sim.maf.db.ResultsDb, optional
        The resultsDb in which to record the summary statistic value at Hmark. Default None.

    Returns
    -------
    ~rubin_sim.maf.metricBundles.MoMetricBundle
    """
    bundle.setSummaryMetrics(completenessMetric)
    # This step adds summary values at each point to the original metric - we use this to populate
    # the completeness values in the next step. However, we may not want them to go into the resultsDb.
    bundle.computeSummaryStats(resultsDb)
    summaryName = completenessMetric.name
    # Make up the bundle, including the metric values.
    completeness = ma.MaskedArray(data=bundle.summaryValues[summaryName]['value'],
                                  mask=np.zeros(len(bundle.summaryValues[summaryName]['value'])),
                                  fill_value=0)
    mb = MoMetricBundle(completenessMetric, bundle.slicer,
                        constraint=bundle.constraint, runName=bundle.runName,
                        metadata=bundle.metadata, displayDict=bundle.displayDict)
    plotDict = {}
    plotDict.update(bundle.plotDict)
    plotDict['label'] = bundle.metadata
    if 'Completeness' not in summaryName:
        plotDict['label'] += ' ' + summaryName.replace('FractionPop_', '')
    mb.metricValues = completeness.reshape(1, len(completeness))
    if Hmark is not None:
        metric = ValueAtHMetric(Hmark=Hmark)
        mb.setSummaryMetrics(metric)
        mb.computeSummaryStats(resultsDb)
        val = mb.summaryValues['Value At H=%.1f' % Hmark]
        if summaryName.startswith('Cumulative'):
            plotDict['label'] += ': @ H(<=%.1f) = %.1f%s' % (Hmark, val * 100, '%')
        else:
            plotDict['label'] += ': @ H(=%.1f) = %.1f%s' % (Hmark, val * 100, '%')
    mb.setPlotDict(plotDict)
    return mb


class MoMetricBundle(MetricBundle):
    def __init__(self, metric, slicer, constraint=None,
                 stackerList=None,
                 runName='opsim', metadata=None,
                 fileRoot=None,
                 plotDict=None, plotFuncs=None,
                 displayDict=None,
                 childMetrics=None,
                 summaryMetrics=None):
        """
        Instantiate moving object metric bundle, save metric/slicer/constraint, etc.
        """
        self.metric = metric
        self.slicer = slicer
        if constraint == '':
            constraint = None
        self.constraint = constraint
        # Set the stackerlist.
        if stackerList is not None:
            if isinstance(stackerList, BaseMoStacker):
                self.stackerList = [stackerList, ]
            else:
                self.stackerList = []
                for s in stackerList:
                    if not isinstance(s, BaseMoStacker):
                        raise ValueError('stackerList must only contain '
                                         'rubin_sim.maf.stackers.BaseMoStacker type objs')
                    self.stackerList.append(s)
        else:
            self.stackerList = []
        # Add the basic 'visibility/mag' stacker if not present.
        magStackerFound = False
        for s in self.stackerList:
            if s.__class__.__name__ == 'MoMagStacker':
                magStackerFound = True
                break
        if not magStackerFound:
            self.stackerList.append(MoMagStacker())
        # Set a mapsList just for compatibility with generic MetricBundle.
        self.mapsList = []
        # Add the summary stats, if applicable.
        self.setSummaryMetrics(summaryMetrics)
        # Set the provenance/metadata.
        self.runName = runName
        self._buildMetadata(metadata)
        # Build the output filename root if not provided.
        if fileRoot is not None:
            self.fileRoot = fileRoot
        else:
            self._buildFileRoot()
        # Set the plotting classes/functions.
        self.setPlotFuncs(plotFuncs)
        # Set the plotDict and displayDicts.
        self.plotDict = {'units': '@H'}
        self.setPlotDict(plotDict)
        # Update/set displayDict.
        self.displayDict = {}
        self.setDisplayDict(displayDict)
        # Set the list of child metrics.
        self.setChildBundles(childMetrics)
        # This is where we store the metric values and summary stats.
        self.metricValues = None
        self.summaryValues = None

    def _resetMetricBundle(self):
        """Reset all properties of MetricBundle.
        """
        self.metric = None
        self.slicer = None
        self.constraint = None
        self.stackerList = [MoMagStacker()]
        self.mapsList = []
        self.summaryMetrics = []
        self.plotFuncs = []
        self.runName = 'opsim'
        self.metadata = ''
        self.dbCols = None
        self.fileRoot = None
        self.plotDict = {}
        self.displayDict = {}
        self.childMetrics = None
        self.metricValues = None
        self.summaryValues = None

    def _buildMetadata(self, metadata):
        """If no metadata is provided, auto-generate it from the obsFile + constraint.
        """
        if metadata is None:
            try:
                self.metadata = self.slicer.obsfile.replace('.txt', '').replace('.dat', '')
                self.metadata = self.metadata.replace('_obs', '').replace('_allObs', '')
            except AttributeError:
                self.metadata = 'noObs'
            # And modify by constraint.
            if self.constraint is not None:
                self.metadata += ' ' + self.constraint
        else:
            self.metadata = metadata

    def _findReqCols(self):
        # Doesn't quite work the same way yet. No stacker list, for example.
        raise NotImplementedError

    def setChildBundles(self, childMetrics=None):
        """
        Identify any child metrics to be run on this (parent) bundle.
        and create the new metric bundles that will hold the child values, linking to this bundle.
        Remove the summaryMetrics from self afterwards.
        """
        self.childBundles = {}
        if childMetrics is None:
            childMetrics = self.metric.childMetrics
        for cName, cMetric in childMetrics.items():
            cBundle = MoMetricBundle(metric=cMetric, slicer=self.slicer,
                                     constraint=self.constraint,
                                     stackerList=self.stackerList,
                                     runName=self.runName, metadata=self.metadata,
                                     plotDict=self.plotDict, plotFuncs=self.plotFuncs,
                                     displayDict=self.displayDict,
                                     summaryMetrics=self.summaryMetrics)
            self.childBundles[cName] = cBundle
        if len(childMetrics) > 0:
            self.summaryMetrics = []

    def computeSummaryStats(self, resultsDb=None):
        """
        Compute summary statistics on metricValues, using summaryMetrics, for self and child bundles.
        """
        if self.summaryValues is None:
            self.summaryValues = {}
        if self.summaryMetrics is not None:
            # Build array of metric values, to use for (most) summary statistics.
            for m in self.summaryMetrics:
                summaryName = m.name
                summaryVal = m.run(self.metricValues, self.slicer.slicePoints['H'])
                self.summaryValues[summaryName] = summaryVal
                # Add summary metric info to results database, if applicable.
                if resultsDb:
                    metricId = resultsDb.updateMetric(self.metric.name, self.slicer.slicerName,
                                                      self.runName, self.constraint, self.metadata, None)
                    resultsDb.updateSummaryStat(metricId, summaryName=summaryName, summaryValue=summaryVal)

    def reduceMetric(self, reduceFunc, reducePlotDict=None, reduceDisplayDict=None):
        raise NotImplementedError


class MoMetricBundleGroup(object):
    def __init__(self, bundleDict, outDir='.', resultsDb=None, verbose=True):
        self.verbose = verbose
        self.bundleDict = bundleDict
        self.outDir = outDir
        if not os.path.isdir(self.outDir):
            os.makedirs(self.outDir)
        self.resultsDb = resultsDb

        self.slicer = list(self.bundleDict.values())[0].slicer
        for b in self.bundleDict.values():
            if b.slicer != self.slicer:
                raise ValueError('Currently, the slicers for the MoMetricBundleGroup must be equal,'
                                 ' using the same observations and Hvals.')
        self.constraints = list(set([b.constraint for b in bundleDict.values()]))

    def _checkCompatible(self, metricBundle1, metricBundle2):
        """Check if two MetricBundles are "compatible".
        Compatible indicates that the constraints, the slicers, and the maps are the same, and
        that the stackers do not interfere with each other
        (i.e. are not trying to set the same column in different ways).
        Returns True if the MetricBundles are compatible, False if not.

        Parameters
        ----------
        metricBundle1 : MetricBundle
        metricBundle2 : MetricBundle

        Returns
        -------
        bool
        """
        if metricBundle1.constraint != metricBundle2.constraint:
            return False
        if metricBundle1.slicer != metricBundle2.slicer:
            return False
        if metricBundle1.mapsList.sort() != metricBundle2.mapsList.sort():
            return False
        for stacker in metricBundle1.stackerList:
            for stacker2 in metricBundle2.stackerList:
                # If the stackers have different names, that's OK, and if they are identical, that's ok.
                if (stacker.__class__.__name__ == stacker2.__class__.__name__) & (stacker != stacker2):
                    return False
        # But if we got this far, everything matches.
        return True

    def _findCompatible(self, testKeys):
        """"Private utility to find which metricBundles with keys in the list 'testKeys' can be calculated
        at the same time -- having the same slicer, constraint, maps, and compatible stackers.

        Parameters
        -----------
        testKeys : list
            List of the dictionary keys (of self.bundleDict) to test for compatibilility.
        Returns
        --------
        list of lists
            Returns testKeys, split into separate lists of compatible metricBundles.
        """
        compatibleLists = []
        for k in testKeys:
            try:
                b = self.bundleDict[k]
            except KeyError:
                warnings.warn('Received %s in testkeys, but this is not present in self.bundleDict.'
                              'Will continue, but this is not expected.')
                continue
            foundCompatible = False
            checkedAll = False
            while not(foundCompatible) and not(checkedAll):
                # Go through the existing lists in compatibleLists, to see if this metricBundle matches.
                for compatibleList in compatibleLists:
                    # Compare to all the metricBundles in this subset, to check all stackers are compatible.
                    foundCompatible = True
                    for comparisonKey in compatibleList:
                        compatible = self._checkCompatible(self.bundleDict[comparisonKey], b)
                        if not compatible:
                            # Found a metricBundle which is not compatible, so stop and go onto the next subset.
                            foundCompatible = False
                            break
                checkedAll = True
            if foundCompatible:
                compatibleList.append(k)
            else:
                compatibleLists.append([k,])
        return compatibleLists

    def runConstraint(self, constraint):
        """Calculate the metric values for all the metricBundles which match this constraint in the
        metricBundleGroup. Also calculates child metrics and summary statistics, and writes all to disk.
        (work is actually done in _runCompatible, so that only completely compatible sets of metricBundles
        run at the same time).

        Parameters
        ----------
        constraint : str
            SQL-where or pandas constraint for the metricBundles.
        """
        # Find the dict keys of the bundles which match this constraint.
        keysMatchingConstraint = []
        for k, b in self.bundleDict.items():
            if b.constraint == constraint:
                keysMatchingConstraint.append(k)
        if len(keysMatchingConstraint) == 0:
            return
        # Identify the observations which are relevant for this constraint.
        # This sets slicer.obs (valid for all H values).
        self.slicer.subsetObs(constraint)
        # Identify the sets of these metricBundles can be run at the same time (also have the same stackers).
        compatibleLists = self._findCompatible(keysMatchingConstraint)

        # And now run each of those subsets of compatible metricBundles.
        for compatibleList in compatibleLists:
            self._runCompatible(compatibleList)

    def _runCompatible(self, compatibleList):
        """Calculate the metric values for set of (parent and child) bundles, as well as the summary stats,
        and write to disk.

        Parameters
        -----------
        compatibleList : list
            List of dictionary keys, of the metricBundles which can be calculated together.
            This means they are 'compatible' and have the same slicer, constraint, and non-conflicting
            mappers and stackers.
        """
        if self.verbose:
            print('Running metrics %s' % compatibleList)

        bDict = self.bundleDict  #  {key: self.bundleDict.get(key) for key in compatibleList}

        # Find the unique stackers and maps. These are already "compatible" (as id'd by compatibleList).
        uniqStackers = []
        allStackers = []
        uniqMaps = []
        allMaps = []
        for b in bDict.values():
            allStackers += b.stackerList
            allMaps += b.mapsList
        for s in allStackers:
            if s not in uniqStackers:
                uniqStackers.append(s)
        for m in allMaps:
            if m not in uniqMaps:
                uniqMaps.append(m)

        if len(uniqMaps) > 0:
            print("Got some maps .. that was unexpected at the moment. Can't use them here yet.")

        # Set up all of the metric values, including for the child bundles.
        for k in compatibleList:
            b = self.bundleDict[k]
            b._setupMetricValues()
            for cb in b.childBundles.values():
                cb._setupMetricValues()
        # Calculate the metric values.
        for i, slicePoint in enumerate(self.slicer):
            ssoObs = slicePoint['obs']
            for j, Hval in enumerate(slicePoint['Hvals']):
                # Run stackers to add extra columns (that depend on Hval)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    for s in uniqStackers:
                        ssoObs = s.run(ssoObs, slicePoint['orbit']['H'], Hval)
                # Run all the parent metrics.
                for k in compatibleList:
                    b = self.bundleDict[k]
                    # Mask the parent metric (and then child metrics) if there was no data.
                    if len(ssoObs) == 0:
                        b.metricValues.mask[i][j] = True
                        for cb in list(b.childBundles.values()):
                            cb.metricValues.mask[i][j] = True
                    # Otherwise, calculate the metric value for the parent, and then child.
                    else:
                        # Calculate for the parent.
                        mVal = b.metric.run(ssoObs, slicePoint['orbit'], Hval)
                        # Mask if the parent metric returned a bad value.
                        if mVal == b.metric.badval:
                            b.metricValues.mask[i][j] = True
                            for cb in b.childBundles.values():
                                cb.metricValues.mask[i][j] = True
                        # Otherwise, set the parent value and calculate the child metric values as well.
                        else:
                            b.metricValues.data[i][j] = mVal
                            for cb in b.childBundles.values():
                                childVal = cb.metric.run(ssoObs, slicePoint['orbit'], Hval, mVal)
                                if childVal == cb.metric.badval:
                                    cb.metricValues.mask[i][j] = True
                                else:
                                    cb.metricValues.data[i][j] = childVal
        for k in compatibleList:
            b = self.bundleDict[k]
            b.computeSummaryStats(self.resultsDb)
            for cB in b.childBundles.values():
                cB.computeSummaryStats(self.resultsDb)
                # Write to disk.
                cB.write(outDir=self.outDir, resultsDb=self.resultsDb)
            # Write to disk.
            b.write(outDir=self.outDir, resultsDb=self.resultsDb)

    def runAll(self):
        """
        Run all constraints and metrics for these moMetricBundles.
        """
        for constraint in self.constraints:
            self.runConstraint(constraint)
        if self.verbose:
            print('Calculated and saved all metrics.')

    def plotAll(self, savefig=True, outfileSuffix=None, figformat='pdf', dpi=600, thumbnail=True,
                closefigs=True):
        """
        Make a few generically desired plots. This needs more flexibility in the future.
        """
        plotHandler = PlotHandler(outDir=self.outDir, resultsDb=self.resultsDb,
                                  savefig=savefig, figformat=figformat, dpi=dpi, thumbnail=thumbnail)
        for b in self.bundleDict.values():
            try:
                b.plot(plotHandler=plotHandler, outfileSuffix=outfileSuffix, savefig=savefig)
            except ValueError as ve:
                message = 'Plotting failed for metricBundle %s.' % (b.fileRoot)
                message += ' Error message: %s' % (ve.message)
                warnings.warn(message)
            if closefigs:
                plt.close('all')
        if self.verbose:
            print('Plotting all metrics.')
