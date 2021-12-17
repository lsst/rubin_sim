from __future__ import print_function
from builtins import object
import os
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from collections import OrderedDict

import rubin_sim.maf.utils as utils
from rubin_sim.maf.plots import PlotHandler
import rubin_sim.maf.maps as maps
import rubin_sim.maf.db as db
from rubin_sim.maf.stackers import BaseDitherStacker
from .metricBundle import MetricBundle, createEmptyMetricBundle
import warnings

__all__ = ["makeBundlesDictFromList", "MetricBundleGroup"]


def makeBundlesDictFromList(bundleList):
    """Utility to convert a list of MetricBundles into a dictionary, keyed by the fileRoot names.

    Raises an exception if the fileroot duplicates another metricBundle.
    (Note this should alert to potential cases of filename duplication).

    Parameters
    ----------
    bundleList : `list` of `MetricBundles`
    """
    bDict = {}
    for b in bundleList:
        if b.fileRoot in bDict:
            raise NameError(
                "More than one metricBundle is using the same fileroot, %s"
                % (b.fileRoot)
            )
        bDict[b.fileRoot] = b
    return bDict


class MetricBundleGroup(object):
    """The MetricBundleGroup exists to calculate the metric values for a group of
    MetricBundles.

    The MetricBundleGroup will query data from a single database table (for multiple
    constraints), use that data to calculate metric values for multiple slicers,
    and calculate summary statistics and generate plots for all metrics included in
    the dictionary passed to the MetricBundleGroup.

    We calculate the metric values here, rather than in the individual MetricBundles,
    because it is much more efficient to step through a slicer once (and calculate all
    the relevant metric values at each point) than it is to repeat this process multiple times.

    The MetricBundleGroup also determines how to efficiently group the MetricBundles
    to reduce the number of sql queries of the database, grabbing larger chunks of data at once.

    Parameters
    ----------
    bundleDict : `dict` or `list` of `MetricBundles`
        Individual MetricBundles should be placed into a dictionary, and then passed to
        the MetricBundleGroup. The dictionary keys can then be used to identify MetricBundles
        if needed -- and to identify new MetricBundles which could be created if 'reduce'
        functions are run on a particular MetricBundle.
        A bundleDict can be conveniently created from a list of MetricBundles using
        makeBundlesDictFromList (done automatically if a list is passed in)
    dbCon : `str` or database connection object
        The database object or sqlite3 filename connected to the data to be used to
        calculate metrics.
        Advanced use: It is possible to set this to None, in which case data should be passed
        directly to the runCurrent method (and runAll should not be used).
    outDir : `str`, optional
        Directory to save the metric results. Default is the current directory.
    resultsDb : `ResultsDb`, optional
        A results database. If not specified, one will be created in the outDir.
        This database saves information about the metrics calculated, including their summary statistics.
    verbose : `bool`, optional
        Flag to turn on/off verbose feedback.
    saveEarly : `bool`, optional
        If True, metric values will be saved immediately after they are first calculated (to prevent
        data loss) as well as after summary statistics are calculated.
        If False, metric values will only be saved after summary statistics are calculated.
    dbTable : `str`, optional
        The name of the table in the dbObj to query for data.
    """

    def __init__(
        self,
        bundleDict,
        dbCon,
        outDir=".",
        resultsDb=None,
        verbose=True,
        saveEarly=True,
        dbTable="observations",
    ):
        """Set up the MetricBundleGroup."""
        if type(bundleDict) is list:
            bundleDict = makeBundlesDictFromList(bundleDict)
        # Print occasional messages to screen.
        self.verbose = verbose
        # Save metric results as soon as possible (in case of crash).
        self.saveEarly = saveEarly
        # Check for output directory, create it if needed.
        self.outDir = outDir
        if not os.path.isdir(self.outDir):
            os.makedirs(self.outDir)

        # Do some type checking on the MetricBundle dictionary.
        if not isinstance(bundleDict, dict):
            raise ValueError(
                "bundleDict should be a dictionary containing MetricBundle objects."
            )
        for b in bundleDict.values():
            if not isinstance(b, MetricBundle):
                raise ValueError("bundleDict should contain only MetricBundle objects.")
        # Identify the series of constraints.
        self.constraints = list(set([b.constraint for b in bundleDict.values()]))
        # Set the bundleDict (all bundles, with all constraints)
        self.bundleDict = bundleDict

        self.dbObj = dbCon
        # Set the table we're going to be querying.
        self.dbTable = dbTable

        # Check the resultsDb (optional).
        if resultsDb is not None:
            if not isinstance(resultsDb, db.ResultsDb):
                raise ValueError("resultsDb should be an ResultsDb object")
        self.resultsDb = resultsDb

        # Dict to keep track of what's been run:
        self.hasRun = {}
        for bk in bundleDict:
            self.hasRun[bk] = False

    def _checkCompatible(self, metricBundle1, metricBundle2):
        """Check if two MetricBundles are "compatible".
        Compatible indicates that the sql constraints, the slicers, and the maps are the same, and
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
                if (stacker.__class__.__name__ == stacker2.__class__.__name__) & (
                    stacker != stacker2
                ):
                    return False
        # But if we got this far, everything matches.
        return True

    def _findCompatibleLists(self):
        """Find sets of compatible metricBundles from the currentBundleDict."""
        # CompatibleLists stores a list of lists;
        #   each (nested) list contains the bundleDict _keys_ of a compatible set of metricBundles.
        #
        compatibleLists = []
        for k, b in self.currentBundleDict.items():
            foundCompatible = False
            for compatibleList in compatibleLists:
                comparisonMetricBundleKey = compatibleList[0]
                compatible = self._checkCompatible(
                    self.bundleDict[comparisonMetricBundleKey], b
                )
                if compatible:
                    # Must compare all metricBundles in each subset (if they are a potential match),
                    #  as the stackers could be different (and one could be incompatible,
                    #  not necessarily the first)
                    for comparisonMetricBundleKey in compatibleList[1:]:
                        compatible = self._checkCompatible(
                            self.bundleDict[comparisonMetricBundleKey], b
                        )
                        if not compatible:
                            # If we find one which is not compatible, stop and go on to the
                            # next subset list.
                            break
                    # Otherwise, we reached the end of the subset and they were all compatible.
                    foundCompatible = True
                    compatibleList.append(k)
            if not foundCompatible:
                # Didn't find a pre-existing compatible set; make a new one.
                compatibleLists.append(
                    [
                        k,
                    ]
                )
        self.compatibleLists = compatibleLists

    def getData(self, constraint):
        """Query the data from the database.

        The currently bundleDict should generally be set before calling getData (using setCurrent).

        Parameters
        ----------
        constraint : `str`
           The constraint for the currently active set of MetricBundles.
        """
        if self.verbose:
            if constraint == "":
                print("Querying database with no constraint.")
            else:
                print("Querying database with constraint %s" % (constraint))
        # Note that we do NOT run the stackers at this point (this must be done in each 'compatible' group).
        self.simData = utils.getSimData(
            self.dbObj,
            constraint,
            self.dbCols,
            tableName=self.dbTable,
        )

        if self.verbose:
            print("Found %i visits" % (self.simData.size))

    def runAll(self, clearMemory=False, plotNow=False, plotKwargs=None):
        """Runs all the metricBundles in the metricBundleGroup, over all constraints.

        Calculates metric values, then runs reduce functions and summary statistics for
        all MetricBundles.

        Parameters
        ----------
        clearMemory : `bool`, optional
            If True, deletes metric values from memory after running each constraint group.
        plotNow : `bool`, optional
            If True, plots the metric values immediately after calculation.
        plotKwargs : `bool`, optional
            kwargs to pass to plotCurrent.
        """
        for constraint in self.constraints:
            # Set the 'currentBundleDict' which is a dictionary of the metricBundles which match this
            #  constraint.
            self.runCurrent(
                constraint,
                clearMemory=clearMemory,
                plotNow=plotNow,
                plotKwargs=plotKwargs,
            )

    def setCurrent(self, constraint):
        """Utility to set the currentBundleDict (i.e. a set of metricBundles with the same SQL constraint).

        Parameters
        ----------
        constraint : `str`
            The subset of MetricBundles with metricBundle.constraint == constraint will be
            included in a subset identified as the currentBundleDict.
            These are the active metrics to be calculated and plotted, etc.
        """
        if constraint is None:
            constraint = ""
        self.currentBundleDict = {}
        for k, b in self.bundleDict.items():
            if b.constraint == constraint:
                self.currentBundleDict[k] = b
        # Build list of all the columns needed from the database.
        self.dbCols = []
        for b in self.currentBundleDict.values():
            self.dbCols.extend(b.dbCols)
        self.dbCols = list(set(self.dbCols))

    def runCurrent(
        self,
        constraint,
        simData=None,
        clearMemory=False,
        plotNow=False,
        plotKwargs=None,
    ):
        """Run all the metricBundles which match this constraint in the metricBundleGroup.

        Calculates the metric values, then runs reduce functions and summary statistics for
        metrics in the current set only (see self.setCurrent).

        Parameters
        ----------
        constraint : `str`
           constraint to use to set the currently active metrics
        simData : `numpy.ndarray`, optional
           If simData is not None, then this numpy structured array is used instead of querying
           data from the dbObj.
        clearMemory : `bool`, optional
           If True, metric values are deleted from memory after they are calculated (and saved to disk).
        plotNow : `bool`, optional
           Plot immediately after calculating metric values (instead of the usual procedure, which
           is to plot after metric values are calculated for all constraints).
        plotKwargs : kwargs, optional
           Plotting kwargs to pass to plotCurrent.
        """
        self.setCurrent(constraint)

        # Can pass simData directly (if had other method for getting data)
        if simData is not None:
            self.simData = simData

        else:
            self.simData = None
            # Query for the data.
            try:
                self.getData(constraint)
            except UserWarning:
                warnings.warn("No data matching constraint %s" % constraint)
                metricsSkipped = []
                for b in self.currentBundleDict.values():
                    metricsSkipped.append(
                        "%s : %s : %s"
                        % (b.metric.name, b.metadata, b.slicer.slicerName)
                    )
                warnings.warn(" This means skipping metrics %s" % metricsSkipped)
                return
            except ValueError:
                warnings.warn(
                    "One or more of the columns requested from the database was not available."
                    + " Skipping constraint %s" % constraint
                )
                metricsSkipped = []
                for b in self.currentBundleDict.values():
                    metricsSkipped.append(
                        "%s : %s : %s"
                        % (b.metric.name, b.metadata, b.slicer.slicerName)
                    )
                warnings.warn(" This means skipping metrics %s" % metricsSkipped)
                return

        # Find compatible subsets of the MetricBundle dictionary,
        # which can be run/metrics calculated/ together.
        self._findCompatibleLists()

        for compatibleList in self.compatibleLists:
            if self.verbose:
                print("Running: ", compatibleList)
            self._runCompatible(compatibleList)
            if self.verbose:
                print("Completed metric generation.")
            for key in compatibleList:
                self.hasRun[key] = True
        # Run the reduce methods.
        if self.verbose:
            print("Running reduce methods.")
        self.reduceCurrent()
        # Run the summary statistics.
        if self.verbose:
            print("Running summary statistics.")
        self.summaryCurrent()
        if self.verbose:
            print("Completed.")
        if plotNow:
            if plotKwargs is None:
                self.plotCurrent()
            else:
                self.plotCurrent(**plotKwargs)
        # Optionally: clear results from memory.
        if clearMemory:
            for b in self.currentBundleDict.values():
                b.metricValues = None
            if self.verbose:
                print("Deleted metricValues from memory.")

    def getData(self, constraint):
        """Query the data from the database.

        The currently bundleDict should generally be set before calling getData (using setCurrent).

        Parameters
        ----------
        constraint : `str`
           The constraint for the currently active set of MetricBundles.
        """
        if self.verbose:
            if constraint == "":
                print(
                    "Querying database %s with no constraint for columns %s."
                    % (self.dbTable, self.dbCols)
                )
            else:
                print(
                    "Querying database %s with constraint %s for columns %s"
                    % (self.dbTable, constraint, self.dbCols)
                )
        # Note that we do NOT run the stackers at this point (this must be done in each 'compatible' group).
        self.simData = utils.getSimData(
            self.dbObj,
            constraint,
            self.dbCols,
            tableName=self.dbTable,
        )

        if self.verbose:
            print("Found %i visits" % (self.simData.size))

        # Query for the fieldData if we need it for the opsimFieldSlicer.
        needFields = [b.slicer.needsFields for b in self.currentBundleDict.values()]
        if True in needFields:
            self.fieldData = utils.getFieldData(self.dbObj, constraint)
        else:
            self.fieldData = None

    def _runCompatible(self, compatibleList):
        """Runs a set of 'compatible' metricbundles in the MetricBundleGroup dictionary,
        identified by 'compatibleList' keys.

        A compatible list of MetricBundles is a subset of the currentBundleDict.
        The currentBundleDict == set of MetricBundles with the same constraint.
        The compatibleBundles == set of MetricBundles with the same constraint, the same
        slicer, the same maps applied to the slicer, and stackers which do not clobber each other's data.

        This is where the work of calculating the metric values is done.
        """

        if len(self.simData) == 0:
            return

        # Grab a dictionary representation of this subset of the dictionary, for easier iteration.
        bDict = {key: self.currentBundleDict.get(key) for key in compatibleList}

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

        # Run stackers.
        # Run dither stackers first. (this is a bit of a hack -- we should probably figure out
        # proper hierarchy and DAG so that stackers run in the order they need to. This will catch 90%).
        ditherStackers = []
        for s in uniqStackers:
            if isinstance(s, BaseDitherStacker):
                ditherStackers.append(s)
        for stacker in ditherStackers:
            self.simData = stacker.run(self.simData, override=True)
            uniqStackers.remove(stacker)

        for stacker in uniqStackers:
            # Note that stackers will clobber previously existing rows with the same name.
            self.simData = stacker.run(self.simData, override=True)

        # Pull out one of the slicers to use as our 'slicer'.
        # This will be forced back into all of the metricBundles at the end (so that they track
        #  the same metadata such as the slicePoints, in case the same actual object wasn't used).
        slicer = list(bDict.values())[0].slicer
        slicer.setupSlicer(self.simData, maps=uniqMaps)
        # Copy the slicer (after setup) back into the individual metricBundles.
        if slicer.slicerName != "HealpixSlicer" or slicer.slicerName != "UniSlicer":
            for b in bDict.values():
                b.slicer = slicer

        # Set up (masked) arrays to store metric data in each metricBundle.
        for b in bDict.values():
            b._setupMetricValues()

        # Set up an ordered dictionary to be the cache if needed:
        # (Currently using OrderedDict, it might be faster to use 2 regular Dicts instead)
        if slicer.cacheSize > 0:
            cacheDict = OrderedDict()
            cache = True
        else:
            cache = False
        # Run through all slicepoints and calculate metrics.
        for slice_i in slicer:
            i = slice_i["slicePoint"]["sid"]
            slicedata = self.simData[slice_i["idxs"]]
            if len(slicedata) == 0:
                # No data at this slicepoint. Mask data values.
                for b in bDict.values():
                    b.metricValues.mask[i] = True
            else:
                # There is data! Should we use our data cache?
                if cache:
                    # Make the data idxs hashable.
                    cacheKey = frozenset(slice_i["idxs"])
                    # If key exists, set flag to use it, otherwise add it
                    if cacheKey in cacheDict:
                        useCache = True
                        cacheVal = cacheDict[cacheKey]
                        # Move this value to the end of the OrderedDict
                        del cacheDict[cacheKey]
                        cacheDict[cacheKey] = cacheVal
                    else:
                        cacheDict[cacheKey] = i
                        useCache = False
                    for b in bDict.values():
                        if useCache:
                            b.metricValues.data[i] = b.metricValues.data[
                                cacheDict[cacheKey]
                            ]
                        else:
                            b.metricValues.data[i] = b.metric.run(
                                slicedata, slicePoint=slice_i["slicePoint"]
                            )
                    # If we are above the cache size, drop the oldest element from the cache dict.
                    if len(cacheDict) > slicer.cacheSize:
                        del cacheDict[list(cacheDict.keys())[0]]

                # Not using memoize, just calculate things normally
                else:
                    for b in bDict.values():
                        b.metricValues.data[i] = b.metric.run(
                            slicedata, slicePoint=slice_i["slicePoint"]
                        )
        # Mask data where metrics could not be computed (according to metric bad value).
        for b in bDict.values():
            if b.metricValues.dtype.name == "object":
                for ind, val in enumerate(b.metricValues.data):
                    if val is b.metric.badval:
                        b.metricValues.mask[ind] = True
            else:
                # For some reason, this doesn't work for dtype=object arrays.
                b.metricValues.mask = np.where(
                    b.metricValues.data == b.metric.badval, True, b.metricValues.mask
                )

        # Save data to disk as we go, although this won't keep summary values, etc. (just failsafe).
        if self.saveEarly:
            for b in bDict.values():
                b.write(outDir=self.outDir, resultsDb=self.resultsDb)
        else:
            # Just write the metric run information to the resultsDb
            for b in bDict.values():
                b.writeDb(resultsDb=self.resultsDb)

    def reduceAll(self, updateSummaries=True):
        """Run the reduce methods for all metrics in bundleDict.

        Running this method, for all MetricBundles at once, assumes that clearMemory was False.

        Parameters
        ----------
        updateSummaries : `bool`, optional
            If True, summary metrics are removed from the top-level (non-reduced)
            MetricBundle. Usually this should be True, as summary metrics are generally
            intended to run on the simpler data produced by reduce metrics.
        """
        for constraint in self.constraints:
            self.setCurrent(constraint)
            self.reduceCurrent(updateSummaries=updateSummaries)

    def reduceCurrent(self, updateSummaries=True):
        """Run all reduce functions for the metricbundle in the currently active set of MetricBundles.

        Parameters
        ----------
        updateSummaries : `bool`, optional
            If True, summary metrics are removed from the top-level (non-reduced)
            MetricBundle. Usually this should be True, as summary metrics are generally
            intended to run on the simpler data produced by reduce metrics.
        """
        # Create a temporary dictionary to hold the reduced metricbundles.
        reduceBundleDict = {}
        for b in self.currentBundleDict.values():
            # If there are no reduce functions associated with the metric, skip this metricBundle.
            if len(b.metric.reduceFuncs) > 0:
                # Apply reduce functions, creating a new metricBundle in the process (new metric values).
                for reduceFunc in b.metric.reduceFuncs.values():
                    newmetricbundle = b.reduceMetric(reduceFunc)
                    # Add the new metricBundle to our metricBundleGroup dictionary.
                    name = newmetricbundle.metric.name
                    if name in self.bundleDict:
                        name = newmetricbundle.fileRoot
                    reduceBundleDict[name] = newmetricbundle
                    if self.saveEarly:
                        newmetricbundle.write(
                            outDir=self.outDir, resultsDb=self.resultsDb
                        )
                    else:
                        newmetricbundle.writeDb(resultsDb=self.resultsDb)
                # Remove summaryMetrics from top level metricbundle if desired.
                if updateSummaries:
                    b.summaryMetrics = []
        # Add the new metricBundles to the MetricBundleGroup dictionary.
        self.bundleDict.update(reduceBundleDict)
        # And add to to the currentBundleDict too, so we run as part of 'summaryCurrent'.
        self.currentBundleDict.update(reduceBundleDict)

    def summaryAll(self):
        """Run the summary statistics for all metrics in bundleDict.

        Calculating all summary statistics, for all MetricBundles, at this
        point assumes that clearMemory was False.
        """
        for constraint in self.constraints:
            self.setCurrent(constraint)
            self.summaryCurrent()

    def summaryCurrent(self):
        """Run summary statistics on all the metricBundles in the currently active set of MetricBundles."""
        for b in self.currentBundleDict.values():
            b.computeSummaryStats(self.resultsDb)

    def plotAll(
        self,
        savefig=True,
        outfileSuffix=None,
        figformat="pdf",
        dpi=600,
        trimWhitespace=True,
        thumbnail=True,
        closefigs=True,
    ):
        """Generate all the plots for all the metricBundles in bundleDict.

        Generating all ploots, for all MetricBundles, at this point, assumes that
        clearMemory was False.

        Parameters
        ----------
        savefig : `bool`, optional
            If True, save figures to disk, to self.outDir directory.
        outfileSuffix : `bool`, optional
            Append outfileSuffix to the end of every plot file generated. Useful for generating
            sequential series of images for movies.
        figformat : `str`, optional
            Matplotlib figure format to use to save to disk. Default pdf.
        dpi : `int`, optional
            DPI for matplotlib figure. Default 600.
        trimWhitespace : `bool`, optional
            If True, trim additional whitespace from final figures. Default True.
        thumbnail : `bool`, optional
            If True, save a small thumbnail jpg version of the output file to disk as well.
            This is useful for showMaf web pages. Default True.
        closefigs : `bool`, optional
            Close the matplotlib figures after they are saved to disk. If many figures are
            generated, closing the figures saves significant memory. Default True.
        """
        for constraint in self.constraints:
            if self.verbose:
                print('Plotting figures with "%s" constraint now.' % (constraint))

            self.setCurrent(constraint)
            self.plotCurrent(
                savefig=savefig,
                outfileSuffix=outfileSuffix,
                figformat=figformat,
                dpi=dpi,
                trimWhitespace=trimWhitespace,
                thumbnail=thumbnail,
                closefigs=closefigs,
            )

    def plotCurrent(
        self,
        savefig=True,
        outfileSuffix=None,
        figformat="pdf",
        dpi=600,
        trimWhitespace=True,
        thumbnail=True,
        closefigs=True,
    ):
        """Generate the plots for the currently active set of MetricBundles.

        Parameters
        ----------
        savefig : `bool`, optional
            If True, save figures to disk, to self.outDir directory.
        outfileSuffix : `str`, optional
            Append outfileSuffix to the end of every plot file generated. Useful for generating
            sequential series of images for movies.
        figformat : `str`, optional
            Matplotlib figure format to use to save to disk. Default pdf.
        dpi : `int`, optional
            DPI for matplotlib figure. Default 600.
        trimWhitespace : `bool`, optional
            If True, trim additional whitespace from final figures. Default True.
        thumbnail : `bool`, optional
            If True, save a small thumbnail jpg version of the output file to disk as well.
            This is useful for showMaf web pages. Default True.
        closefigs : `bool`, optional
            Close the matplotlib figures after they are saved to disk. If many figures are
            generated, closing the figures saves significant memory. Default True.
        """
        plotHandler = PlotHandler(
            outDir=self.outDir,
            resultsDb=self.resultsDb,
            savefig=savefig,
            figformat=figformat,
            dpi=dpi,
            trimWhitespace=trimWhitespace,
            thumbnail=thumbnail,
        )

        for b in self.currentBundleDict.values():
            try:
                b.plot(
                    plotHandler=plotHandler,
                    outfileSuffix=outfileSuffix,
                    savefig=savefig,
                )
            except ValueError as ve:
                message = "Plotting failed for metricBundle %s." % (b.fileRoot)
                message += " Error message: %s" % (ve)
                warnings.warn(message)
            if closefigs:
                plt.close("all")
        if self.verbose:
            print("Plotting complete.")

    def writeAll(self):
        """Save all the MetricBundles to disk.

        Saving all MetricBundles to disk at this point assumes that clearMemory was False.
        """
        for constraint in self.constraints:
            self.setCurrent(constraint)
            self.writeCurrent()

    def writeCurrent(self):
        """Save all the MetricBundles in the currently active set to disk."""
        if self.verbose:
            if self.saveEarly:
                print("Re-saving metric bundles.")
            else:
                print("Saving metric bundles.")
        for b in self.currentBundleDict.values():
            b.write(outDir=self.outDir, resultsDb=self.resultsDb)

    def readAll(self):
        """Attempt to read all MetricBundles from disk.

        You must set the metrics/slicer/constraint/runName for a metricBundle appropriately;
        then this method will search for files in the location self.outDir/metricBundle.fileRoot.
        Reads all the files associated with all metricbundles in self.bundleDict.
        """
        reduceBundleDict = {}
        removeBundles = []
        for b in self.bundleDict:
            bundle = self.bundleDict[b]
            filename = os.path.join(self.outDir, bundle.fileRoot + ".npz")
            try:
                # Create a temporary metricBundle to read the data into.
                #  (we don't use b directly, as this overrides plotDict/etc).
                tmpBundle = createEmptyMetricBundle()
                tmpBundle.read(filename)
                # Copy the tmpBundle metricValues into bundle.
                bundle.metricValues = tmpBundle.metricValues
                # And copy the slicer into b, to get slicePoints.
                bundle.slicer = tmpBundle.slicer
                if self.verbose:
                    print("Read %s from disk." % (bundle.fileRoot))
            except IOError:
                warnings.warn(
                    "Warning: file %s not found, bundle not restored." % filename
                )
                removeBundles.append(b)

            # Look to see if this is a complex metric, with associated 'reduce' functions,
            # and read those in too.
            if len(bundle.metric.reduceFuncs) > 0:
                origMetricName = bundle.metric.name
                for reduceFunc in bundle.metric.reduceFuncs.values():
                    reduceName = (
                        origMetricName + "_" + reduceFunc.__name__.replace("reduce", "")
                    )
                    # Borrow the fileRoot in b (we'll reset it appropriately afterwards).
                    bundle.metric.name = reduceName
                    bundle._buildFileRoot()
                    filename = os.path.join(self.outDir, bundle.fileRoot + ".npz")
                    tmpBundle = createEmptyMetricBundle()
                    try:
                        tmpBundle.read(filename)
                        # This won't necessarily recreate the plotDict and displayDict exactly
                        # as they would have been made if you calculated the reduce metric from scratch.
                        # Perhaps update these metric reduce dictionaries after reading them in?
                        newmetricBundle = MetricBundle(
                            metric=bundle.metric,
                            slicer=bundle.slicer,
                            constraint=bundle.constraint,
                            stackerList=bundle.stackerList,
                            runName=bundle.runName,
                            metadata=bundle.metadata,
                            plotDict=bundle.plotDict,
                            displayDict=bundle.displayDict,
                            summaryMetrics=bundle.summaryMetrics,
                            mapsList=bundle.mapsList,
                            fileRoot=bundle.fileRoot,
                            plotFuncs=bundle.plotFuncs,
                        )
                        newmetricBundle.metric.name = reduceName
                        newmetricBundle.metricValues = ma.copy(tmpBundle.metricValues)
                        # Add the new metricBundle to our metricBundleGroup dictionary.
                        name = newmetricBundle.metric.name
                        if name in self.bundleDict:
                            name = newmetricBundle.fileRoot
                        reduceBundleDict[name] = newmetricBundle
                        if self.verbose:
                            print("Read %s from disk." % (newmetricBundle.fileRoot))
                    except IOError:
                        warnings.warn(
                            'Warning: file %s not found, bundle not restored ("reduce" metric).'
                            % filename
                        )

                    # Remove summaryMetrics from top level metricbundle.
                    bundle.summaryMetrics = []
                    # Update parent MetricBundle name.
                    bundle.metric.name = origMetricName
                    bundle._buildFileRoot()

        # Add the reduce bundles into the bundleDict.
        self.bundleDict.update(reduceBundleDict)
        # And remove the bundles which were not found on disk, so we don't try to make (blank) plots.
        for b in removeBundles:
            del self.bundleDict[b]
