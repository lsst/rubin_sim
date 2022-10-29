import os
import sys
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from collections import OrderedDict
import tqdm

import rubin_sim.maf.utils as utils
from rubin_sim.maf.plots import PlotHandler
import rubin_sim.maf.maps as maps
import rubin_sim.maf.db as db
from rubin_sim.maf.stackers import BaseDitherStacker
from .metric_bundle import MetricBundle, create_empty_metric_bundle
import warnings

__all__ = ["make_bundles_dict_from_list", "MetricBundleGroup"]


def make_bundles_dict_from_list(bundleList):
    """Utility to convert a list of MetricBundles into a dictionary, keyed by the fileRoot names.

    Raises an exception if the fileroot duplicates another metricBundle.
    (Note this should alert to potential cases of filename duplication).

    Parameters
    ----------
    bundleList : `list` of `MetricBundles`
    """
    b_dict = {}
    for b in bundleList:
        if b.file_root in b_dict:
            raise NameError(
                "More than one metricBundle is using the same fileroot, %s"
                % (b.file_root)
            )
        b_dict[b.file_root] = b
    return b_dict


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
    bundle_dict : `dict` or `list` of `MetricBundles`
        Individual MetricBundles should be placed into a dictionary, and then passed to
        the MetricBundleGroup. The dictionary keys can then be used to identify MetricBundles
        if needed -- and to identify new MetricBundles which could be created if 'reduce'
        functions are run on a particular MetricBundle.
        A bundleDict can be conveniently created from a list of MetricBundles using
        makeBundlesDictFromList (done automatically if a list is passed in)
    db_con : `str` or database connection object
        A str that is the path to a sqlite3 file or a database object that can be used by pandas.read_sql.
        Advanced use: It is possible to set this to None, in which case data should be passed
        directly to the runCurrent method (and runAll should not be used).
    out_dir : `str`, optional
        Directory to save the metric results. Default is the current directory.
    results_db : `ResultsDb`, optional
        A results database. If not specified, one will be created in the outDir.
        This database saves information about the metrics calculated, including their summary statistics.
    verbose : `bool`, optional
        Flag to turn on/off verbose feedback.
    save_early : `bool`, optional
        If True, metric values will be saved immediately after they are first calculated (to prevent
        data loss) as well as after summary statistics are calculated.
        If False, metric values will only be saved after summary statistics are calculated.
    db_table : `str`, optional
        The name of the table in the dbObj to query for data.
    """

    def __init__(
        self,
        bundle_dict,
        db_con,
        out_dir=".",
        results_db=None,
        verbose=False,
        save_early=True,
        db_table=None,
    ):
        """Set up the MetricBundleGroup."""
        if type(bundle_dict) is list:
            bundle_dict = make_bundles_dict_from_list(bundle_dict)
        # Print occasional messages to screen.
        self.verbose = verbose
        # Save metric results as soon as possible (in case of crash).
        self.saveEarly = save_early
        # Check for output directory, create it if needed.
        self.outDir = out_dir
        if not os.path.isdir(self.outDir):
            os.makedirs(self.outDir)

        # Do some type checking on the MetricBundle dictionary.
        if not isinstance(bundle_dict, dict):
            raise ValueError(
                "bundleDict should be a dictionary containing MetricBundle objects."
            )
        for b in bundle_dict.values():
            if not isinstance(b, MetricBundle):
                raise ValueError("bundleDict should contain only MetricBundle objects.")
        # Identify the series of constraints.
        self.constraints = list(set([b.constraint for b in bundle_dict.values()]))
        # Set the bundleDict (all bundles, with all constraints)
        self.bundleDict = bundle_dict

        self.dbObj = db_con
        # Set the table we're going to be querying.
        self.dbTable = db_table

        # Check the resultsDb (optional).
        if results_db is not None:
            if not isinstance(results_db, db.ResultsDb):
                raise ValueError("resultsDb should be an ResultsDb object")
        self.resultsDb = results_db

        # Dict to keep track of what's been run:
        self.hasRun = {}
        for bk in bundle_dict:
            self.hasRun[bk] = False

    def _check_compatible(self, metric_bundle1, metric_bundle2):
        """Check if two MetricBundles are "compatible".
        Compatible indicates that the sql constraints, the slicers, and the maps are the same, and
        that the stackers do not interfere with each other
        (i.e. are not trying to set the same column in different ways).
        Returns True if the MetricBundles are compatible, False if not.

        Parameters
        ----------
        metric_bundle1 : MetricBundle
        metric_bundle2 : MetricBundle

        Returns
        -------
        bool
        """
        if metric_bundle1.constraint != metric_bundle2.constraint:
            return False
        if metric_bundle1.slicer != metric_bundle2.slicer:
            return False
        if metric_bundle1.maps_list.sort() != metric_bundle2.maps_list.sort():
            return False
        for stacker in metric_bundle1.stacker_list:
            for stacker2 in metric_bundle2.stacker_list:
                # If the stackers have different names, that's OK, and if they are identical, that's ok.
                if (stacker.__class__.__name__ == stacker2.__class__.__name__) & (
                    stacker != stacker2
                ):
                    return False
        # But if we got this far, everything matches.
        return True

    def _find_compatible_lists(self):
        """Find sets of compatible metricBundles from the currentBundleDict."""
        # CompatibleLists stores a list of lists;
        #   each (nested) list contains the bundleDict _keys_ of a compatible set of metricBundles.
        #
        compatible_lists = []
        for k, b in self.currentBundleDict.items():
            found_compatible = False
            for compatibleList in compatible_lists:
                comparison_metric_bundle_key = compatibleList[0]
                compatible = self._check_compatible(
                    self.bundleDict[comparison_metric_bundle_key], b
                )
                if compatible:
                    # Must compare all metricBundles in each subset (if they are a potential match),
                    #  as the stackers could be different (and one could be incompatible,
                    #  not necessarily the first)
                    for comparison_metric_bundle_key in compatibleList[1:]:
                        compatible = self._check_compatible(
                            self.bundleDict[comparison_metric_bundle_key], b
                        )
                        if not compatible:
                            # If we find one which is not compatible, stop and go on to the
                            # next subset list.
                            break
                    # Otherwise, we reached the end of the subset and they were all compatible.
                    found_compatible = True
                    compatibleList.append(k)
            if not found_compatible:
                # Didn't find a pre-existing compatible set; make a new one.
                compatible_lists.append(
                    [
                        k,
                    ]
                )
        self.compatibleLists = compatible_lists

    def run_all(self, clear_memory=False, plot_now=False, plot_kwargs=None):
        """Runs all the metricBundles in the metricBundleGroup, over all constraints.

        Calculates metric values, then runs reduce functions and summary statistics for
        all MetricBundles.

        Parameters
        ----------
        clear_memory : `bool`, optional
            If True, deletes metric values from memory after running each constraint group.
        plot_now : `bool`, optional
            If True, plots the metric values immediately after calculation.
        plot_kwargs : `bool`, optional
            kwargs to pass to plotCurrent.
        """
        for constraint in self.constraints:
            # Set the 'currentBundleDict' which is a dictionary of the metricBundles which match this
            #  constraint.
            self.run_current(
                constraint,
                clear_memory=clear_memory,
                plot_now=plot_now,
                plot_kwargs=plot_kwargs,
            )

    def set_current(self, constraint):
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
        self.current_bundle_dict = {}
        for k, b in self.bundleDict.items():
            if b.constraint == constraint:
                self.currentBundleDict[k] = b
        # Build list of all the columns needed from the database.
        self.db_cols = []
        for b in self.currentBundleDict.values():
            self.dbCols.extend(b.dbCols)
        self.db_cols = list(set(self.dbCols))

    def run_current(
        self,
        constraint,
        sim_data=None,
        clear_memory=False,
        plot_now=False,
        plot_kwargs=None,
    ):
        """Run all the metricBundles which match this constraint in the metricBundleGroup.

        Calculates the metric values, then runs reduce functions and summary statistics for
        metrics in the current set only (see self.setCurrent).

        Parameters
        ----------
        constraint : `str`
           constraint to use to set the currently active metrics
        sim_data : `numpy.ndarray`, optional
           If simData is not None, then this numpy structured array is used instead of querying
           data from the dbObj.
        clear_memory : `bool`, optional
           If True, metric values are deleted from memory after they are calculated (and saved to disk).
        plot_now : `bool`, optional
           Plot immediately after calculating metric values (instead of the usual procedure, which
           is to plot after metric values are calculated for all constraints).
        plot_kwargs : kwargs, optional
           Plotting kwargs to pass to plotCurrent.
        """
        self.set_current(constraint)

        # Can pass simData directly (if had other method for getting data)
        if sim_data is not None:
            self.sim_data = sim_data

        else:
            self.sim_data = None
            # Query for the data.
            try:
                self.get_data(constraint)
            except UserWarning:
                warnings.warn("No data matching constraint %s" % constraint)
                metrics_skipped = []
                for b in self.currentBundleDict.values():
                    metrics_skipped.append(
                        "%s : %s : %s"
                        % (b.metric.name, b.info_label, b.slicer.slicerName)
                    )
                warnings.warn(" This means skipping metrics %s" % metrics_skipped)
                return
            except ValueError:
                warnings.warn(
                    "One or more of the columns requested from the database was not available."
                    + " Skipping constraint %s" % constraint
                )
                metrics_skipped = []
                for b in self.currentBundleDict.values():
                    metrics_skipped.append(
                        "%s : %s : %s"
                        % (b.metric.name, b.info_label, b.slicer.slicerName)
                    )
                warnings.warn(" This means skipping metrics %s" % metrics_skipped)
                return

        # Find compatible subsets of the MetricBundle dictionary,
        # which can be run/metrics calculated/ together.
        self._find_compatible_lists()

        for compatibleList in self.compatibleLists:
            if self.verbose:
                print("Running: ", compatibleList)
            self._run_compatible(compatibleList)
            if self.verbose:
                print("Completed metric generation.")
            for key in compatibleList:
                self.hasRun[key] = True
        # Run the reduce methods.
        if self.verbose:
            print("Running reduce methods.")
        self.reduce_current()
        # Run the summary statistics.
        if self.verbose:
            print("Running summary statistics.")
        self.summary_current()
        if self.verbose:
            print("Completed.")
        if plot_now:
            if plot_kwargs is None:
                self.plot_current()
            else:
                self.plot_current(**plot_kwargs)
        # Optionally: clear results from memory.
        if clear_memory:
            for b in self.currentBundleDict.values():
                b.metricValues = None
            if self.verbose:
                print("Deleted metricValues from memory.")

    def get_data(self, constraint):
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
                    "Querying table %s with no constraint for columns %s."
                    % (self.dbTable, self.dbCols)
                )
            else:
                print(
                    "Querying table %s with constraint %s for columns %s"
                    % (self.dbTable, constraint, self.dbCols)
                )
        # Note that we do NOT run the stackers at this point (this must be done in each 'compatible' group).
        self.sim_data = utils.getSimData(
            self.dbObj,
            constraint,
            self.dbCols,
            tableName=self.dbTable,
        )

        if self.verbose:
            print("Found %i visits" % (self.sim_data.size))

    def _run_compatible(self, compatibleList):
        """Runs a set of 'compatible' metricbundles in the MetricBundleGroup dictionary,
        identified by 'compatibleList' keys.

        A compatible list of MetricBundles is a subset of the currentBundleDict.
        The currentBundleDict == set of MetricBundles with the same constraint.
        The compatibleBundles == set of MetricBundles with the same constraint, the same
        slicer, the same maps applied to the slicer, and stackers which do not clobber each other's data.

        This is where the work of calculating the metric values is done.
        """

        if len(self.sim_data) == 0:
            return

        # Grab a dictionary representation of this subset of the dictionary, for easier iteration.
        b_dict = {key: self.currentBundleDict.get(key) for key in compatibleList}

        # Find the unique stackers and maps. These are already "compatible" (as id'd by compatibleList).
        uniq_stackers = []
        all_stackers = []
        uniq_maps = []
        all_maps = []
        for b in b_dict.values():
            all_stackers += b.stacker_list
            all_maps += b.maps_list
        for s in all_stackers:
            if s not in uniq_stackers:
                uniq_stackers.append(s)
        for m in all_maps:
            if m not in uniq_maps:
                uniq_maps.append(m)

        # Run stackers.
        # Run dither stackers first. (this is a bit of a hack -- we should probably figure out
        # proper hierarchy and DAG so that stackers run in the order they need to. This will catch 90%).
        dither_stackers = []
        for s in uniq_stackers:
            if isinstance(s, BaseDitherStacker):
                dither_stackers.append(s)
        for stacker in dither_stackers:
            self.sim_data = stacker.run(self.sim_data, override=True)
            uniq_stackers.remove(stacker)

        for stacker in uniq_stackers:
            # Note that stackers will clobber previously existing rows with the same name.
            self.sim_data = stacker.run(self.sim_data, override=True)

        # Pull out one of the slicers to use as our 'slicer'.
        # This will be forced back into all of the metricBundles at the end (so that they track
        #  the same info_label such as the slicePoints, in case the same actual object wasn't used).
        slicer = list(b_dict.values())[0].slicer
        slicer.setupSlicer(self.sim_data, maps=uniq_maps)
        # Copy the slicer (after setup) back into the individual metricBundles.
        if slicer.slicerName != "HealpixSlicer" or slicer.slicerName != "UniSlicer":
            for b in b_dict.values():
                b.slicer = slicer

        # Set up (masked) arrays to store metric data in each metricBundle.
        for b in b_dict.values():
            b._setup_metric_values()

        # Set up an ordered dictionary to be the cache if needed:
        # (Currently using OrderedDict, it might be faster to use 2 regular Dicts instead)
        if slicer.cacheSize > 0:
            cache_dict = OrderedDict()
            cache = True
        else:
            cache = False
        # Run through all slicepoints and calculate metrics.
        if self.verbose:
            slicer_iter = tqdm.tqdm(
                slicer,
                desc="Processing slices",
                ncols=79,
                file=sys.stdout,
            )
        else:
            slicer_iter = slicer

        for slice_i in slicer_iter:
            i = slice_i["slicePoint"]["sid"]
            slicedata = self.sim_data[slice_i["idxs"]]
            if len(slicedata) == 0:
                # No data at this slicepoint. Mask data values.
                for b in b_dict.values():
                    b.metricValues.mask[i] = True
            else:
                # There is data! Should we use our data cache?
                if cache:
                    # Make the data idxs hashable.
                    cacheKey = frozenset(slice_i["idxs"])
                    # If key exists, set flag to use it, otherwise add it
                    if cacheKey in cache_dict:
                        useCache = True
                        cacheVal = cache_dict[cacheKey]
                        # Move this value to the end of the OrderedDict
                        del cache_dict[cacheKey]
                        cache_dict[cacheKey] = cacheVal
                    else:
                        cache_dict[cacheKey] = i
                        useCache = False
                    for b in b_dict.values():
                        if useCache:
                            b.metricValues.data[i] = b.metricValues.data[
                                cache_dict[cacheKey]
                            ]
                        else:
                            b.metricValues.data[i] = b.metric.run(
                                slicedata, slicePoint=slice_i["slicePoint"]
                            )
                    # If we are above the cache size, drop the oldest element from the cache dict.
                    if len(cache_dict) > slicer.cacheSize:
                        del cache_dict[list(cache_dict.keys())[0]]

                # Not using memoize, just calculate things normally
                else:
                    for b in b_dict.values():
                        try:
                            b.metricValues.data[i] = b.metric.run(
                                slicedata, slicePoint=slice_i["slicePoint"]
                            )
                        except BaseException as e:
                            print(f"Failed at slicePoint {slice_i}, sid {i}")
                            raise e
        # Mask data where metrics could not be computed (according to metric bad value).
        for b in b_dict.values():
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
            for b in b_dict.values():
                b.write(out_dir=self.outDir, results_db=self.resultsDb)
        else:
            # Just write the metric run information to the resultsDb
            for b in b_dict.values():
                b.write_db(results_db=self.resultsDb)

    def reduce_all(self, update_summaries=True):
        """Run the reduce methods for all metrics in bundleDict.

        Running this method, for all MetricBundles at once, assumes that clearMemory was False.

        Parameters
        ----------
        update_summaries : `bool`, optional
            If True, summary metrics are removed from the top-level (non-reduced)
            MetricBundle. Usually this should be True, as summary metrics are generally
            intended to run on the simpler data produced by reduce metrics.
        """
        for constraint in self.constraints:
            self.set_current(constraint)
            self.reduce_current(update_summaries=update_summaries)

    def reduce_current(self, update_summaries=True):
        """Run all reduce functions for the metricbundle in the currently active set of MetricBundles.

        Parameters
        ----------
        update_summaries : `bool`, optional
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
                for r in b.metric.reduceFuncs:
                    newmetricbundle = b.reduce_metric(
                        b.metric.reduceFuncs[r], reduce_func_name=r
                    )
                    # Add the new metricBundle to our metricBundleGroup dictionary.
                    name = newmetricbundle.metric.name
                    if name in self.bundleDict:
                        name = newmetricbundle.file_root
                    reduceBundleDict[name] = newmetricbundle
                    if self.saveEarly:
                        newmetricbundle.write(
                            out_dir=self.outDir, results_db=self.resultsDb
                        )
                    else:
                        newmetricbundle.write_db(results_db=self.resultsDb)
                # Remove summaryMetrics from top level metricbundle if desired.
                if update_summaries:
                    b.summary_metrics = []
        # Add the new metricBundles to the MetricBundleGroup dictionary.
        self.bundleDict.update(reduceBundleDict)
        # And add to to the currentBundleDict too, so we run as part of 'summaryCurrent'.
        self.currentBundleDict.update(reduceBundleDict)

    def summary_all(self):
        """Run the summary statistics for all metrics in bundleDict.

        Calculating all summary statistics, for all MetricBundles, at this
        point assumes that clearMemory was False.
        """
        for constraint in self.constraints:
            self.set_current(constraint)
            self.summary_current()

    def summary_current(self):
        """Run summary statistics on all the metricBundles in the currently active set of MetricBundles."""
        for b in self.currentBundleDict.values():
            b.compute_summary_stats(self.resultsDb)

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

            self.set_current(constraint)
            self.plot_current(
                savefig=savefig,
                outfileSuffix=outfileSuffix,
                figformat=figformat,
                dpi=dpi,
                trimWhitespace=trimWhitespace,
                thumbnail=thumbnail,
                closefigs=closefigs,
            )

    def plot_current(
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
        plot_handler = PlotHandler(
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
                    plotHandler=plot_handler,
                    outfileSuffix=outfileSuffix,
                    savefig=savefig,
                )
            except ValueError as ve:
                message = "Plotting failed for metricBundle %s." % (b.file_root)
                message += " Error message: %s" % (ve)
                warnings.warn(message)
            if closefigs:
                plt.close("all")
        if self.verbose:
            print("Plotting complete.")

    def write_all(self):
        """Save all the MetricBundles to disk.

        Saving all MetricBundles to disk at this point assumes that clearMemory was False.
        """
        for constraint in self.constraints:
            self.set_current(constraint)
            self.write_current()

    def write_current(self):
        """Save all the MetricBundles in the currently active set to disk."""
        if self.verbose:
            if self.saveEarly:
                print("Re-saving metric bundles.")
            else:
                print("Saving metric bundles.")
        for b in self.currentBundleDict.values():
            b.write(out_dir=self.outDir, results_db=self.resultsDb)

    def read_all(self):
        """Attempt to read all MetricBundles from disk.

        You must set the metrics/slicer/constraint/runName for a metricBundle appropriately;
        then this method will search for files in the location self.outDir/metricBundle.fileRoot.
        Reads all the files associated with all metricbundles in self.bundleDict.
        """
        reduce_bundle_dict = {}
        remove_bundles = []
        for b in self.bundleDict:
            bundle = self.bundleDict[b]
            filename = os.path.join(self.outDir, bundle.file_root + ".npz")
            try:
                # Create a temporary metricBundle to read the data into.
                #  (we don't use b directly, as this overrides plotDict/etc).
                tmp_bundle = create_empty_metric_bundle()
                tmp_bundle.read(filename)
                # Copy the tmp_bundle metricValues into bundle.
                bundle.metricValues = tmp_bundle.metricValues
                # And copy the slicer into b, to get slicePoints.
                bundle.slicer = tmp_bundle.slicer
                if self.verbose:
                    print("Read %s from disk." % (bundle.file_root))
            except IOError:
                warnings.warn(
                    "Warning: file %s not found, bundle not restored." % filename
                )
                remove_bundles.append(b)

            # Look to see if this is a complex metric, with associated 'reduce' functions,
            # and read those in too.
            if len(bundle.metric.reduceFuncs) > 0:
                orig_metric_name = bundle.metric.name
                for reduceFunc in bundle.metric.reduceFuncs.values():
                    reduce_name = (
                        orig_metric_name + "_" + reduceFunc.__name__.replace("reduce", "")
                    )
                    # Borrow the fileRoot in b (we'll reset it appropriately afterwards).
                    bundle.metric.name = reduce_name
                    bundle._build_file_root()
                    filename = os.path.join(self.outDir, bundle.file_root + ".npz")
                    tmp_bundle = create_empty_metric_bundle()
                    try:
                        tmp_bundle.read(filename)
                        # This won't necessarily recreate the plotDict and displayDict exactly
                        # as they would have been made if you calculated the reduce metric from scratch.
                        # Perhaps update these metric reduce dictionaries after reading them in?
                        newmetric_bundle = MetricBundle(
                            metric=bundle.metric,
                            slicer=bundle.slicer,
                            constraint=bundle.constraint,
                            stacker_list=bundle.stacker_list,
                            run_name=bundle.run_name,
                            info_label=bundle.info_label,
                            plot_dict=bundle.plotDict,
                            display_dict=bundle.displayDict,
                            summary_metrics=bundle.summary_metrics,
                            maps_list=bundle.maps_list,
                            file_root=bundle.file_root,
                            plot_funcs=bundle.plot_funcs,
                        )
                        newmetric_bundle.metric.name = reduce_name
                        newmetric_bundle.metricValues = ma.copy(tmp_bundle.metricValues)
                        # Add the new metricBundle to our metricBundleGroup dictionary.
                        name = newmetric_bundle.metric.name
                        if name in self.bundleDict:
                            name = newmetric_bundle.file_root
                        reduce_bundle_dict[name] = newmetric_bundle
                        if self.verbose:
                            print("Read %s from disk." % (newmetric_bundle.file_root))
                    except IOError:
                        warnings.warn(
                            'Warning: file %s not found, bundle not restored ("reduce" metric).'
                            % filename
                        )

                    # Remove summaryMetrics from top level metricbundle.
                    bundle.summary_metrics = []
                    # Update parent MetricBundle name.
                    bundle.metric.name = orig_metric_name
                    bundle._build_file_root()

        # Add the reduce bundles into the bundleDict.
        self.bundleDict.update(reduce_bundle_dict)
        # And remove the bundles which were not found on disk, so we don't try to make (blank) plots.
        for b in remove_bundles:
            del self.bundleDict[b]
