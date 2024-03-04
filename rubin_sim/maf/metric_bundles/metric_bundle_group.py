__all__ = ("make_bundles_dict_from_list", "MetricBundleGroup")

import os
import sys
import warnings
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import tqdm

import rubin_sim.maf.db as db
import rubin_sim.maf.utils as utils
from rubin_sim.maf.plots import PlotHandler

from .metric_bundle import MetricBundle, create_empty_metric_bundle


def make_bundles_dict_from_list(bundle_list):
    """Utility to convert a list of MetricBundles into a dictionary,
    keyed by the file_root names.

    Raises an exception if the file_root duplicates another metricBundle.
    (Note this should alert to potential cases of filename duplication).

    Parameters
    ----------
    bundle_list : `list` [`MetricBundles`]
        List of metric bundles to convert into a dict.
    """
    b_dict = {}
    for b in bundle_list:
        if b.file_root in b_dict:
            raise NameError("More than one metric_bundle is using the same file_root, %s" % (b.file_root))
        b_dict[b.file_root] = b
    return b_dict


class MetricBundleGroup:
    """Calculate all values for a group of MetricBundles.

    Parameters
    ----------
    bundle_dict : `dict` or `list` [`MetricBundles`]
        Individual MetricBundles should be placed into a dictionary,
        and then passed to the MetricBundleGroup.
        The dictionary keys can then be used to identify MetricBundles
        if needed -- and to identify new MetricBundles which could be
        created if 'reduce' functions are run on a particular MetricBundle.
        A bundle_dict can be conveniently created from a list of MetricBundles
        using makeBundlesDictFromList (done automatically if a list is passed).
    db_con : `str` or database connection object
        A str that is the path to a sqlite3 file or a database object
        that can be used by pandas.read_sql.
        Advanced use: It is possible to set this to None, in which case
        data should be passed directly to the runCurrent method
        (and runAll should not be used).
    out_dir : `str`, opt
        Directory to save the metric results. Default is the current directory.
    results_db : `ResultsDb`, opt
        A results database to store summary stat information.
        If not specified, one will be created in the out_dir.
        This database saves information about the metrics calculated,
        including their summary statistics.
    verbose : `bool`, opt
        Flag to turn on/off verbose feedback.
    save_early : `bool`, opt
        If True, metric values will be saved immediately after
        they are first calculated (to prevent data loss) as well as after
        summary statistics are calculated.
        If False, metric values will only be saved after summary statistics
        are calculated.
    db_table : `str`, opt
        The name of the table in the db_obj to query for data.
        For modern opsim outputs, this table is `observations` (default None).

    Notes
    -----
    The MetricBundleGroup will query data from a single database table
    (for multiple constraints), use that data to calculate metric values
    for multiple slicers, and calculate summary statistics and
    generate plots for all metrics included in
    the dictionary passed to the MetricBundleGroup.

    We calculate the metric values here, rather than in the
    individual MetricBundles, because it is much more efficient to step
    through a slicer once (and calculate all the relevant metric values
    at each point) than it is to repeat this process multiple times.

    The MetricBundleGroup also determines how to efficiently group
    the MetricBundles to reduce the number of sql queries of the database,
    grabbing larger chunks of data at once.
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
        if isinstance(bundle_dict, list):
            bundle_dict = make_bundles_dict_from_list(bundle_dict)
        # Print occasional messages to screen.
        self.verbose = verbose
        # Save metric results as soon as possible (in case of crash).
        self.save_early = save_early
        # Check for output directory, create it if needed.
        self.out_dir = out_dir
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)

        # Do some type checking on the MetricBundle dictionary.
        if not isinstance(bundle_dict, dict):
            raise ValueError("bundleDict should be a dictionary containing MetricBundle objects.")
        for b in bundle_dict.values():
            if not isinstance(b, MetricBundle):
                raise ValueError("bundleDict should contain only MetricBundle objects.")
        # Identify the series of constraints.
        self.constraints = list(set([b.constraint for b in bundle_dict.values()]))
        # Set the bundleDict (all bundles, with all constraints)
        self.bundle_dict = bundle_dict

        self.db_obj = db_con
        # Set the table we're going to be querying.
        self.db_table = db_table

        # Check the results_db (optional).
        if results_db is not None:
            if not isinstance(results_db, db.ResultsDb):
                raise ValueError("results_db should be an ResultsDb object")
        self.results_db = results_db

        # Dict to keep track of what's been run:
        self.has_run = {}
        for bk in bundle_dict:
            self.has_run[bk] = False

    def _check_compatible(self, metric_bundle1, metric_bundle2):
        """Check if two MetricBundles are "compatible".

        Parameters
        ----------
        metric_bundle1 : `MetricBundle`
        metric_bundle2 : `MetricBundle`

        Returns
        -------
        match : `bool`

        Notes
        -----
        Compatible indicates that the sql constraints, the slicers,
        and the maps are the same, and
        that the stackers do not interfere with each other
        (i.e. are not trying to set the same column in different ways).
        Returns True if the MetricBundles are compatible, False if not.
        """
        if metric_bundle1.constraint != metric_bundle2.constraint:
            return False
        if metric_bundle1.slicer != metric_bundle2.slicer:
            return False
        if metric_bundle1.maps_list.sort() != metric_bundle2.maps_list.sort():
            return False
        for stacker in metric_bundle1.stacker_list:
            for stacker2 in metric_bundle2.stacker_list:
                # If the stackers have different names, that's OK,
                # and if they are identical, that's ok.
                if (stacker.__class__.__name__ == stacker2.__class__.__name__) & (stacker != stacker2):
                    return False
        # But if we got this far, everything matches.
        return True

    def _find_compatible_lists(self):
        """Find sets of compatible metricBundles from the currentBundleDict."""
        # CompatibleLists stores a list of lists;
        #  each (nested) list contains the bundleDict _keys_
        #  of a compatible set of metricBundles.
        compatible_lists = []
        for k, b in self.current_bundle_dict.items():
            found_compatible = False
            for compatible_list in compatible_lists:
                comparison_metric_bundle_key = compatible_list[0]
                compatible = self._check_compatible(self.bundle_dict[comparison_metric_bundle_key], b)
                if compatible:
                    # Must compare all metricBundles in each subset
                    # (if they are a potential match),
                    #  as the stackers could be different
                    #  (and one could be incompatible,
                    #  not necessarily the first)
                    for comparison_metric_bundle_key in compatible_list[1:]:
                        compatible = self._check_compatible(self.bundle_dict[comparison_metric_bundle_key], b)
                        if not compatible:
                            # If we find one which is not compatible,
                            # stop and go on to the
                            # next subset list.
                            break
                    # Otherwise, we reached the end of the subset
                    # and they were all compatible.
                    found_compatible = True
                    compatible_list.append(k)
            if not found_compatible:
                # Didn't find a pre-existing compatible set; make a new one.
                compatible_lists.append(
                    [
                        k,
                    ]
                )
        self.compatible_lists = compatible_lists

    def run_all(self, clear_memory=False, plot_now=False, plot_kwargs=None):
        """Calculates metric values, then runs reduce functions and summary
        statistics for all MetricBundles, over all constraints.

        Parameters
        ----------
        clear_memory : `bool`, optional
            If True, deletes metric values from memory after running
            each constraint group.
        plot_now : `bool`, optional
            If True, plots the metric values immediately after calculation.
        plot_kwargs : `bool`, optional
            kwargs to pass to plotCurrent.
        """
        for constraint in self.constraints:
            # Set the 'currentBundleDict' which is a dictionary of the
            # metricBundles which match this constraint.
            self.run_current(
                constraint,
                clear_memory=clear_memory,
                plot_now=plot_now,
                plot_kwargs=plot_kwargs,
            )

    def set_current(self, constraint):
        """Utility to set the currentBundleDict
        (i.e. a set of metricBundles with the same SQL constraint).

        Parameters
        ----------
        constraint : `str`
            The subset of MetricBundles with metricBundle.constraint ==
            constraint will be included in a subset identified as the
            currentBundleDict.
            These are the active metrics to be calculated and plotted, etc.

        Notes
        -----
        This is useful, for the context of running only a specific set
        of metric bundles so that the user can provide `sim_data` directly.
        """
        if constraint is None:
            constraint = ""
        self.current_bundle_dict = {}
        for k, b in self.bundle_dict.items():
            if b.constraint == constraint:
                self.current_bundle_dict[k] = b
        # Build list of all the columns needed from the database.
        self.db_cols = []
        for b in self.current_bundle_dict.values():
            self.db_cols.extend(b.db_cols)
        self.db_cols = list(set(self.db_cols))

    def run_current(
        self,
        constraint,
        sim_data=None,
        clear_memory=False,
        plot_now=False,
        plot_kwargs=None,
    ):
        """Calculates the metric values, then runs reduce functions and
        summary statistics for metrics in the current set only
        (see self.setCurrent).

        Parameters
        ----------
        constraint : `str`
           constraint to use to set the currently active metrics
        sim_data : `np.ndarray`, opt
           If simData is not None, then this numpy structured array is used
           instead of querying data from the dbObj.
        clear_memory : `bool`, opt
           If True, metric values are deleted from memory after they are
           calculated (and saved to disk).
        plot_now : `bool`, opt
           Plot immediately after calculating metric values
           (instead of the usual procedure, which is to plot after metric
           values are calculated for all constraints).
        plot_kwargs : kwargs, opt
           Plotting kwargs to pass to plotCurrent.

        Notes
        -----
        This is useful, for the context of running only a specific set
        of metric bundles so that the user can provide `sim_data` directly.
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
                for b in self.current_bundle_dict.values():
                    metrics_skipped.append(
                        "%s : %s : %s" % (b.metric.name, b.info_label, b.slicer.slicer_name)
                    )
                warnings.warn(" This means skipping metrics %s" % metrics_skipped)
                return
            except ValueError:
                warnings.warn(
                    "One or more of the columns requested from the database was not available."
                    + " Skipping constraint %s" % constraint
                )
                metrics_skipped = []
                for b in self.current_bundle_dict.values():
                    metrics_skipped.append(
                        "%s : %s : %s" % (b.metric.name, b.info_label, b.slicer.slicer_name)
                    )
                warnings.warn(" This means skipping metrics %s" % metrics_skipped)
                return

        # Find compatible subsets of the MetricBundle dictionary,
        # which can be run/metrics calculated/ together.
        self._find_compatible_lists()

        for compatible_list in self.compatible_lists:
            if self.verbose:
                print("Running: ", compatible_list)
            self._run_compatible(compatible_list)
            if self.verbose:
                print("Completed metric generation.")
            for key in compatible_list:
                self.has_run[key] = True
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
            for b in self.current_bundle_dict.values():
                b.metric_values = None
            if self.verbose:
                print("Deleted metric_values from memory.")

    def get_data(self, constraint):
        """Query the data from the database.

        The currently bundleDict should generally be set
        before calling getData (using setCurrent).

        Parameters
        ----------
        constraint : `str`
           The constraint for the currently active set of MetricBundles.
        """
        if self.verbose:
            if constraint == "":
                print("Querying table %s with no constraint for columns %s." % (self.db_table, self.db_cols))
            else:
                print(
                    "Querying table %s with constraint %s for columns %s"
                    % (self.db_table, constraint, self.db_cols)
                )
        # Note that we do NOT run the stackers at this point
        # (this must be done in each 'compatible' group).
        self.sim_data = utils.get_sim_data(
            self.db_obj,
            constraint,
            self.db_cols,
            table_name=self.db_table,
        )

        if self.verbose:
            print("Found %i visits" % (self.sim_data.size))

    def _run_compatible(self, compatible_list):
        """Runs a set of 'compatible' metric_bundles in the MetricBundleGroup
        dictionary identified by 'compatible_list' keys.

        A compatible list of MetricBundles is a subset of the
        currentBundleDict.
        The currentBundleDict == set of MetricBundles with the same constraint.
        The compatibleBundles == set of MetricBundles with the same constraint,
        AND the same slicer, the same maps applied to the slicer,
        and stackers which do not clobber each other's data.

        This is where the work of calculating the metric values is done.
        """

        if len(self.sim_data) == 0:
            return

        # Grab a dictionary representation of this subset of the dictionary,
        # for easier iteration.
        b_dict = {key: self.current_bundle_dict.get(key) for key in compatible_list}

        # Find the unique stackers and maps.
        # These are already "compatible" (as id'd by compatible_list).
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
        for stacker in uniq_stackers:
            # Note that stackers will clobber previously existing columns
            self.sim_data = stacker.run(self.sim_data, override=True)

        # Pull out one of the slicers to use as our 'slicer'.
        # This will be forced back into all of the metricBundles
        # at the end (so that they track the same info_label such as the
        # slice_points, in case the same actual object wasn't used).
        slicer = list(b_dict.values())[0].slicer
        slicer.setup_slicer(self.sim_data, maps=uniq_maps)
        # Copy the slicer (after setup) back into the individual metricBundles.
        if slicer.slicer_name != "HealpixSlicer" or slicer.slicer_name != "UniSlicer":
            for b in b_dict.values():
                b.slicer = slicer

        # Set up (masked) arrays to store metric data in each metricBundle.
        for b in b_dict.values():
            b._setup_metric_values()

        # Set up an ordered dictionary to be the cache if needed:
        # (Currently using OrderedDict,
        # it might be faster to use 2 regular Dicts instead)
        if slicer.cache_size > 0:
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
            i = slice_i["slice_point"]["sid"]
            slicedata = self.sim_data[slice_i["idxs"]]
            if len(slicedata) == 0:
                # No data at this slice_point. Mask data values.
                for b in b_dict.values():
                    b.metric_values.mask[i] = True
            else:
                # There is data! Should we use our data cache?
                if cache:
                    # Make the data idxs hashable.
                    cache_key = frozenset(slice_i["idxs"])
                    # If key exists, set flag to use it, otherwise add it
                    if cache_key in cache_dict:
                        use_cache = True
                        cache_val = cache_dict[cache_key]
                        # Move this value to the end of the OrderedDict
                        del cache_dict[cache_key]
                        cache_dict[cache_key] = cache_val
                    else:
                        cache_dict[cache_key] = i
                        use_cache = False
                    for b in b_dict.values():
                        if use_cache:
                            b.metric_values.data[i] = b.metric_values.data[cache_dict[cache_key]]
                        else:
                            b.metric_values.data[i] = b.metric.run(
                                slicedata, slice_point=slice_i["slice_point"]
                            )
                    # If we are above the cache size,
                    # drop the oldest element from the cache dict.
                    if len(cache_dict) > slicer.cache_size:
                        del cache_dict[list(cache_dict.keys())[0]]

                # Not using memoize, just calculate things normally
                else:
                    for b in b_dict.values():
                        try:
                            b.metric_values.data[i] = b.metric.run(
                                slicedata, slice_point=slice_i["slice_point"]
                            )
                        except BaseException as e:
                            print(f"Failed at slice_point {slice_i}, sid {i}")
                            raise e
        # Mask data where metrics could not be computed
        # (according to metric bad value).
        for b in b_dict.values():
            if b.metric_values.dtype.name == "object":
                for ind, val in enumerate(b.metric_values.data):
                    if val is b.metric.badval:
                        b.metric_values.mask[ind] = True
            else:
                # For some reason, this doesn't work for dtype=object arrays.
                b.metric_values.mask = np.where(
                    b.metric_values.data == b.metric.badval, True, b.metric_values.mask
                )

        # Save data to disk as we go (just failsafe).
        if self.save_early:
            for b in b_dict.values():
                b.write(out_dir=self.out_dir, results_db=self.results_db)
        else:
            # Just write the metric run information to the results_db
            for b in b_dict.values():
                b.write_db(results_db=self.results_db)

    def reduce_all(self, update_summaries=True):
        """Run the reduce methods for all metrics in bundleDict.

        Running this method, for all MetricBundles at once,
        assumes that clearMemory was False.

        Parameters
        ----------
        update_summaries : `bool`, optional
            If True, summary metrics are removed from the top-level
            (non-reduced) MetricBundle. Usually this should be True,
            as summary metrics are generally intended to run on the simpler
            data produced by reduce metrics.
        """
        for constraint in self.constraints:
            self.set_current(constraint)
            self.reduce_current(update_summaries=update_summaries)

    def reduce_current(self, update_summaries=True):
        """Run all reduce functions for the metricbundle in the
        currently active set of MetricBundles.

        Parameters
        ----------
        update_summaries : `bool`, optional
            If True, summary metrics are removed from the top-level
            (non-reduced) MetricBundle. Usually this should be True,
            as summary metrics are generally intended to run on the simpler
            data produced by reduce metrics.
        """
        # Create a temporary dictionary to hold the reduced metricbundles.
        reduce_bundle_dict = {}
        for b in self.current_bundle_dict.values():
            # If there are no reduce functions associated with the metric,
            # skip this metricBundle.
            if len(b.metric.reduce_funcs) > 0:
                # Apply reduce functions, creating a new metricBundle in
                # the process (new metric values).
                for r in b.metric.reduce_funcs:
                    newmetricbundle = b.reduce_metric(b.metric.reduce_funcs[r], reduce_func_name=r)
                    # Add the new metricBundle to our metricBundleGroup
                    # dictionary.
                    name = newmetricbundle.metric.name
                    if name in self.bundle_dict:
                        name = newmetricbundle.file_root
                    reduce_bundle_dict[name] = newmetricbundle
                    if self.save_early:
                        newmetricbundle.write(out_dir=self.out_dir, results_db=self.results_db)
                    else:
                        newmetricbundle.write_db(results_db=self.results_db)
                # Remove summaryMetrics from top level metricbundle if desired.
                if update_summaries:
                    b.summary_metrics = []
        # Add the new metricBundles to the MetricBundleGroup dictionary.
        self.bundle_dict.update(reduce_bundle_dict)
        # And add to to the currentBundleDict too, so we run as part
        # of 'summaryCurrent'.
        self.current_bundle_dict.update(reduce_bundle_dict)

    def summary_all(self):
        """Run the summary statistics for all metrics in bundleDict.

        Calculating all summary statistics, for all MetricBundles, at this
        point assumes that clearMemory was False.
        """
        for constraint in self.constraints:
            self.set_current(constraint)
            self.summary_current()

    def summary_current(self):
        """Run summary statistics on all the metricBundles in the
        currently active set of MetricBundles.
        """
        for b in self.current_bundle_dict.values():
            b.compute_summary_stats(self.results_db)

    def plot_all(
        self,
        save_figs=True,
        outfile_suffix=None,
        fig_format="pdf",
        dpi=600,
        trim_whitespace=True,
        thumbnail=True,
        closefigs=True,
    ):
        """Generate all the plots for all the metricBundles in bundleDict.

        Generating all plots, for all MetricBundles,
        At this point, assumes that clearMemory was False.

        Parameters
        ----------
        savefig : `bool`, optional
            If True, save figures to disk, to self.out_dir directory.
        outfile_suffix : `str`, optional
            Append outfile_suffix to the end of every plot file generated.
            Useful for generating sequential series of images for movies.
        fig_format : `str`, optional
            Matplotlib figure format to use to save to disk.
        dpi : `int`, optional
            DPI for matplotlib figure.
        trim_whitespace : `bool`, optional
            If True, trim additional whitespace from final figures.
        thumbnail : `bool`, optional
            If True, save a small thumbnail jpg version of the output file
            to disk as well. This is useful for show_maf web pages.
        closefigs : `bool`, optional
            Close the matplotlib figures after they are saved to disk.
            If many figures are generated, closing the figures saves
            significant memory.
        """
        for constraint in self.constraints:
            if self.verbose:
                print('Plotting figures with "%s" constraint now.' % (constraint))

            self.set_current(constraint)
            self.plot_current(
                savefig=save_figs,
                outfile_suffix=outfile_suffix,
                fig_format=fig_format,
                dpi=dpi,
                trim_whitespace=trim_whitespace,
                thumbnail=thumbnail,
                closefigs=closefigs,
            )

    def plot_current(
        self,
        savefig=True,
        outfile_suffix=None,
        fig_format="pdf",
        dpi=600,
        trim_whitespace=True,
        thumbnail=True,
        closefigs=True,
    ):
        """Generate the plots for the currently active set of MetricBundles.

        Parameters
        ----------
        savefig : `bool`, optional
            If True, save figures to disk, to self.out_dir directory.
        outfile_suffix : `str`, optional
            Append outfile_suffix to the end of every plot file generated.
            Useful for generating sequential series of images for movies.
        fig_format : `str`, optional
            Matplotlib figure format to use to save to disk.
        dpi : `int`, optional
            DPI for matplotlib figure.
        trim_whitespace : `bool`, optional
            If True, trim additional whitespace from final figures.
        thumbnail : `bool`, optional
            If True, save a small thumbnail jpg version of the output file
            to disk as well. This is useful for show_maf web pages.
        closefigs : `bool`, optional
            Close the matplotlib figures after they are saved to disk.
            If many figures are generated, closing the figures saves
            significant memory.
        """
        plot_handler = PlotHandler(
            out_dir=self.out_dir,
            results_db=self.results_db,
            savefig=savefig,
            fig_format=fig_format,
            dpi=dpi,
            trim_whitespace=trim_whitespace,
            thumbnail=thumbnail,
        )

        for b in self.current_bundle_dict.values():
            try:
                b.plot(
                    plot_handler=plot_handler,
                    outfile_suffix=outfile_suffix,
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

        Saving all MetricBundles to disk at this point assumes that
        clearMemory was False.
        """
        for constraint in self.constraints:
            self.set_current(constraint)
            self.write_current()

    def write_current(self):
        """Save all the MetricBundles in the currently active set to disk."""
        if self.verbose:
            if self.save_early:
                print("Re-saving metric bundles.")
            else:
                print("Saving metric bundles.")
        for b in self.current_bundle_dict.values():
            b.write(out_dir=self.out_dir, results_db=self.results_db)

    def read_all(self):
        """Attempt to read all MetricBundles from disk.

        You must set the metrics/slicer/constraint/run_name for a metricBundle
        appropriately, so that the file_root is correct.
        """
        reduce_bundle_dict = {}
        remove_bundles = []
        for b in self.bundle_dict:
            bundle = self.bundle_dict[b]
            filename = os.path.join(self.out_dir, bundle.file_root + ".npz")
            try:
                # Create a temporary metricBundle to read the data into.
                # We are not using "b" directly, as we are trying NOT
                # to override the plot_dict or display_dict.
                tmp_bundle = create_empty_metric_bundle()
                tmp_bundle.read(filename)
                # Copy the tmp_bundle metric_values into bundle.
                bundle.metric_values = tmp_bundle.metric_values
                # And copy the slicer into b, to get slice_points.
                bundle.slicer = tmp_bundle.slicer
                # Copy the summary stats though.
                bundle.summary_values = tmp_bundle.summary_values
                if self.verbose:
                    print("Read %s from disk." % (bundle.file_root))
            except IOError:
                warnings.warn("Warning: file %s not found, bundle not restored." % filename)
                remove_bundles.append(b)

            # Look to see if this is a complex metric,
            # with associated 'reduce' functions, and read those in too.
            if len(bundle.metric.reduce_funcs) > 0:
                orig_metric_name = bundle.metric.name
                for reduce_func in bundle.metric.reduce_funcs.values():
                    reduce_name = orig_metric_name + "_" + reduce_func.__name__.replace("reduce", "")
                    # Borrow the fileRoot in b (we'll reset it appropriately)
                    bundle.metric.name = reduce_name
                    bundle._build_file_root()
                    filename = os.path.join(self.out_dir, bundle.file_root + ".npz")
                    tmp_bundle = create_empty_metric_bundle()
                    try:
                        tmp_bundle.read(filename)
                        # This won't necessarily recreate the plot_dict and
                        # display_dict exactly as they would have been made
                        # if you calculated the reduce metric from scratch.
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
                        newmetric_bundle.metric_values = ma.copy(tmp_bundle.metric_values)
                        newmetric_bundle.summary_values = tmp_bundle.summary_values
                        # Add the new metricBundle to metricBundleGroup dict.
                        name = newmetric_bundle.metric.name
                        if name in self.bundle_dict:
                            name = newmetric_bundle.file_root
                        reduce_bundle_dict[name] = newmetric_bundle
                        if self.verbose:
                            print("Read %s from disk." % (newmetric_bundle.file_root))
                    except IOError:
                        warnings.warn(
                            'Warning: file %s not found, bundle not restored ("reduce" metric).' % filename
                        )

                    # Remove summaryMetrics from top level metricbundle.
                    bundle.summary_metrics = []
                    # Update parent MetricBundle name.
                    bundle.metric.name = orig_metric_name
                    bundle._build_file_root()

        # Add the reduce bundles into the bundleDict.
        self.bundle_dict.update(reduce_bundle_dict)
        # And remove the bundles which were not found on disk,
        # so we don't try to make (blank) plots.
        for b in remove_bundles:
            del self.bundle_dict[b]
