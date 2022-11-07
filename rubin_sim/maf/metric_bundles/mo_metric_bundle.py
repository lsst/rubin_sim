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

from .metric_bundle import MetricBundle

__all__ = [
    "MoMetricBundle",
    "MoMetricBundleGroup",
    "create_empty_mo_metric_bundle",
    "make_completeness_bundle",
]


def create_empty_mo_metric_bundle():
    """Create an empty metric bundle.

    Returns
    -------
    ~rubin_sim.maf.metricBundles.MoMetricBundle
        An empty metric bundle, configured with just the :class:`BaseMetric` and :class:`BaseSlicer`.
    """
    return MoMetricBundle(BaseMoMetric(), MoObjSlicer(), None)


def make_completeness_bundle(bundle, completeness_metric, h_mark=None, results_db=None):
    """
    Make a mock metric bundle from a bundle which had MoCompleteness or MoCumulativeCompleteness summary
    metrics run. This lets us use the plotHandler + plots.MetricVsH to generate plots.
    Will also work with completeness metric run in order to calculate fraction of the population,
    or with MoCompletenessAtTime metric.

    Parameters
    ----------
    bundle : ~rubin_sim.maf.metricBundles.MetricBundle
        The metric bundle with a completeness summary statistic.
    completeness_metric : ~rubin_sim.maf.metric
        The summary (completeness) metric to run on the bundle.
    h_mark : float, optional
        The Hmark value to add to the plotting dictionary of the new mock bundle. Default None.
    results_db : ~rubin_sim.maf.db.ResultsDb, optional
        The resultsDb in which to record the summary statistic value at Hmark. Default None.

    Returns
    -------
    ~rubin_sim.maf.metricBundles.MoMetricBundle
    """
    bundle.set_summary_metrics(completeness_metric)
    # This step adds summary values at each point to the original metric - we use this to populate
    # the completeness values in the next step. However, we may not want them to go into the resultsDb.
    bundle.compute_summary_stats(results_db)
    summary_name = completeness_metric.name
    # Make up the bundle, including the metric values.
    completeness = ma.MaskedArray(
        data=bundle.summary_values[summary_name]["value"],
        mask=np.zeros(len(bundle.summary_values[summary_name]["value"])),
        fill_value=0,
    )
    mb = MoMetricBundle(
        completeness_metric,
        bundle.slicer,
        constraint=bundle.constraint,
        run_name=bundle.run_name,
        info_label=bundle.info_label,
        display_dict=bundle.displayDict,
    )
    plot_dict = {}
    plot_dict.update(bundle.plotDict)
    plot_dict["label"] = bundle.info_label
    if "Completeness" not in summary_name:
        plot_dict["label"] += " " + summary_name.replace("FractionPop_", "")
    mb.metric_values = completeness.reshape(1, len(completeness))
    if h_mark is not None:
        metric = ValueAtHMetric(Hmark=h_mark)
        mb.set_summary_metrics(metric)
        mb.compute_summary_stats(results_db)
        val = mb.summary_values["Value At H=%.1f" % h_mark]
        if val is None:
            val = 0
        if summary_name.startswith("Cumulative"):
            plot_dict["label"] += ": @ H(<=%.1f) = %.1f%s" % (h_mark, val * 100, "%")
        else:
            plot_dict["label"] += ": @ H(=%.1f) = %.1f%s" % (h_mark, val * 100, "%")
    mb.set_plot_dict(plot_dict)
    return mb


class MoMetricBundle(MetricBundle):
    def __init__(
        self,
        metric,
        slicer,
        constraint=None,
        stacker_list=None,
        run_name="run name",
        info_label=None,
        file_root=None,
        plot_dict=None,
        plot_funcs=None,
        display_dict=None,
        child_metrics=None,
        summary_metrics=None,
    ):
        """
        Instantiate moving object metric bundle, save metric/slicer/constraint, etc.
        """
        self.metric = metric
        self.slicer = slicer
        if constraint == "":
            constraint = None
        self.constraint = constraint
        # Set the stackerlist.
        if stacker_list is not None:
            if isinstance(stacker_list, BaseMoStacker):
                self.stacker_list = [
                    stacker_list,
                ]
            else:
                self.stacker_list = []
                for s in stacker_list:
                    if not isinstance(s, BaseMoStacker):
                        raise ValueError(
                            "stackerList must only contain "
                            "rubin_sim.maf.stackers.BaseMoStacker type objs"
                        )
                    self.stacker_list.append(s)
        else:
            self.stacker_list = []
        # Add the basic 'visibility/mag' stacker if not present.
        mag_stacker_found = False
        for s in self.stacker_list:
            if s.__class__.__name__ == "MoMagStacker":
                mag_stacker_found = True
                break
        if not mag_stacker_found:
            self.stacker_list.append(MoMagStacker())
        # Set a mapsList just for compatibility with generic MetricBundle.
        self.maps_list = []
        # Add the summary stats, if applicable.
        self.set_summary_metrics(summary_metrics)
        # Set the provenance/info_label.
        self.run_name = run_name
        self._build_metadata(info_label)
        # Build the output filename root if not provided.
        if file_root is not None:
            self.file_root = file_root
        else:
            self._build_file_root()
        # Set the plotting classes/functions.
        self.set_plot_funcs(plot_funcs)
        # Set the plot_dict and displayDicts.
        self.plot_dict = {"units": "@H"}
        self.set_plot_dict(plot_dict)
        # Update/set display_dict.
        self.display_dict = {}
        self.set_display_dict(display_dict)
        # Set the list of child metrics.
        self.set_child_bundles(child_metrics)
        # This is where we store the metric values and summary stats.
        self.metric_values = None
        self.summary_values = None

    def _reset_metric_bundle(self):
        """Reset all properties of MetricBundle."""
        self.metric = None
        self.slicer = None
        self.constraint = None
        self.stacker_list = [MoMagStacker()]
        self.maps_list = []
        self.summary_metrics = []
        self.plot_funcs = []
        self.run_name = "opsim"
        self.info_label = ""
        self.db_cols = None
        self.file_root = None
        self.plot_dict = {}
        self.display_dict = {}
        self.child_metrics = None
        self.metric_values = None
        self.summary_values = None

    def _build_metadata(self, info_label):
        """If no info_label is provided, auto-generate it from the obs_file + constraint."""
        if info_label is None:
            try:
                self.info_label = self.slicer.obsfile.replace(".txt", "").replace(
                    ".dat", ""
                )
                self.info_label = self.info_label.replace("_obs", "").replace(
                    "_allObs", ""
                )
            except AttributeError:
                self.info_label = "noObs"
            # And modify by constraint.
            if self.constraint is not None:
                self.info_label += " " + self.constraint
        else:
            self.info_label = info_label

    def _find_req_cols(self):
        # Doesn't quite work the same way yet. No stacker list, for example.
        raise NotImplementedError

    def set_child_bundles(self, child_metrics=None):
        """
        Identify any child metrics to be run on this (parent) bundle.
        and create the new metric bundles that will hold the child values, linking to this bundle.
        Remove the summaryMetrics from self afterwards.
        """
        self.child_bundles = {}
        if child_metrics is None:
            child_metrics = self.metric.child_metrics
        for c_name, cMetric in child_metrics.items():
            c_bundle = MoMetricBundle(
                metric=cMetric,
                slicer=self.slicer,
                constraint=self.constraint,
                stacker_list=self.stacker_list,
                run_name=self.run_name,
                info_label=self.info_label,
                plot_dict=self.plot_dict,
                plot_funcs=self.plot_funcs,
                display_dict=self.display_dict,
                summary_metrics=self.summary_metrics,
            )
            self.child_bundles[c_name] = c_bundle
        if len(child_metrics) > 0:
            self.summary_metrics = []

    def compute_summary_stats(self, results_db=None):
        """
        Compute summary statistics on metric_values, using summaryMetrics, for self and child bundles.
        """
        if self.summary_values is None:
            self.summary_values = {}
        if self.summary_metrics is not None:
            # Build array of metric values, to use for (most) summary statistics.
            for m in self.summary_metrics:
                summary_name = m.name
                summary_val = m.run(self.metric_values, self.slicer.slicePoints["H"])
                self.summary_values[summary_name] = summary_val
                # Add summary metric info to results database, if applicable.
                if results_db:
                    metric_id = results_db.update_metric(
                        self.metric.name,
                        self.slicer.slicerName,
                        self.run_name,
                        self.constraint,
                        self.info_label,
                        None,
                    )
                    results_db.update_summary_stat(
                        metric_id, summary_name=summary_name, summary_value=summary_val
                    )

    def reduce_metric(self, reduce_func, reduce_plot_dict=None, reduce_display_dict=None):
        raise NotImplementedError


class MoMetricBundleGroup(object):
    def __init__(self, bundle_dict, out_dir=".", results_db=None, verbose=True):
        self.verbose = verbose
        self.bundle_dict = bundle_dict
        self.out_dir = out_dir
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)
        self.results_db = results_db

        self.slicer = list(self.bundle_dict.values())[0].slicer
        for b in self.bundle_dict.values():
            if b.slicer != self.slicer:
                raise ValueError(
                    "Currently, the slicers for the MoMetricBundleGroup must be equal,"
                    " using the same observations and Hvals."
                )
        self.constraints = list(set([b.constraint for b in bundle_dict.values()]))

    def _check_compatible(self, metric_bundle1, metric_bundle2):
        """Check if two MetricBundles are "compatible".
        Compatible indicates that the constraints, the slicers, and the maps are the same, and
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

    def _find_compatible(self, test_keys):
        """ "Private utility to find which metricBundles with keys in the list 'test_keys' can be calculated
        at the same time -- having the same slicer, constraint, maps, and compatible stackers.

        Parameters
        -----------
        test_keys : list
            List of the dictionary keys (of self.bundle_dict) to test for compatibilility.
        Returns
        --------
        list of lists
            Returns test_keys, split into separate lists of compatible metricBundles.
        """
        compatible_lists = []
        for k in test_keys:
            try:
                b = self.bundle_dict[k]
            except KeyError:
                warnings.warn(
                    "Received %s in testkeys, but this is not present in self.bundle_dict."
                    "Will continue, but this is not expected."
                )
                continue
            found_compatible = False
            checked_all = False
            while not (found_compatible) and not (checked_all):
                # Go through the existing lists in compatible_lists, to see if this metricBundle matches.
                for compatible_list in compatible_lists:
                    # Compare to all the metricBundles in this subset, to check all stackers are compatible.
                    found_compatible = True
                    for comparison_key in compatible_list:
                        compatible = self._check_compatible(
                            self.bundle_dict[comparison_key], b
                        )
                        if not compatible:
                            # Found a metricBundle which is not compatible, so stop and go onto the next subset.
                            found_compatible = False
                            break
                checked_all = True
            if found_compatible:
                compatible_list.append(k)
            else:
                compatible_lists.append(
                    [
                        k,
                    ]
                )
        return compatible_lists

    def run_constraint(self, constraint):
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
        keys_matching_constraint = []
        for k, b in self.bundle_dict.items():
            if b.constraint == constraint:
                keys_matching_constraint.append(k)
        if len(keys_matching_constraint) == 0:
            return
        # Identify the observations which are relevant for this constraint.
        # This sets slicer.obs (valid for all H values).
        self.slicer.subsetObs(constraint)
        # Identify the sets of these metricBundles can be run at the same time (also have the same stackers).
        compatible_lists = self._find_compatible(keys_matching_constraint)

        # And now run each of those subsets of compatible metricBundles.
        for compatible_list in compatible_lists:
            self._run_compatible(compatible_list)

    def _run_compatible(self, compatible_list):
        """Calculate the metric values for set of (parent and child) bundles, as well as the summary stats,
        and write to disk.

        Parameters
        -----------
        compatible_list : list
            List of dictionary keys, of the metricBundles which can be calculated together.
            This means they are 'compatible' and have the same slicer, constraint, and non-conflicting
            mappers and stackers.
        """
        if self.verbose:
            print("Running metrics %s" % compatible_list)

        b_dict = (
            self.bundle_dict
        )  #  {key: self.bundle_dict.get(key) for key in compatible_list}

        # Find the unique stackers and maps. These are already "compatible" (as id'd by compatible_list).
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

        if len(uniq_maps) > 0:
            print(
                "Got some maps .. that was unexpected at the moment. Can't use them here yet."
            )

        # Set up all of the metric values, including for the child bundles.
        for k in compatible_list:
            b = self.bundle_dict[k]
            b._setup_metric_values()
            for cb in b.child_bundles.values():
                cb._setup_metric_values()
        # Calculate the metric values.
        for i, slicePoint in enumerate(self.slicer):
            sso_obs = slicePoint["obs"]
            for j, Hval in enumerate(slicePoint["Hvals"]):
                # Run stackers to add extra columns (that depend on Hval)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for s in uniq_stackers:
                        sso_obs = s.run(sso_obs, slicePoint["orbit"]["H"], Hval)
                # Run all the parent metrics.
                for k in compatible_list:
                    b = self.bundle_dict[k]
                    # Mask the parent metric (and then child metrics) if there was no data.
                    if len(sso_obs) == 0:
                        b.metricValues.mask[i][j] = True
                        for cb in list(b.child_bundles.values()):
                            cb.metricValues.mask[i][j] = True
                    # Otherwise, calculate the metric value for the parent, and then child.
                    else:
                        # Calculate for the parent.
                        m_val = b.metric.run(sso_obs, slicePoint["orbit"], Hval)
                        # Mask if the parent metric returned a bad value.
                        if m_val == b.metric.badval:
                            b.metricValues.mask[i][j] = True
                            for cb in b.child_bundles.values():
                                cb.metricValues.mask[i][j] = True
                        # Otherwise, set the parent value and calculate the child metric values as well.
                        else:
                            b.metricValues.data[i][j] = m_val
                            for cb in b.child_bundles.values():
                                child_val = cb.metric.run(
                                    sso_obs, slicePoint["orbit"], Hval, m_val
                                )
                                if child_val == cb.metric.badval:
                                    cb.metricValues.mask[i][j] = True
                                else:
                                    cb.metricValues.data[i][j] = child_val
        for k in compatible_list:
            b = self.bundle_dict[k]
            b.compute_summary_stats(self.results_db)
            for c_b in b.child_bundles.values():
                c_b.compute_summary_stats(self.results_db)
                # Write to disk.
                c_b.write(out_dir=self.out_dir, results_db=self.results_db)
            # Write to disk.
            b.write(out_dir=self.out_dir, results_db=self.results_db)

    def run_all(self):
        """
        Run all constraints and metrics for these moMetricBundles.
        """
        for constraint in self.constraints:
            self.run_constraint(constraint)
        if self.verbose:
            print("Calculated and saved all metrics.")

    def plot_all(
        self,
        savefig=True,
        outfile_suffix=None,
        figformat="pdf",
        dpi=600,
        thumbnail=True,
        closefigs=True,
    ):
        """
        Make a few generically desired plots. This needs more flexibility in the future.
        """
        plot_handler = PlotHandler(
            outDir=self.out_dir,
            resultsDb=self.results_db,
            savefig=savefig,
            figformat=figformat,
            dpi=dpi,
            thumbnail=thumbnail,
        )
        for b in self.bundle_dict.values():
            try:
                b.plot(
                    plot_handler=plot_handler,
                    outfile_suffix=outfile_suffix,
                    savefig=savefig,
                )
            except ValueError as ve:
                message = "Plotting failed for metricBundle %s." % (b.file_root)
                message += " Error message: %s" % (ve.message)
                warnings.warn(message)
            if closefigs:
                plt.close("all")
        if self.verbose:
            print("Plotting all metrics.")
