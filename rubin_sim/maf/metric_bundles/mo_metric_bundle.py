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
    summaryName = completeness_metric.name
    # Make up the bundle, including the metric values.
    completeness = ma.MaskedArray(
        data=bundle.summary_values[summaryName]["value"],
        mask=np.zeros(len(bundle.summary_values[summaryName]["value"])),
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
    plotDict = {}
    plotDict.update(bundle.plotDict)
    plotDict["label"] = bundle.info_label
    if "Completeness" not in summaryName:
        plotDict["label"] += " " + summaryName.replace("FractionPop_", "")
    mb.metricValues = completeness.reshape(1, len(completeness))
    if h_mark is not None:
        metric = ValueAtHMetric(Hmark=h_mark)
        mb.set_summary_metrics(metric)
        mb.compute_summary_stats(results_db)
        val = mb.summaryValues["Value At H=%.1f" % h_mark]
        if val is None:
            val = 0
        if summaryName.startswith("Cumulative"):
            plotDict["label"] += ": @ H(<=%.1f) = %.1f%s" % (h_mark, val * 100, "%")
        else:
            plotDict["label"] += ": @ H(=%.1f) = %.1f%s" % (h_mark, val * 100, "%")
    mb.set_plot_dict(plotDict)
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
                self.stackerList = [
                    stacker_list,
                ]
            else:
                self.stackerList = []
                for s in stacker_list:
                    if not isinstance(s, BaseMoStacker):
                        raise ValueError(
                            "stackerList must only contain "
                            "rubin_sim.maf.stackers.BaseMoStacker type objs"
                        )
                    self.stackerList.append(s)
        else:
            self.stackerList = []
        # Add the basic 'visibility/mag' stacker if not present.
        mag_stacker_found = False
        for s in self.stackerList:
            if s.__class__.__name__ == "MoMagStacker":
                mag_stacker_found = True
                break
        if not mag_stacker_found:
            self.stackerList.append(MoMagStacker())
        # Set a mapsList just for compatibility with generic MetricBundle.
        self.mapsList = []
        # Add the summary stats, if applicable.
        self.set_summary_metrics(summary_metrics)
        # Set the provenance/info_label.
        self.runName = run_name
        self._build_metadata(info_label)
        # Build the output filename root if not provided.
        if file_root is not None:
            self.fileRoot = file_root
        else:
            self._build_file_root()
        # Set the plotting classes/functions.
        self.set_plot_funcs(plot_funcs)
        # Set the plotDict and displayDicts.
        self.plotDict = {"units": "@H"}
        self.set_plot_dict(plot_dict)
        # Update/set displayDict.
        self.displayDict = {}
        self.set_display_dict(display_dict)
        # Set the list of child metrics.
        self.set_child_bundles(child_metrics)
        # This is where we store the metric values and summary stats.
        self.metricValues = None
        self.summaryValues = None

    def _reset_metric_bundle(self):
        """Reset all properties of MetricBundle."""
        self.metric = None
        self.slicer = None
        self.constraint = None
        self.stackerList = [MoMagStacker()]
        self.mapsList = []
        self.summary_metrics = []
        self.plotFuncs = []
        self.runName = "opsim"
        self.info_label = ""
        self.dbCols = None
        self.fileRoot = None
        self.plotDict = {}
        self.displayDict = {}
        self.childMetrics = None
        self.metricValues = None
        self.summaryValues = None

    def _build_metadata(self, info_label):
        """If no info_label is provided, auto-generate it from the obsFile + constraint."""
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
            child_metrics = self.metric.childMetrics
        for cName, cMetric in child_metrics.items():
            c_bundle = MoMetricBundle(
                metric=cMetric,
                slicer=self.slicer,
                constraint=self.constraint,
                stacker_list=self.stackerList,
                run_name=self.runName,
                info_label=self.info_label,
                plot_dict=self.plotDict,
                plot_funcs=self.plotFuncs,
                display_dict=self.displayDict,
                summary_metrics=self.summary_metrics,
            )
            self.child_bundles[cName] = c_bundle
        if len(child_metrics) > 0:
            self.summary_metrics = []

    def compute_summary_stats(self, results_db=None):
        """
        Compute summary statistics on metricValues, using summaryMetrics, for self and child bundles.
        """
        if self.summaryValues is None:
            self.summaryValues = {}
        if self.summary_metrics is not None:
            # Build array of metric values, to use for (most) summary statistics.
            for m in self.summary_metrics:
                summary_name = m.name
                summary_val = m.run(self.metricValues, self.slicer.slicePoints["H"])
                self.summaryValues[summary_name] = summary_val
                # Add summary metric info to results database, if applicable.
                if results_db:
                    metric_id = results_db.update_metric(
                        self.metric.name,
                        self.slicer.slicerName,
                        self.runName,
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
    def __init__(self, bundle_dict, out_dir=".", resultsDb=None, verbose=True):
        self.verbose = verbose
        self.bundleDict = bundle_dict
        self.outDir = out_dir
        if not os.path.isdir(self.outDir):
            os.makedirs(self.outDir)
        self.resultsDb = resultsDb

        self.slicer = list(self.bundleDict.values())[0].slicer
        for b in self.bundleDict.values():
            if b.slicer != self.slicer:
                raise ValueError(
                    "Currently, the slicers for the MoMetricBundleGroup must be equal,"
                    " using the same observations and Hvals."
                )
        self.constraints = list(set([b.constraint for b in bundle_dict.values()]))

    def _check_compatible(self, metricBundle1, metricBundle2):
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
        if metricBundle1.maps_list.sort() != metricBundle2.maps_list.sort():
            return False
        for stacker in metricBundle1.stacker_list:
            for stacker2 in metricBundle2.stacker_list:
                # If the stackers have different names, that's OK, and if they are identical, that's ok.
                if (stacker.__class__.__name__ == stacker2.__class__.__name__) & (
                    stacker != stacker2
                ):
                    return False
        # But if we got this far, everything matches.
        return True

    def _find_compatible(self, testKeys):
        """ "Private utility to find which metricBundles with keys in the list 'testKeys' can be calculated
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
                warnings.warn(
                    "Received %s in testkeys, but this is not present in self.bundleDict."
                    "Will continue, but this is not expected."
                )
                continue
            foundCompatible = False
            checkedAll = False
            while not (foundCompatible) and not (checkedAll):
                # Go through the existing lists in compatibleLists, to see if this metricBundle matches.
                for compatibleList in compatibleLists:
                    # Compare to all the metricBundles in this subset, to check all stackers are compatible.
                    foundCompatible = True
                    for comparisonKey in compatibleList:
                        compatible = self._check_compatible(
                            self.bundleDict[comparisonKey], b
                        )
                        if not compatible:
                            # Found a metricBundle which is not compatible, so stop and go onto the next subset.
                            foundCompatible = False
                            break
                checkedAll = True
            if foundCompatible:
                compatibleList.append(k)
            else:
                compatibleLists.append(
                    [
                        k,
                    ]
                )
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
        compatibleLists = self._find_compatible(keysMatchingConstraint)

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
            print("Running metrics %s" % compatibleList)

        bDict = (
            self.bundleDict
        )  #  {key: self.bundleDict.get(key) for key in compatibleList}

        # Find the unique stackers and maps. These are already "compatible" (as id'd by compatibleList).
        uniqStackers = []
        allStackers = []
        uniqMaps = []
        allMaps = []
        for b in bDict.values():
            allStackers += b.stacker_list
            allMaps += b.maps_list
        for s in allStackers:
            if s not in uniqStackers:
                uniqStackers.append(s)
        for m in allMaps:
            if m not in uniqMaps:
                uniqMaps.append(m)

        if len(uniqMaps) > 0:
            print(
                "Got some maps .. that was unexpected at the moment. Can't use them here yet."
            )

        # Set up all of the metric values, including for the child bundles.
        for k in compatibleList:
            b = self.bundleDict[k]
            b._setup_metric_values()
            for cb in b.child_bundles.values():
                cb._setup_metric_values()
        # Calculate the metric values.
        for i, slicePoint in enumerate(self.slicer):
            ssoObs = slicePoint["obs"]
            for j, Hval in enumerate(slicePoint["Hvals"]):
                # Run stackers to add extra columns (that depend on Hval)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for s in uniqStackers:
                        ssoObs = s.run(ssoObs, slicePoint["orbit"]["H"], Hval)
                # Run all the parent metrics.
                for k in compatibleList:
                    b = self.bundleDict[k]
                    # Mask the parent metric (and then child metrics) if there was no data.
                    if len(ssoObs) == 0:
                        b.metricValues.mask[i][j] = True
                        for cb in list(b.child_bundles.values()):
                            cb.metricValues.mask[i][j] = True
                    # Otherwise, calculate the metric value for the parent, and then child.
                    else:
                        # Calculate for the parent.
                        mVal = b.metric.run(ssoObs, slicePoint["orbit"], Hval)
                        # Mask if the parent metric returned a bad value.
                        if mVal == b.metric.badval:
                            b.metricValues.mask[i][j] = True
                            for cb in b.child_bundles.values():
                                cb.metricValues.mask[i][j] = True
                        # Otherwise, set the parent value and calculate the child metric values as well.
                        else:
                            b.metricValues.data[i][j] = mVal
                            for cb in b.child_bundles.values():
                                childVal = cb.metric.run(
                                    ssoObs, slicePoint["orbit"], Hval, mVal
                                )
                                if childVal == cb.metric.badval:
                                    cb.metricValues.mask[i][j] = True
                                else:
                                    cb.metricValues.data[i][j] = childVal
        for k in compatibleList:
            b = self.bundleDict[k]
            b.compute_summary_stats(self.resultsDb)
            for cB in b.child_bundles.values():
                cB.compute_summary_stats(self.resultsDb)
                # Write to disk.
                cB.write(out_dir=self.outDir, results_db=self.resultsDb)
            # Write to disk.
            b.write(out_dir=self.outDir, results_db=self.resultsDb)

    def runAll(self):
        """
        Run all constraints and metrics for these moMetricBundles.
        """
        for constraint in self.constraints:
            self.runConstraint(constraint)
        if self.verbose:
            print("Calculated and saved all metrics.")

    def plotAll(
        self,
        savefig=True,
        outfileSuffix=None,
        figformat="pdf",
        dpi=600,
        thumbnail=True,
        closefigs=True,
    ):
        """
        Make a few generically desired plots. This needs more flexibility in the future.
        """
        plotHandler = PlotHandler(
            outDir=self.outDir,
            resultsDb=self.resultsDb,
            savefig=savefig,
            figformat=figformat,
            dpi=dpi,
            thumbnail=thumbnail,
        )
        for b in self.bundleDict.values():
            try:
                b.plot(
                    plotHandler=plotHandler,
                    outfileSuffix=outfileSuffix,
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
