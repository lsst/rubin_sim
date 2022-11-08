import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
import rubin_sim.maf.utils as utils

__all__ = ["apply_zp_norm", "PlotHandler", "BasePlotter"]


def apply_zp_norm(metric_value, plot_dict):
    if "zp" in plot_dict:
        if plot_dict["zp"] is not None:
            metric_value = metric_value - plot_dict["zp"]
    if "normVal" in plot_dict:
        if plot_dict["normVal"] is not None:
            metric_value = metric_value / plot_dict["normVal"]
    return metric_value


class BasePlotter(object):
    """
    Serve as the base type for MAF plotters and example of API.
    """

    def __init__(self):
        self.plot_type = None
        # This should be included in every subsequent defaultPlotDict (assumed to be present).
        self.default_plot_dict = {
            "title": None,
            "xlabel": None,
            "label": None,
            "labelsize": None,
            "fontsize": None,
            "figsize": None,
        }

    def __call__(self, metric_value, slicer, user_plot_dict, fignum=None):
        pass


class PlotHandler(object):
    def __init__(
        self,
        out_dir=".",
        results_db=None,
        savefig=True,
        figformat="pdf",
        dpi=600,
        thumbnail=True,
        trim_whitespace=True,
    ):
        self.out_dir = out_dir
        self.results_db = results_db
        self.savefig = savefig
        self.figformat = figformat
        self.dpi = dpi
        self.trim_whitespace = trim_whitespace
        self.thumbnail = thumbnail
        self.filtercolors = {
            "u": "cyan",
            "g": "g",
            "r": "y",
            "i": "r",
            "z": "m",
            "y": "k",
            " ": None,
        }
        self.filterorder = {" ": -1, "u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5}

    def set_metric_bundles(self, m_bundles):
        """
        Set the metric bundle or bundles (list or dictionary).
        Reuse the PlotHandler by resetting this reference.
        The metric bundles have to have the same slicer.
        """
        self.m_bundles = []
        # Try to add the metricBundles in filter order.
        if isinstance(m_bundles, dict):
            for m_b in m_bundles.values():
                vals = m_b.file_root.split("_")
                forder = [self.filterorder.get(f, None) for f in vals if len(f) == 1]
                forder = [o for o in forder if o is not None]
                if len(forder) == 0:
                    forder = len(self.m_bundles)
                else:
                    forder = forder[-1]
                self.m_bundles.insert(forder, m_b)
            self.slicer = self.m_bundles[0].slicer
        else:
            for m_b in m_bundles:
                vals = m_b.file_root.split("_")
                forder = [self.filterorder.get(f, None) for f in vals if len(f) == 1]
                forder = [o for o in forder if o is not None]
                if len(forder) == 0:
                    forder = len(self.m_bundles)
                else:
                    forder = forder[-1]
                self.m_bundles.insert(forder, m_b)
            self.slicer = self.m_bundles[0].slicer
        for m_b in self.m_bundles:
            if m_b.slicer.slicerName != self.slicer.slicerName:
                raise ValueError("MetricBundle items must have the same type of slicer")
        self._combine_metric_names()
        self._combine_run_names()
        self._combine_metadata()
        self._combine_constraints()
        self.set_plot_dicts(reset=True)

    def set_plot_dicts(self, plot_dicts=None, plot_func=None, reset=False):
        """
        Set or update (or 'reset') the plot_dict for the (possibly joint) plots.

        Resolution is:
        auto-generated items (colors/labels/titles)
        < anything previously set in the plot_handler
        < defaults set by the plotter
        < explicitly set items in the metricBundle plot_dict
        < explicitly set items in the plot_dicts list passed to this method.
        """
        if reset:
            # Have to explicitly set each dictionary to a (separate) blank dictionary.
            self.plot_dicts = [{} for b in self.m_bundles]

        if isinstance(plot_dicts, dict):
            # We were passed a single dictionary, not a list.
            plot_dicts = [plot_dicts] * len(self.m_bundles)

        auto_label_list = self._build_legend_labels()
        auto_color_list = self._build_colors()
        auto_cbar = self._build_cbar_format()
        auto_title = self._build_title()
        if plot_func is not None:
            auto_xlabel, auto_ylabel = self._build_x_ylabels(plot_func)

        # Loop through each bundle and generate a plot_dict for it.
        for i, bundle in enumerate(self.m_bundles):
            # First use the auto-generated values.
            tmp_plot_dict = {}
            tmp_plot_dict["title"] = auto_title
            tmp_plot_dict["label"] = auto_label_list[i]
            tmp_plot_dict["color"] = auto_color_list[i]
            tmp_plot_dict["cbarFormat"] = auto_cbar
            # Then update that with anything previously set in the plot_handler.
            tmp_plot_dict.update(self.plot_dicts[i])
            # Then override with plot_dict items set explicitly based on the plot type.
            if plot_func is not None:
                tmp_plot_dict["xlabel"] = auto_xlabel
                tmp_plot_dict["ylabel"] = auto_ylabel
                # Replace auto-generated plot dict items with things
                #  set by the plotter_defaults, if they are not None.
                plotter_defaults = plot_func.defaultPlotDict
                for k, v in plotter_defaults.items():
                    if v is not None:
                        tmp_plot_dict[k] = v
            # Then add/override based on the bundle plot_dict parameters if they are set.
            tmp_plot_dict.update(bundle.plotDict)
            # Finally, override with anything set explicitly by the user right now.
            if plot_dicts is not None:
                tmp_plot_dict.update(plot_dicts[i])
            # And save this new dictionary back in the class.
            self.plot_dicts[i] = tmp_plot_dict

        # Check that the plot_dicts do not conflict.
        self._check_plot_dicts()

    def _combine_metric_names(self):
        """
        Combine metric names.
        """
        # Find the unique metric names.
        self.metric_names = set()
        for m_b in self.m_bundles:
            self.metric_names.add(m_b.metric.name)
        # Find a pleasing combination of the metric names.
        order = ["u", "g", "r", "i", "z", "y"]
        if len(self.metric_names) == 1:
            joint_name = " ".join(self.metric_names)
        else:
            # Split each unique name into a list to see if we can merge the names.
            name_lengths = [len(x.split()) for x in self.metric_names]
            name_lists = [x.split() for x in self.metric_names]
            # If the metric names are all the same length, see if we can combine any parts.
            if len(set(name_lengths)) == 1:
                joint_name = []
                for i in range(name_lengths[0]):
                    tmp = set([x[i] for x in name_lists])
                    # Try to catch special case of filters and put them in order.
                    if tmp.intersection(order) == tmp:
                        filterlist = ""
                        for f in order:
                            if f in tmp:
                                filterlist += f
                        joint_name.append(filterlist)
                    else:
                        # Otherwise, just join and put into joint_name.
                        joint_name.append("".join(tmp))
                joint_name = " ".join(joint_name)
            # If the metric names are not the same length, just join everything.
            else:
                joint_name = " ".join(self.metric_names)
        self.joint_metric_names = joint_name

    def _combine_run_names(self):
        """
        Combine runNames.
        """
        self.run_names = set()
        for m_b in self.m_bundles:
            self.run_names.add(m_b.run_name)
        self.joint_run_names = " ".join(self.run_names)

    def _combine_metadata(self):
        """
        Combine info_label.
        """
        info_label = set()
        for m_b in self.m_bundles:
            info_label.add(m_b.info_label)
        self.info_label = info_label
        # Find a pleasing combination of the info_label.
        if len(info_label) == 1:
            self.joint_metadata = " ".join(info_label)
        else:
            order = ["u", "g", "r", "i", "z", "y"]
            # See if there are any subcomponents we can combine,
            # splitting on some values we expect to separate info_label clauses.
            splitmetas = []
            for m in self.info_label:
                # Try to split info_label into separate phrases (filter / proposal / constraint..).
                if " and " in m:
                    m = m.split(" and ")
                elif ", " in m:
                    m = m.split(", ")
                else:
                    m = [
                        m,
                    ]
                # Strip white spaces from individual elements.
                m = set([im.strip() for im in m])
                splitmetas.append(m)
            # Look for common elements and separate from the general info_label.
            common = set.intersection(*splitmetas)
            diff = [x.difference(common) for x in splitmetas]
            # Now look within the 'diff' elements and see if there are any common words to split off.
            diffsplit = []
            for d in diff:
                if len(d) > 0:
                    m = set([x.split() for x in d][0])
                else:
                    m = set()
                diffsplit.append(m)
            diffcommon = set.intersection(*diffsplit)
            diffdiff = [x.difference(diffcommon) for x in diffsplit]
            # If the length of any of the 'differences' is 0, then we should stop and not try to subdivide.
            lengths = [len(x) for x in diffdiff]
            if min(lengths) == 0:
                # Sort them in order of length (so it goes 'g', 'g dithered', etc.)
                tmp = []
                for d in diff:
                    tmp.append(list(d)[0])
                diff = tmp
                xlengths = [len(x) for x in diff]
                idx = np.argsort(xlengths)
                diffdiff = [diff[i] for i in idx]
                diffcommon = []
            else:
                # diffdiff is the part where we might expect our filter values to appear;
                # try to put this in order.
                diffdiff_ordered = []
                diffdiff_end = []
                for f in order:
                    for d in diffdiff:
                        if len(d) == 1:
                            if list(d)[0] == f:
                                diffdiff_ordered.append(d)
                for d in diffdiff:
                    if d not in diffdiff_ordered:
                        diffdiff_end.append(d)
                diffdiff = diffdiff_ordered + diffdiff_end
                diffdiff = [" ".join(c) for c in diffdiff]
            # And put it all back together.
            combo = (
                ", ".join(["".join(c) for c in diffdiff])
                + " "
                + " ".join(["".join(d) for d in diffcommon])
                + " "
                + " ".join(["".join(e) for e in common])
            )
            self.joint_metadata = combo

    def _combine_constraints(self):
        """
        Combine the constraints.
        """
        constraints = set()
        for m_b in self.m_bundles:
            if m_b.constraint is not None:
                constraints.add(m_b.constraint)
        self.constraints = "; ".join(constraints)

    def _build_title(self):
        """
        Build a plot title from the metric names, runNames and info_label.
        """
        # Create a plot title from the unique parts of the metric/run_name/info_label.
        plot_title = ""
        if len(self.run_names) == 1:
            plot_title += list(self.run_names)[0]
        if len(self.info_label) == 1:
            plot_title += " " + list(self.info_label)[0]
        if len(self.metric_names) == 1:
            plot_title += ": " + list(self.metric_names)[0]
        if plot_title == "":
            # If there were more than one of everything above, use joint info_label and metricNames.
            plot_title = self.joint_metadata + " " + self.joint_metric_names
        return plot_title

    def _build_x_ylabels(self, plot_func, len_max=25):
        """
        Build a plot x and y label.

        Parameters
        ----------
        len_max : `int` (30)
            If the xlabel starts longer than this, add the units as a newline.
        """
        if plot_func.plotType == "BinnedData":
            if len(self.m_bundles) == 1:
                m_b = self.m_bundles[0]
                if len(m_b.slicer.sliceColName) < len_max:
                    xlabel = (
                        m_b.slicer.sliceColName + " (" + m_b.slicer.sliceColUnits + ")"
                    )
                else:
                    xlabel = (
                        m_b.slicer.sliceColName
                        + " \n("
                        + m_b.slicer.sliceColUnits
                        + ")"
                    )
                ylabel = m_b.metric.name + " (" + m_b.metric.units + ")"
            else:
                xlabel = set()
                for m_b in self.m_bundles:
                    xlabel.add(m_b.slicer.sliceColName)
                xlabel = ", ".join(xlabel)
                ylabel = self.joint_metric_names
        elif plot_func.plotType == "MetricVsH":
            if len(self.m_bundles) == 1:
                m_b = self.m_bundles[0]
                ylabel = m_b.metric.name + " (" + m_b.metric.units + ")"
            else:
                ylabel = self.joint_metric_names
            xlabel = "H (mag)"
        else:
            if len(self.m_bundles) == 1:
                m_b = self.m_bundles[0]
                xlabel = m_b.metric.name
                if m_b.metric.units is not None:
                    if len(m_b.metric.units) > 0:
                        if len(xlabel) < len_max:
                            xlabel += " (" + m_b.metric.units + ")"
                        else:
                            xlabel += "\n(" + m_b.metric.units + ")"
                ylabel = None
            else:
                xlabel = self.joint_metric_names
                ylabel = set()
                for m_b in self.m_bundles:
                    if "ylabel" in m_b.plotDict:
                        ylabel.add(m_b.plotDict["ylabel"])
                if len(ylabel) == 1:
                    ylabel = list(ylabel)[0]
                else:
                    ylabel = None
        return xlabel, ylabel

    def _build_legend_labels(self):
        """
        Build a set of legend labels, using parts of the run_name/info_label/metricNames that change.
        """
        if len(self.m_bundles) == 1:
            return [None]
        labels = []
        for m_b in self.m_bundles:
            if "label" in m_b.plotDict:
                label = m_b.plotDict["label"]
            else:
                label = ""
                if len(self.run_names) > 1:
                    label += m_b.run_name
                if len(self.info_label) > 1:
                    label += " " + m_b.info_label
                if len(self.metric_names) > 1:
                    label += " " + m_b.metric.name
            labels.append(label)
        return labels

    def _build_colors(self):
        """
        Try to set an appropriate range of colors for the metric Bundles.
        """
        if len(self.m_bundles) == 1:
            if "color" in self.m_bundles[0].plotDict:
                return [self.m_bundles[0].plotDict["color"]]
            else:
                return ["b"]
        colors = []
        for m_b in self.m_bundles:
            color = "b"
            if "color" in m_b.plotDict:
                color = m_b.plotDict["color"]
            else:
                if m_b.constraint is not None:
                    # If the filter is part of the sql constraint, we'll
                    #  try to use that first.
                    if "filter" in m_b.constraint:
                        vals = m_b.constraint.split('"')
                        for v in vals:
                            if len(v) == 1:
                                # Guess that this is the filter value
                                if v in self.filtercolors:
                                    color = self.filtercolors[v]
            colors.append(color)
        # If we happened to end up with the same color throughout
        #  (say, the metrics were all in the same filter)
        #  then go ahead and generate random colors.
        if (len(self.m_bundles) > 1) and (len(np.unique(colors)) == 1):
            colors = [
                np.random.rand(
                    3,
                )
                for m_b in self.m_bundles
            ]
        return colors

    def _build_cbar_format(self):
        """
        Set the color bar format.
        """
        cbar_format = None
        if len(self.m_bundles) == 1:
            if self.m_bundles[0].metric.metricDtype == "int":
                cbar_format = "%d"
        else:
            metric_dtypes = set()
            for m_b in self.m_bundles:
                metric_dtypes.add(m_b.metric.metricDtype)
            if len(metric_dtypes) == 1:
                if list(metric_dtypes)[0] == "int":
                    cbar_format = "%d"
        return cbar_format

    def _build_file_root(self, outfile_suffix=None):
        """
        Build a root filename for plot outputs.
        If there is only one metricBundle, this is equal to the metricBundle fileRoot + outfile_suffix.
        For multiple metricBundles, this is created from the runNames, info_label and metric names.

        If you do not wish to use the automatic filenames, then you could set 'savefig' to False and
          save the file manually to disk, using the plot figure numbers returned by 'plot'.
        """
        if len(self.m_bundles) == 1:
            outfile = self.m_bundles[0].file_root
        else:
            outfile = "_".join(
                [self.joint_run_names, self.joint_metric_names, self.joint_metadata]
            )
            outfile += "_" + self.m_bundles[0].slicer.slicerName[:4].upper()
        if outfile_suffix is not None:
            outfile += "_" + outfile_suffix
        outfile = utils.nameSanitize(outfile)
        return outfile

    def _build_display_dict(self):
        """
        Generate a display dictionary.
        This is most useful for when there are many metricBundles being combined into a single plot.
        """
        if len(self.m_bundles) == 1:
            return self.m_bundles[0].displayDict
        else:
            display_dict = {}
            group = set()
            subgroup = set()
            order = 0
            for m_b in self.m_bundles:
                group.add(m_b.displayDict["group"])
                subgroup.add(m_b.displayDict["subgroup"])
                if order < m_b.displayDict["order"]:
                    order = m_b.displayDict["order"] + 1
            display_dict["order"] = order
            if len(group) > 1:
                display_dict["group"] = "Comparisons"
            else:
                display_dict["group"] = list(group)[0]
            if len(subgroup) > 1:
                display_dict["subgroup"] = "Comparisons"
            else:
                display_dict["subgroup"] = list(subgroup)[0]

            display_dict["caption"] = (
                "%s metric(s) calculated on a %s grid, for opsim runs %s, for info_label values of %s."
                % (
                    self.joint_metric_names,
                    self.m_bundles[0].slicer.slicerName,
                    self.joint_run_names,
                    self.joint_metadata,
                )
            )

            return display_dict

    def _check_plot_dicts(self):
        """
        Check to make sure there are no conflicts in the plotDicts that are being used in the same subplot.
        """
        # Check that the length is OK
        if len(self.plot_dicts) != len(self.m_bundles):
            raise ValueError(
                "plotDicts (%i) must be same length as mBundles (%i)"
                % (len(self.plot_dicts), len(self.m_bundles))
            )

        # These are the keys that need to match (or be None)
        keys2_check = ["xlim", "ylim", "colorMin", "colorMax", "title"]

        # Identify how many subplots there are. If there are more than one, just don't change anything.
        # This assumes that if there are more than one, the plotDicts are actually all compatible.
        subplots = set()
        for pd in self.plot_dicts:
            if "subplot" in pd:
                subplots.add(pd["subplot"])

        # Now check subplots are consistent.
        if len(subplots) <= 1:
            reset_keys = []
            for key in keys2_check:
                values = [pd[key] for pd in self.plot_dicts if key in pd]
                if len(np.unique(values)) > 1:
                    # We will reset some of the keys to the default, but for some we should do better.
                    if key.endswith("Max"):
                        for pd in self.plot_dicts:
                            pd[key] = np.max(values)
                    elif key.endswith("Min"):
                        for pd in self.plot_dicts:
                            pd[key] = np.min(values)
                    elif key == "title":
                        title = self._build_title()
                        for pd in self.plot_dicts:
                            pd["title"] = title
                    else:
                        warnings.warn(
                            'Found more than one value to be set for "%s" in the plotDicts.'
                            % (key)
                            + " Will reset to default value. (found values %s)" % values
                        )
                        reset_keys.append(key)
            # Reset the most of the keys to defaults; this can generally be done safely.
            for key in reset_keys:
                for pd in self.plot_dicts:
                    pd[key] = None

    def plot(
        self,
        plot_func,
        plot_dicts=None,
        display_dict=None,
        outfile_root=None,
        outfile_suffix=None,
    ):
        """
        Create plot for mBundles, using plot_func.

        plot_dicts:  List of plot_dicts if one wants to use a _new_ plot_dict per MetricBundle.
        """
        if not plot_func.objectPlotter:
            # Check that metric_values type and plotter are compatible (most are float/float, but
            #  some plotters expect object data .. and some only do sometimes).
            for m_b in self.m_bundles:
                if m_b.metric.metricDtype == "object":
                    metric_is_color = m_b.plotDict.get("metric_is_color", False)
                    if not metric_is_color:
                        warnings.warn(
                            "Cannot plot object metric values with this plotter."
                        )
                        return

        # Update x/y labels using plot_type.
        self.set_plot_dicts(plot_dicts=plot_dicts, plot_func=plot_func, reset=False)
        # Set outfile name.
        if outfile_root is None:
            outfile = self._build_file_root(outfile_suffix)
        else:
            outfile = outfile_root
        plot_type = plot_func.plotType
        if len(self.m_bundles) > 1:
            plot_type = "Combo" + plot_type
        # Make plot.
        fignum = None
        for m_b, plotDict in zip(self.m_bundles, self.plot_dicts):
            if m_b.metricValues is None:
                # Skip this metricBundle.
                msg = 'MetricBundle (%s) has no attribute "metric_values".' % (
                    m_b.file_root
                )
                msg += " Either the values have not been calculated or they have been deleted."
                warnings.warn(msg)
            else:
                fignum = plot_func(
                    m_b.metricValues, m_b.slicer, plotDict, fignum=fignum
                )
        # Add a legend if more than one metricValue is being plotted or if legendloc is specified.
        legendloc = None
        if "legendloc" in self.plot_dicts[0]:
            legendloc = self.plot_dicts[0]["legendloc"]
        if len(self.m_bundles) > 1:
            try:
                legendloc = self.plot_dicts[0]["legendloc"]
            except KeyError:
                legendloc = "upper right"
        if legendloc is not None:
            plt.figure(fignum)
            plt.legend(loc=legendloc, fancybox=True, fontsize="smaller")
        # Add the super title if provided.
        if "suptitle" in self.plot_dicts[0]:
            plt.suptitle(self.plot_dicts[0]["suptitle"])
        # Save to disk and file info to results_db if desired.
        if self.savefig:
            if display_dict is None:
                display_dict = self._build_display_dict()
            self.save_fig(
                fignum,
                outfile,
                plot_type,
                self.joint_metric_names,
                self.slicer.slicerName,
                self.joint_run_names,
                self.constraints,
                self.joint_metadata,
                display_dict,
            )
        return fignum

    def save_fig(
        self,
        fignum,
        outfile_root,
        plot_type,
        metric_name,
        slicer_name,
        run_name,
        constraint,
        info_label,
        display_dict=None,
    ):
        fig = plt.figure(fignum)
        plot_file = outfile_root + "_" + plot_type + "." + self.figformat
        if self.trim_whitespace:
            fig.savefig(
                os.path.join(self.out_dir, plot_file),
                dpi=self.dpi,
                bbox_inches="tight",
                format=self.figformat,
            )
        else:
            fig.savefig(
                os.path.join(self.out_dir, plot_file),
                dpi=self.dpi,
                format=self.figformat,
            )
        # Generate a png thumbnail.
        if self.thumbnail:
            thumb_file = "thumb." + outfile_root + "_" + plot_type + ".png"
            plt.savefig(
                os.path.join(self.out_dir, thumb_file), dpi=72, bbox_inches="tight"
            )
        # Save information about the file to results_db.
        if self.results_db:
            if display_dict is None:
                display_dict = {}
            metric_id = self.results_db.update_metric(
                metric_name, slicer_name, run_name, constraint, info_label, None
            )
            self.results_db.update_display(
                metric_id=metric_id, display_dict=display_dict, overwrite=False
            )
            self.results_db.update_plot(
                metric_id=metric_id, plot_type=plot_type, plot_file=plot_file
            )
