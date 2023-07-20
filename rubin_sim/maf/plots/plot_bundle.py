__all__ = ("PlotBundle",)


import matplotlib.pylab as plt

from .plot_handler import PlotHandler


class PlotBundle:
    """
    Object designed to help organize multiple MetricBundles that will be plotted
    together using the PlotHandler.
    """

    def __init__(self, bundle_list=None, plot_dicts=None, plot_func=None):
        """
        Init object and set things if desired.
        bundle_list: A list of bundleDict objects
        plot_dicts: A list of dictionaries with plotting kwargs
        plot_func: A single MAF plotting function
        """
        if bundle_list is None:
            self.bundle_list = []
        else:
            self.bundle_list = bundle_list

        if plot_dicts is None:
            if len(self.bundle_list) > 0:
                self.plot_dicts = [{}]
            else:
                self.plot_dicts = []
        else:
            self.plot_dicts = plot_dicts

        self.plot_func = plot_func

    def add_bundle(self, bundle, plot_dict=None, plot_func=None):
        """
        Add bundle to the object.
        Optionally add a plot_dict and/or replace the plot_func
        """
        self.bundle_list.append(bundle)
        if plot_dict is not None:
            self.plot_dicts.append(plot_dict)
        else:
            self.plot_dicts.append({})
        if plot_func is not None:
            self.plot_func = plot_func

    def increment_plot_order(self):
        """
        Find the maximium order number in the display dicts, and set them to +1 that
        """
        max_order = 0
        for m_b in self.bundle_list:
            if "order" in list(m_b.displayDict.keys()):
                max_order = max([max_order, m_b.displayDict["order"]])

        for m_b in self.bundle_list:
            m_b.displayDict["order"] = max_order + 1

    def percentile_legend(self):
        """
        Go through the bundles and change the lables if there are the correct summary stats
        """
        for i, mB in enumerate(self.bundle_list):
            if mB.summary_values is not None:
                keys = list(mB.summary_values.keys())
                if ("25th%ile" in keys) & ("75th%ile" in keys) & ("Median" in keys):
                    if "label" not in list(self.plot_dicts[i].keys()):
                        self.plot_dicts[i]["label"] = ""
                    newstr = "%0.1f/%0.1f/%0.1f " % (
                        mB.summary_values["25th%ile"],
                        mB.summary_values["Median"],
                        mB.summary_values["75th%ile"],
                    )
                    self.plot_dicts[i]["label"] = newstr + self.plot_dicts[i]["label"]

    def plot(self, out_dir="Out", results_db=None, closefigs=True):
        ph = PlotHandler(out_dir=out_dir, results_db=results_db)
        ph.set_metric_bundles(self.bundle_list)
        # Auto-generate labels and things
        ph.set_plot_dicts(plot_dicts=self.plot_dicts, plot_func=self.plot_func)
        ph.plot(self.plot_func, plot_dicts=self.plot_dicts)
        if closefigs:
            plt.close("all")
