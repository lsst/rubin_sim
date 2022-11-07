import matplotlib.pyplot as plt
from .plot_handler import BasePlotter


__all__ = ["XyPlotter"]


class XyPlotter(BasePlotter):
    """Bare-bones plotter for making scatter plots. Expects single metric value
    (e.g, from UniSlicer or UserPointSlicer with one point)"""

    def __init__(self):
        self.object_plotter = True
        self.plot_type = "simple"
        self.default_plot_dict = {"title": None, "xlabel": "", "ylabel": ""}

    def __call__(self, metric_value_in, slicer, user_plot_dict, fignum=None):

        plot_dict = {}
        plot_dict.update(self.default_plot_dict)
        plot_dict.update(user_plot_dict)
        plot_dict.update(metric_value_in[0]["plot_dict"])

        fig = plt.figure(fignum)
        ax = fig.add_subplot(111)
        x = metric_value_in[0]["x"]
        y = metric_value_in[0]["y"]
        ax.plot(x, y)
        ax.set_title(plot_dict["title"])
        ax.set_xlabel(plot_dict["xlabel"])
        ax.set_ylabel(plot_dict["ylabel"])
        return fig.number
