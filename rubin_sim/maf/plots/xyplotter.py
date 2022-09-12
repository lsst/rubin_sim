import matplotlib.pyplot as plt
from .plot_handler import BasePlotter


__all__ = ["XyPlotter"]


class XyPlotter(BasePlotter):
    """Bare-bones plotter for making scatter plots. Expects single metric value
    (e.g, from UniSlicer or UserPointSlicer with one point)"""

    def __init__(self):
        self.objectPlotter = True
        self.plotType = "simple"
        self.defaultPlotDict = {"title": None, "xlabel": "", "ylabel": ""}

    def __call__(self, metricValueIn, slicer, userPlotDict, fignum=None):

        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        plotDict.update(userPlotDict)
        plotDict.update(metricValueIn[0]["plotDict"])

        fig = plt.figure(fignum)
        ax = fig.add_subplot(111)
        x = metricValueIn[0]["x"]
        y = metricValueIn[0]["y"]
        ax.plot(x, y)
        ax.set_title(plotDict["title"])
        ax.set_xlabel(plotDict["xlabel"])
        ax.set_ylabel(plotDict["ylabel"])
        return fig.number
