from builtins import object
from .plotHandler import PlotHandler
import matplotlib.pylab as plt

__all__ = ['PlotBundle']

class PlotBundle(object):
    """
    Object designed to help organize multiple MetricBundles that will be plotted
    together using the PlotHandler.
    """

    def __init__(self, bundleList=None, plotDicts=None, plotFunc=None):
        """
        Init object and set things if desired.
        bundleList: A list of bundleDict objects
        plotDicts: A list of dictionaries with plotting kwargs
        plotFunc: A single MAF plotting function
        """
        if bundleList is None:
            self.bundleList = []
        else:
            self.bundleList = bundleList

        if plotDicts is None:
            if len(self.bundleList) > 0:
                self.plotDicts = [{}]
            else:
                self.plotDicts = []
        else:
            self.plotDicts = plotDicts

        self.plotFunc = plotFunc

    def addBundle(self, bundle, plotDict=None, plotFunc=None):
        """
        Add bundle to the object.
        Optionally add a plotDict and/or replace the plotFunc
        """
        self.bundleList.append(bundle)
        if plotDict is not None:
            self.plotDicts.append(plotDict)
        else:
            self.plotDicts.append({})
        if plotFunc is not None:
            self.plotFunc = plotFunc

    def incrementPlotOrder(self):
        """
        Find the maximium order number in the display dicts, and set them to +1 that
        """
        maxOrder = 0
        for mB in self.bundleList:
            if 'order' in list(mB.displayDict.keys()):
                maxOrder = max([maxOrder, mB.displayDict['order']])

        for mB in self.bundleList:
            mB.displayDict['order'] = maxOrder + 1

    def percentileLegend(self):
        """
        Go through the bundles and change the lables if there are the correct summary stats
        """
        for i, mB in enumerate(self.bundleList):
            if mB.summaryValues is not None:
                keys = list(mB.summaryValues.keys())
                if ('25th%ile' in keys) & ('75th%ile' in keys) & ('Median' in keys):
                    if 'label' not in list(self.plotDicts[i].keys()):
                        self.plotDicts[i]['label'] = ''
                    newstr = '%0.1f/%0.1f/%0.1f ' % (mB.summaryValues['25th%ile'],
                                                     mB.summaryValues['Median'],
                                                     mB.summaryValues['75th%ile'])
                    self.plotDicts[i]['label'] = newstr + self.plotDicts[i]['label']

    def plot(self, outDir='Out', resultsDb=None, closeFigs=True):
        ph = PlotHandler(outDir=outDir, resultsDb=resultsDb)
        ph.setMetricBundles(self.bundleList)
        # Auto-generate labels and things
        ph.setPlotDicts(plotDicts=self.plotDicts, plotFunc=self.plotFunc)
        ph.plot(self.plotFunc, plotDicts=self.plotDicts)
        if closeFigs:
            plt.close('all')
