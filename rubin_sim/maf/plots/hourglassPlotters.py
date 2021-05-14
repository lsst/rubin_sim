import numpy as np
import matplotlib.pyplot as plt

from .plotHandler import BasePlotter

__all__ = ['HourglassPlot']

class HourglassPlot(BasePlotter):
    def __init__(self):
        self.plotType = 'Hourglass'
        self.objectPlotter = True
        self.defaultPlotDict = {'title': None, 'xlabel': 'Night - min(Night)',
                                'ylabel': 'Hours from local midnight'}
        self.filter2color = {'u': 'purple', 'g': 'blue', 'r': 'green',
                             'i': 'cyan', 'z': 'orange', 'y': 'red'}

    def __call__(self, metricValue, slicer, userPlotDict, fignum=None):
        """
        Generate the hourglass plot
        """
        if slicer.slicerName != 'HourglassSlicer':
            raise ValueError('HourglassPlot is for use with hourglass slicers')

        fig = plt.figure(fignum)
        ax = fig.add_subplot(111)

        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        plotDict.update(userPlotDict)

        pernight = metricValue[0]['pernight']
        perfilter = metricValue[0]['perfilter']

        y = (perfilter['mjd'] - perfilter['midnight']) * 24.0
        dmin = np.floor(np.min(perfilter['mjd']))
        for i in np.arange(0, perfilter.size, 2):
            ax.plot([perfilter['mjd'][i] - dmin, perfilter['mjd'][i + 1] - dmin], [y[i], y[i + 1]],
                    self.filter2color[perfilter['filter'][i]])
        for i, key in enumerate(['u', 'g', 'r', 'i', 'z', 'y']):
            ax.text(1.05, .9 - i * 0.07, key, color=self.filter2color[key], transform=ax.transAxes)
        ax.plot(pernight['mjd'] - dmin, (pernight['twi6_rise'] - pernight['midnight']) * 24.,
                'blue', label=r'6$^\circ$ twilight')
        ax.plot(pernight['mjd'] - dmin, (pernight['twi6_set'] - pernight['midnight']) * 24.,
                'blue')
        ax.plot(pernight['mjd'] - dmin, (pernight['twi12_rise'] - pernight['midnight']) * 24.,
                'yellow', label=r'12$^\circ$ twilight')
        ax.plot(pernight['mjd'] - dmin, (pernight['twi12_set'] - pernight['midnight']) * 24.,
                'yellow')
        ax.plot(pernight['mjd'] - dmin, (pernight['twi18_rise'] - pernight['midnight']) * 24.,
                'red', label=r'18$^\circ$ twilight')
        ax.plot(pernight['mjd'] - dmin, (pernight['twi18_set'] - pernight['midnight']) * 24.,
                'red')
        ax.plot(pernight['mjd'] - dmin, pernight['moonPer'] / 100. - 7, 'black', label='Moon')
        ax.set_xlabel(plotDict['xlabel'])
        ax.set_ylabel(plotDict['ylabel'])
        ax.set_title(plotDict['title'])

        # draw things in with lines if we are only plotting one night
        if len(pernight['mjd']) == 1:
            ax.axhline((pernight['twi6_rise'] - pernight['midnight']) * 24.,
                       color='blue', label=r'6$^\circ$ twilight')
            ax.axhline((pernight['twi6_set'] - pernight['midnight']) * 24.,
                       color='blue')
            ax.axhline((pernight['twi12_rise'] - pernight['midnight']) * 24.,
                       color='yellow', label=r'12$^\circ$ twilight')
            ax.axhline((pernight['twi12_set'] - pernight['midnight']) * 24.,
                       color='yellow')
            ax.axhline((pernight['twi18_rise'] - pernight['midnight']) * 24.,
                       color='red', label=r'18$^\circ$ twilight')
            ax.axhline((pernight['twi18_set'] - pernight['midnight']) * 24.,
                       color='red')

        return fig.number
