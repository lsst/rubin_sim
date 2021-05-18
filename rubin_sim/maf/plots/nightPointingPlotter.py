import numpy as np
import matplotlib.pyplot as plt

from .plotHandler import BasePlotter

__all__ = ['NightPointingPlotter']


class NightPointingPlotter(BasePlotter):

    def __init__(self, mjdCol='observationStartMJD', altCol='alt', azCol='az'):

        # Just call it Hourglass so it gets treated the same way
        self.plotType = 'Hourglass'
        self.mjdCol = mjdCol
        self.altCol = altCol
        self.azCol = azCol
        self.objectPlotter = True
        self.defaultPlotDict = {'title': None, 'xlabel': 'MJD',
                                'ylabels': ['Alt', 'Az']}
        self.filter2color = {'u': 'purple', 'g': 'blue', 'r': 'green',
                             'i': 'cyan', 'z': 'orange', 'y': 'red'}

    def __call__(self, metricValue, slicer, userPlotDict, fignum=None):

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        mv = metricValue[0]

        u_filters = np.unique(mv['dataSlice']['filter'])
        for filt in u_filters:
            good = np.where(mv['dataSlice']['filter'] == filt)
            ax1.plot(mv['dataSlice'][self.mjdCol][good],
                     mv['dataSlice'][self.altCol][good],
                     'o', color=self.filter2color[filt], markersize=5, alpha=.5)
            ax2.plot(mv['dataSlice'][self.mjdCol][good],
                     mv['dataSlice'][self.azCol][good],
                     'o', color=self.filter2color[filt], markersize=5, alpha=.5)

        good = np.where(np.degrees(mv['moon_alts']) > -10.)
        ax1.plot(mv['mjds'][good], np.degrees(mv['moon_alts'][good]), 'ko', markersize=3, alpha=.1)
        ax2.plot(mv['mjds'][good], np.degrees(mv['moon_azs'][good]), 'ko', markersize=3, alpha=.1)
        ax2.set_xlabel('MJD')
        ax1.set_ylabel('Altitude (deg)')
        ax2.set_ylabel('Azimuth (deg)')

        good = np.where(np.degrees(mv['sun_alts']) > -20.)
        ax1.plot(mv['mjds'][good], np.degrees(mv['sun_alts'][good]), 'yo', markersize=3)
        ax2.plot(mv['mjds'][good], np.degrees(mv['sun_azs'][good]), 'yo', markersize=3)

        ax1.set_ylim([-20., 90.])
        ax2.set_ylim([0., 360.])

        for i, key in enumerate(['u', 'g', 'r', 'i', 'z', 'y']):
            ax1.text(1.05, .9 - i * 0.07, key, color=self.filter2color[key], transform=ax1.transAxes)

        return fig.number
