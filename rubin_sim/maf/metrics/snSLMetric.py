import numpy as np
import healpy as hp

import rubin_sim.maf.metrics as metrics
from rubin_sim.photUtils import Dust_values
from rubin_sim.maf.utils import collapse_night
from rubin_sim.utils import calcSeason

__all__ = ['SNSLMetric']


class SNSLMetric(metrics.BaseMetric):
    """Calculate  the number of expected well-measured strongly lensed SN (per dataslice).

    The number of expected strongly lensed SN detections with a well-measured time delay is given by:

    N (lensed SNe Ia with well measured time delay) = 45.7 *
    survey_area / (20000 deg^2) *
    cumulative_season_length / (2.5 years) /
    (2.15 * exp(0.37 * gap_median_all_filter))

    where:
    survey_area: survey area (in deg2)
    cumulative_season_length: cumulative season length (in years)
    gap_median_all_filter: median gap (all filters) (in days)

    (reference? metric originated from Simon Huber and Phillipe Gris)

    Parameters
    ----------
    metricName : str, optional
        metric name
        Default : SNCadenceMetric
    mjdCol : str, optional
        mjd column name
        Default : observationStartMJD,
    filterCol : str, optional
        filter column name
        Default: filter
    nightCol : str, optional
        night column name
        Default : night
    m5Col : str, optional
        individual visit five-sigma limiting magnitude (m5) column name
        Default : fiveSigmaDepth
    season: int (list) or -1, optional
        season to process (default: -1: all seasons)
    nfilters_min : int, optional
        The number of filters to demand in a season
        Default: 4.
    min_season_obs : int, optional
        Minimum number of observations per season. Default 5.
    m5mins : dict, optional
        Minimum individual image depth for visit to 'count'.
        Default None uses {'u': 22.7, 'g': 24.1, 'r': 23.7, 'i': 23.1, 'z': 22.2, 'y': 21.4}.
    maps : list, optional
        List of maps to use. Default is the dustmap, to reduce m5 limiting mags accordingly.

    Returns
    -------
    float
        Number of expected well-measured strongly lensed SN
    """
    def __init__(self, metricName='SNSLMetric',
                 mjdCol='observationStartMJD', filterCol='filter',
                 nightCol='night', m5Col='fiveSigmaDepth',
                 season=[-1], nfilters_min=4, min_season_obs=5,
                 m5mins=None,
                 maps=['DustMap'], **kwargs):

        self.mjdCol = mjdCol
        self.filterCol = filterCol
        self.nightCol = nightCol
        self.m5Col = m5Col
        self.maps= maps

        cols = [self.nightCol, self.filterCol, self.mjdCol, self.m5Col]
        super().__init__(col=cols, metricName=metricName, maps=self.maps, units='N SL', **kwargs)

        self.badVal = 0
        self.season = season
        self.bands = 'ugrizy'
        if m5mins is None:
            self.m5mins = {'u': 22.7, 'g': 24.1, 'r': 23.7, 'i': 23.1, 'z': 22.2, 'y': 21.4}
        else:
            self.m5mins = m5mins
        self.min_season_obs = min_season_obs
        self.nfilters_min = nfilters_min
        # Set up dust-extinction values to use to interpret the dust map.
        self.phot_properties = Dust_values()

    def n_lensed(self, area, cadence, season_length):
        """
        Parameters
        -----------
        area : float
            Area in square degrees related to this dataslice (sq deg)
        gap_median : float
            median gap between nights with visits (days) - any filter
        cumul_season : float
            length of the season or period of consideration (years)

        Returns
        -------
        float
            Number of strongly lensed SN expected in this area
        """
        # estimate the number of lensed supernovae
        N_lensed_SNe_Ia = (45.7
                           * area / 20000.
                           * season_length / 2.5
                           / (2.15 * np.exp(0.37 * cadence)))
        return N_lensed_SNe_Ia

    def run(self, dataSlice, slicePoint=None):
        """
        Runs the metric for each dataSlice

        Parameters
        ---------------
        dataSlice : simulation data
        slicePoint:  slicePoint(default None)

        Returns
        -----------
        number of SL time delay supernovae

        """
        # If we had no incoming data - just return with badVal.
        if len(dataSlice) == 0:
            return self.badVal

        # Crop it down so things are coadded per night per filter at the median MJD time
        nightSlice = collapse_night(dataSlice, nightCol=self.nightCol, filterCol=self.filterCol,
                                   m5Col=self.m5Col, mjdCol=self.mjdCol)
        # Calculate the dust extinction-corrected m5 values and cut visits which don't meet self.m5mins
        for f in np.unique(nightSlice[self.filterCol]):
            in_filt = np.where(nightSlice[self.filterCol] == f)[0]
            A_x = self.phot_properties.Ax1[f] * slicePoint['ebv']
            nightSlice[self.m5Col][in_filt] = nightSlice[self.m5Col][in_filt] - A_x
            # Set the visits which fall below the minimum to an obvious non-valid value
            nightSlice[self.m5Col][in_filt] = np.where(nightSlice[self.m5Col][in_filt] > self.m5mins[f],
                                                       nightSlice[self.m5Col][in_filt], -999)
        idxs = np.where(nightSlice[self.m5Col] > -998)
        # If nothing survived these cuts, just return with badVal.
        if len(idxs[0]) == 0:
            return self.badval

        # Reset, with coadded per-night/per-filter values, skipping any too-shallow visits.
        nightSlice = np.sort(nightSlice[idxs], order=self.mjdCol)

        # get the pixel area
        area = hp.nside2pixarea(slicePoint['nside'], degrees=True)

        # Note that 'seasons' is the same length as nightSlice, and contains integer (season) + float (day)
        seasons = calcSeason(np.degrees(slicePoint['ra']), nightSlice[self.mjdCol])
        season_ints = np.floor(seasons)

        if self.season == [-1]:
            season_loop = np.unique(season_ints)
        else:
            season_loop = self.season

        N_lensed_SNe_Ia = 0
        for s in season_loop:
            s_idx = np.where(season_ints == s)[0]
            u_filters = np.unique(nightSlice[s_idx][self.filterCol])
            if (len(s_idx) < self.min_season_obs) | (np.size(u_filters) < self.nfilters_min):
                # Skip this season
                N_lensed_SNe_Ia += 0
            else:
                # Find the cadence (days) between visits within the season
                cadence = np.diff(nightSlice['observationStartMJD'][s_idx])
                # But only the values between nights, not within nights
                cadence = np.median(cadence[np.where(cadence > 0.4)])
                # Season length in years
                season_length = seasons[s_idx][-1] - seasons[s_idx][0]
                N_lensed_SNe_Ia += self.n_lensed(area, cadence, season_length)

        return N_lensed_SNe_Ia
