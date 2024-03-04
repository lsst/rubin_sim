__all__ = ("SNSLMetric",)

import healpy as hp
import numpy as np
from rubin_scheduler.utils import calc_season

import rubin_sim.maf.metrics as metrics
from rubin_sim.maf.utils import collapse_night
from rubin_sim.phot_utils import DustValues


class SNSLMetric(metrics.BaseMetric):
    """Calculate  the number of expected well-measured strongly lensed SN
    (per data_slice).

    Parameters
    ----------
    metric_name : `str`, optional
        metric name
        Default : SNCadenceMetric
    mjd_col : `str`, optional
        mjd column name
        Default : observationStartMJD,
    filter_col : `str`, optional
        filter column name
        Default: filter
    night_col : `str`, optional
        night column name
        Default : night
    m5_col : `str`, optional
        individual visit five-sigma limiting magnitude (m5) column name
        Default : fiveSigmaDepth
    season : `list` [`int`] or None, optional
        season to process (default: None: all seasons)
        A list with [-1] processes all seasons, as does None.
    nfilters_min : `int`, optional
        The number of filters to demand in a season
        Default: 4.
    min_season_obs : `int`, optional
        Minimum number of observations per season. Default 5.
    m5mins : `dict`, optional
        Minimum individual image depth for visit to 'count'.
        Default None uses
        {'u': 22.7, 'g': 24.1, 'r': 23.7, 'i': 23.1, 'z': 22.2, 'y': 21.4}.
    maps : `list`, optional
        List of maps to use.
        Default is the dustmap, to reduce m5 limiting mags accordingly.

    Returns
    -------
    n_slsn : `float`
        Number of expected well-measured strongly lensed SN

    Notes
    -----
    The number of expected strongly lensed SN detections with a
    well-measured time delay is given by:

    N (lensed SNe Ia with well measured time delay) = 45.7 *
    survey_area / (20000 deg^2) *
    cumulative_season_length / (2.5 years) /
    (2.15 * exp(0.37 * gap_median_all_filter))

    where:
    survey_area: survey area (in deg2)
    cumulative_season_length: cumulative season length (in years)
    gap_median_all_filter: median gap (all filters) (in days)

    (reference? metric originated from Simon Huber and Phillipe Gris)
    """

    def __init__(
        self,
        metric_name="SNSLMetric",
        mjd_col="observationStartMJD",
        filter_col="filter",
        night_col="night",
        m5_col="fiveSigmaDepth",
        season=None,
        nfilters_min=4,
        min_season_obs=5,
        m5mins=None,
        maps=["DustMap"],
        **kwargs,
    ):
        self.mjd_col = mjd_col
        self.filter_col = filter_col
        self.night_col = night_col
        self.m5_col = m5_col
        self.maps = maps

        cols = [self.night_col, self.filter_col, self.mjd_col, self.m5_col]
        super().__init__(col=cols, metric_name=metric_name, maps=self.maps, units="N SL", **kwargs)

        self.bad_val = 0
        if season is None:
            self.season = [-1]
        else:
            self.season = season
        self.bands = "ugrizy"
        if m5mins is None:
            self.m5mins = {
                "u": 22.7,
                "g": 24.1,
                "r": 23.7,
                "i": 23.1,
                "z": 22.2,
                "y": 21.4,
            }
        else:
            self.m5mins = m5mins
        self.min_season_obs = min_season_obs
        self.nfilters_min = nfilters_min
        # Set up dust-extinction values to use to interpret the dust map.
        self.phot_properties = DustValues()

    def n_lensed(self, area, cadence, season_length):
        """Estimate the number of lensed supernovae.

        Parameters
        -----------
        area : `float`
            Area in square degrees related to this data_slice (sq deg)
        gap_median : `float`
            median gap between nights with visits (days) - any filter
        cumul_season : `float`
            length of the season or period of consideration (years)

        Returns
        -------
        n_lensed_s_ne__ia : `float`
            Number of strongly lensed SN expected in this area
        """
        # estimate the number of lensed supernovae
        n_lensed_s_ne__ia = 45.7 * area / 20000.0 * season_length / 2.5 / (2.15 * np.exp(0.37 * cadence))
        return n_lensed_s_ne__ia

    def run(self, data_slice, slice_point=None):
        """
        Runs the metric for each data_slice

        Parameters
        ---------------
        data_slice : simulation data
        slice_point:  slice_point(default None)

        Returns
        -----------
        number of SL time delay supernovae

        """
        # If we had no incoming data - just return with badVal.
        if len(data_slice) == 0:
            return self.bad_val

        # Crop it down so things are coadded per night per
        # filter at the median MJD time
        night_slice = collapse_night(
            data_slice,
            night_col=self.night_col,
            filter_col=self.filter_col,
            m5_col=self.m5_col,
            mjd_col=self.mjd_col,
        )
        # Calculate the dust extinction-corrected m5 values
        # and cut visits which don't meet self.m5mins
        for f in np.unique(night_slice[self.filter_col]):
            in_filt = np.where(night_slice[self.filter_col] == f)[0]
            a_x = self.phot_properties.ax1[f] * slice_point["ebv"]
            night_slice[self.m5_col][in_filt] = night_slice[self.m5_col][in_filt] - a_x
            # Set the visits which fall below the minimum
            # to an obvious non-valid value
            night_slice[self.m5_col][in_filt] = np.where(
                night_slice[self.m5_col][in_filt] > self.m5mins[f],
                night_slice[self.m5_col][in_filt],
                -999,
            )
        idxs = np.where(night_slice[self.m5_col] > -998)
        # If nothing survived these cuts, just return with badVal.
        if len(idxs[0]) == 0:
            return self.badval

        # Reset, with coadded per-night/per-filter values,
        # skipping any too-shallow visits.
        night_slice = np.sort(night_slice[idxs], order=self.mjd_col)

        # get the pixel area
        area = hp.nside2pixarea(slice_point["nside"], degrees=True)

        # Note that 'seasons' is the same length as night_slice,
        # and contains integer (season) + float (day)
        seasons = calc_season(np.degrees(slice_point["ra"]), night_slice[self.mjd_col])
        season_ints = np.floor(seasons)

        if self.season == [-1]:
            season_loop = np.unique(season_ints)
        else:
            season_loop = self.season

        n_lensed_s_ne__ia = 0
        for s in season_loop:
            s_idx = np.where(season_ints == s)[0]
            u_filters = np.unique(night_slice[s_idx][self.filter_col])
            if (len(s_idx) < self.min_season_obs) | (np.size(u_filters) < self.nfilters_min):
                # Skip this season
                n_lensed_s_ne__ia += 0
            else:
                # Find the cadence (days) between visits within the season
                cadence = np.diff(night_slice["observationStartMJD"][s_idx])
                # But only the values between nights, not within nights
                cadence = np.median(cadence[np.where(cadence > 0.4)])
                # Season length in years
                season_length = seasons[s_idx][-1] - seasons[s_idx][0]
                n_lensed_s_ne__ia += self.n_lensed(area, cadence, season_length)

        return n_lensed_s_ne__ia
