__all__ = ("TdeLc", "TdePopMetric")

import glob
import os

import numpy as np
from rubin_scheduler.data import get_data_dir
from rubin_scheduler.utils import SURVEY_START_MJD, uniform_sphere

from rubin_sim.maf_proto.utils import eb_v_hp
from rubin_sim.phot_utils import DustValues

from .metrics import BaseMetric


class TdeLc:
    """
    Read in some TDE lightcurves

    Parameters
    ----------
    file_list : `list` [`str`], opt
        List of file paths to load.
        If None, loads up all the files from $RUBIN_SIM_DATA/maf/tde/
    """

    def __init__(self, file_list=None):
        if file_list is None:
            sims_maf_contrib_dir = get_data_dir()
            file_list = glob.glob(os.path.join(sims_maf_contrib_dir, "maf/tde/*.dat"))

        lcs = []
        for filename in file_list:
            lcs.append(np.genfromtxt(filename, dtype=[("ph", "f8"), ("mag", "f8"), ("filter", "U1")]))

        # Let's organize the data in to a list of dicts for easy lookup
        self.data = []
        filternames = "ugrizy"
        for lc in lcs:
            new_dict = {}
            for filtername in filternames:
                infilt = np.where(lc["filter"] == filtername)
                new_dict[filtername] = lc[infilt]
            self.data.append(new_dict)

    def interp(self, t, filtername, lc_indx=0):
        """
        t : array of floats
            The times to interpolate the light curve to.
        filtername : str
            The filter. one of ugrizy
        lc_index : int (0)
            Which file to use.
        """

        result = np.interp(
            t,
            self.data[lc_indx][filtername]["ph"],
            self.data[lc_indx][filtername]["mag"],
            left=99,
            right=99,
        )
        return result


class TdePopMetric(BaseMetric):
    """ """

    def __init__(
        self,
        name="TDEsPopMetric",
        mjd_col="observationStartMJD",
        m5_col="fiveSigmaDepth",
        filter_col="filter",
        night_col="night",
        pts_needed=2,
        file_list=None,
        tmin=-30,
        tmax=100,
        mjd0=SURVEY_START_MJD,
        dust_nside=128,
        badval=0,
        **kwargs,
    ):
        self.mjd_col = mjd_col
        self.m5_col = m5_col
        self.filter_col = filter_col
        self.night_col = night_col
        self.pts_needed = pts_needed
        self.tmin = tmin
        self.tmax = tmax

        self.lightcurves = TdeLc(file_list=file_list)
        self.mjd0 = mjd0

        dust_properties = DustValues()
        self.ax1 = dust_properties.ax1
        self.dust_nside = dust_nside

        super(TdePopMetric, self).__init__(unit="Detected, 0 or 1", name=name, **kwargs)

        names = [
            "pre_peak",
            "some_color",
            "some_color_pu",
            "some_color_pnum",
            "some_color_pu_pnum",
        ]
        types = [float] * len(names)
        self.shape = None
        self.dtype = list(zip(names, types))
        self.empty = np.empty(1, dtype=self.dtype)
        self.badval = np.empty(1, dtype=self.dtype)
        self.badval.fill(badval)

    def _some_color_pnum_detect(self, data_slice, mags, t):
        # 1 detection pre peak
        pre_peak_detected = np.where((t < -10) & (mags < data_slice[self.m5_col]))[0]
        if np.size(pre_peak_detected) < 1:
            return 0

        # At least 3 filters within 10 days of peak
        around_peak = np.where((np.abs(t) < 5) & (mags < data_slice[self.m5_col]))[0]
        if np.size(np.unique(data_slice[self.filter_col][around_peak])) < 3:
            return 0

        # At least 2 bands after peak
        post_peak = np.where((t > 10) & (t < 30) & (mags < data_slice[self.m5_col]))[0]
        if np.size(np.unique(data_slice[self.filter_col][post_peak])) < 2:
            return 0

        # count number of data points in the light curve
        obs_points = np.where((t > self.tmin) & (t < self.tmax) & (mags < data_slice[self.m5_col]))[0]

        # define the time range around peak in which the number of
        # data points is measured
        t_range = self.tmax - self.tmin

        # number of data points / time range gives a "score" for
        # light curve quality
        # 0: did not pass some_color requirements;
        # 1: passed some_color requirements and has 1 data point
        # every other night
        nresult = np.size(obs_points) / t_range

        return nresult

    def _some_color_pu_pnum_detect(self, data_slice, mags, t):
        # 1 detection pre peak
        pre_peak_detected = np.where((t < -10) & (mags < data_slice[self.m5_col]))[0]
        if np.size(pre_peak_detected) < 1:
            return 0

        # 1 detection in u and any other band near peak
        around_peak = np.where((np.abs(t) < 5) & (mags < data_slice[self.m5_col]))[0]
        filters = np.unique(data_slice[self.filter_col][around_peak])
        if np.size(filters) < 2:
            return 0
        if "u" not in filters:
            return 0

        # 1 detecion in u and any other band post peak
        post_peak = np.where((t > 10) & (t < 30) & (mags < data_slice[self.m5_col]))[0]
        filters = np.unique(data_slice[self.filter_col][post_peak])
        if np.size(filters) < 2:
            return 0
        if "u" not in filters:
            return 0

        # count number of data points in the light curve
        obs_points = np.where((t > self.tmin) & (t < self.tmax) & (mags < data_slice[self.m5_col]))[0]

        # define the time range around peak in which the number of
        # data points is measured
        t_range = self.tmax - self.tmin

        # number of data points / time range gives a "score" for
        # light curve quality
        # 0: did not pass some_color_pu requirements;
        # 1: passed some_color_pu requirements and has 1 data point
        # every other night
        nresult = np.size(obs_points) / t_range

        return nresult

    def _pre_peak_detect(self, data_slice, slice_point, mags, t):
        """
        Simple detection criteria
        """
        result = 0
        # Simple alert criteria.
        # Could make more in depth, or use reduce functions
        # to have multiple criteria checked.
        pre_peak_detected = np.where((t < 0) & (mags < data_slice[self.m5_col]))[0]

        if pre_peak_detected.size > self.pts_needed:
            result = 1
        return result

    def _some_color_detect(self, data_slice, slice_point, mags, t):
        result = 1
        # 1 detection pre peak
        pre_peak_detected = np.where((t < -10) & (mags < data_slice[self.m5_col]))[0]
        if np.size(pre_peak_detected) < 1:
            return 0

        # At least 3 filters within 10 days of peak
        around_peak = np.where((np.abs(t) < 5) & (mags < data_slice[self.m5_col]))[0]
        if np.size(np.unique(data_slice[self.filter_col][around_peak])) < 3:
            return 0

        # At least 2 bands after peak
        post_peak = np.where((t > 10) & (t < 30) & (mags < data_slice[self.m5_col]))[0]
        if np.size(np.unique(data_slice[self.filter_col][post_peak])) < 2:
            return 0

        return result

    def _some_color_pu_detect(self, data_slice, slice_point, mags, t):
        result = 1
        # 1 detection pre peak
        pre_peak_detected = np.where((t < -10) & (mags < data_slice[self.m5_col]))[0]
        if np.size(pre_peak_detected) < 1:
            return 0

        # 1 detection in u and any other band near peak
        around_peak = np.where((np.abs(t) < 5) & (mags < data_slice[self.m5_col]))[0]
        filters = np.unique(data_slice[self.filter_col][around_peak])
        if np.size(filters) < 2:
            return 0
        if "u" not in filters:
            return 0

        post_peak = np.where((t > 10) & (t < 30) & (mags < data_slice[self.m5_col]))[0]
        filters = np.unique(data_slice[self.filter_col][post_peak])
        if np.size(filters) < 2:
            return 0
        if "u" not in filters:
            return 0

        return result

    def __call__(self, data_slice, slice_point=None):
        indx = slice_point["sid"]
        result = self.empty.copy()
        t = data_slice[self.mjd_col] - self.mjd0 - self.peak_times[indx]
        mags = np.zeros(t.size, dtype=float)

        for filtername in np.unique(data_slice[self.filter_col]):
            infilt = np.where(data_slice[self.filter_col] == filtername)
            mags[infilt] = self.lightcurves.interp(t[infilt], filtername, lc_indx=self.file_indx[indx])
            # Apply dust extinction on the light curve
            extinction = eb_v_hp(self.dust_nside, ra=self.ra[indx], dec=self.dec[indx])
            mags[infilt] += self.ax1[filtername] * extinction

        result["pre_peak"] = self._pre_peak_detect(data_slice, slice_point, mags, t)
        result["some_color"] = self._some_color_detect(data_slice, slice_point, mags, t)
        result["some_color_pu"] = self._some_color_pu_detect(data_slice, slice_point, mags, t)
        result["some_color_pnum"] = self._some_color_pnum_detect(data_slice, mags, t)
        result["some_color_pu_pnum"] = self._some_color_pu_pnum_detect(data_slice, mags, t)

        return result

    def generate_tde_pop(self, t_start=1, t_end=3652, n_events=10000, seed=42, n_files=7):
        """Generate a population of TDE events,
        and put the info about them into a UserPointSlicer object.

        Parameters
        ----------
        t_start : `float`, opt
            The night to start tde events on (days)
        t_end : `float`, opt
            The final night of TDE events
        n_events : `int`, opt
            The number of TDE events to generate
        seed : `float`, opt
            The seed passed to np.random
        n_files : `int`, opt
            The number of different TDE lightcurves to use
        """

        self.ra, self.dec = uniform_sphere(n_events, seed=seed)
        self.ra = np.radians(self.ra)
        self.dec = np.radians(self.dec)
        self.peak_times = np.random.uniform(low=t_start, high=t_end, size=n_events)
        self.file_indx = np.floor(np.random.uniform(low=0, high=n_files, size=n_events)).astype(int)
