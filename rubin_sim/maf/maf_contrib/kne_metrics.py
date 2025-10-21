__all__ = ("get_kne_filename", "KnLc", "KNePopMetric", "generate_kn_pop_slicer")

import glob
import os

import numpy as np
from rubin_scheduler.data import get_data_dir
from rubin_scheduler.utils import SURVEY_START_MJD, uniform_sphere

from rubin_sim.maf.metrics import BaseMetric
from rubin_sim.maf.slicers import UserPointsSlicer
from rubin_sim.maf.utils import m52snr
from rubin_sim.phot_utils import DustValues


def get_kne_filename(inj_params_list=None):
    """Given kilonova parameters, get the filename from the grid of models
    developed by M. Bulla

    Parameters
    ----------
    inj_params_list : `list` [`dict`]
        parameters for the kilonova model such as
        mass of the dynamical ejecta (mej_dyn), mass of the disk wind ejecta
        (mej_wind), semi opening angle of the cylindrically-symmetric ejecta
        fan ('phi'), and viewing angle ('theta'). For example
        inj_params_list =
        [{'mej_dyn': 0.005, 'mej_wind': 0.050, 'phi': 30, 'theta': 25.8}]
    """
    # Get files, model grid developed by M. Bulla
    datadir = get_data_dir()
    file_list = glob.glob(os.path.join(datadir, "maf", "bns", "*.dat"))

    # If no specific parameters passed - return everything.
    if inj_params_list is None or len(inj_params_list) == 0:
        return file_list

    # Otherwise find the parameters for each file and
    # then find the relevant matches.
    params = {}
    matched_files = []
    for filename in file_list:
        key = filename.replace(".dat", "").split("/")[-1]
        params[key] = {}
        params[key]["filename"] = filename
        key_split = key.split("_")
        # Binary neutron star merger models
        if key_split[0] == "nsns":
            mejdyn = float(key_split[2].replace("mejdyn", ""))
            mejwind = float(key_split[3].replace("mejwind", ""))
            phi0 = float(key_split[4].replace("phi", ""))
            theta = float(key_split[5])
            params[key]["mej_dyn"] = mejdyn
            params[key]["mej_wind"] = mejwind
            params[key]["phi"] = phi0
            params[key]["theta"] = theta
        # Neutron star--black hole merger models
        elif key_split[0] == "nsbh":
            mej_dyn = float(key_split[2].replace("mejdyn", ""))
            mej_wind = float(key_split[3].replace("mejwind", ""))
            phi = float(key_split[4].replace("phi", ""))
            theta = float(key_split[5])
            params[key]["mej_dyn"] = mej_dyn
            params[key]["mej_wind"] = mej_wind
            params[key]["phi"] = phi
            params[key]["theta"] = theta
    for key in params.keys():
        for inj_params in inj_params_list:
            match = all([np.isclose(params[key][var], inj_params[var]) for var in inj_params.keys()])
            if match:
                matched_files.append(params[key]["filename"])
                print(f"Found match for {inj_params}")
    print(
        f"Found matches for {len(matched_files)}/{len(inj_params_list)} \
          sets of parameters"
    )

    return matched_files


class KnLc:
    """Read in some KNe lightcurves

    Parameters
    ----------
    file_list : `list` [`str`] or None
        List of file paths to load. If None, loads up all the files
        from data/bns/
    """

    def __init__(self, file_list=None):
        if file_list is None:
            datadir = get_data_dir()
            # Get files, model grid developed by M. Bulla
            file_list = glob.glob(os.path.join(datadir, "maf", "bns", "*.dat"))

        filts = ["u", "g", "r", "i", "z", "y"]
        magidxs = [1, 2, 3, 4, 5, 6]

        # Let's organize the data in to a list of dicts for easy lookup
        self.data = []
        for filename in file_list:
            mag_ds = np.loadtxt(filename)
            t = mag_ds[:, 0]
            new_dict = {}
            for ii, (filt, magidx) in enumerate(zip(filts, magidxs)):
                new_dict[filt] = {"ph": t, "mag": mag_ds[:, magidx]}
            self.data.append(new_dict)

    def interp(self, t, filtername, lc_indx=0):
        """Do the interpolation of the lightcurve for a given time and filter.

        Parameters
        ----------
        t : `np.ndarray`, (N,)
            The times to interpolate the light curve to.
        filtername : `str`
            The filter. one of ugrizy
        lc_index : `int`, optional
            Which file to use.

        Returns
        -------
        result : `np.ndarray`, (N,)
            Array of lightcurve brightnesses at the times of t.=
        """

        result = np.interp(
            t,
            self.data[lc_indx][filtername]["ph"],
            self.data[lc_indx][filtername]["mag"],
            left=99,
            right=99,
        )
        return result


class KNePopMetric(BaseMetric):
    def __init__(
        self,
        metric_name="KNePopMetric",
        mjd_col="observationStartMJD",
        m5_col="fiveSigmaDepth",
        filter_col="filter",
        night_col="night",
        pts_needed=2,
        file_list=None,
        mjd0=SURVEY_START_MJD,
        output_lc=False,
        badval=-666,
        **kwargs,
    ):
        maps = ["DustMap"]
        self.mjd_col = mjd_col
        self.m5_col = m5_col
        self.filter_col = filter_col
        self.night_col = night_col
        self.pts_needed = pts_needed
        # `bool` variable, if True the light curve will be exported
        self.output_lc = output_lc

        self.lightcurves = KnLc(file_list=file_list)

        self.mjd0 = mjd0

        dust_properties = DustValues()
        self.ax1 = dust_properties.ax1

        cols = [self.mjd_col, self.m5_col, self.filter_col, self.night_col]
        super(KNePopMetric, self).__init__(
            col=cols,
            units="Detected, 0 or 1",
            metric_name=metric_name,
            maps=maps,
            badval=badval,
            **kwargs,
        )

    def _multi_detect(self, around_peak):
        """Simple detection criteria:
        detect at least a certain number of times
        """
        result = 1
        # Detected data points
        if np.size(around_peak) < self.pts_needed:
            return 0

        return result

    def _ztfrest_simple(
        self,
        around_peak,
        mags,
        mags_unc,
        t,
        filters,
        min_dt=0.125,
        min_fade=0.3,
        max_rise=-1.0,
        select_red=False,
        select_blue=False,
    ):
        """Selection criteria based on rise or decay rate;
        simplified version of the methods employed by the ZTFReST project
        (Andreoni & Coughlin et al., 2021)

        Parameters
        ----------
        around_peak : `np.ndarray`, (N,)
            indexes corresponding to 5sigma detections
        mags : `np.ndarray`, (N,)
            magnitudes obtained interpolating models on the data_slice
        t : `np.ndarray`, (N,)
            relative times
        filters : `np.ndarray`, (N,)
            filters in which detections happened
        min_dt : `float`
            minimum time gap between first and last detection in a given band
        min_fade : `float`
            fade rate threshold (positive, mag/day)
        max_rise : `float`
            rise rate threshold (negative, mag/day)
        select_red : `bool`
            if True, only red 'izy' filters will be considered
        select_blue : `bool`
            if True, only blue 'ugr' filters will be considered

        Examples
        ----------
        A transient:
            rising by 0.74 mag/day will pass a threshold max_rise=-0.5
            rising by 0.74 mag/day will not pass a threshold max_rise=-1.0
            fading by 0.6 mag/day will pass a threshold min_fade=0.3
            fading by 0.2 mag/day will not pass a threshold min_fade=0.3
        """
        result = 1

        # Quick check on the number of detected points
        if np.size(around_peak) < self.pts_needed:
            return 0
        # Quick check on the time gap between first and last detection
        elif np.max(t[around_peak]) - np.min(t[around_peak]) < min_dt:
            return 0
        else:
            evol_rate = []
            fil = []
            # Check time gaps and rise or fade rate for each band
            for f in set(filters):
                if select_red is True and f not in "izy":
                    continue
                elif select_blue is True and f not in "ugr":
                    continue
                times_f = t[around_peak][np.where(filters == f)[0]]
                mags_f = mags[around_peak][np.where(filters == f)[0]]
                mags_unc_f = mags_unc[around_peak][np.where(filters == f)[0]]

                # Check if the evolution is significant enough
                idx_max = np.argmax(mags_f)
                idx_min = np.argmin(mags_f)
                if mags_f[idx_min] + mags_unc_f[idx_min] < mags_f[idx_max] - mags_unc_f[idx_max]:
                    signif = True
                else:
                    signif = False

                # Time difference between max and min
                dt_f = np.abs(times_f[idx_max] - times_f[idx_min])

                # Get the evolution rate, if the time gap condition is met
                if dt_f > min_dt and signif is True:
                    # Calculate evolution rate
                    evol_rate_f = (np.max(mags_f) - np.min(mags_f)) / (times_f[idx_max] - times_f[idx_min])
                    evol_rate.append(evol_rate_f)
                else:
                    evol_rate.append(0)
                fil.append(f)
            if len(evol_rate) == 0:
                return 0
            # Check if the conditions on the evolution rate are met
            if np.max(evol_rate) < min_fade and np.min(evol_rate) > max_rise:
                return 0

        return result

    def _multi_color_detect(self, filters):
        """Color-based simple detection criteria:
        detect at least twice, with at least two filters
        """
        result = 1
        # detected in at least two filters
        if np.size(np.unique(filters)) < 2:
            return 0

        return result

    def _red_color_detect(self, filters, min_det=4):
        """Detected at least min_det times in either izy colors

        Parameters
        ----------
        filters : `np.ndarray`, (N,)
            filters in which detections happened
        min_det : `float` or `int`
            minimum number of detections required in izy bands
        """
        result = 1
        # Number of detected points in izy bands
        n_red_det = (
            np.size(np.where(filters == "i")[0])
            + np.size(np.where(filters == "z")[0])
            + np.size(np.where(filters == "y")[0])
        )
        # Condition
        if n_red_det < min_det:
            return 0

        return result

    def _blue_color_detect(self, filters, min_det=4):
        """Detected at least min_det times in either ugr colors

        Parameters
        ----------
        filters : `np.ndarray`, (N,)
            filters in which detections happened
        min_det : `float` or `int`
            minimum number of detections required in ugr bands
        """
        result = 1
        # Number of detected points in ugr bands
        n_blue_det = (
            np.size(np.where(filters == "u")[0])
            + np.size(np.where(filters == "g")[0])
            + np.size(np.where(filters == "r")[0])
        )
        # Condition
        if n_blue_det < min_det:
            return 0

        return result

    def run(self, data_slice, slice_point=None):
        result = {}
        t = data_slice[self.mjd_col] - self.mjd0 - slice_point["peak_time"]
        mags = np.zeros(t.size, dtype=float)

        for filtername in np.unique(data_slice[self.filter_col]):
            infilt = np.where(data_slice[self.filter_col] == filtername)
            mags[infilt] = self.lightcurves.interp(t[infilt], filtername, lc_indx=slice_point["file_indx"])
            # Apply dust extinction on the light curve
            a_x = self.ax1[filtername] * slice_point["ebv"]
            mags[infilt] += a_x

            distmod = 5 * np.log10(slice_point["distance"] * 1e6) - 5.0
            mags[infilt] += distmod

        # Find the detected points
        around_peak = np.where((t > 0) & (t < 30) & (mags < data_slice[self.m5_col]))[0]
        # Filters in which the detections happened
        filters = data_slice[self.filter_col][around_peak]
        # Magnitude uncertainties with Gaussian approximation
        snr = m52snr(mags, data_slice[self.m5_col])
        mags_unc = 2.5 * np.log10(1.0 + 1.0 / snr)

        result["multi_detect"] = self._multi_detect(around_peak)
        result["ztfrest_simple"] = self._ztfrest_simple(
            around_peak, mags, mags_unc, t, filters, select_red=False
        )
        result["ztfrest_simple_red"] = self._ztfrest_simple(
            around_peak, mags, mags_unc, t, filters, select_red=True
        )
        result["ztfrest_simple_blue"] = self._ztfrest_simple(
            around_peak, mags, mags_unc, t, filters, select_blue=True
        )
        result["multi_color_detect"] = self._multi_color_detect(filters)
        result["red_color_detect"] = self._red_color_detect(filters)
        result["blue_color_detect"] = self._blue_color_detect(filters)

        # Export the light curve
        if self.output_lc is True:
            mags[np.where(mags > 50)[0]] = 99.0
            result["lc"] = [
                data_slice[self.mjd_col],
                mags,
                mags_unc,
                data_slice[self.m5_col],
                data_slice[self.filter_col],
            ]
            result["lc_colnames"] = ("t", "mag", "mag_unc", "maglim", "filter")

        return result

    def reduce_multi_detect(self, metric):
        return metric["multi_detect"]

    def reduce_ztfrest_simple(self, metric):
        return metric["ztfrest_simple"]

    def reduce_ztfrest_simple_red(self, metric):
        return metric["ztfrest_simple_red"]

    def reduce_ztfrest_simple_blue(self, metric):
        return metric["ztfrest_simple_blue"]

    def reduce_multi_color_detect(self, metric):
        return metric["multi_color_detect"]

    def reduce_red_color_detect(self, metric):
        return metric["red_color_detect"]

    def reduce_blue_color_detect(self, metric):
        return metric["blue_color_detect"]


def generate_kn_pop_slicer(
    t_start=1,
    t_end=3652,
    n_events=10000,
    seed=42,
    n_files=308,
    d_min=10,
    d_max=300,
    ra=None,
    dec=None,
):
    """Generate a population of KNe events, and put the info about them
    into a UserPointSlicer object

    Parameters
    ----------
    t_start : `float`, optional
        The night to start kilonova events on (days)
    t_end : `float`, optional
        The final night of kilonova events
    n_events : `int`, optional
        The number of kilonova events to generate
    seed : `float`, optional
        The seed passed to np.random
    n_files : `int`, optional
        The number of different kilonova lightcurves to use
        This should match the length of the filenames list passed
        to the KNePopMetric directly.
    d_min : `float` or `int`, optional
        Minimum luminosity distance (Mpc)
    d_max : `float` or `int`, optional
        Maximum luminosity distance (Mpc)
    ra, dec : `np.ndarray`, (N,) or None
        The ra and dec to use for event positions.
        Generates uniformly on the spehere if None. (degrees)
    """

    def rndm(a, b, g, size=1):
        """Power-law gen for pdf(x) propto x^{g-1} for a<=x<=b"""
        r = np.random.random(size=size)
        ag, bg = a**g, b**g
        return (ag + (bg - ag) * r) ** (1.0 / g)

    if ra is None:
        ra, dec = uniform_sphere(n_events, seed=seed)

    peak_times = np.random.uniform(low=t_start, high=t_end, size=n_events)
    file_indx = np.floor(np.random.uniform(low=0, high=n_files, size=n_events)).astype(int)

    # Define the distance
    distance = rndm(d_min, d_max, 4, size=n_events)

    # Set up the slicer to evaluate the catalog we just made
    slicer = UserPointsSlicer(ra, dec, lat_lon_deg=True, badval=0)
    # Add any additional information about each object to the slicer
    slicer.slice_points["peak_time"] = peak_times
    slicer.slice_points["file_indx"] = file_indx
    slicer.slice_points["distance"] = distance

    return slicer
