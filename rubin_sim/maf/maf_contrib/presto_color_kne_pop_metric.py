__all__ = ("PrestoColorKNePopMetric", "generate_presto_pop_slicer")

import os
import pickle
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from rubin_scheduler.data import get_data_dir
from rubin_scheduler.utils import SURVEY_START_MJD, uniform_sphere

import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers
from rubin_sim.phot_utils import DustValues

from .kne_metrics import KnLc


def radec2gal(ra, dec):
    """convert from ra/dec to galactic l/b"""
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    c = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree))

    gal_l = c.galactic.l.degree
    gal_b = c.galactic.b.degree
    return gal_l, gal_b


def _load_hash(
    file_galactic="TotalCubeNorm_1000Obj.pkl",
    file_extragalactic="TotalCubeNorm_1000Obj.pkl",
    skyregion="extragalactic",
):
    """Helper function to load large hash table.

    Because this is kept outside the metric attributes, it allows easy resuse.
    Note that this does mean running different sky regions at the same
    time may end up thrashing the data in the hash/hash table.

    Parameters
    ----------
    skyregion : `str`
        The skyregion of interst.
        Only two options: 'galactic' and 'extragalactic'
    filePathGalactic : `str`
        File containing galactic Presto-Color phase space information
    filePathExtragalactic : `str`
        File containing galactic Presto-Color phase space information
    """

    if hasattr(_load_hash, "InfoDict"):
        if skyregion == _load_hash.skyregion:
            return _load_hash.InfoDict, _load_hash.HashTable

    data_dir = get_data_dir()
    if skyregion == "galactic":
        file_path = os.path.join(data_dir, "maf", file_galactic)
    elif skyregion == "extragalactic":
        file_path = os.path.join(data_dir, "maf", file_extragalactic)

    with open(file_path, "rb") as f:
        _load_hash.InfoDict = pickle.load(f)
        _load_hash.HashTable = pickle.load(f)
    _load_hash.skyregion = skyregion
    return _load_hash.InfoDict, _load_hash.HashTable


def generate_presto_pop_slicer(
    skyregion="galactic",
    t_start=1,
    t_end=3652,
    n_events=10000,
    seed=42,
    n_files=100,
    d_min=10,
    d_max=300,
    gb_cut=20,
):
    """Generate a population of KNe events, and put the info about them
    into a UserPointSlicer object

    Parameters
    ----------
    skyregion : `str`
        The skyregion of interest.
        Only two options: 'galactic' and 'extragalactic'
    t_start : `float`
        The night to start kilonova events on (days)
    t_end : `float`
        The final night of kilonova events
    n_events : `int`
        The number of kilonova events to generate
    seed : `float`
        The seed passed to np.random
    n_files : `int`
        The number of different kilonova lightcurves to use
    d_min : `float` or `int`
        Minimum luminosity distance (Mpc)
    d_max : `float` or `int`
        Maximum luminosity distance (Mpc)

    Returns
    -------
    kne_slicer : `~.maf.UserPointsSlicer`
    """

    def rndm(a, b, g, size=1):
        """Power-law gen for pdf(x) proportional to x^{g-1} for a<=x<=b"""
        r = np.random.random(size=size)
        ag, bg = a**g, b**g
        return (ag + (bg - ag) * r) ** (1.0 / g)

    ra, dec = uniform_sphere(n_events, seed=seed)

    # Convert ra, dec to gl, gb
    gl, gb = radec2gal(ra, dec)

    # Determine if the object is in the Galaxy plane
    if skyregion == "galactic":  # keep the galactic events
        ra = ra[np.abs(gb) < gb_cut]
        dec = dec[np.abs(gb) < gb_cut]
    elif skyregion == "extragalactic":  # keep the extragalactic events.
        ra = ra[np.abs(gb) > gb_cut]
        dec = dec[np.abs(gb) > gb_cut]
    else:
        warnings.warn("Skyregion %s not recognized, using whole sky" % skyregion)

    n_events = len(ra)

    peak_times = np.random.uniform(low=t_start, high=t_end, size=n_events)
    file_indx = np.floor(np.random.uniform(low=0, high=n_files, size=n_events)).astype(int)

    # Define the distance
    distance = rndm(d_min, d_max, 4, size=n_events)

    # Set up the slicer to evaluate the catalog we just made
    slicer = slicers.UserPointsSlicer(ra, dec, lat_lon_deg=True, badval=0)
    # Add any additional information about each object to the slicer
    slicer.slice_points["peak_time"] = peak_times
    slicer.slice_points["file_indx"] = file_indx
    slicer.slice_points["distance"] = distance
    return slicer


class PrestoColorKNePopMetric(metrics.BaseMetric):
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
        skyregion="galactic",
        thr=0.003,
        **kwargs,
    ):
        """
        Parameters
        ----------
        file_list : `str` or None, optional
            File containing input lightcurves
        mjd0 : `float`, optional
            MJD of the start of the survey.
        output_lc : `bool`, optional
            Flag to whether or not to output lightcurve for each object.
        skyregion : `str`, optional
            The skyregion of interest.
            Only two options: 'galactic' and 'extragalactic'
        thr : `float`, optional
            Threshold for "classification" of events via the Score_S
        """
        maps = ["DustMap"]
        self.mjd_col = mjd_col
        self.m5_col = m5_col
        self.filter_col = filter_col
        self.night_col = night_col
        # Boolean variable, if True the light curve will be exported
        self.output_lc = output_lc
        self.thr = thr
        self.skyregion = skyregion
        # read in file as light curve object;
        self.lightcurves = KnLc(file_list=file_list)

        self.mjd0 = mjd0

        dust_properties = DustValues()
        self.ax1 = dust_properties.ax1

        cols = [self.mjd_col, self.m5_col, self.filter_col, self.night_col]
        super().__init__(col=cols, units="Detected, 0 or 1", metric_name=metric_name, maps=maps, **kwargs)

        # Unused ..
        self.pts_needed = pts_needed

    def _presto_color_detect(self, around_peak, filters):
        """Detection criteria of presto cadence:
        at least three detections at two filters;

        Parameters
        ----------
        around_peak : `np.ndarray`, (N,)
            indexes corresponding to 5sigma detections
        filters : `np.ndarray`, (N,)
            filters in which detections happened
        """
        result = 1

        if np.size(around_peak) < 3:
            result = 0

        flts, flts_count = np.unique(
            filters,
            return_counts=True,
        )
        if np.size(flts) < 2:
            result = 0
        elif np.max(flts_count) < 2:
            # if no filters have visits larger than 2, set detection false
            result = 0

        return result

    def _enquiry(self, hash_table, info_dict, band1, band2, d_t1, d_t2, d_mag, color):
        """
        Return the value in the probability cube provided the coordinates
        in the Presto-color phase space of an observation triplet.

        Parameters
        ----------
        hash_table : `np.ndarray`, (N,)
            Contains the values of the 6-D Presto-color phase space
        info_dict : `dict`
            Contains the essential information of the hash_table abobe.
        band1, band2 : `str`, `str`
            The two filters that comprise the Presto-color observation triplet.
            The filters are the 6 bands of LSST: u, g, r, i, z, y.
            Band1 and band2 should be different.
        d_t1, d_t2 : `float`, `float`
            The time gaps of the Presto-color observation triplet.
        d_mag : `float`
            The magnitude change between from the observations of the same band
        color : `float`
            The difference in magnitude of observations in different bands.

        hash_table and info_dict have to be loaded from premade data
        Presto-color data file.
        """

        #         if abs(d_t1) > abs(d_t1-d_t2):
        #             d_t1, d_t2 = d_t1-d_t2, -d_t2

        if not (
            info_dict["BinMag"][0] <= d_mag < info_dict["BinMag"][-1]
            and info_dict["BinColor"][0] <= color < info_dict["BinColor"][-1]
        ):
            return 0

        ind1 = info_dict["BandPairs"].index(band1 + band2)

        time_pair_grid = [
            info_dict["dT1s"][abs(d_t1 - info_dict["dT1s"]).argmin()],
            info_dict["dT2s"][abs(d_t2 - info_dict["dT2s"]).argmin()],
        ]

        ind2 = np.where((info_dict["TimePairs"] == time_pair_grid).all(axis=1))[0][0]
        ind3 = np.where(d_mag >= info_dict["BinMag"])[0][-1]
        ind4 = np.where(color >= info_dict["BinColor"])[0][-1]

        return hash_table[ind1, ind2, ind3, ind4]

    def _get_score(self, result, hash_table, info_dict, thr):
        """Get the score of a strategy from the Presto-color perspective.

        Parameters
        ----------
        result : `pd.DataFrame`
            Dataframe that contains the results of the observations.
            The columns include
            t: the time of the observation
            mag: the detected magnitude
            maglim: the limit fiveSigmaDepth that can be detected
            filter: the filter used for the observation
        hash_table : `np.ndarray`, (N,)
            Contains the values of the 6-D Presto-color phase space
        info_dict : `dict`
            Contains the essential information of the hash_table abobe.
        scoreType : `str`
            Two types of scores were designed:
            'S' type involves a threshold,
            'P' type work without a threshold.
        thr : `float`
            The threashold need for type 'S' score.
            The default value is 0.003 (3-sigma)

        hash_table and info_dict have to be loaded from the premade
        Presto-color data file.
        """

        time_lim1 = 8.125 / 24  # 8 h 7.5 min
        time_lim2 = 32.25 / 24  # 32 h 15 min

        detects = result[result.mag < result.maglim]

        # reset index
        detects = detects.reset_index(drop=True)

        # Times for valid detections
        ts = detects.t.values
        # Find out the differences between each pair
        d_ts = ts.reshape(1, len(ts)) - ts.reshape(len(ts), 1)

        # The time differences should be within 32 hours (2 nights)
        d_tindex0, d_tindex1 = np.where(abs(d_ts) < time_lim2)

        phase_space_coords = []

        # loop through the rows of the matrix of valid time differences
        for ii in range(d_ts.shape[0]):
            groups_of_three = np.array(
                [
                    [ii] + list(jj)
                    for jj in list(combinations(d_tindex1[(d_tindex0 == ii) * (d_tindex1 > ii)], 2))
                ]
            )

            for indices in groups_of_three:
                bands = detects["filter"][indices].values

                # print('bands: ', bands)
                if len(np.unique(bands)) != 2:
                    continue

                # The band appears once will be band2
                occurence = np.array([np.count_nonzero(ii == bands) for ii in bands])
                # The index of observation in band2
                index2 = indices[occurence == 1][0]
                # The index of the first observation in band1
                index11 = indices[occurence == 2][0]
                # The index of the second observation in band1
                index12 = indices[occurence == 2][1]

                if (
                    abs(d_ts[index12, index2]) < abs(d_ts[index11, index2])
                    and abs(d_ts[index12, index2]) < time_lim1
                ):
                    index11, index12 = index12, index11
                elif abs(d_ts[index11, index2]) > time_lim1:
                    continue

                d_t1 = d_ts[index11, index2]
                d_t2 = d_ts[index11, index12]

                band1 = bands[occurence == 2][0]
                band2 = bands[occurence == 1][0]

                if band1 + band2 == "uy" or band1 + band2 == "yu":
                    continue

                d_mag = (detects.mag[index11] - detects.mag[index12]) * np.sign(d_t2)
                color = detects.mag[index11] - detects.mag[index2]

                phase_space_coords.append([band1, band2, d_t1, d_t2, d_mag, color])

        score_s = 0
        score_p = [0]

        for phase_space_coord in phase_space_coords:
            rate = self._enquiry(hash_table, info_dict, *phase_space_coord)

            if score_s == 0 and rate < thr:
                score_s = 1

            score_p.append((1 - rate))

        return score_s, max(score_p)

    def run(self, data_slice, slice_point=None):
        data_slice.sort(order=self.mjd_col)
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

        # presto color
        result["presto_color_detect"] = self._presto_color_detect(around_peak, filters)

        # Export the light curve
        idx = np.where(mags < 100)[0]
        lc = {
            "t": data_slice[self.mjd_col][idx],
            "mag": mags[idx],
            "maglim": data_slice[self.m5_col][idx],
            "filter": data_slice[self.filter_col][idx],
        }

        if self.output_lc is True:
            result["lc"] = lc
            result["slice_point"] = slice_point

        if result["presto_color_detect"] == 1:
            info_dict, hash_table = _load_hash(skyregion=self.skyregion)
            result["scoreS"], result["scoreP"] = self._get_score(
                pd.DataFrame(lc),
                hash_table=hash_table,
                info_dict=info_dict,
                thr=self.thr,
            )
        else:
            result["scoreS"] = 0
            result["scoreP"] = 0
        return result

    def reduce_presto_color_detect(self, metric):
        return metric["presto_color_detect"]

    def reduce_score_s(self, metric):
        return metric["scoreS"]

    def reduce_score_p(self, metric):
        return metric["scoreP"]
