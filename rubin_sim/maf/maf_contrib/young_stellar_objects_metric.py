"""Young Stellar Objects metric.
Converted from notebook 211116_yso_3D_90b.ipynb.
Formatted with black."""

__all__ = ("NYoungStarsMetric",)

import healpy as hp
import numpy as np
import scipy.integrate as integrate

from rubin_sim.maf.maps import DustMap, DustMap3D, StellarDensityMap
from rubin_sim.maf.metrics import BaseMetric, CrowdingM5Metric
from rubin_sim.phot_utils import DustValues


class StarDensity:
    """integrate from zero to some max distance, then multiply by angular area

    Parameters
    ----------
    l : float
        Galactic longitude, radians
    b : float
        Galactic latitude, radians
    """

    def __init__(self, gall, galb):
        """Calculate the expected number of stars along a line of site"""
        self.r_thin = 2.6  # scale length of the thin disk, kpc
        self.d_gc = 8.178  # Distance to the galactic center, kpc
        self.h_thin = 0.300  # scale height of the thin disk, kpc

        self.gall = gall
        self.galb = galb

        self.A = 0.8e8 / (4.0 * np.pi * self.h_thin * self.r_thin**2)

    def __call__(self, r):
        """
        Parameters
        ----------
        r : float
            Distance in kpc
        """
        r_galac = ((self.d_gc - r * np.cos(self.gall)) ** 2 + (r * np.sin(self.gall)) ** 2) ** 0.5

        exponent = -1.0 * r * np.abs(np.sin(self.galb)) / self.h_thin - r_galac / self.r_thin

        result = self.A * r**2 * np.exp(exponent)
        return result


class NYoungStarsMetric(BaseMetric):
    """Calculate the distance or number of stars with
    color uncertainty defined by mags/snrs.

    Parameters
    ----------
    metric_name : `str`, opt
        Default 'young_stars'.
    m5_col : `str`, opt
        The default column name for m5 information in the input data.
        Default fiveSigmaDepth.
    filter_col : `str`, opt
        The column name for the filter information. Default filter.
    mags : `dict`, opt
        The absolute magnitude of the object in question.
        Keys of filter name, values in mags.
        Default is for a 0.3 solar mass star at age = 100 Myr.
    snrs : `dict`, opt
        The SNR to demand for each filter.
    galb_limit : `float`, opt
        The galactic latitude above which to return zero (degrees).
        Default 90.
    badval : `float`, opt
        The value to return when the metric value cannot be calculated.
        Default 0.
    return_distance : `bool`, opt
        Whether the metric will return the maximum distance that
        can be reached for each slice_point,
        or the total number of stars down to mags/snrs.
    crowding_error : `float`, opt
        Crowding error that gets passed to CrowdingM5Metric. Default 0.25.
    use_2D_extinction : `bool`, opt
        Uses the 2D extinction map instead of the 3D one. Default False.
    """

    def __init__(
        self,
        metric_name="young_stars",
        m5_col="fiveSigmaDepth",
        filter_col="filter",
        seeing_col="seeingFwhmGeom",
        mags={"g": 10.32, "r": 9.28, "i": 7.97},
        snrs={"g": 5.0, "r": 5.0, "i": 5.0},
        galb_limit=90.0,
        badval=0,
        return_distance=False,
        crowding_error=0.25,
        use_2D_extinction=False,
        **kwargs,
    ):
        cols = [m5_col, filter_col, seeing_col]
        maps = [
            DustMap3D(),
            StellarDensityMap(filtername="g"),
            StellarDensityMap(filtername="r"),
            StellarDensityMap(filtername="i"),
        ]
        self.use_2D_extinction = use_2D_extinction
        if self.use_2D_extinction:
            maps[0] = DustMap()
        # This will give us access to the dust map get_distance_at_dmag routine
        # but does not require loading another copy of the map
        self.ebvmap = DustMap3D()
        self.return_distance = return_distance
        units = "kpc" if self.return_distance else "N stars"
        super().__init__(cols, metric_name=metric_name, maps=maps, units=units, badval=badval, **kwargs)
        # Save R_x values for on-the-fly calculation of dust extinction
        self.r_x = DustValues().r_x.copy()
        # set return type
        self.m5_col = m5_col
        self.filter_col = filter_col
        self.galb_limit = np.radians(galb_limit)
        self.mags = mags
        self.filters = list(self.mags.keys())
        self.snrs = snrs
        self.crowding_error = crowding_error
        self.m5crowding = {
            f: CrowdingM5Metric(
                crowding_error=crowding_error,
                filtername=f,
                seeing_col=seeing_col,
                maps=maps,
            )
            for f in self.filters
        }
        dust_properties = DustValues()
        self.ax1 = dust_properties.ax1

    def run(self, data_slice, slice_point=None):
        # Evaluate area on sky for this slice_point, in radians
        if "nside" in slice_point:
            # Best area /pixel calculation, appropriate for healpix slicers
            sky_area = hp.nside2pixarea(slice_point["nside"], degrees=False)
        else:
            # Assume single, approximate, circular FOV
            sky_area = np.pi * (np.radians(1.75)) ** 2

        # if we are outside the galb_limit, return nothing
        # Note we could make this a more complicated function that
        # returns an expected density of star forming regions
        if np.abs(slice_point["galb"]) > self.galb_limit:
            return self.badval

        # Compute depth for each filter
        depths = {}
        # ignore the divide by zero warnings
        with np.errstate(divide="ignore"):
            for filtername in self.filters:
                in_filt = np.where(data_slice[self.filter_col] == filtername)[0]
                # Calculate coadded depth per filter
                depth_m5 = 1.25 * np.log10(np.sum(10.0 ** (0.8 * data_slice[self.m5_col][in_filt])))
                if self.crowding_error > 0:
                    depth_crowding = self.m5crowding[filtername].run(data_slice, slice_point)
                    depths[filtername] = min(depth_m5, depth_crowding)
                else:
                    depths[filtername] = depth_m5

        # solve for the distances in each filter where we hit the required SNR
        distances = []
        for filtername in self.filters:
            # Apparent magnitude at the SNR requirement
            m_app = -2.5 * np.log10(self.snrs[filtername] / 5.0)
            m_app += depths[filtername]
            if self.use_2D_extinction:
                A_x = self.ax1[filtername] * slice_point["ebv"]
                # Assuming all the dust along the line of sight matters.
                m_app = m_app - A_x
                dist = 10.0 * (100 ** ((m_app - self.mags[filtername]) / 5.0)) ** 0.5
            else:
                dist = self.ebvmap.distance_at_dmag(
                    dmag=m_app - self.mags[filtername],
                    dists=slice_point["ebv3d_dists"],
                    ebvs=slice_point["ebv3d_ebvs"],
                    filtername=filtername,
                )
            distances.append(dist)
        # compute the final distance, limited by whichever filter is
        # most shallow
        final_distance = np.min(distances, axis=-1) / 1e3  # to kpc
        if self.return_distance:
            return final_distance
        else:
            # Resorting to numerical integration of ugly function
            sd = StarDensity(slice_point["gall"], slice_point["galb"])
            stars_per_sterr, _err = integrate.quad(sd, 0, final_distance)
            stars_tot = stars_per_sterr * sky_area
            return stars_tot
