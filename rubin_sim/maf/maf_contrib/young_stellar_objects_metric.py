"""Young Stellar Objects metric.
Converted from notebook 211116_yso_3D_90b.ipynb.
Formatted with black."""

import healpy as hp
import numpy as np
import scipy.integrate as integrate
from rubin_sim.maf.metrics.base_metric import BaseMetric
from rubin_sim.phot_utils import DustValues
from rubin_sim.maf.maps import DustMap3D

__all__ = ["NYoungStarsMetric"]


class star_density(object):
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
        self.D_gc = 8.178  # Distance to the galactic center, kpc
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
        R_galac = (
            (self.D_gc - r * np.cos(self.gall)) ** 2 + (r * np.sin(self.gall)) ** 2
        ) ** 0.5

        exponent = (
            -1.0 * r * np.abs(np.sin(self.galb)) / self.h_thin - R_galac / self.r_thin
        )

        result = self.A * r**2 * np.exp(exponent)
        return result


class NYoungStarsMetric(BaseMetric):
    """Calculate the distance to which one could reach color uncertainties
    Parameters
    ----------
    metricName : str, opt
        Default 'young_stars'.
    m5Col : str, opt
        The default column name for m5 information in the input data. Default fiveSigmaDepth.
    filterCol : str, opt
        The column name for the filter information. Default filter.
    mags : dict
        The absolute magnitude of the object in question. Keys of filter name, values in mags.
        Default is for a 0.3 solar mass star at age = 100 Myr.
    snrs : dict
        The SNR to demand for each filter.
    galb_limit : float (25.)
        The galactic latitude above which to return zero (degrees).
    badval : float, opt
        The value to return when the metric value cannot be calculated. Default 0.
    """

    def __init__(
        self,
        metricName="young_stars",
        m5Col="fiveSigmaDepth",
        filterCol="filter",
        badval=0,
        mags={"g": 10.32, "r": 9.28, "i": 7.97},
        galb_limit=90.0,
        snrs={"g": 5.0, "r": 5.0, "i": 5.0},
        nside=64,
        **kwargs
    ):
        Cols = [m5Col, filterCol]
        maps = ["DustMap3D"]
        # This will give us access to the dust map get_distance_at_dmag routine
        # but does not require loading another copy of the map
        self.ebvmap = DustMap3D()
        units = "N stars"
        super().__init__(
            Cols, metricName=metricName, maps=maps, units=units, badval=badval, *kwargs
        )
        # Save R_x values for on-the-fly calculation of dust extinction with map
        self.R_x = DustValues().R_x.copy()
        # set return type
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.galb_limit = np.radians(galb_limit)
        self.mags = mags
        self.filters = list(self.mags.keys())
        self.snrs = snrs

    def run(self, dataSlice, slicePoint=None):

        # Is there another way to calculate sky_area, for non-healpix slicers?
        sky_area = hp.nside2pixarea(slicePoint["nside"], degrees=False)

        # if we are outside the galb_limit, return nothing
        # Note we could make this a more comlicated function that returns an expected density of
        # star forming regions
        if np.abs(slicePoint["galb"]) > self.galb_limit:
            return self.badval

        # Coadd depths for each filter
        depths = {}
        # ignore the divide by zero warnings
        with np.errstate(divide="ignore"):
            for filtername in self.filters:
                in_filt = np.where(dataSlice[self.filterCol] == filtername)[0]
                # Calculate coadded depth per filter
                depths[filtername] = 1.25 * np.log10(
                    np.sum(10.0 ** (0.8 * dataSlice[self.m5Col][in_filt]))
                )

        # solve for the distances in each filter where we hit the required SNR
        distances = []
        for filtername in self.filters:
            # Apparent magnitude at the SNR requirement
            m_app = -2.5 * np.log10(self.snrs[filtername] / 5.0)
            m_app += depths[filtername]
            dist_dmag = self.ebvmap.distance_at_dmag(
                dmag=m_app - self.mags[filtername],
                dists=slicePoint["ebv3d_dists"],
                ebvs=slicePoint["ebv3d_ebvs"],
                filtername=filtername,
            )
            distances.append(dist_dmag)
        # compute the final distance, limited by whichever filter is most shallow
        final_distance = np.min(distances, axis=-1) / 1e3  # to kpc
        # print(final_distance)

        # Resorting to numerical integration of ugly function
        sd = star_density(slicePoint["gall"], slicePoint["galb"])
        stars_per_sterr, _err = integrate.quad(sd, 0, final_distance)
        stars_tot = stars_per_sterr * sky_area

        return stars_tot
