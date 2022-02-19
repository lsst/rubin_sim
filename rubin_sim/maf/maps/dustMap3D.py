import warnings
import numpy as np

from rubin_sim.maf.maps import BaseMap
from rubin_sim.photUtils import Dust_values
from .ebv3Dhp import ebv_3d_hp, get_x_at_nearest_y

__all__ = ["DustMap3D"]


class DustMap3D(BaseMap):
    """The DustMap3D provides a `~rubin_sim.maf.map` to hold 3d EBV data.

    Adds the following keys to the slicePoints:
    ebv3d_dists - the distances from the 3d dust map at each slicePoint (in pc)
    ebv3d_ebvs - the E(B-V) values corresponding to each distance at each slicePoint
    ebv3d_ebv_at_<distPc> - the (single) ebv value at the nearest distance to distPc
    ebv3d_dist_at_<dMag> - the (single) distance value corresponding to where extinction and
    distance modulus combine to create a m-Mo value of dMag, for the filter specified in filtername (in pc).
    Note that <distPc> and <dMag> will be formatted with a single decimal place.

    The additional method 'distance_at_mag' can be called either with the distances and ebv values for the
    entire map or with the values from a single slicePoint, in order to calculate the distance at which
    extinction and distance modulus combine to create a m-Mo value closest to 'dmag' in any filter.
    This is the same value as would be reported in ebv3d_dist_at_<dMag>, but can be calculated on the fly,
    allowing variable filters and dmag values.

    Parameters
    ----------
    nside: `int`
        Healpixel resolution (2^x).
    mapFile : `str`, opt
        Path to dust map file.
    interp : `bool`, opt
        Should returned values be interpolated (True) or just nearest neighbor (False).
        Default True, but is ignored if 'pixels' is provided.
    filtername : 'str', opt
        Name of the filter (to match the lsst filter names in rubin_sim.photUtils.Dust_values)
        in which to calculate dust extinction magnitudes
    distPc : `float`, opt
        Distance at which to precalculate the nearest ebv value
    dMag : `float`, opt
        Calculate the maximum distance which matches this 'dMag'
        dMag == m-mO (dust extinction + distance modulus)
    R_x : `dict` {`str`: `float`}, opt
        Per-filter dust extinction curve coefficients.
        Calculated by rubin_sim.photUtils.Dust_values if "None".
    """

    def __init__(
        self,
        nside=64,
        mapFile=None,
        interp=True,
        filtername="r",
        distPc=3000,
        dMag=15.2,
        R_x=None,
    ):
        self.nside = nside
        self.interp = interp
        self.mapFile = mapFile
        self.filtername = filtername
        self.distPc = distPc
        self.dMag = dMag
        # R_x is the extinction coefficient (A_v = R_v * E(B-V) .. A_x = R_x * E(B-V)) per filter
        # This is equivalent to calculating A_x (using rubin_sim.photUtils.Sed.addDust) in each
        # filter and setting E(B-V) to 1 [so similar to the values calculated in Dust_values ..
        # we probably should rename those (from Ax1 to R_x)
        if R_x is None:
            self.R_x = Dust_values().R_x.copy()
        else:
            self.R_x = R_x
        # The values that will be added to the slicepoints
        self.keynames = [
            "ebv3d_ebvs",
            "ebv3d_dists",
            f"ebv3d_ebv_at_{self.distPc:.1f}",
            f"ebv3d_dist_at_{self.dMag:.1f}",
        ]

    def run(self, slicePoints):
        # If the slicer has nside, it's a healpix slicer so we can read the map directly
        if "nside" in slicePoints:
            if slicePoints["nside"] != self.nside:
                warnings.warn(
                    f"Slicer value of nside {slicePoints['nside']} different "
                    f"from map value {self.nside}, will use correct slicer value here."
                )
            dists, ebvs = ebv_3d_hp(
                slicePoints["nside"], pixels=slicePoints["sid"], mapFile=self.mapFile
            )
        # Not a healpix slicer, look up values based on RA,dec with possible interpolation
        else:
            dists, ebvs = ebv_3d_hp(
                self.nside,
                ra=slicePoints["ra"],
                dec=slicePoints["dec"],
                interp=self.interp,
                mapFile=self.mapFile,
            )

        # Calculate the map ebv and dist values at the initialized distance
        dist_closest, ebv_at_dist = get_x_at_nearest_y(dists, ebvs, self.distPc)

        # Calculate the distances at which m_minus_Mo values of 'dmag' are reached
        dist_dmag = self.distance_at_dmag(self.dMag, dists, ebvs, self.filtername)

        slicePoints["ebv3d_dists"] = dists
        slicePoints["ebv3d_ebvs"] = ebvs
        slicePoints[f"ebv3d_ebv_at_{self.distPc:.1f}"] = ebv_at_dist
        slicePoints[f"ebv3d_dist_at_{self.dMag:.1f}"] = dist_dmag

        return slicePoints

    def distance_at_dmag(self, dmag, dists, ebvs, filtername=None):
        # Provide this as a method which could be used for a single slicePoint as well as for whole map
        # (single slicePoint means you could calculate this for any arbitrary magnitude or filter if needed)
        if filtername is None:
            filtername = self.filtername
        # calculate distance modulus for each distance
        dmods = 5.0 * np.log10(dists) - 5.0
        # calculate dust extinction at each distance, for the filtername
        A_x = self.R_x[filtername] * ebvs
        # calculate the (m-Mo) = distmod + A_x -- combination of extinction due to distance and dust
        m_minus_Mo = dmods + A_x

        # Maximum distance for the given m-Mo (dmag) value
        # first do the 'within the map' closest distance
        m_minus_Mo_at_mag, dist_closest = get_x_at_nearest_y(m_minus_Mo, dists, dmag)
        # calculate distance modulus for an object with the maximum dust extinction (and then the distance)
        if A_x.ndim == 2:
            distModsFar = dmag - A_x[:, -1]
        else:
            distModsFar = dmag - A_x.max()
        distsFar = 10.0 ** (0.2 * distModsFar + 1.0)
        # Use furthest of the two
        dist_dmag = np.where(distsFar > dist_closest, distsFar, dist_closest)
        return dist_dmag
