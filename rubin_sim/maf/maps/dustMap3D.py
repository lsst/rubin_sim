import warnings
import numpy as np
import healpy as hp

from rubin_sim.maf.maps import BaseMap
from rubin_sim.photUtils import Dust_values
from .ebv3Dhp import ebv_3d_hp, get_ebv_at_distance

__all__ = ["DustMap3D"]


class DustMap3D(BaseMap):
    """The DustMap3D provides a `~rubin_sim.maf.map` to hold 3d EBV data.

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
        # The values that will be added to the slicepoints
        self.keynames = [
            "ebv3d_ebvs",
            "ebv3d_dists",
            "ebv3d_ebv_at_{self.distPc:.1f}",
            "ebv3d_dist_at_{self.dmag:.1f}",
        ]
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
            dust = Dust_values()
            self.R_x = dust.Ax1.copy()
        else:
            self.R_x = R_x

    def run(self, slicePoints):
        # If the slicer has nside, it's a healpix slicer so we can read the map directly
        if "nside" in slicePoints:
            if slicePoints["nside"] != self.nside:
                warnings.warn(
                    f"Slicer value of nside {slicePoints['nside']} different "
                    f"from map value {self.nside}, will use slicer value here as appropriate."
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

        # calculate distance modulus for each distance
        dmods = 5.0 * np.log10(dists) - 5.0
        # calculate dust extinction at each distance, for the filtername
        A_x = self.R_x[self.filtername] * ebvs
        # calculate the (m-Mo) = distmod + A_x -- combination of extinction due to distance and dust
        m_minus_Mo = dmods + A_x

        # Maximum distance for the given m-Mo (dmag) value
        # first do the 'within the map' closest distance
        m_minus_Mo_at_mag, dist_closest = get_x_at_nearest_y(
            m_minus_Mo, dists, self.dmag
        )
        # calculate distance modulus for an object with the maximum dust extinction (and then the distance)
        distModsFar = self.dmag - A_x[:, -1]
        distsFar = 10.0 ** (0.2 * distModsFar + 1.0)
        # Use furthest of the two
        dist_dmag = np.where(distsFar > dist_closest, distsFar, dist_closest)

        slicePoints["ebv3d_dists"] = dists
        slicePoints["ebv3d_ebvs"] = ebvs
        slicePoints["ebv3d_ebv_at_{self.distPc:.1f}"] = ebv_at_dist
        slicePoints["ebv3d_dist_at_{self.dmag:.1f}"] = dist_dmag

        return slicePoints

    def getMaxDistDeltaMag(self, dmagVec, sfilt="r", ipix=None):
        """Compute the maximum distance for an apparent m-M using the maximum
        value of extinction from the map.
        """

        # We do distance modulus = (m-M) - A_x, and calculate the
        # distance from the result. We do this for every sightline
        # at once.
        if ipix is not None:
            ebvsMax = self.R_x[sfilt] * self.ebvs[ipix, -1]
        else:
            ebvsMax = self.R_x[sfilt] * self.ebvs[:, -1]
        distModsFar = dmagVec - ebvsMax

        distsFar = 10.0 ** (0.2 * distModsFar + 1.0)

        return distsFar

    def getDistanceAtMag(
        self, deltamag=15.2, sfilt="r", ipix=None, extrapolateFar=True
    ):

        """Returns the distances at which the combination of distance and
        extinction produces the input magnitude difference (m-M) = deltamag. Arguments:
                ARGUMENTS:
                deltamag = target (m-m_0). Can be scalar or array. If array,
                must have the same number of elements as the healpix map
                (i.e. hp.nside2npix(64) )
                sfilt = filter at which we want deltamag
                ipix = Pixels for which to perform the evaluation.
                extrapolateFar - for distances beyond the maximum
                distance in the model, treat the extinction as constant beyond
                that maximum distance and compute the distance at which
                delta-mag is achieved. (Defaults to True: only set to False if
                you know what you are doing!!)
                RETURNS:
                distances, mMinusM, bFar   -- where:
                distances = npix-length array giving the distances in parsecs
                mMinusM = npix-length array giving the magnitude differences
                bFar = npix-length boolean indicating whether the maximum
                distance indicated by a sight line was beyond the range of
                validity of the extinction model.
                    If ipix is provided, either as a single int or a list|array or
                    ints representing Healpix pixel indices, only the number of
                    pixels requested will be queried. Arrays will be returne in any
                    case, even when one single pixel is requested.
        """

        # A little bit of parsing... if deltamag is a scalar,
        # replicate it into an array. Otherwise just reference the
        # array that was passed in. For the moment, trust the user to
        # have inputted a deltamag vector of the right shape.
        if ipix is not None:
            ipix = np.atleast_1d(ipix)
            npix = ipix.shape[0]
        else:
            npix = self.ebvs.shape[0]
        if np.isscalar(deltamag):
            dmagVec = np.repeat(deltamag, npix)
        else:
            dmagVec = deltamag

        if np.size(dmagVec) != npix:
            print(
                "ebv3d.getDistanceAtMag WARN - size mismatch:", npix, np.shape(dmagVec)
            )
            return np.array([]), np.array([]), np.array([])

        # Now we need apparent minus absolute magnitude:
        mMinusM = self.getDeltaMag(sfilt, ipix=ipix)

        # Now we find elements in each row that are closest to the
        # requested deltamag:
        iMin = np.argmin(np.abs(mMinusM - dmagVec[:, np.newaxis]), axis=1)
        iExpand = np.expand_dims(iMin, axis=-1)

        # select only m-M at needed distance
        mMinusM = np.take_along_axis(mMinusM, iMin[:, np.newaxis], -1).flatten()

        # now find the closest distance...
        if ipix is not None:
            distsClosest = np.take_along_axis(
                self.dists[ipix], iExpand, axis=-1
            ).squeeze()
            # To keep things similar to the case of querying the whole map,
            #  we need to return an array also in case ipix is a single pixel.
            distsClosest = np.atleast_1d(distsClosest)
            # if npix>1:
            #     distsClosest = distsClosest.squeeze()
        else:
            distsClosest = np.take_along_axis(self.dists, iExpand, axis=-1).squeeze()

        # 2021-04-09: started implementing distances at or beyond the
        # maximum distance. Points for which the closest delta-mag is
        # in the maximum distance bin are picked.
        if ipix is not None:
            bFar = iMin == self.dists[ipix].shape[-1] - 1
        else:
            bFar = iMin == self.dists.shape[-1] - 1
        if not extrapolateFar:
            return distsClosest, mMinusM, bFar

        # For distances beyond the max, we use the maximum E(B-V)
        # along the line of sight to compute the distance.

        distsFar = self.getMaxDistDeltaMag(mMinusM, sfilt, ipix)

        # Now we swap in the far distances
        distsClosest[bFar] = distsFar[bFar]

        # ... Let's return both the closest distances and the map of
        # (m-M), since the user might want both.
        return distsClosest, mMinusM, bFar
