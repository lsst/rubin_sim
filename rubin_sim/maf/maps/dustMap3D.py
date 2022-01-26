import warnings
import os
import numpy as np
import healpy as hp
from astropy.io import fits

from rubin_sim.data import get_data_dir
from rubin_sim.maf.maps import BaseMap
from rubin_sim.photUtils import Dust_values

__all__ = ["DustMap3D"]


class DustMap3D(BaseMap):
    """The DustMap3D provides a `~rubin_sim.maf.map` to hold 3d EBV data."""

    def __init__(self, nside=64, filtername=None, distPc=3000, mapPath=None, R_x=None):
        # The values that will be added to the slicepoints
        self.keynames = ["ebv3d_ebv", "ebv3d_distance", "ebv3d_"]
        self.nside = nside
        self.filtername = filtername
        self.distPc = distPc
        self.mapPath = mapPath
        # R_x is the extinction coefficient (A_v = R_v * E(B-V) .. A_x = R_x * E(B-V)) per filter
        # This is equivalent to calculating A_x (using rubin_sim.photUtils.Sed.addDust) in each
        # filter and setting E(B-V) to 1 [so similar to the values calculated in Dust_values ..
        # we probably should rename those (from Ax1 to R_x)
        if R_x is None:
            dust = Dust_values()
            self.R_x = dust.Ax1.copy()
        else:
            self.R_x = R_x

    def _readMap(self):
        # Set up path to map
        if self.mapPath is None:
            if nside == 64:
                mapName = "merged_ebv3d_nside64.fits"
                self.mapPath = os.path.join(get_data_dir(), "DustMaps3d", mapName)
            elif nside == 128:
                mapName = "merged_ebv3d_nside128_defaults.fits"
                self.mapPath = os.path.join(get_data_dir(), "DustMaps3d", mapName)
            else:
                raise Exception(
                    f"mapPath not specified, and nside {self.nside} not one of 64 or 128. "
                    "Please specify mapPath."
                )
        else:
            # Check if user specified map name but not full map path
            if not os.path.isfile(self.mapPath):
                testPath = os.path.join(get_data_dir(), "DustMaps3d", self.mapPath)
                if os.path.isfile(testPath):
                    self.mapPath = testPath
        # Read map from disk
        hdul = fits.open(self.mapPath)
        self.hpids = hdul[0].data
        self.dists = hdul[1].data
        self.ebvs = hdul[2].data
        self.hdr = hdul[0].header
        self.sfacs = None
        self.mask = None
        if len(hdul) > 3:
            self.sfacs = hdul[3].data
        if len(hdul) > 4:
            self.mask = hdul[4].data
        # Close map file
        hdul.close()

        # Check additional healpix information from the header
        map_nside = self.hdr["NSIDE"]
        self.nested = self.hdr["NESTED"]
        if self.nside != map_nside:
            warnings.warn(
                f"Map nside {map_nside} did not match expected nside {self.nside}; "
                f"Will use nside from map data."
            )
            self.nside = map_nside
        # Nested healpix data will not match the healpix arrays for the slicers (at this time)
        if self.nested:
            warnings.warn("Map has nested (not ring order) data; will have to reorder.")
            for i in np.arange(0, len(self.dists[0])):
                self.dists[:, i] = hp.reorder(self.dists[:, i], "nest", "ring")
                self.ebvs[:, i] = hp.reorder(self.ebvs[:, i], "nest", "ring")
                if self.sfacs is not None:
                    self.sfacs[:, i] = hp.reorder(self.sfacs[:, i], "nest", "ring")
                if self.mask is not None:
                    self.mask[:, i] = hp.reorder(self.mask[:, i], "nest", "ring")
            self.nested = False

        # Add distance modulus for each 'distance'
        self.dmods = 5.0 * np.log10(self.dists) - 5.0

        # Calculate the map ebv and dist values at the initialized distance
        self.ebvClosest, self.distClosest = self.getMapNearestDist(self.distPc)


    def getMapNearestDist(self, distPc=3000):
        """Given a distance, find E(B-V) at the closest distance in the map.
        This function operates on all pixels in the map at once.
        To provide this as a slicePoint map key means that distPc must be determined (once) at map setup.

        Parameters
        ----------
        distPc : `float', opt
            Default 30000 parsecs

        Returns
        -------
        ebv, dist : `np.array`, `np.array`
        """
        # find the location of the closest distance at each healpixel
        dist_idx_min = np.argmin(np.abs(self.dists - distPc), axis=1)
        # now lift the ebv values at this distance
        iExpand = np.expand_dims(dist_idx_min, axis=-1)
        distsClosest = np.take_along_axis(self.dists, iExpand, axis=-1).squeeze()
        ebvsClosest = np.take_along_axis(self.ebvs, iExpand, axis=-1).squeeze()
        return ebvsClosest, distsClosest

    def getNearestDist(self, hp_idx, distPc=3000):
        """Given a distance, find E(B-V) at the closest distance.
         This function operates on a single slicePoint or healpixel, so distPc can be modified at call time.

         Worth noting that this method (if you looped over each healpixel) is about 2x slower
         than doing the whole map at once as above -- but both are quick and the difference is about 100ms
         at nside=64.

        Parameters
        ----------
        hp_idx : 'int'
            Healpix index at which to find the nearest distance and E(B-V) value
        distPc : `float', opt
            Default 30000 parsecs

        Returns
        -------
        ebv, dist : `float`, `float`
        """
        idx_min = np.argmin(np.abs(self.dists[hp_idx, :] - distPc))
        dist = self.dists[hp_idx, idx_min]
        ebv = self.ebvs[hp_idx, idx_min]
        return dist, ebv

    def getEBVatSightline(
        self, l=0.0, b=0.0, ebvMap=np.array([]), interp=False, showEBVdist=False
    ):  ## KEEP? REMOVE? (slicePoint?)

        """Utility - returns E(B-V) for one or more sightlines in Galactic
        coordinates. Takes as an argument a 2D healpix array of
        quantities (usually this will be reddening returned by
        getMapNearestDist() ). Also returns the nearest coords to the
        requested coords for debug purposes. Arguments:
        l, b = Galactic coordinates of the sight-line(s). Can be
        scalar or vector.
        ebvMap = 2D reddening map to use
        interp: Interpolate using healpy? If False, the nearest
        healpix is used instead.
        showEBVdist = Will usually be used only for debugging
        purposes. If True, this method will plot the run of E(B-V) vs
        distance for the nearest hpid (ignored if interp=True)
        """

        if np.size(ebvMap) < 1:
            return 0.0, -99.0, -99.0

        # find the coords on the sky of the requested sight line, and
        # convert this to healpix
        coo = SkyCoord(l * u.deg, b * u.deg, frame="galactic")

        # Equatorial coordinates of the requested position(s)
        ra = coo.icrs.ra.deg
        dec = coo.icrs.dec.deg

        if interp:
            ebvRet = hp.get_interp_val(ebvMap, ra, dec, nest=self.nested, lonlat=True)

            # For backwards compatibility with the "Test" return values
            lTest = np.copy(l)
            bTest = np.copy(b)

        else:
            hpid = hp.ang2pix(self.nside, ra, dec, nest=self.nested, lonlat=True)
            ebvRet = ebvMap[hpid]

            # For debugging: determine the coordinates at this nearest pixel
            raTest, decTest = hp.pix2ang(
                self.nside, hpid, nest=self.nested, lonlat=True
            )
            cooTest = SkyCoord(raTest * u.deg, decTest * u.deg, frame="icrs")
            lTest = cooTest.galactic.l.degree
            bTest = cooTest.galactic.b.degree
        return ebvRet, lTest, bTest

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

    def getDeltaMag(self, sFilt="r", ipix=None):

        """Converts the reddening map into an (m-m0) map for the given
        filter"""

        if not sFilt in self.R_x.keys():
            sFilt = "r"
        Rx = self.R_x[sFilt]
        if ipix is not None:
            mMinusm0 = self.dmods[np.newaxis, ipix] + Rx * self.ebvs[ipix, :]
            # make 3d so that form the outside nothing changes
        else:
            mMinusm0 = self.dmods[np.newaxis, :] + Rx * self.ebvs

        return mMinusm0[0]

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

    def getInterpolatedProfile(self, gall, galb, dist): ### KEEP? REMOVE? (slicePoint?)
        gall = np.atleast_1d(gall)
        galb = np.atleast_1d(galb)
        dist = np.atleast_2d(
            dist
        ).T  # required for subtraction to 2d array with shape (N,N_samples)
        N = len(dist)

        coo = SkyCoord(gall * u.deg, galb * u.deg, frame="galactic")
        RAs = coo.icrs.ra.deg
        DECs = coo.icrs.dec.deg
        hpids, weights = hp.get_interp_weights(
            self.nside, RAs, DECs, self.nested, lonlat=True
        )
        # hpids, weights = hp.get_interp_weights(64, gall, galb, False, lonlat=True)
        ebvout = np.zeros(N)
        distout = np.zeros(N)
        for i in range(hpids.shape[0]):
            pid = hpids[i]
            w = weights[i]
            distID = np.argmin(np.abs(self.dists[pid] - dist), axis=1)
            ebvout_i = self.ebvs[pid, distID] * w
            ebvout += ebvout_i
            distout_i = self.dists[pid, distID] * w
            distout += distout_i
