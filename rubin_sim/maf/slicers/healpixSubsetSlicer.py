"""A HealpixSubsetSlicer - define the subset of healpixels to use to calculate metrics."""

from functools import wraps
import numpy as np
import healpy as hp

import rubin_sim.utils as simsUtils

from rubin_sim.maf.plots.spatialPlotters import HealpixSkyMap, HealpixHistogram, HealpixPowerSpectrum

from .baseSpatialSlicer import BaseSpatialSlicer


__all__ = ['HealpixSubsetSlicer']


class HealpixSubsetSlicer(BaseSpatialSlicer):
    """
    A spatial slicer that evaluates pointings on a subset of a healpix-based grid.
    The advantage of using this healpixSubsetSlicer (rather than just putting the RA/Dec values into
    the UserPointsSlicer, which is another valid approach) is that you preserve the full healpix array.
    This means you could do things like calculate the power spectrum and plot without remapping into
    healpixels first. The downside is that you must first (externally) define the healpixels that you
    wish to use - the rubin_sim.featureScheduler.footprints is a useful add-on here.

    When plotting with RA/Dec, the default HealpixSkyMap can be used, corresponding to
    {'rot': (0, 0, 0), 'flip': 'astro'}.

    Parameters
    ----------
    nside : int
        The nside parameter of the healpix grid. Must be a power of 2.
    hpid : np.ndarray
        The subset of healpix id's to use to calculate the metric.
        Because the hpid should be defined based on a particular nside, these first two
        arguments are not optional for this slicer.
    lonCol : str, optional
        Name of the longitude (RA equivalent) column to use from the input data.
        Default fieldRA
    latCol : str, optional
        Name of the latitude (Dec equivalent) column to use from the input data.
        Default fieldDec
    latLonDeg : `bool`, optional
        Flag indicating whether the lat and lon values in the input data are in
        degrees (True) or radians (False).
        Default True.
    verbose : `bool`, optional
        Flag to indicate whether or not to write additional information to stdout during runtime.
        Default True.
    badval : float, optional
        Bad value flag, relevant for plotting. Default the hp.UNSEEN value (in order to properly flag
        bad data points for plotting with the healpix plotting routines). This should not be changed.
    useCache : `bool`
        Flag allowing the user to indicate whether or not to cache (and reuse) metric results
        calculated with the same set of simulated data pointings.
        This can be safely set to True for slicers not using maps and will result in increased speed.
        When calculating metric results using maps, the metadata at each individual ra/dec point may
        influence the metric results and so useCache should be set to False.
        Default True.
    leafsize : int, optional
        Leafsize value for kdtree. Default 100.
    radius : float, optional
        Radius for matching in the kdtree. Equivalent to the radius of the FOV. Degrees.
        Default 1.75.
    useCamera : `bool`, optional
        Flag to indicate whether to use the LSST camera footprint or not.
        Default False.
    rotSkyPosColName : str, optional
        Name of the rotSkyPos column in the input  data. Only used if useCamera is True.
        Describes the orientation of the camera orientation compared to the sky.
        Default rotSkyPos.
    mjdColName : str, optional
        Name of the exposure time column. Only used if useCamera is True.
        Default observationStartMJD.
    chipNames : array-like, optional
        List of chips to accept, if useCamera is True. This lets users turn 'on' only a subset of chips.
        Default 'all' - this uses all chips in the camera.
    """
    def __init__(self, nside, hpid, lonCol ='fieldRA',
                 latCol='fieldDec', latLonDeg=True, verbose=True, badval=hp.UNSEEN,
                 useCache=True, leafsize=100, radius=2.45,
                 useCamera=True, rotSkyPosColName='rotSkyPos',
                 mjdColName='observationStartMJD', chipNames='all'):
        """Instantiate and set up healpix slicer object."""
        super().__init__(verbose=verbose,
                         lonCol=lonCol, latCol=latCol,
                         badval=badval, radius=radius, leafsize=leafsize,
                         useCamera=useCamera, rotSkyPosColName=rotSkyPosColName,
                         mjdColName=mjdColName, chipNames=chipNames, latLonDeg=latLonDeg)
        # Valid values of nside are powers of 2.
        # nside=64 gives about 1 deg resolution
        # nside=256 gives about 13' resolution (~1 CCD)
        # nside=1024 gives about 3' resolution
        # Check validity of nside:
        if not(hp.isnsideok(nside)):
            raise ValueError('Valid values of nside are powers of 2.')
        if len(hpid) > hp.nside2npix(nside):
            raise ValueError('Nside (%d) and length of hpid (%d) seem incompatible.' % (nside,
                                                                                        hp.nside2npix(nside)))
        self.nside = int(nside)
        self.hpid = hpid
        self.pixArea = hp.nside2pixarea(self.nside)
        self.nslice = hp.nside2npix(self.nside)
        self.spatialExtent = [0, self.nslice-1]
        self.shape = self.nslice
        if self.verbose:
            print('HealpixSubsetSlicer using NSIDE=%d, ' % (self.nside) + \
                  'approximate resolution %f arcminutes' % (hp.nside2resol(self.nside, arcmin=True)))
        # Set variables so slicer can be re-constructed
        self.slicer_init = {'nside': nside, 'hpid': hpid, 'lonCol': lonCol, 'latCol': latCol,
                            'radius': radius}
        if useCache:
            # useCache set the size of the cache for the memoize function in sliceMetric.
            binRes = hp.nside2resol(nside)  # Pixel size in radians
            # Set the cache size to be ~2x the circumference
            self.cacheSize = int(np.round(4.*np.pi/binRes))
        # Set up slicePoint metadata.
        self.slicePoints['nside'] = nside
        self.slicePoints['sid'] = np.arange(self.nslice)
        self.slicePoints['ra'], self.slicePoints['dec'] = self._pix2radec(self.slicePoints['sid'])
        # Set the default plotting functions.
        self.plotFuncs = [HealpixSkyMap, HealpixHistogram, HealpixPowerSpectrum]

    def __eq__(self, otherSlicer):
        """Evaluate if two slicers are equivalent."""
        # If the two slicers are both healpix slicers, check nsides value.
        result = False
        if isinstance(otherSlicer, HealpixSubsetSlicer):
            if otherSlicer.nside == self.nside:
                if np.all(otherSlicer.hpid == self.hpid):
                    if (otherSlicer.lonCol == self.lonCol and otherSlicer.latCol == self.latCol):
                        if otherSlicer.radius == self.radius:
                            if otherSlicer.useCamera == self.useCamera:
                                if otherSlicer.chipsToUse == self.chipsToUse:
                                    if otherSlicer.rotSkyPosColName == self.rotSkyPosColName:
                                        if np.all(otherSlicer.shape == self.shape):
                                            result = True
        return result

    def _pix2radec(self, islice):
        """Given the pixel number / sliceID, return the RA/Dec of the pointing, in radians."""
        # Calculate RA/Dec in RADIANS of pixel in this healpix slicer.
        # Note that ipix could be an array,
        # in which case RA/Dec values will be an array also.
        lat, ra = hp.pix2ang(self.nside, islice)
        # Move dec to +/- 90 degrees
        dec = np.pi/2.0 - lat
        return ra, dec

    # This slicer does iterate over all of the slicepoints - mainly so it can return a masked value for
    # non-calculated healpixels.
    def setupSlicer(self, simData, maps=None):
        """Use simData[self.lonCol] and simData[self.latCol] (in radians) to set up KDTree.

        Parameters
        -----------
        simData : numpy.recarray
            The simulated data, including the location of each pointing.
        maps : list of rubin_sim.maf.maps objects, optional
            List of maps (such as dust extinction) that will run to build up additional metadata at each
            slicePoint. This additional metadata is available to metrics via the slicePoint dictionary.
            Default None.
        """
        if maps is not None:
            if self.cacheSize != 0 and len(maps) > 0:
                warnings.warn('Warning:  Loading maps but cache on.'
                              'Should probably set useCache=False in slicer.')
            self._runMaps(maps)
        self._setRad(self.radius)
        if self.useCamera:
            self._setupLSSTCamera()
            self._presliceFootprint(simData)
        else:
            if self.latLonDeg:
                self._buildTree(np.radians(simData[self.lonCol]),
                                np.radians(simData[self.latCol]), self.leafsize)
            else:
                self._buildTree(simData[self.lonCol], simData[self.latCol], self.leafsize)

        @wraps(self._sliceSimData)
        def _sliceSimData(islice):
            """Return indexes for relevant opsim data at slicepoint
            (slicepoint=lonCol/latCol value .. usually ra/dec)."""
            if islice not in self.hpid:
                return {'idxs': [], 'slicePoint': {self.slicePoints['sid'][islice],
                                                   self.slicePoints['ra'][islice],
                                                   self.slicePoints['dec'][islice]}}
            # Build dict for slicePoint info
            slicePoint = {}
            if self.useCamera:
                indices = self.sliceLookup[islice]
                slicePoint['chipNames'] = self.chipNames[islice]
            else:
                sx, sy, sz = simsUtils._xyz_from_ra_dec(self.slicePoints['ra'][islice],
                                                        self.slicePoints['dec'][islice])
                # Query against tree.
                indices = self.opsimtree.query_ball_point((sx, sy, sz), self.rad)

            # Loop through all the slicePoint keys. If the first dimension of slicepoint[key] has
            # the same shape as the slicer, assume it is information per slicepoint.
            # Otherwise, pass the whole slicePoint[key] information. Useful for stellar LF maps
            # where we want to pass only the relevant LF and the bins that go with it.
            for key in self.slicePoints:
                if len(np.shape(self.slicePoints[key])) == 0:
                    keyShape = 0
                else:
                    keyShape = np.shape(self.slicePoints[key])[0]
                if (keyShape == self.nslice):
                    slicePoint[key] = self.slicePoints[key][islice]
                else:
                    slicePoint[key] = self.slicePoints[key]
            return {'idxs': indices, 'slicePoint': slicePoint}
        setattr(self, '_sliceSimData', _sliceSimData)
