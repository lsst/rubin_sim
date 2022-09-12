"""A HealpixSubsetSlicer - define the subset of healpixels to use to calculate metrics."""

from functools import wraps
import numpy as np
import healpy as hp
import warnings
import rubin_sim.utils as simsUtils

from rubin_sim.maf.plots.spatial_plotters import (
    HealpixSkyMap,
    HealpixHistogram,
    HealpixPowerSpectrum,
)

from .healpix_slicer import HealpixSlicer


__all__ = ["HealpixSubsetSlicer"]


class HealpixSubsetSlicer(HealpixSlicer):
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
    nside : `int`
        The nside parameter of the healpix grid. Must be a power of 2.
    hpid : `np.ndarray`
        The subset of healpix id's to use to calculate the metric.
        Because the hpid should be defined based on a particular nside, these first two
        arguments are not optional for this slicer.
    lonCol : `str`, optional
        Name of the longitude (RA equivalent) column to use from the input data.
        Default fieldRA
    latCol : `str`, optional
        Name of the latitude (Dec equivalent) column to use from the input data.
        Default fieldDec
    latLonDeg : `bool`, optional
        Flag indicating whether the lat and lon values in the input data are in
        degrees (True) or radians (False).
        Default True.
    verbose : `bool`, optional
        Flag to indicate whether or not to write additional information to stdout during runtime.
        Default True.
    badval : `float`, optional
        Bad value flag, relevant for plotting. Default the hp.UNSEEN value (in order to properly flag
        bad data points for plotting with the healpix plotting routines). This should not be changed.
    useCache : `bool`, optional
        Flag allowing the user to indicate whether or not to cache (and reuse) metric results
        calculated with the same set of simulated data pointings.
        This can be safely set to True for slicers not using maps and will result in increased speed.
        When calculating metric results using maps, the metadata at each individual ra/dec point may
        influence the metric results and so useCache should be set to False.
        Default True.
    leafsize : `int`, optional
        Leafsize value for kdtree. Default 100.
    radius : `float`, optional
        Radius for matching in the kdtree. Equivalent to the radius of the FOV. Degrees.
        Default 1.75.
    useCamera : `bool`, optional
        Flag to indicate whether to use the LSST camera footprint or not.
        Default False.
    cameraFootprintFile : `str`, optional
        Name of the camera footprint map to use. Can be None, which will use the default.
    rotSkyPosColName : `str`, optional
        Name of the rotSkyPos column in the input  data. Only used if useCamera is True.
        Describes the orientation of the camera orientation compared to the sky.
        Default rotSkyPos.
    """

    def __init__(
        self,
        nside,
        hpid,
        lonCol="fieldRA",
        latCol="fieldDec",
        latLonDeg=True,
        verbose=True,
        badval=hp.UNSEEN,
        useCache=True,
        leafsize=100,
        radius=2.45,
        useCamera=True,
        cameraFootprintFile=None,
        rotSkyPosColName="rotSkyPos",
    ):
        """Instantiate and set up healpix slicer object."""
        super().__init__(
            nside=nside,
            verbose=verbose,
            lonCol=lonCol,
            latCol=latCol,
            badval=badval,
            radius=radius,
            leafsize=leafsize,
            useCache=useCache,
            useCamera=useCamera,
            rotSkyPosColName=rotSkyPosColName,
            cameraFootprintFile=cameraFootprintFile,
            latLonDeg=latLonDeg,
        )
        self.hpid = hpid
        self.len_hpid = len(self.hpid)
        # Set up a mask for the metric values to use
        self.mask = np.ones(hp.nside2npix(self.nside), bool)
        self.mask[self.hpid] = False

    def __eq__(self, otherSlicer):
        """Evaluate if two slicers are equivalent."""
        # If the two slicers are both healpix slicers, check nsides value.
        result = False
        if isinstance(otherSlicer, HealpixSubsetSlicer):
            if otherSlicer.nside == self.nside:
                if np.array_equal(otherSlicer.hpid, self.hpid):
                    if (
                        otherSlicer.lonCol == self.lonCol
                        and otherSlicer.latCol == self.latCol
                    ):
                        if otherSlicer.radius == self.radius:
                            if otherSlicer.useCamera == self.useCamera:
                                if (
                                    otherSlicer.rotSkyPosColName
                                    == self.rotSkyPosColName
                                ):
                                    if np.all(otherSlicer.shape == self.shape):
                                        result = True
        return result

    def __iter__(self):
        """Iterate over the slices."""
        self.hpid_counter = 0
        return self

    def __next__(self):
        """Returns results of self._sliceSimData when iterating over slicer.

        Results of self._sliceSimData should be dictionary of
        {'idxs': the data indexes relevant for this slice of the slicer,
        'slicePoint': the metadata for the slicePoint, which always includes 'sid' key for ID of slicePoint.}
        """
        if self.hpid_counter >= self.len_hpid:
            raise StopIteration
        # Set up 'current'
        islice = self.hpid[self.hpid_counter]
        # Set up 'next'
        self.hpid_counter += 1
        # Return 'current'
        return self._sliceSimData(islice)

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
        super().setupSlicer(simData=simData, maps=maps)

        @wraps(self._sliceSimData)
        def _sliceSimData(islice):
            """Return indexes for relevant opsim data at slicepoint
            (slicepoint=lonCol/latCol value .. usually ra/dec)."""
            # Subclass this method, just to make sure we return no data for points not in self.hpid
            slicePoint = {"sid": islice, "nside": self.nside}
            if islice not in self.hpid:
                indices = []
            else:
                sx, sy, sz = simsUtils._xyz_from_ra_dec(
                    self.slicePoints["ra"][islice], self.slicePoints["dec"][islice]
                )
                # Query against tree.
                indices = self.opsimtree.query_ball_point((sx, sy, sz), self.rad)
                if (self.useCamera) & (len(indices) > 0):
                    # Find the indices *of those indices* which fall in the camera footprint
                    camera_idx = self.camera(
                        self.slicePoints["ra"][islice],
                        self.slicePoints["dec"][islice],
                        self.data_ra[indices],
                        self.data_dec[indices],
                        self.data_rot[indices],
                    )
                    indices = np.array(indices)[camera_idx]
                # Loop through all the slicePoint keys. If the first dimension of slicepoint[key] has
                # the same shape as the slicer, assume it is information per slicepoint.
                # Otherwise, pass the whole slicePoint[key] information. Useful for stellar LF maps
                # where we want to pass only the relevant LF and the bins that go with it.
                for key in self.slicePoints:
                    if len(np.shape(self.slicePoints[key])) == 0:
                        keyShape = 0
                    else:
                        keyShape = np.shape(self.slicePoints[key])[0]
                    if keyShape == self.nslice:
                        slicePoint[key] = self.slicePoints[key][islice]
                    else:
                        slicePoint[key] = self.slicePoints[key]

            return {"idxs": indices, "slicePoint": slicePoint}

        setattr(self, "_sliceSimData", _sliceSimData)
