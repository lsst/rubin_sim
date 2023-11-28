"""A HealpixSubsetSlicer - define the subset of healpixels to use to calculate metrics."""

__all__ = ("HealpixSubsetSlicer",)

from functools import wraps

import healpy as hp
import numpy as np
import rubin_scheduler.utils as simsUtils

from .healpix_slicer import HealpixSlicer


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
    lon_col : `str`, optional
        Name of the longitude (RA equivalent) column to use from the input data.
        Default fieldRA
    lat_col : `str`, optional
        Name of the latitude (Dec equivalent) column to use from the input data.
        Default fieldDec
    lat_lon_deg : `bool`, optional
        Flag indicating whether the lat and lon values in the input data are in
        degrees (True) or radians (False).
        Default True.
    verbose : `bool`, optional
        Flag to indicate whether or not to write additional information to stdout during runtime.
        Default True.
    badval : `float`, optional
        Bad value flag, relevant for plotting. Default the np.nan value (in order to properly flag
        bad data points for plotting with the healpix plotting routines). This should not be changed.
    use_cache : `bool`, optional
        Flag allowing the user to indicate whether or not to cache (and reuse) metric results
        calculated with the same set of simulated data pointings.
        This can be safely set to True for slicers not using maps and will result in increased speed.
        When calculating metric results using maps, the metadata at each individual ra/dec point may
        influence the metric results and so use_cache should be set to False.
        Default True.
    leafsize : `int`, optional
        Leafsize value for kdtree. Default 100.
    radius : `float`, optional
        Radius for matching in the kdtree. Equivalent to the radius of the FOV. Degrees.
        Default 1.75.
    use_camera : `bool`, optional
        Flag to indicate whether to use the LSST camera footprint or not.
        Default False.
    camera_footprint_file : `str`, optional
        Name of the camera footprint map to use. Can be None, which will use the default.
    rot_sky_pos_col_name : `str`, optional
        Name of the rotSkyPos column in the input  data. Only used if use_camera is True.
        Describes the orientation of the camera orientation compared to the sky.
        Default rotSkyPos.
    """

    def __init__(
        self,
        nside,
        hpid,
        lon_col="fieldRA",
        lat_col="fieldDec",
        lat_lon_deg=True,
        verbose=True,
        badval=np.nan,
        use_cache=True,
        leafsize=100,
        radius=2.45,
        use_camera=True,
        camera_footprint_file=None,
        rot_sky_pos_col_name="rotSkyPos",
    ):
        """Instantiate and set up healpix slicer object."""
        super().__init__(
            nside=nside,
            verbose=verbose,
            lon_col=lon_col,
            lat_col=lat_col,
            badval=badval,
            radius=radius,
            leafsize=leafsize,
            use_cache=use_cache,
            use_camera=use_camera,
            rot_sky_pos_col_name=rot_sky_pos_col_name,
            camera_footprint_file=camera_footprint_file,
            lat_lon_deg=lat_lon_deg,
        )
        self.hpid = hpid
        # add hpid to the slicer_init values
        self.slicer_init["hpid"] = hpid
        self.len_hpid = len(self.hpid)
        # Set up a mask for the metric values to use
        self.mask = np.ones(hp.nside2npix(self.nside), bool)
        self.mask[self.hpid] = False

    def __eq__(self, other_slicer):
        """Evaluate if two slicers are equivalent."""
        # If the two slicers are both healpix slicers, check nsides value.
        result = False
        if isinstance(other_slicer, HealpixSubsetSlicer):
            if other_slicer.nside == self.nside:
                if np.array_equal(other_slicer.hpid, self.hpid):
                    if other_slicer.lon_col == self.lon_col and other_slicer.lat_col == self.lat_col:
                        if other_slicer.radius == self.radius:
                            if other_slicer.use_camera == self.use_camera:
                                if other_slicer.rotSkyPosColName == self.rotSkyPosColName:
                                    if np.all(other_slicer.shape == self.shape):
                                        result = True
        return result

    def __iter__(self):
        """Iterate over the slices."""
        self.hpid_counter = 0
        return self

    def __next__(self):
        """Returns results of self._slice_sim_data when iterating over slicer.

        Results of self._slice_sim_data should be dictionary of
        {'idxs': the data indexes relevant for this slice of the slicer,
        'slice_point': the metadata for the slice_point, which always includes 'sid' key for ID of slice_point.}
        """
        if self.hpid_counter >= self.len_hpid:
            raise StopIteration
        # Set up 'current'
        islice = self.hpid[self.hpid_counter]
        # Set up 'next'
        self.hpid_counter += 1
        # Return 'current'
        return self._slice_sim_data(islice)

    def setup_slicer(self, sim_data, maps=None):
        """Use sim_data[self.lon_col] and sim_data[self.lat_col] (in radians) to set up KDTree.

        Parameters
        -----------
        sim_data : numpy.recarray
            The simulated data, including the location of each pointing.
        maps : list of rubin_sim.maf.maps objects, optional
            List of maps (such as dust extinction) that will run to build up additional metadata at each
            slice_point. This additional metadata is available to metrics via the slice_point dictionary.
            Default None.
        """
        super().setup_slicer(sim_data=sim_data, maps=maps)

        @wraps(self._slice_sim_data)
        def _slice_sim_data(islice):
            """Return indexes for relevant opsim data at slice_point
            (slice_point=lon_col/lat_col value .. usually ra/dec)."""
            # Subclass this method, just to make sure we return no data for points not in self.hpid
            slice_point = {"sid": islice, "nside": self.nside}
            if islice not in self.hpid:
                indices = []
            else:
                sx, sy, sz = simsUtils._xyz_from_ra_dec(
                    self.slice_points["ra"][islice], self.slice_points["dec"][islice]
                )
                # Query against tree.
                indices = self.opsimtree.query_ball_point((sx, sy, sz), self.rad)
                if (self.use_camera) & (len(indices) > 0):
                    # Find the indices *of those indices* which fall in the camera footprint
                    camera_idx = self.camera(
                        self.slice_points["ra"][islice],
                        self.slice_points["dec"][islice],
                        self.data_ra[indices],
                        self.data_dec[indices],
                        self.data_rot[indices],
                    )
                    indices = np.array(indices)[camera_idx]
                # Loop through all the slice_point keys. If the first dimension of slice_point[key] has
                # the same shape as the slicer, assume it is information per slice_point.
                # Otherwise, pass the whole slice_point[key] information. Useful for stellar LF maps
                # where we want to pass only the relevant LF and the bins that go with it.
                for key in self.slice_points:
                    if len(np.shape(self.slice_points[key])) == 0:
                        keyShape = 0
                    else:
                        keyShape = np.shape(self.slice_points[key])[0]
                    if keyShape == self.nslice:
                        slice_point[key] = self.slice_points[key][islice]
                    else:
                        slice_point[key] = self.slice_points[key]

            return {"idxs": indices, "slice_point": slice_point}

        setattr(self, "_slice_sim_data", _slice_sim_data)
