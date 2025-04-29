__all__ = ("Slicer", "UserSlicer")

import copy
import warnings

import healpy as hp
import numpy as np
import rubin_scheduler.utils as utils


class Slicer(object):
    """

    Parameters
    ----------
    lon_col : `str`, optional
        Name of the longitude (RA equivalent) column.
    lat_col : `str`, optional
        Name of the latitude (Dec equivalent) column.
    rot_sky_pos_col_name : `str`, optional
        Name of the rotSkyPos column in the input  data.
        Only used if use_camera is True.
        Describes the orientation of the camera orientation on the sky.
    missing : `float`, optional
        Bad value flag, relevant for plotting.
    leafsize : `int`, optional
        Leafsize value for kdtree.
    radius : `float`, optional
        Radius for matching in the kdtree.
        Equivalent to the radius of the FOV, in degrees.
    use_camera : `bool`, optional
        Flag to indicate whether to use the LSST camera footprint or not.
    camera_footprint_file : `str`, optional
        Name of the camera footprint map to use.
        Can be None, which will use the default file.
    """

    def __init__(
        self,
        nside=128,
        lon_col="fieldRA",
        lat_col="fieldDec",
        leafsize=100,
        radius=2.45,
        use_camera=True,
        camera_footprint_file=None,
        rot_sky_pos_col_name="rotSkyPos",
        missing=np.nan,
        cache=False,
    ):

        self.nside = int(nside)
        self.nslice = hp.nside2npix(self.nside)
        self.shape = self.nslice

        self.lon_col = lon_col
        self.lat_col = lat_col
        self.rot_sky_pos_col_name = rot_sky_pos_col_name
        self.use_camera = use_camera
        self.camera_footprint_file = camera_footprint_file
        self.leafsize = leafsize

        self.missing = missing

        self.radius = radius

        self.cache = cache

        # Set up slice_point
        self.slice_points = {}
        self.slice_points["nside"] = nside
        self.slice_points["sid"] = np.arange(self.nslice)
        self.slice_points["ra"], self.slice_points["dec"] = utils._hpid2_ra_dec(
            self.nside, self.slice_points["sid"]
        )

    def __len__(self):
        """Return nslice, the number of slice_points in the slicer."""
        return self.nslice

    def __iter__(self):
        """Iterate over the slices."""
        self.islice = 0
        return self

    def __next__(self):
        """Returns results of self._slice_data when iterating over slicer.

        Results of self._slice_data should be dictionary of
        {'idxs': the data indexes relevant for this slice of the slicer,
        'slice_point': the metadata for the slice_point, which always
        includes 'sid' key for ID of slice_point.}
        """
        if self.islice >= self.nslice:
            raise StopIteration
        islice = self.islice
        self.islice += 1
        return self._slice_data(islice)

    def __getitem__(self, islice):
        return self._slice_data(islice)

    def setup_slicer(self, pointings_data):
        """set up KDTree.

        Parameters
        -----------
        pointings_data : `numpy.ndarray`
            Array with location and camera rotation of each pointing.
        """
        self._set_rad(self.radius)

        self.data_ra_rad = np.radians(pointings_data[self.lon_col])
        self.data_dec_rad = np.radians(pointings_data[self.lat_col])
        self.data_rot_rad = np.radians(pointings_data[self.rot_sky_pos_col_name])
    
        if self.use_camera:
            self._setupLSSTCamera()

        self._build_tree(self.data_ra_rad, self.data_dec_rad, self.leafsize)

        def _slice_data(islice):
            """Return indexes for relevant opsim data at slice_point
            (slice_point=lon_col/lat_col value .. usually ra/dec).
            """

            # Build dict for slice_point info
            slice_point = {"sid": islice}
            sx, sy, sz = utils._xyz_from_ra_dec(
                self.slice_points["ra"][islice], self.slice_points["dec"][islice]
            )
            # Query against tree.
            indices = self.opsimtree.query_ball_point((sx, sy, sz), self.rad_deg)

            if (self.use_camera) & (len(indices) > 0):
                # Find the indices *of those indices*
                # which fall in the camera footprint
                camera_idx = self.camera(
                    self.slice_points["ra"][islice],
                    self.slice_points["dec"][islice],
                    self.data_ra_rad[indices],
                    self.data_dec_rad[indices],
                    self.data_rot_rad[indices],
                )
                indices = np.array(indices)[camera_idx]

            # Loop through all the slice_point keys.
            # If the first dimension of slice_point[key] has the same shape
            # as the slicer, assume it is information per slice_point.
            # Otherwise, pass the whole slice_point[key] information.
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

        setattr(self, "_slice_data", _slice_data)

    def _setupLSSTCamera(self):
        """If we want to include the camera chip gaps, etc."""
        self.camera = utils.LsstCameraFootprint(units="radians", footprint_file=self.camera_footprint_file)

    def _build_tree(self, ra_rad, dec_rad, leafsize=100):
        """Build KD tree on sim_dataRA/Dec.
        
        Parameters
        ----------
        ra_rad : `numpy.array`
            RA of points (radians)
        dec_rad : `numpy.array`
            Dec of points (radians)
        leafsize : `int`
            Number of Ra/Dec pointings in each leaf node.
        """
        self.opsimtree = utils._build_tree(ra_rad, dec_rad, leafsize)

    def _set_rad(self, radius=1.75):
        """Set radius (in degrees) for kdtree search."""
        self.rad_deg = utils.xyz_angular_radius(radius)

    def add_info(self, info):
        info["slicer: nside"] = self.nside

        return info

    def __call__(self, input_visits, metric_s, info=None):
        """

        Parameters
        ----------
        input_vistis : `np.array`
            Array with the visit information. If a pandas
            DataFrame gets passed in, it gets converted
            via .to_records method for slicing efficiency.
        metric_s : callable
            A callable function/class or list of callables
            that take an array of visits and slicepoints as
            input
        info : `dict`
            Dict or list of dicts for holding information
            about the analysis process.
        """

        if hasattr(input_visits, "to_records"):
            visits_array = input_visits.to_records(index=False)
        else:
            visits_array = input_visits

        if not isinstance(visits_array, np.ndarray):
            raise ValueError("input_visits should be numpy array or pandas DataFrame.")

        # No data in, just toss back
        if np.size(input_visits) == 0:
            result = np.empty(self.shape, dtype=float)
            result.fill(self.missing)
            return result, info

        orig_info = copy.copy(info)
        # Construct the KD Tree for this dataset
        self.setup_slicer(visits_array)

        # Check metric_s and info are same length
        if info is not None:
            if isinstance(metric_s, list):
                matching_len = len(metric_s) == len(info)
                if not matching_len:
                    raise ValueError("Length of metric_s must match info length")

        # Naked metric sent in, wrap as a 1-element list
        if not isinstance(metric_s, list):
            metric_s = [metric_s]
            info = [info]

        for metric in metric_s:
            if self.cache:
                if not hasattr(metric, "call_cached"):
                    warnings.warn("Metric does not support cache, turning cache off")
                    self.cache = False

        results = []
        final_info = []
        # See what dtype the metric will return,
        # make an array to hold it.
        for metric, single_info in zip(metric_s, info):
            if hasattr(metric, "shape"):
                if metric.shape is None:
                    result = np.empty(self.shape, dtype=metric.dtype)
                else:
                    result = np.empty((self.shape, metric.shape), dtype=metric.dtype)
            else:
                result = np.empty(self.shape, dtype=float)
            result.fill(self.missing)
            results.append(result)

        for i, slice_i in enumerate(self):
            if len(slice_i["idxs"]) != 0:
                slicedata = visits_array[slice_i["idxs"]]
                for j, metric in enumerate(metric_s):
                    if self.cache:
                        results[j][i] = metric.call_cached(
                            frozenset(slicedata["observationId"].tolist()),
                            slicedata,
                            slice_point=slice_i["slice_point"],
                        )
                    else:
                        results[j][i] = metric(slicedata, slice_point=slice_i["slice_point"])

        if orig_info is not None:
            for single_info, metric in zip(info, metric_s):
                if single_info is not None:
                    single_info = self.add_info(single_info)
                    if hasattr(metric, "add_info"):
                        single_info = metric.add_info(single_info)
                final_info.append(single_info)

        # Unwrap if single metric sent in
        if orig_info is None:
            if len(results) == 1:
                return results[0]
            return results
        else:
            if len(results) == 1:
                return results[0], final_info[0]
            return results, final_info


class UserSlicer(Slicer):
    """For looping over a user-defined set of points.

    Parameters
    ----------
    ra : `np.array`
        The RA points to loop over (degrees).
    dec : `np.array`
        The dec points to loop over (degrees).
    """

    def __init__(
        self,
        ra,
        dec,
        lon_col="fieldRA",
        lat_col="fieldDec",
        leafsize=100,
        radius=2.45,
        use_camera=True,
        camera_footprint_file=None,
        rot_sky_pos_col_name="rotSkyPos",
        missing=np.nan,
        cache=False,
    ):

        super().__init__(
            lon_col=lon_col,
            lat_col=lat_col,
            leafsize=leafsize,
            radius=radius,
            use_camera=use_camera,
            camera_footprint_file=camera_footprint_file,
            rot_sky_pos_col_name=rot_sky_pos_col_name,
            missing=missing,
            cache=cache,
        )

        # Remove the things that were set for a
        # specific nside
        del self.nside
        self.nslice = np.size(ra)
        self.shape = self.nslice

        # Set up slice_point
        self.slice_points = {}
        self.slice_points["nside"] = None
        self.slice_points["sid"] = np.arange(self.nslice)
        self.slice_points["ra"] = np.radians(ra)
        self.slice_points["dec"] = np.radians(dec)
