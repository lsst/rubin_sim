__all__ = ("DustMap",)

import warnings

from rubin_sim.maf.maps import BaseMap

from .ebv_hp import eb_vhp


class DustMap(BaseMap):
    """Add the E(B-V) values to the slice points.

    Primarily, this calls eb_vhp to read a healpix map of E(B-V) values over
    the sky, then assigns ebv values to each slice_point.
    If the slicer is a healpix slicer, this is trivial.
    Otherwise, it either uses the nearest healpix grid point or interpolates.

    The key added to the slice points is `ebv`.

    Parameters
    ----------
    interp : `bool`, opt
        Interpolate the dust map at each slice_point (True)
        or just use the nearest value (False).
        Default is False.
    nside : `int`, opt
        Default nside value to read the dust map from disk.
        Primarily useful if the slicer is not a healpix slicer.
        Default 128.
    map_path : `str`, opt
        Define a path to the directory holding the dust map files.
        Default None, which uses RUBIN_SIM_DATA_DIR.
    """

    def __init__(self, interp=False, nside=128, map_path=None):
        self.keynames = ["ebv"]
        self.interp = interp
        self.nside = nside
        self.map_path = map_path

    def run(self, slice_points):
        # If the slicer has nside,
        # it's a healpix slicer so we can read the map directly
        if "nside" in slice_points:
            if slice_points["nside"] != self.nside:
                warnings.warn(
                    f"Slicer value of nside {slice_points['nside']} different "
                    f"from map value {self.nside}, using slicer value"
                )
            slice_points["ebv"] = eb_vhp(
                slice_points["nside"],
                pixels=slice_points["sid"],
                map_path=self.map_path,
            )
        # Not a healpix slicer,
        # look up values based on RA,dec with possible interpolation
        else:
            slice_points["ebv"] = eb_vhp(
                self.nside,
                ra=slice_points["ra"],
                dec=slice_points["dec"],
                interp=self.interp,
                map_path=self.map_path,
            )

        return slice_points
