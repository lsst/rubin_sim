__all__ = (
    "generate_all_sky",
    "filter_count_ratios",
    "SkyAreaGenerator",
    "SkyAreaGeneratorGalplane",
    "EuclidOverlapFootprint",
)

import os
import warnings

import astropy.units as u
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
from numpy.lib import recfunctions as rfn
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from rubin_sim import data as rs_data
from rubin_sim.utils import Site, _angular_separation, angular_separation

from .footprints import ra_dec_hp_map
from .utils import IntRounded, set_default_nside


def generate_all_sky(nside=None, elevation_limit=20, mask=hp.UNSEEN):
    """Set up a healpix map over the entire sky.
    Calculate RA & Dec, Galactic l & b, Ecliptic l & b, for all healpixels.
    Calculate max altitude, to define areas which LSST cannot reach.

    This is intended to be a useful tool to use to set up target maps,
    beyond the standard maps provided in the various SkyArea generator maps.
    Masking based on RA, Dec, Galactic or Ecliptic lat and lon is easier.

    Parameters
    ----------
    nside : `int`, optional
        Resolution for the healpix maps.
        Default None uses rubin_sim.scheduler.utils.set_default_nside
        to set default (often 32).
    elevation_limit : `float`, optional
        Elevation limit for map.
        Parts of the sky which do not reach this elevation limit
        will be set to `mask.`
    mask : `float`, optional
        Mask value for 'unreachable' parts of the sky,
        defined as elevation < 20.

    Returns
    -------
    maps : `dict` {`str`: `np.ndarray`, (N,)}
        A dictionary of `map` (the skymap healpix array, with `mask` values),
        `ra`, `dec`, eclip_lat`, `eclip_lon`, `gal_lat`, `gal_lon` values.
        All coordinates are in radians.
    """
    if nside is None:
        nside = set_default_nside()

    # Calculate coordinates of everything.
    skymap = np.zeros(hp.nside2npix(nside), float)
    ra, dec = ra_dec_hp_map(nside=nside)
    coord = SkyCoord(ra=ra * u.rad, dec=dec * u.rad, frame="icrs")
    eclip_lat = coord.barycentrictrueecliptic.lat.deg
    eclip_lon = coord.barycentrictrueecliptic.lon.deg
    gal_lon = coord.galactic.l.deg
    gal_lat = coord.galactic.b.deg

    # Calculate max altitude (when on meridian).
    lsst_site = Site("LSST")
    elev_max = np.pi / 2.0 - np.abs(dec - lsst_site.latitude_rad)
    skymap = np.where(IntRounded(elev_max) >= IntRounded(np.radians(elevation_limit)), skymap, mask)

    return {
        "map": skymap,
        "ra": np.degrees(ra),
        "dec": np.degrees(dec),
        "eclip_lat": eclip_lat,
        "eclip_lon": eclip_lon,
        "gal_lat": gal_lat,
        "gal_lon": gal_lon,
    }


def filter_count_ratios(target_maps):
    """Compute the desired ratio of observations per filter.

    Given a goal map that includes multiple filters, sum the number of
    pixels in each map and return this per-filter, normalized so the sum
    across all filters is 1.
    If the map is constant over all healpixels, this is the
    ratio of filters at the max/constant value.
    """
    results = {}
    all_norm = 0.0
    for key in target_maps:
        good = target_maps[key] > 0
        results[key] = np.sum(target_maps[key][good])
        all_norm += results[key]
    for key in results:
        results[key] /= all_norm
    return results


class SkyAreaGenerator:
    """
    Generate survey footprint maps in each filter.

    Parameters
    ----------
    nside : `int`
        Healpix nside
    dust_limit : `float`
        E(B-V) limit for dust extinction. Default of 0.199.
    smoothing_cutoff : `float`
       We apply a smoothing filter to the defined dust-free region to
       avoid sharp edges. Larger values = less area, but guaranteed
       less dust extinction. Reflects the value to cut at, after smoothing.
    smoothing_beam : `float`
        The size of the smoothing filter, in degrees.
    lmc_ra, lmc_dec : `float`, `float`
        RA and Dec locations of the LMC, in degrees.
    lmc_radius : `float`
        The radius to use around the LMC, in degrees.
    smc_ra, smc_dec : `float`, `float`
        RA and Dec locations for the center of the SMC, in degrees.
    smc_radius : `float`
        The radius to use around the SMC, degrees.
    scp_dec_max : `float`
        Maximum declination for the south celestial pole region, degrees.
    gal_long1 : `float`
        Longitude at which to start the GP region, in degrees.
    gal_long2 : `float`
        Longitude at which to stop the GP region, degrees.
        Order matters for gal_long1 / gal_long2!
    gal_lat_width_max : `float`
        Max width of the galactic plane, in degrees.
    center_width : `float`
        Width at the center of the galactic plane region, in degrees.
    end_width: `float`
        Width at the remainder of the galactic plane region, in degrees.
    gal_dec_max : `float`
        Maximum declination for the galactic plane region, degrees.
    dusty_dec_min : `float`
        The minimum dec for the dusty plane region, in degrees.
    dusty_dec_max : `float`
        The maximum dec for the dusty plane, degrees.
    eclat_min : `float`
        Ecliptic latitutde minimum for the NES, degrees.
    eclat_max : `float`
        Ecliptic latitude maximum for the NES, degrees.
    eclip_dec_min : `float`
        Declination minimum for the NES, degrees.
    nes_glon_limit : `float`
        Galactic longitude limit for the NES, degrees.
    virgo_ra, virgo_dec : `float`, `float`
        RA and Dec values for the Virgo coverage center, in degrees.
    virgo_radius : `float`
        Radius for the virgo coverage, in degrees.
    """

    def __init__(
        self,
        nside=32,
        dust_limit=0.199,
        smoothing_cutoff=0.45,
        smoothing_beam=10,
        lmc_ra=80.893860,
        lmc_dec=-69.756126,
        lmc_radius=8,
        smc_ra=13.186588,
        smc_dec=-72.828599,
        smc_radius=5,
        scp_dec_max=-60,
        gal_long1=335,
        gal_long2=25,
        gal_lat_width_max=23,
        center_width=12,
        end_width=4,
        gal_dec_max=12,
        low_dust_dec_min=-70,
        low_dust_dec_max=15,
        adjust_halves=12,
        dusty_dec_min=-90,
        dusty_dec_max=15,
        eclat_min=-10,
        eclat_max=10,
        eclip_dec_min=0,
        nes_glon_limit=45.0,
        virgo_ra=186.75,
        virgo_dec=12.717,
        virgo_radius=8.75,
    ):
        self.nside = nside
        self.hpid = np.arange(0, hp.nside2npix(nside))
        self.read_dustmap()

        self.lmc_ra = lmc_ra
        self.lmc_dec = lmc_dec
        self.lmc_radius = lmc_radius
        self.smc_ra = smc_ra
        self.smc_dec = smc_dec
        self.smc_radius = smc_radius

        self.virgo_ra = virgo_ra
        self.virgo_dec = virgo_dec
        self.virgo_radius = virgo_radius

        self.scp_dec_max = scp_dec_max

        self.gal_long1 = gal_long1
        self.gal_long2 = gal_long2
        self.gal_lat_width_max = gal_lat_width_max
        self.center_width = center_width
        self.end_width = end_width
        self.gal_dec_max = gal_dec_max

        self.low_dust_dec_min = low_dust_dec_min
        self.low_dust_dec_max = low_dust_dec_max
        self.adjust_halves = adjust_halves

        self.dusty_dec_min = dusty_dec_min
        self.dusty_dec_max = dusty_dec_max

        self.eclat_min = eclat_min
        self.eclat_max = eclat_max
        self.eclip_dec_min = eclip_dec_min
        self.nes_glon_limit = nes_glon_limit

        # Ra/dec in degrees and other coordinates
        self.ra, self.dec = hp.pix2ang(nside, self.hpid, lonlat=True)
        self.coord = SkyCoord(ra=self.ra * u.deg, dec=self.dec * u.deg, frame="icrs")
        self.eclip_lat = self.coord.barycentrictrueecliptic.lat.deg
        self.eclip_lon = self.coord.barycentrictrueecliptic.lon.deg
        self.gal_lon = self.coord.galactic.l.deg
        self.gal_lat = self.coord.galactic.b.deg

        # Set the low extinction area
        self.low_dust = np.where((self.dustmap < dust_limit), 1, 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            self.low_dust = hp.smoothing(self.low_dust, fwhm=np.radians(smoothing_beam))
        self.low_dust = np.where(self.low_dust > smoothing_cutoff, 1, 0)

    def read_dustmap(self, dustmap_file=None):
        """Read the dustmap from rubin_sim, in the appropriate resolution."""
        # Dustmap from rubin_sim_data
        if dustmap_file is None:
            datadir = rs_data.get_data_dir()
            if datadir is None:
                raise Exception('Cannot find datadir, please set "RUBIN_SIM_DATA_DIR"')
            datadir = os.path.join(datadir, "scheduler", "dust_maps")
            filename = os.path.join(datadir, "dust_nside_%i.npz" % self.nside)
        self.dustmap = np.load(filename)["ebvMap"]

    def _set_circular_region(self, ra_center, dec_center, radius):
        """Define a circular region centered on ra_center, dec_center."""
        # find the healpixels that cover a circle of radius
        # "radius" around ra/dec center (deg).
        result = np.zeros(len(self.ra))
        distance = _angular_separation(
            np.radians(ra_center),
            np.radians(dec_center),
            np.radians(self.ra),
            np.radians(self.dec),
        )
        result[np.where(distance < np.radians(radius))] = 1
        return result

    def _set_bulge_diamond(self, center_width, end_width, gal_long1, gal_long2):
        """
        Define a Galactic Bulge diamond-ish region.

        Parameters
        ----------
        center_width : `float`
            Width at the center of the galactic plane region.
        end_width : `float`
            Width at the remainder of the galactic plane region.
        gal_long1 : `float`
            Longitude at which to start the GP region.
        gal_long2 : `float`
            Longitude at which to stop the GP region.
            Order matters for gal_long1 / gal_long2!

        Returns
        -------
        bulge : `np.ndarray`
            HEALpix array with 1 for bulge pixels, 0 otherwise
        """
        # Reject anything beyond the central width.
        bulge = np.where(np.abs(self.gal_lat) < center_width, 1, 0)
        # Apply the galactic longitude cuts,
        # so that plane goes between gal_long1 to gal_long2.
        # This is NOT the shortest distance between the angles.
        gp_length = (gal_long2 - gal_long1) % 360
        # If the length is greater than 0 then we can add additional cuts.
        if gp_length > 0:
            # First, remove anything outside the gal_long1/gal_long2 region.
            bulge = np.where(((self.gal_lon - gal_long1) % 360) < gp_length, bulge, 0)
            # Add the tapers.
            # These slope from the center (gp_center @ center_width)
            # to the edges (gp_center + gp_length/2 @ end_width).
            half_width = gp_length / 2.0
            slope = (center_width - end_width) / half_width
            # The 'center' can have a wrap-around 0 problem
            gp_center = (gal_long1 + half_width) % 360
            # Calculate the longitude-distance btwn any point and the 'center'
            gp_dist = (self.gal_lon - gp_center) % 360
            gp_dist = np.abs(np.where((gp_dist > 180), (180 - gp_dist) % 180, gp_dist))
            lat_limit = np.abs(center_width - slope * gp_dist)
            bulge = np.where((np.abs(self.gal_lat)) < lat_limit, bulge, 0)
        return bulge

    def _set_bulge_rectangle(self, lat_width, gal_long1, gal_long2):
        """
        Define a Galactic Bulge as a simple rectangle in galactic coordinates,
        centered on the plane

        Parameters
        ----------
        lat_width : `float`
            Latitude with for the galactic bulge
        gal_long1 : float
            Longitude at which to start the GP region.
        gal_long2 : float
            Longitude at which to stop the GP region.
            Order matters for gal_long1 / gal_long2!

        Returns
        -------
        bulge : `np.ndarray`
            HEALpix array with 1 for bulge pixels, 0 otherwise
        """
        bulge = np.where(np.abs(self.gal_lat) < lat_width, 1, 0)
        # This is NOT the shortest distance between the angles.
        gp_length = (gal_long2 - gal_long1) % 360
        # If the length is greater than 0 then we can add additional cuts.
        if gp_length > 0:
            # First, remove anything outside the gal_long1/gal_long2 region.
            bulge = np.where(((self.gal_lon - gal_long1) % 360) < gp_length, bulge, 0)
        return bulge

    def add_virgo_cluster(self, filter_ratios, label="virgo"):
        """Define a circular region around the Virgo Cluster.
        Updates self.healmaps and self.pix_labels

        Parameters
        ----------
        filter_ratios : `dict` {`str`: `float`}
            Dictionary of weights per filter for the footprint.
        label : `str`, optional
            Label to apply to the resulting footprint
        """
        temp_map = np.zeros(hp.nside2npix(self.nside))
        temp_map += self._set_circular_region(self.virgo_ra, self.virgo_dec, self.virgo_radius)
        # Don't overide any pixels that have already been designated
        indx = np.where((temp_map > 0) & (self.pix_labels == ""))
        self.pix_labels[indx] = label
        for filtername in filter_ratios:
            self.healmaps[filtername][indx] = filter_ratios[filtername]

    def add_magellanic_clouds(
        self,
        filter_ratios,
        label="LMC_SMC",
    ):
        """Define circular regions around the Magellanic Clouds.
        Updates self.healmaps and self.pix_labels.

        Parameters
        -----------
        filter_ratios : `dict` {`str`: `float`}
            Dictionary of weights per filter for the footprint.
        label : `str`, optional
            Label to apply to the resulting footprint
        """
        temp_map = np.zeros(hp.nside2npix(self.nside))
        # Define the LMC pixels
        temp_map += self._set_circular_region(self.lmc_ra, self.lmc_dec, self.lmc_radius)
        # Define the SMC pixels
        temp_map += self._set_circular_region(self.smc_ra, self.smc_dec, self.smc_radius)
        # Add a simple bridge between the two - to remove the gap
        mc_dec_min = self.dec[np.where(temp_map > 0)].min()
        mc_dec_max = self.dec[np.where(temp_map > 0)].max()
        temp_map += np.where(
            ((self.ra > self.smc_ra) & (self.ra < self.lmc_ra))
            & ((self.dec > mc_dec_min) & (self.dec < mc_dec_max)),
            1,
            0,
        )

        # Don't overide any pixels that have already been designated
        indx = np.where((temp_map > 0) & (self.pix_labels == ""))

        self.pix_labels[indx] = label
        for filtername in filter_ratios:
            self.healmaps[filtername][indx] = filter_ratios[filtername]

    def add_scp(self, filter_ratios, label="scp"):
        """Define a south celestial pole cap region.
        Updates self.healmaps and self.pix_labels.

        Parameters
        ----------
        filter_ratios : `dict` {`str`: `float`}
            Dictionary of weights per filter for the footprint.
        label : `str`, optional
            Label to apply to the resulting footprint
        """
        indx = np.where((self.dec < self.scp_dec_max) & (self.pix_labels == ""))

        self.pix_labels[indx] = label
        for filtername in filter_ratios:
            self.healmaps[filtername][indx] = filter_ratios[filtername]

    def add_bulge(self, filter_ratios, label="bulge"):
        """Define a bulge region, where the 'bulge' is a large "diamond"
        centered on the galactic center.
        Updates self.healmaps and self.pix_labels.

        Parameters
        ----------
        filter_ratios : `dict` {`str`: `float`}
            Dictionary of weights per filter for the footprint.
        label : `str`, optional
            Label to apply to the resulting footprint
        """
        b1 = self._set_bulge_diamond(
            center_width=self.center_width,
            end_width=self.end_width,
            gal_long1=self.gal_long1,
            gal_long2=self.gal_long2,
        )
        b2 = self._set_bulge_rectangle(self.gal_lat_width_max, self.gal_long1, self.gal_long2)
        b2[np.where(self.gal_lat > 0)] = 0

        bulge = b1 + b2

        bulge[np.where(np.abs(self.gal_lat) > self.gal_lat_width_max)] = 0
        indx = np.where((bulge > 0) & (self.pix_labels == ""))
        self.pix_labels[indx] = label
        for filtername in filter_ratios:
            self.healmaps[filtername][indx] = filter_ratios[filtername]

    def add_lowdust_wfd(self, filter_ratios, label="lowdust"):
        """Define a low-dust WFD region.
        Updates self.healmaps and self.pix_labels.

        Parameters
        ----------
        filter_ratios : `dict` {`str`: `float`}
            Dictionary of weights per filter for the footprint.
        label : `str`, optional
            Label to apply to the resulting footprint
        """
        dustfree = np.where(
            (self.dec > self.low_dust_dec_min) & (self.dec < self.low_dust_dec_max) & (self.low_dust == 1),
            1,
            0,
        )

        dustfree[np.where(self.low_dust == 0)] = 0

        if self.adjust_halves > 0:
            dustfree = np.where(
                (self.gal_lat < 0) & (self.dec > self.low_dust_dec_max - self.adjust_halves),
                0,
                dustfree,
            )

        indx = np.where((dustfree > 0) & (self.pix_labels == ""))
        self.pix_labels[indx] = label
        for filtername in filter_ratios:
            self.healmaps[filtername][indx] = filter_ratios[filtername]

    def add_dusty_plane(self, filter_ratios, label="dusty_plane"):
        """Define high-dust region of the map.
        Updates self.healmaps and self.pix_labels.

        Parameters
        ----------
        filter_ratios : `dict` {`str`: `float`}
            Dictionary of weights per filter for the footprint.
        label : `str`, optional
            Label to apply to the resulting footprint
        """
        dusty = np.where(
            ((self.dec > self.dusty_dec_min) & (self.dec < self.dusty_dec_max) & (self.low_dust == 0)),
            1,
            0,
        )

        indx = np.where((dusty > 0) & (self.pix_labels == ""))
        self.pix_labels[indx] = label
        for filtername in filter_ratios:
            self.healmaps[filtername][indx] = filter_ratios[filtername]

    def add_nes(self, filter_ratios, label="nes"):
        """Define a North Ecliptic Plane region.
        Updates self.healmaps and self.pix_labels.

        Parameters
        ----------
        filter_ratios : `dict` {`str`: `float`}
            Dictionary of weights per filter for the footprint.
        label : `str`, optional
            Label to apply to the resulting footprint
        """
        nes = np.where(
            ((self.eclip_lat > self.eclat_min) | (self.dec > self.eclip_dec_min))
            & (self.eclip_lat < self.eclat_max),
            1,
            0,
        )

        nes[np.where(self.gal_lon < self.nes_glon_limit)] = 0
        nes[np.where(self.gal_lon > (360 - self.nes_glon_limit))] = 0

        indx = np.where((nes > 0) & (self.pix_labels == ""))
        self.pix_labels[indx] = label
        for filtername in filter_ratios:
            self.healmaps[filtername][indx] = filter_ratios[filtername]

    def return_maps(
        self,
        magellenic_clouds_ratios={
            "u": 0.32,
            "g": 0.4,
            "r": 1.0,
            "i": 1.0,
            "z": 0.9,
            "y": 0.9,
        },
        scp_ratios={"u": 0.1, "g": 0.1, "r": 0.1, "i": 0.1, "z": 0.1, "y": 0.1},
        nes_ratios={"g": 0.28, "r": 0.4, "i": 0.4, "z": 0.28},
        dusty_plane_ratios={
            "u": 0.1,
            "g": 0.28,
            "r": 0.28,
            "i": 0.28,
            "z": 0.28,
            "y": 0.1,
        },
        low_dust_ratios={"u": 0.32, "g": 0.4, "r": 1.0, "i": 1.0, "z": 0.9, "y": 0.9},
        bulge_ratios={"u": 0.18, "g": 1.0, "r": 1.05, "i": 1.05, "z": 1.0, "y": 0.23},
        virgo_ratios={"u": 0.32, "g": 0.4, "r": 1.0, "i": 1.0, "z": 0.9, "y": 0.9},
    ):
        """
        Return the survey sky maps and labels.

        Parameters
        ----------
        magellanic_clouds_ratios :  `dict` {`str`: `float`}
            Magellanic clouds filter ratios.
        scp_ratios :  `dict` {`str`: `float`}
            SCP filter ratios.
        nes_ratios :  `dict` {`str`: `float`}
            NES filter ratios
        dusty_plane-ratios :  `dict` {`str`: `float`}
            dusty plane filter ratios
        low_dust_ratios :  `dict` {`str`: `float`}
            Low Dust WFD filter ratios.
        bulge_ratios :  `dict` {`str`: `float`}
            Bulge region filter ratios.
        virgo_ratios :  `dict` {`str`: `float`}
            Virgo cluster coverage filter ratios.

        Returns
        --------
        self.healmaps, self.pix_labels : `np.ndarray`, (N,), `np.ndarray`, (N,)
            HEALPix target survey maps for ugrizy,
            and string labels for each healpix to indicate the "region".


        Notes
        -----
        Each healpix point can only belong to one region. Which region it is
        assigned to first will be used for its definition, thus order
        matters within this method. The region defines the filter ratios.

        The filter ratios contain information about the *ratio* of visits
        in that region (compared to some reference point in the entire map)
        in each particular filter. By convention, the low_dust_wfd ratio
        in r band is set to "1" and all other values are then in reference
        to that.

        For example: if scp_ratios['u'] = 0.1 and the low_dust_wfd['r'] = 1,
        then when the low-dust WFD has 10 visits in r band, the SCP should
        have obtained 1 visits in u band (per pixel).
        """

        # Array to hold the labels for each pixel
        self.pix_labels = np.zeros(hp.nside2npix(self.nside), dtype="U20")
        self.healmaps = np.zeros(
            hp.nside2npix(self.nside),
            dtype=list(zip(["u", "g", "r", "i", "z", "y"], [float] * 7)),
        )

        # Note, order here matters.
        # Once a HEALpix is set and labled, subsequent add_ methods
        # will not override that pixel.
        self.add_magellanic_clouds(magellenic_clouds_ratios)
        self.add_lowdust_wfd(low_dust_ratios)
        self.add_virgo_cluster(virgo_ratios)
        self.add_bulge(bulge_ratios)
        self.add_nes(nes_ratios)
        self.add_dusty_plane(dusty_plane_ratios)
        self.add_scp(scp_ratios)

        return self.healmaps, self.pix_labels

    def estimate_visits(self, nvis_total, fov_area=9.6, **kwargs):
        """Convience method for converting relative maps into number of visits

        Parameters
        ----------
        nvis_total : `int`
            The total number of visits in the survey
        fov_area : `float`
            The area of a single visit (sq degrees)
        **kwargs :
            Gets passed to self.return_maps if one wants to change the
            default ratios.

        Returns
        -------
        result : `np.array`, (N,)
            array with filtername dtypes that have HEALpix arrays with the
            number of expected visits of each HEALpix center
        sum_map : `np.array`, (N,)
            The number of visits summed over all the filters
        labels : `np.ndarray`, (N,)
            Array string labels for each HEALpix
        """
        # Note that there really ought to be a fudge factor here;
        # typically the number of visits per pixel will be slightly
        # more than requested in the map, due to dithering.
        # The fudge factor will depend on the complexity of the region's
        # boundaries, but can be on the order of 1.3.
        healmaps, labels = self.return_maps(**kwargs)

        sum_map = rfn.structured_to_unstructured(healmaps).sum(axis=1)

        norm = np.sum(sum_map)
        pix_area = hp.nside2pixarea(self.nside, degrees=True)
        pix_per_visit = fov_area / pix_area

        result = np.zeros_like(healmaps)
        for key in result.dtype.names:
            result[key] = healmaps[key] / norm * pix_per_visit * nvis_total

        return result, sum_map / norm * pix_per_visit * nvis_total, labels

    def estimate_visits_per_label(self, nvis_total, **kwargs):
        """Estimate how many visits would be used for each region

        Parameters
        ----------
        nvis_total : `int`
            The total number of visits in the survey
        **kwargs :
            Gets passed to self.return_maps if one wants to change the
            default ratios.

        Returns
        -------
        result : `dict`
            Dictionary with keys that are label names and values that are the
            expected number of visits for that region if nvis_total is reached.
        """

        healmaps, labels = self.return_maps(**kwargs)
        sum_map = rfn.structured_to_unstructured(healmaps).sum(axis=1)
        ulabels = np.unique(labels)
        label_sums = {}
        norm = 0
        for label in ulabels:
            in_region = np.where(labels == label)
            label_sums[label] = sum_map[in_region].sum()
            norm += label_sums[label]
        result = {}
        for key in ulabels:
            result[key] = label_sums[key] / norm * nvis_total
        return result


class SkyAreaGeneratorGalplane(SkyAreaGenerator):
    """
    Generate survey footprint maps in each filter.
    Adds a 'bulgy' galactic plane coverage map.

    Parameters
    ----------
    nside : `int`
        Healpix nside
    dust_limit : `float`
        E(B-V) limit for dust extinction. Default of 0.199.
    smoothing_cutoff : `float`
       We apply a smoothing filter to the defined dust-free region to
       avoid sharp edges. Larger values = less area, but guaranteed
       less dust extinction. Reflects the value to cut at, after smoothing.
    smoothing_beam : `float`
        The size of the smoothing filter, in degrees.
    lmc_ra, lmc_dec : `float`, `float`
        RA and Dec locations of the LMC, in degrees.
    lmc_radius : `float`
        The radius to use around the LMC, in degrees.
    smc_ra, smc_dec : `float`, `float`
        RA and Dec locations for the center of the SMC, in degrees.
    smc_radius : `float`
        The radius to use around the SMC, degrees.
    scp_dec_max : `float`
        Maximum declination for the south celestial pole region, degrees.
    gal_long1 : `float`
        Longitude at which to start the GP region, in degrees.
    gal_long2 : `float`
        Longitude at which to stop the GP region, degrees.
        Order matters for gal_long1 / gal_long2!
    gal_lat_width_max : `float`
        Max width of the galactic plane, in degrees.
    center_width : `float`
        Width at the center of the galactic plane region, in degrees.
    end_width: `float`
        Width at the remainder of the galactic plane region, in degrees.
    gal_dec_max : `float`
        Maximum declination for the galactic plane region, degrees.
    dusty_dec_min : `float`
        The minimum dec for the dusty plane region, in degrees.
    dusty_dec_max : `float`
        The maximum dec for the dusty plane, degrees.
    eclat_min : `float`
        Ecliptic latitutde minimum for the NES, degrees.
    eclat_max : `float`
        Ecliptic latitude maximum for the NES, degrees.
    eclip_dec_min : `float`
        Declination minimum for the NES, degrees.
    nes_glon_limit : `float`
        Galactic longitude limit for the NES, degrees.
    virgo_ra, virgo_dec : `float`, `float`
        RA and Dec values for the Virgo coverage center, in degrees.
    virgo_radius : `float`
        Radius for the virgo coverage, in degrees.
    """

    def __init__(self, lmc_ra=89.0, lmc_dec=-70, **kwargs):
        super().__init__(lmc_ra=lmc_ra, lmc_dec=lmc_dec, **kwargs)

    def add_bulgy(self, filter_ratios, label="bulgy"):
        """Define a bulge region, where the 'bulge' is a series of
        circles set by points defined to match as best as possible the
        map requested by the SMWLV working group on galactic plane coverage.
        Implemented in v3.0.
        Updates self.healmaps and self.pix_labels.

        Parameters
        ----------
        filter_ratios : `dict` {`str`: `float`}
            Dictionary of weights per filter for the footprint.
        label : `str`, optional
            Label to apply to the resulting footprint
        """
        # Some RA, dec, radius points that
        # seem to cover the areas that are desired
        points = [
            [100.90, 9.55, 3],
            [84.92, -5.71, 3],
            [288.84, 9.18, 3.8],
            [266.3, -29, 14.5],
            [279, -13, 10],
            [256, -45, 5],
            [155, -56.5, 6.5],
            [172, -62, 5],
            [190, -65, 5],
            [210, -64, 5],
            [242, -58, 5],
            [225, -60, 6.5],
        ]
        for point in points:
            dist = angular_separation(self.ra, self.dec, point[0], point[1])
            # Only change pixels where the label isn't already set.
            indx = np.where((dist < point[2]) & (self.pix_labels == ""))
            self.pix_labels[indx] = label
            for filtername in filter_ratios:
                self.healmaps[filtername][indx] = filter_ratios[filtername]

    def return_maps(
        self,
        magellenic_clouds_ratios={
            "u": 0.32,
            "g": 0.4,
            "r": 1.0,
            "i": 1.0,
            "z": 0.9,
            "y": 0.9,
        },
        low_dust_ratios={"u": 0.32, "g": 0.4, "r": 1.0, "i": 1.0, "z": 0.9, "y": 0.9},
        virgo_ratios={"u": 0.32, "g": 0.4, "r": 1.0, "i": 1.0, "z": 0.9, "y": 0.9},
        scp_ratios={"u": 0.08, "g": 0.15, "r": 0.08, "i": 0.15, "z": 0.08, "y": 0.06},
        nes_ratios={"g": 0.23, "r": 0.33, "i": 0.33, "z": 0.23},
        bulge_ratios={"u": 0.19, "g": 0.57, "r": 1.15, "i": 1.05, "z": 0.78, "y": 0.57},
        dusty_plane_ratios={
            "u": 0.07,
            "g": 0.13,
            "r": 0.28,
            "i": 0.28,
            "z": 0.25,
            "y": 0.18,
        },
    ):
        """
        Return the survey sky maps and labels.

        Parameters
        ----------
        magellanic_clouds_ratios :  `dict` {`str`: `float`}
            Magellanic clouds filter ratios.
        scp_ratios :  `dict` {`str`: `float`}
            SCP filter ratios.
        nes_ratios :  `dict` {`str`: `float`}
            NES filter ratios
        dusty_plane-ratios :  `dict` {`str`: `float`}
            dusty plane filter ratios
        low_dust_ratios :  `dict` {`str`: `float`}
            Low Dust WFD filter ratios.
        bulge_ratios :  `dict` {`str`: `float`}
            Bulge region filter ratios (note this is the 'bulgy' bulge).
        virgo_ratios :  `dict` {`str`: `float`}
            Virgo cluster coverage filter ratios.

        Returns
        --------
        self.healmaps, self.pix_labels : `np.ndarray`, (N,), `np.ndarray`, (N,)
            HEALPix target survey maps for ugrizy,
            and string labels for each healpix to indicate the "region".


        Notes
        -----
        Each healpix point can only belong to one region. Which region it is
        assigned to first will be used for its definition, thus order
        matters within this method. The region defines the filter ratios.

        The filter ratios contain information about the *ratio* of visits
        in that region (compared to some reference point in the entire map)
        in each particular filter. By convention, the low_dust_wfd ratio
        in r band is set to "1" and all other values are then in reference
        to that.

        For example: if scp_ratios['u'] = 0.1 and the low_dust_wfd['r'] = 1,
        then when the low-dust WFD has 10 visits in r band, the SCP should
        have obtained 1 visits in u band (per pixel).
        """
        self.pix_labels = np.zeros(hp.nside2npix(self.nside), dtype="U20")
        dt = list(zip(["u", "g", "r", "i", "z", "y"], [float] * 7))
        self.healmaps = np.zeros(hp.nside2npix(self.nside), dtype=dt)

        self.add_magellanic_clouds(magellenic_clouds_ratios)
        self.add_lowdust_wfd(low_dust_ratios)
        self.add_virgo_cluster(virgo_ratios)
        self.add_bulgy(bulge_ratios)
        self.add_nes(nes_ratios)
        self.add_dusty_plane(dusty_plane_ratios)
        self.add_scp(scp_ratios)

        return self.healmaps, self.pix_labels


class EuclidOverlapFootprint(SkyAreaGeneratorGalplane):
    """
    Generate survey footprint maps in each filter.
    This uses a bulgy coverage in the galactic plane, plus small
    Euclid footprint extension to the low-dust WFD.

    Parameters
    ----------
    nside : `int`
        Healpix nside
    dust_limit : `float`
        E(B-V) limit for dust extinction. Default of 0.199.
    smoothing_cutoff : `float`
       We apply a smoothing filter to the defined dust-free region to
       avoid sharp edges. Larger values = less area, but guaranteed
       less dust extinction. Reflects the value to cut at, after smoothing.
    smoothing_beam : `float`
        The size of the smoothing filter, in degrees.
    lmc_ra, lmc_dec : `float`, `float`
        RA and Dec locations of the LMC, in degrees.
    lmc_radius : `float`
        The radius to use around the LMC, in degrees.
    smc_ra, smc_dec : `float`, `float`
        RA and Dec locations for the center of the SMC, in degrees.
    smc_radius : `float`
        The radius to use around the SMC, degrees.
    scp_dec_max : `float`
        Maximum declination for the south celestial pole region, degrees.
    gal_long1 : `float`
        Longitude at which to start the GP region, in degrees.
    gal_long2 : `float`
        Longitude at which to stop the GP region, degrees.
        Order matters for gal_long1 / gal_long2!
    gal_lat_width_max : `float`
        Max width of the galactic plane, in degrees.
    center_width : `float`
        Width at the center of the galactic plane region, in degrees.
    end_width: `float`
        Width at the remainder of the galactic plane region, in degrees.
    gal_dec_max : `float`
        Maximum declination for the galactic plane region, degrees.
    dusty_dec_min : `float`
        The minimum dec for the dusty plane region, in degrees.
    dusty_dec_max : `float`
        The maximum dec for the dusty plane, degrees.
    eclat_min : `float`
        Ecliptic latitutde minimum for the NES, degrees.
    eclat_max : `float`
        Ecliptic latitude maximum for the NES, degrees.
    eclip_dec_min : `float`
        Declination minimum for the NES, degrees.
    nes_glon_limit : `float`
        Galactic longitude limit for the NES, degrees.
    virgo_ra, virgo_dec : `float`, `float`
        RA and Dec values for the Virgo coverage center, in degrees.
    virgo_radius : `float`
        Radius for the virgo coverage, in degrees.
    euclid_contour_file : `str`
        File containing a Euclid footprint contour file.
        Default of none uses the file in rubin_sim_data/scheduler.
    """

    def __init__(self, euclid_contour_file=None, lmc_radius=6, smc_radius=4, **kwargs):
        super().__init__(lmc_radius=lmc_radius, smc_radius=smc_radius, **kwargs)
        self.euclid_contour_file = euclid_contour_file

    def add_euclid_overlap(
        self,
        filter_ratios,
        label="euclid_overlap",
    ):
        """Define a small extension (few degrees) to the low-dust WFD
        to accomodate the Euclid footprint.
        Updates self.healmaps and self.pix_labels.

        Parameters
        ----------
        filter_ratios : `dict` {`str`: `float`}
            Dictionary of weights per filter for the footprint.
        label : `str`, optional
            Label to apply to the resulting footprint
        """
        names = ["RA", "dec"]
        types = [float, float]
        if self.euclid_contour_file is None:
            self.euclid_contour_file = os.path.join(
                rs_data.get_data_dir(), "scheduler/EWS.SGC.Mainland.ROI.2022.RADEC.txt"
            )
        euclid_contours = np.genfromtxt(self.euclid_contour_file, dtype=list(zip(names, types)))

        wrap_ra = self.ra + 0
        wrap_ra[np.where(wrap_ra > 180)] -= 360

        polygon = Polygon(zip(euclid_contours["RA"], euclid_contours["dec"]))
        in_poly = [polygon.contains(Point(x, y)) for x, y in zip(wrap_ra, self.dec)]

        # find which map points are inside the contour
        indx = np.where((np.array(in_poly) == True) & (self.pix_labels == ""))[0]
        self.pix_labels[indx] = label
        for filtername in filter_ratios:
            self.healmaps[filtername][indx] = filter_ratios[filtername]

    def return_maps(
        self,
        magellenic_clouds_ratios={
            "u": 0.32,
            "g": 0.4,
            "r": 1.0,
            "i": 1.0,
            "z": 0.9,
            "y": 0.9,
        },
        low_dust_ratios={"u": 0.32, "g": 0.4, "r": 1.0, "i": 1.0, "z": 0.9, "y": 0.9},
        virgo_ratios={"u": 0.32, "g": 0.4, "r": 1.0, "i": 1.0, "z": 0.9, "y": 0.9},
        scp_ratios={"u": 0.08, "g": 0.15, "r": 0.08, "i": 0.15, "z": 0.08, "y": 0.06},
        nes_ratios={"g": 0.23, "r": 0.33, "i": 0.33, "z": 0.23},
        bulge_ratios={"u": 0.19, "g": 0.57, "r": 1.15, "i": 1.05, "z": 0.78, "y": 0.57},
        dusty_plane_ratios={
            "u": 0.07,
            "g": 0.13,
            "r": 0.28,
            "i": 0.28,
            "z": 0.25,
            "y": 0.18,
        },
        euclid_ratios={"u": 0.32, "g": 0.4, "r": 1.0, "i": 1.0, "z": 0.9, "y": 0.9},
    ):
        """
        Return the survey sky maps and labels.

        Parameters
        ----------
        magellanic_clouds_ratios :  `dict` {`str`: `float`}
            Magellanic clouds filter ratios.
        scp_ratios :  `dict` {`str`: `float`}
            SCP filter ratios.
        nes_ratios :  `dict` {`str`: `float`}
            NES filter ratios
        dusty_plane-ratios :  `dict` {`str`: `float`}
            dusty plane filter ratios
        low_dust_ratios :  `dict` {`str`: `float`}
            Low Dust WFD filter ratios.
        bulge_ratios :  `dict` {`str`: `float`}
            Bulge region filter ratios (note this is the 'bulgy' bulge).
        virgo_ratios :  `dict` {`str`: `float`}
            Virgo cluster coverage filter ratios.
        euclid_ratios : `dict` {`str`: `float`}
            Euclid footprint overlap ratios.

        Returns
        --------
        self.healmaps, self.pix_labels : `np.ndarray`, (N,), `np.ndarray`, (N,)
            HEALPix target survey maps for ugrizy,
            and string labels for each healpix to indicate the "region".


        Notes
        -----
        Each healpix point can only belong to one region. Which region it is
        assigned to first will be used for its definition, thus order
        matters within this method. The region defines the filter ratios.

        The filter ratios contain information about the *ratio* of visits
        in that region (compared to some reference point in the entire map)
        in each particular filter. By convention, the low_dust_wfd ratio
        in r band is set to "1" and all other values are then in reference
        to that.

        For example: if scp_ratios['u'] = 0.1 and the low_dust_wfd['r'] = 1,
        then when the low-dust WFD has 10 visits in r band, the SCP should
        have obtained 1 visits in u band (per pixel).
        """

        # Array to hold the labels for each pixel
        self.pix_labels = np.zeros(hp.nside2npix(self.nside), dtype="U20")
        self.healmaps = np.zeros(
            hp.nside2npix(self.nside),
            dtype=list(zip(["u", "g", "r", "i", "z", "y"], [float] * 7)),
        )

        # Note, order here matters.
        # Once a HEALpix is set and labled, subsequent add_ methods
        # will not override that pixel.
        self.add_magellanic_clouds(magellenic_clouds_ratios)
        self.add_lowdust_wfd(low_dust_ratios)
        self.add_virgo_cluster(virgo_ratios)
        self.add_bulgy(bulge_ratios)
        self.add_nes(nes_ratios)
        self.add_dusty_plane(dusty_plane_ratios)
        self.add_euclid_overlap(euclid_ratios)
        self.add_scp(scp_ratios)

        return self.healmaps, self.pix_labels
