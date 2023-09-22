__all__ = ("SkyAreaGenerator", "SkyAreaGeneratorGalplane", "EuclidOverlapFootprint")

import os
import warnings

import astropy.units as u
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
from numpy.lib import recfunctions as rfn
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import rubin_sim.utils as rs_utils
from rubin_sim import data as rs_data
from rubin_sim.utils import angular_separation


class SkyAreaGenerator:
    """
    Generate survey footprint maps in each filter.

    Parameters
    ----------
    nside : `int` (32)
        Healpix nside (32)
    dust_limit : `float` (0.199)
        E(B-V) limit for dust extinction. Default of 0.199.
    smoothing_cutoff : `float` (0.45)
       We apply a smoothing filter to the defined dust-free region to avoid sharp edges.
       Larger values = less area, but guaranteed less dust extinction. Default 0.45 (degrees).
    smoothing_beam : `float` (10)
        The size of the smoothing filter, in degrees. Default 10.
    lmc_radius : `float` (8)
        The radius to use around the LMC (degrees).
    smc_radius : `float` (5)
        The radius to use around the SMC (degrees)
    scp_dec_max : `float` (-60)
        Maximum declination for the south celestial pole region (degrees)
    gal_long1 : `float` (335)
        Longitude at which to start the GP region (degrees).
    gal_long2 : `float` (25)
        Longitude at which to stop the GP region (degrees).
        Order matters for gal_long1 / gal_long2!
    gal_lat_width_max : `float` (23)
        Max width of the galactic plane (degrees)
    center_width : `float` (12)
        Width at the center of the galactic plane region (degrees).
    end_width: `float` (4)
        Width at the remainder of the galactic plane region.
    gal_dec_max : `float` (12)
        Maximum declination for the galactic plane region (degrees).
    dusty_dec_min : `float` (-90)
        The minimum dec for the dusty plane region (degrees)
    dusty_dec_max : `float` (15)
        The maximum dec for the dusty plane (degrees)
    eclat_min : `float` (-10)
        Ecliptic latitutde minimum for the NES (degrees).
    eclat_max : `float` (10)
        Ecliptic latitude maximum for the NES (degrees).
    eclip_dec_min : `float` (0)
        Declination minimum for the NES (degrees)
    nes_glon_limit : `float` (45.)
        Galactic longitude limit for the NES (degrees).
    """

    def __init__(
        self,
        nside=32,
        dust_limit=0.199,
        smoothing_cutoff=0.45,
        smoothing_beam=10,
        lmc_radius=8,
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

        self.lmc_radius = lmc_radius
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
        # Dustmap from rubin_sim_data  - this is basically just a data directory
        if dustmap_file is None:
            datadir = rs_data.get_data_dir()
            if datadir is None:
                raise Exception('Cannot find datadir, please set "RUBIN_SIM_DATA_DIR"')
            datadir = os.path.join(datadir, "scheduler", "dust_maps")
            filename = os.path.join(datadir, "dust_nside_%i.npz" % self.nside)
        self.dustmap = np.load(filename)["ebvMap"]

    def _set_circular_region(self, ra_center, dec_center, radius):
        # find the healpixels that cover a circle of radius radius around ra/dec center (deg)
        result = np.zeros(len(self.ra))
        distance = rs_utils._angular_separation(
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
        center_width : float
            Width at the center of the galactic plane region.
        end_width : float
            Width at the remainder of the galactic plane region.
        gal_long1 : float
            Longitude at which to start the GP region.
        gal_long2 : float
            Longitude at which to stop the GP region.
            Order matters for gal_long1 / gal_long2!
        Returns
        -------
        bulge : np.ndarray
            HEALpix array with 1 for bulge pixels, 0 otherwise
        """
        # Reject anything beyond the central width.
        bulge = np.where(np.abs(self.gal_lat) < center_width, 1, 0)
        # Apply the galactic longitude cuts, so that plane goes between gal_long1 to gal_long2.
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
            # Calculate the longitude-distance between any point and the 'center'
            gp_dist = (self.gal_lon - gp_center) % 360
            gp_dist = np.abs(np.where((gp_dist > 180), (180 - gp_dist) % 180, gp_dist))
            lat_limit = np.abs(center_width - slope * gp_dist)
            bulge = np.where((np.abs(self.gal_lat)) < lat_limit, bulge, 0)
        return bulge

    def _set_bulge_rectangle(self, lat_width, gal_long1, gal_long2):
        """
        Define a Galactic Bulge as a simple rectangle in galactic coordinates, centered on the plane

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
        bulge : np.ndarray
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
        lmc_ra=80.893860,
        lmc_dec=-69.756126,
        smc_ra=13.186588,
        smc_dec=-72.828599,
    ):
        temp_map = np.zeros(hp.nside2npix(self.nside))
        # Define the LMC pixels
        temp_map += self._set_circular_region(lmc_ra, lmc_dec, self.lmc_radius)
        # Define the SMC pixels
        temp_map += self._set_circular_region(smc_ra, smc_dec, self.smc_radius)
        # Add a simple bridge between the two - to remove the gap
        mc_dec_min = self.dec[np.where(temp_map > 0)].min()
        mc_dec_max = self.dec[np.where(temp_map > 0)].max()
        temp_map += np.where(
            ((self.ra > smc_ra) & (self.ra < lmc_ra)) & ((self.dec > mc_dec_min) & (self.dec < mc_dec_max)),
            1,
            0,
        )

        # Don't overide any pixels that have already been designated
        indx = np.where((temp_map > 0) & (self.pix_labels == ""))

        self.pix_labels[indx] = label
        for filtername in filter_ratios:
            self.healmaps[filtername][indx] = filter_ratios[filtername]

    def add_scp(self, filter_ratios, label="scp"):
        indx = np.where((self.dec < self.scp_dec_max) & (self.pix_labels == ""))

        self.pix_labels[indx] = label
        for filtername in filter_ratios:
            self.healmaps[filtername][indx] = filter_ratios[filtername]

    def add_bulge(self, filter_ratios, label="bulge"):
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
        Parameters:
        various_ratios : `dict`
            Dict with filternames for keys and floats for values that are the desired ratio
            of observations in each filter. By conventions, I usually set the low_dust_ratios['r']=1,
            then all the other values can be interpreted relative to that. E.g., if scp_ratios['u']=0.1, then
            when the low_dust r has 10 visits (per pixel) the scp should have 1 vist (per pixel).

        Returns
        --------
        self.healmaps : `np.ndarray`
            HEALPix maps for ugrizy
        self.pix_labels : `np.ndarray`
            Array string labels for each HEALpix
        """

        # Array to hold the labels for each pixel
        self.pix_labels = np.zeros(hp.nside2npix(self.nside), dtype="U20")
        self.healmaps = np.zeros(
            hp.nside2npix(self.nside),
            dtype=list(zip(["u", "g", "r", "i", "z", "y"], [float] * 7)),
        )

        # Note, order here matters. Once a HEALpix is set and labled, subsequent add_ methods
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
        fov_area : `float` (9.6)
            The area of a single visit (sq degrees)
        **kwargs :
            Gets passed to self.return_maps if one wants to change the
            default ratios.

        Returns
        -------
        result : `np.array`
            array with filtername dtypes that have HEALpix arrays with the
            number of expected visits of each HEALpix center
        sum_map : `np.array`
            The number of visits summed over all the filters
        labels : `np.ndarray`
            Array string labels for each HEALpix
        """

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
    def add_bulgy(self, filter_ratios, label="bulgy"):
        """Properly set the self.healmaps and self.pix_labels for the bulgy area."""
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
        lmc_ra=89.0,
        lmc_dec=-70,
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
        self.pix_labels = np.zeros(hp.nside2npix(self.nside), dtype="U20")
        dt = list(zip(["u", "g", "r", "i", "z", "y"], [float] * 7))
        self.healmaps = np.zeros(hp.nside2npix(self.nside), dtype=dt)

        self.add_magellanic_clouds(magellenic_clouds_ratios, lmc_ra=lmc_ra, lmc_dec=lmc_dec)
        self.add_lowdust_wfd(low_dust_ratios)
        self.add_virgo_cluster(virgo_ratios)
        self.add_bulgy(bulge_ratios)
        self.add_nes(nes_ratios)
        self.add_dusty_plane(dusty_plane_ratios)
        self.add_scp(scp_ratios)

        return self.healmaps, self.pix_labels


class EuclidOverlapFootprint(SkyAreaGeneratorGalplane):
    def add_euclid_overlap(
        self,
        filter_ratios,
        label="euclid_overlap",
        contour_file=None,
        south_limit=-50.0,
    ):
        names = ["RA", "dec"]
        types = [float, float]
        if contour_file is None:
            contour_file = os.path.join(
                rs_data.get_data_dir(), "scheduler/EWS.SGC.Mainland.ROI.2022.RADEC.txt"
            )
        euclid_contours = np.genfromtxt(contour_file, dtype=list(zip(names, types)))

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
        lmc_ra=89.0,
        lmc_dec=-70,
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
        Parameters:
        various_ratios : `dict`
            Dict with filternames for keys and floats for values that are the desired ratio
            of observations in each filter. By conventions, I usually set the low_dust_ratios['r']=1,
            then all the other values can be interpreted relative to that. E.g., if scp_ratios['u']=0.1, then
            when the low_dust r has 10 visits (per pixel) the scp should have 1 vist (per pixel).

        Returns
        --------
        self.healmaps : `np.ndarray`
            HEALPix maps for ugrizy
        self.pix_labels : `np.ndarray`
            Array string labels for each HEALpix
        """

        # Array to hold the labels for each pixel
        self.pix_labels = np.zeros(hp.nside2npix(self.nside), dtype="U20")
        self.healmaps = np.zeros(
            hp.nside2npix(self.nside),
            dtype=list(zip(["u", "g", "r", "i", "z", "y"], [float] * 7)),
        )

        # Note, order here matters. Once a HEALpix is set and labled, subsequent add_ methods
        # will not override that pixel.
        self.add_magellanic_clouds(magellenic_clouds_ratios, lmc_ra=lmc_ra, lmc_dec=lmc_dec)
        self.add_lowdust_wfd(low_dust_ratios)
        self.add_virgo_cluster(virgo_ratios)
        self.add_bulgy(bulge_ratios)
        self.add_nes(nes_ratios)
        self.add_dusty_plane(dusty_plane_ratios)
        self.add_euclid_overlap(euclid_ratios)
        self.add_scp(scp_ratios)

        return self.healmaps, self.pix_labels
