import numbers
import os
import unittest

import numpy as np

import rubin_sim
from rubin_sim.data import get_data_dir
from rubin_sim.utils import (
    HalfSpace,
    _angular_separation,
    angular_separation,
    arcsec_from_radians,
    basic_trixels,
    cartesian_from_spherical,
    find_htmid,
    get_all_trixels,
    half_space_from_points,
    half_space_from_ra_dec,
    intersect_half_spaces,
    level_from_htmid,
    rot_about_x,
    rot_about_y,
    rot_about_z,
    spherical_from_cartesian,
    trixel_from_htmid,
)
from rubin_sim.utils.htm_module import _find_htmid_fast, _find_htmid_slow


def trixel_intersects_half_space(trix, hspace):
    """
    This is a brute force method to determine whether a trixel
    is inside, or at least intersects, a halfspace.
    """
    if hspace.phi > 0.25 * np.pi:
        raise RuntimeError("trixel_intersects_half_space is not safe for " "large HalfSpaces")

    # if any of the trixel's corners are within the
    # HalfSpace, return True
    ra_rad, dec_rad = spherical_from_cartesian(hspace.vector)
    for corner in trix.corners:
        ra_rad1, dec_rad1 = spherical_from_cartesian(corner)
        if _angular_separation(ra_rad, dec_rad, ra_rad1, dec_rad1) < hspace.phi:
            return True

    # if the trixel contains the HalfSpace's center,
    # return True
    if trix.contains_pt(hspace.vector):
        return True

    sinphi = np.abs(np.sin(hspace.phi))

    # Iterate over each pair of corners (c1, c2).  For each pair,
    # construct a coordinate basis in which +z is in the
    # direction of c3, and +x is along the
    # unit vector defining c_i such that the angle
    # phi of c_j in the x,y plane is positive.  This coordinate
    # system is such that the trixel edge defined by c1, c2 is
    # now along the equator of the unit sphere.  Find the point
    # of closest approach of the HalfSpace's center to the equator.
    # If that point is between c1 and c2, return True.
    for i_c_1 in range(3):
        c1 = trix.corners[i_c_1]
        for i_c_2 in range(3):
            if i_c_2 <= i_c_1:
                continue
            c2 = trix.corners[i_c_2]
            i_c_3 = 3 - (i_c_1 + i_c_2)
            c3 = trix.corners[i_c_3]
            assert i_c_3 != i_c_2
            assert i_c_3 != i_c_1
            assert i_c_1 != i_c_2

            z_axis = np.array(
                [
                    c1[1] * c2[2] - c1[2] * c2[1],
                    c2[0] * c1[2] - c1[0] * c2[2],
                    c1[0] * c2[1] - c2[0] * c1[1],
                ]
            )
            z_axis = z_axis / np.sqrt((z_axis**2).sum())

            if np.dot(z_axis, c3) < 0.0:
                z_axis *= -1.0

            assert np.abs(1.0 - np.dot(z_axis, z_axis)) < 1.0e-10
            assert np.abs(1.0 - np.dot(c1, c1)) < 1.0e-10
            assert np.abs(1.0 - np.dot(c2, c2)) < 1.0e-10
            assert np.abs(np.dot(z_axis, c1)) < 1.0e-10
            assert np.abs(np.dot(z_axis, c2)) < 1.0e-10

            # if the dot product of the center of the HalfSpace
            # with the z axis of the new coordinate system is
            # greater than the sine of the radius of the
            # halfspace, then there is no way that the halfspace
            # intersects the equator of the unit sphere in this
            # coordinate system
            if np.abs(np.dot(z_axis, hspace.vector)) > sinphi:
                continue

            x_axis = c1
            y_axis = -1.0 * np.array(
                [
                    x_axis[1] * z_axis[2] - x_axis[2] * z_axis[1],
                    z_axis[0] * x_axis[2] - x_axis[0] * z_axis[2],
                    x_axis[0] * z_axis[1] - z_axis[0] * x_axis[1],
                ]
            )

            cos_a = np.dot(x_axis, c2)
            sin_a = np.dot(y_axis, c2)

            if sin_a < 0.0:
                x_axis = c2
                y_axis = -1.0 * np.array(
                    [
                        x_axis[1] * z_axis[2] - x_axis[2] * z_axis[1],
                        z_axis[0] * x_axis[2] - x_axis[0] * z_axis[2],
                        x_axis[0] * z_axis[1] - z_axis[0] * x_axis[1],
                    ]
                )

                cos_a = np.dot(x_axis, c1)
                sin_a = np.dot(y_axis, c1)

            assert cos_a >= 0.0
            assert sin_a >= 0.0
            assert np.abs(1.0 - cos_a**2 - sin_a**2) < 1.0e-10
            assert np.abs(np.dot(x_axis, z_axis)) < 1.0e-10
            assert np.abs(np.dot(x_axis, y_axis)) < 1.0e-10
            assert np.abs(np.dot(y_axis, z_axis)) < 1.0e-10

            x_center = np.dot(x_axis, hspace.vector)

            # if the x-coordinate of the HalfSpace's center is
            # negative, the HalfSpace is on the opposite side
            # of the unit sphere; ignore this pair c1, c2
            if x_center < 0.0:
                continue

            y_center = np.dot(y_axis, hspace.vector)

            # tan_a is the tangent of the angle between
            # the x_axis and the other trixel corner in
            # the x, y plane
            tan_a = sin_a / cos_a

            # tan_extreme is the tangent of the angle in
            # the x, y plane defining the point of closest
            # approach of the HalfSpace's center to the
            # equator.  If this point is between c1, c2,
            # return True.
            tan_extreme = y_center / x_center
            if tan_extreme > 0.0 and tan_extreme < tan_a:
                return True

    return False


class HalfSpaceTest(unittest.TestCase):
    long_message = True

    def test_half_space_contains_pt(self):
        hs = HalfSpace(np.array([0.0, 0.0, 1.0]), 0.1)
        nhs = HalfSpace(np.array([0.0, 0.0, -1.0]), -0.1)
        theta = np.arcsin(0.1)
        rng = np.random.RandomState(88)
        n_tests = 200
        ra_list = rng.random_sample(n_tests) * 2.0 * np.pi
        dec_list = rng.random_sample(n_tests) * (0.5 * np.pi - theta) + theta
        for (
            ra,
            dec,
        ) in zip(ra_list, dec_list):
            xyz = cartesian_from_spherical(ra, dec)
            self.assertTrue(hs.contains_pt(xyz))
            self.assertFalse(nhs.contains_pt(xyz))

        ra_list = rng.random_sample(n_tests) * 2.0 * np.pi
        dec_list = theta - rng.random_sample(n_tests) * (0.5 * np.pi + theta)
        for (
            ra,
            dec,
        ) in zip(ra_list, dec_list):
            xyz = cartesian_from_spherical(ra, dec)
            self.assertFalse(hs.contains_pt(xyz))
            self.assertTrue(nhs.contains_pt(xyz))

        hs = HalfSpace(np.array([1.0, 0.0, 0.0]), 0.2)
        nhs = HalfSpace(np.array([-1.0, 0.0, 0.0]), -0.2)
        theta = np.arcsin(0.2)
        ra_list = rng.random_sample(n_tests) * 2.0 * np.pi
        dec_list = rng.random_sample(n_tests) * (0.5 * np.pi - theta) + theta
        for ra, dec in zip(ra_list, dec_list):
            xyz_rot = cartesian_from_spherical(ra, dec)
            xyz = rot_about_y(xyz_rot, 0.5 * np.pi)
            self.assertTrue(hs.contains_pt(xyz))
            self.assertFalse(nhs.contains_pt(xyz))

        ra_list = rng.random_sample(n_tests) * 2.0 * np.pi
        dec_list = theta - rng.random_sample(n_tests) * (0.5 * np.pi + theta)
        for (
            ra,
            dec,
        ) in zip(ra_list, dec_list):
            xyz_rot = cartesian_from_spherical(ra, dec)
            xyz = rot_about_y(xyz_rot, 0.5 * np.pi)
            self.assertFalse(hs.contains_pt(xyz))
            self.assertTrue(nhs.contains_pt(xyz))

        vv = np.array([0.5 * np.sqrt(2), -0.5 * np.sqrt(2), 0.0])
        hs = HalfSpace(vv, 0.3)
        nhs = HalfSpace(-1.0 * vv, -0.3)
        theta = np.arcsin(0.3)
        ra_list = rng.random_sample(n_tests) * 2.0 * np.pi
        dec_list = rng.random_sample(n_tests) * (0.5 * np.pi - theta) + theta

        for ra, dec in zip(ra_list, dec_list):
            xyz_rot = cartesian_from_spherical(ra, dec)
            xyz_rot = rot_about_x(xyz_rot, 0.5 * np.pi)
            xyz = rot_about_z(xyz_rot, 0.25 * np.pi)
            self.assertTrue(hs.contains_pt(xyz))
            self.assertFalse(nhs.contains_pt(xyz))

        ra_list = rng.random_sample(n_tests) * 2.0 * np.pi
        dec_list = theta - rng.random_sample(n_tests) * (0.5 * np.pi + theta)
        for (
            ra,
            dec,
        ) in zip(ra_list, dec_list):
            xyz_rot = cartesian_from_spherical(ra, dec)
            xyz_rot = rot_about_x(xyz_rot, 0.5 * np.pi)
            xyz = rot_about_z(xyz_rot, 0.25 * np.pi)
            self.assertFalse(hs.contains_pt(xyz))
            self.assertTrue(nhs.contains_pt(xyz))

    def test_halfspace_contains_pt_scaled(self):
        """
        Test that HalfSpace.contains_pt returns the same answer
        for points on and off the unit sphere
        """
        vv = np.array([0.5 * np.sqrt(2), -0.5 * np.sqrt(2), 0.0])
        hs = HalfSpace(vv, 0.3)

        ct_inside = 0
        ct_outside = 0
        rng = np.random.RandomState(8812)
        random_pts = rng.random_sample((100, 3)) * 5.0
        for pt in random_pts:
            norm = np.sqrt(np.power(pt, 2).sum())
            self.assertGreater(np.abs(1.0 - norm), 0.01)
            unnormed_ans = hs.contains_pt(pt)
            normed_pt = pt / norm
            normed_ans = hs.contains_pt(normed_pt)
            self.assertEqual(unnormed_ans, normed_ans)

            if normed_ans:
                ct_inside += 1
            else:
                ct_outside += 1

        self.assertGreater(ct_inside, 0)
        self.assertGreater(ct_outside, 0)

    def test_halfspace_contains_trixel(self):
        # test half space that is on the equator wher N3 and S0 meet
        hs = HalfSpace(np.array([1.0, 1.0, 0.0]), 0.8)
        for tx in basic_trixels:
            status = hs.contains_trixel(basic_trixels[tx])
            msg = "Failed on %s" % tx
            if tx not in ("S0", "N3"):
                self.assertEqual(status, "outside", msg=msg)
            else:
                self.assertEqual(status, "partial", msg=msg)

        # test halfspace that is centered on vertex where S0, S3, N0, N3 meet
        hs = HalfSpace(np.array([1.0, 0.0, 0.0]), 0.8)
        for tx in basic_trixels:
            status = hs.contains_trixel(basic_trixels[tx])
            msg = "Failed on %s" % tx
            if tx not in ("S0", "S3", "N0", "N3"):
                self.assertEqual(status, "outside", msg=msg)
            else:
                self.assertEqual(status, "partial", msg=msg)

        # test halfspace fully contained in N3
        hs = HalfSpace(np.array([1.0, 1.0, 1.0]), 0.9)
        for tx in basic_trixels:
            status = hs.contains_trixel(basic_trixels[tx])
            msg = "Failed on %s" % tx
            if tx != "N3":
                self.assertEqual(status, "outside", msg=msg)
            else:
                self.assertEqual(status, "partial", msg=msg)

        # test halfspace that totally contains N3
        ra, dec = basic_trixels["N3"].get_center()
        hs = HalfSpace(np.array([1.0, 1.0, 1.0]), np.cos(0.31 * np.pi))
        for tx in basic_trixels:
            status = hs.contains_trixel(basic_trixels[tx])
            msg = "Failed on %s" % tx
            if tx == "N3":
                self.assertEqual(status, "full", msg=msg)
            elif tx in ("N1", "N2", "N0", "S0", "S1", "S3"):
                self.assertEqual(status, "partial", msg=msg)
            else:
                self.assertEqual(status, "outside", msg=msg)

    def test_half_space_eq(self):
        """
        Test that __eq__() works for HalfSpace
        """
        vv = np.array([1.0, 0.9, 2.4])
        hs1 = HalfSpace(vv, 0.1)
        hs2 = HalfSpace(2.0 * vv, 0.1)
        self.assertEqual(hs1, hs2)
        hs2 = HalfSpace(vv, 0.09)
        self.assertNotEqual(hs1, hs2)
        hs2 = HalfSpace(vv - 1.0e-4 * np.array([1.0, 0.0, 0.0]), 0.1)
        self.assertNotEqual(hs1, hs2)

    def test_find_all_trixels_radius(self):
        """
        Test the method that attempts to find all of the trixels
        inside a given half space by approximating the angular
        scale of the trixels and verifying that all returned
        trixels are within radius+angular scale of the center
        of the half space.
        """
        level = 5

        # approximate the linear angular scale (in degrees)
        # of a trixel grid using the fact that there are
        # 8*4**(level-1) trixels in the grid as per equation 2.5 of
        #
        # https://www.microsoft.com/en-us/research/wp-content/uploads/2005/09/tr-2005-123.pdf
        angular_scale = np.sqrt(4.0 * np.pi * (180.0 / np.pi) ** 2 / (8.0 * 4.0 ** (level - 1)))

        ra = 43.0
        dec = 22.0
        radius = 20.0
        half_space = half_space_from_ra_dec(ra, dec, radius)
        trixel_list = half_space.find_all_trixels(level)
        self.assertGreater(len(trixel_list), 2)

        # first, check that all of the returned trixels are
        # inside the HalfSpace
        good_htmid_list = []
        for i_limit, limits in enumerate(trixel_list):
            # verify that the tuples have been sorted by
            # htmid_min
            if i_limit > 0:
                self.assertGreater(limits[0], trixel_list[i_limit - 1][1])

            for htmid in range(limits[0], limits[1] + 1):
                test_trixel = trixel_from_htmid(htmid)
                ra_trix, dec_trix = test_trixel.get_center()
                good_htmid_list.append(htmid)
                self.assertNotEqual(half_space.contains_trixel(test_trixel), "outside")

                # check that the returned trixels are within
                # radius+angular_scale of the center of the HalfSpace
                self.assertLess(
                    angular_separation(ra, dec, ra_trix, dec_trix),
                    radius + angular_scale,
                )

        # next, verify that all of the possible trixels that
        # were not returned are outside the HalfSpace
        for base_htmid in range(8, 16):
            htmid_0 = base_htmid << 2 * (level - 1)
            self.assertEqual(level_from_htmid(htmid_0), level)
            for ii in range(2 ** (2 * level - 2)):
                htmid = htmid_0 + ii
                self.assertEqual(level_from_htmid(htmid), level)
                if htmid not in good_htmid_list:
                    test_trixel = trixel_from_htmid(htmid)
                    self.assertEqual(half_space.contains_trixel(test_trixel), "outside")
                    ra_trix, dec_trix = test_trixel.get_center()
                    self.assertGreater(angular_separation(ra, dec, ra_trix, dec_trix), radius)

    def test_find_all_trixels_brute(self):
        """
        Use the method trixel_intersects_half_space defined at the
        top of this script to verify that HalfSpace.find_all_trixels works
        """
        level = 7
        trixel_dict = get_all_trixels(level)
        all_htmid = []
        for htmid in trixel_dict.keys():
            if level_from_htmid(htmid) == level:
                all_htmid.append(htmid)

        hspace = half_space_from_ra_dec(36.0, 22.1, 2.0)

        # make sure that the two methods of determining if
        # a HalfSpace contains a trixel (HalfSpace.contains_trixel
        # and trixel_interects_half_space) agree
        for htmid in all_htmid:
            trix = trixel_dict[htmid]
            msg = "offending htmid %d" % htmid
            if trixel_intersects_half_space(trix, hspace):
                self.assertNotEqual(hspace.contains_trixel(trix), "outside", msg=msg)
            else:
                self.assertEqual(hspace.contains_trixel(trix), "outside", msg=msg)

        trixel_limits = hspace.find_all_trixels(level)
        intersecting_htmid = set()

        # check that all of the trixels included in the limits
        # do, in fact, intersect or exist in the HalfSpace
        for lim in trixel_limits:
            for htmid in range(lim[0], lim[1] + 1):
                trix = trixel_dict[htmid]
                self.assertTrue(trixel_intersects_half_space(trix, hspace))
                intersecting_htmid.add(htmid)

        # check that all of the trixels not included in the limits
        # are, in fact, outside of the HalfSpace
        self.assertLess(len(intersecting_htmid), len(all_htmid))
        self.assertGreater(len(intersecting_htmid), 0)
        for htmid in all_htmid:
            if htmid in intersecting_htmid:
                continue
            trix = trixel_dict[htmid]
            self.assertFalse(trixel_intersects_half_space(trix, hspace))

    def test_half_space_from_points(self):
        rng = np.random.RandomState(88)
        for ii in range(10):
            pt1 = (rng.random_sample() * 360.0, rng.random_sample() * 180.0 - 90.0)
            pt2 = (rng.random_sample() * 360.0, rng.random_sample() * 180.0 - 90.0)
            pt3 = (rng.random_sample() * 360.0, rng.random_sample() * 180.0 - 90.0)
            hs = half_space_from_points(pt1, pt2, pt3)

            # check that the HalfSpace contains pt3
            vv3 = cartesian_from_spherical(np.radians(pt3[0]), np.radians(pt3[1]))
            self.assertTrue(hs.contains_pt(vv3))

            # check that the HalfSpace encompasses 1/2 of the unit sphere
            self.assertAlmostEqual(hs.phi, 0.5 * np.pi, 10)
            self.assertAlmostEqual(hs.dd, 0.0, 10)

            # check that pt1 and pt2 are 90 degrees away from the center
            # of the HalfSpace
            vv1 = cartesian_from_spherical(np.radians(pt1[0]), np.radians(pt1[1]))
            vv2 = cartesian_from_spherical(np.radians(pt2[0]), np.radians(pt2[1]))
            self.assertAlmostEqual(np.dot(vv1, hs.vector), 0.0, 10)
            self.assertAlmostEqual(np.dot(vv2, hs.vector), 0.0, 10)

    def test__half_space_intersection(self):
        # Test that the two roots of an intersection are the
        # correct angular distance from the centers of the
        # half spaces
        ra1 = 22.0
        dec1 = 45.0
        rad1 = 10.0
        ra2 = 23.5
        dec2 = 37.9
        rad2 = 9.2
        hs1 = half_space_from_ra_dec(ra1, dec1, rad1)
        hs2 = half_space_from_ra_dec(ra2, dec2, rad2)
        roots = intersect_half_spaces(hs1, hs2)
        self.assertEqual(len(roots), 2)
        self.assertAlmostEqual(np.sqrt(np.sum(roots[0] ** 2)), 1.0, 10)
        self.assertAlmostEqual(np.sqrt(np.sum(roots[1] ** 2)), 1.0, 10)
        ra_r1, dec_r1 = np.degrees(spherical_from_cartesian(roots[0]))
        ra_r2, dec_r2 = np.degrees(spherical_from_cartesian(roots[1]))
        dd = angular_separation(ra1, dec1, ra_r1, dec_r1)
        self.assertAlmostEqual(dd, rad1, 10)
        dd = angular_separation(ra1, dec1, ra_r2, dec_r2)
        self.assertAlmostEqual(dd, rad1, 10)
        dd = angular_separation(ra2, dec2, ra_r1, dec_r1)
        self.assertAlmostEqual(dd, rad2, 10)
        dd = angular_separation(ra2, dec2, ra_r2, dec_r2)
        self.assertAlmostEqual(dd, rad2, 10)

        # test that two non-intersecting HalfSpaces return no roots
        hs1 = half_space_from_ra_dec(0.0, 90.0, 1.0)
        hs2 = half_space_from_ra_dec(20.0, -75.0, 5.0)
        roots = intersect_half_spaces(hs1, hs2)
        self.assertEqual(len(roots), 0)

        # test that two half spaces that are inside each other
        # return no roots
        hs1 = half_space_from_ra_dec(77.0, 10.0, 20.0)
        hs2 = half_space_from_ra_dec(75.0, 8.0, 0.2)
        roots = intersect_half_spaces(hs1, hs2)
        self.assertEqual(len(roots), 0)

        # test that two half spaces with identical centers
        # return no roots
        hs1 = half_space_from_ra_dec(11.0, -23.0, 1.0)
        hs2 = half_space_from_ra_dec(11.0, -23.0, 0.2)
        roots = intersect_half_spaces(hs1, hs2)
        self.assertEqual(len(roots), 0)

        roots = intersect_half_spaces(hs1, hs1)
        self.assertEqual(len(roots), 0)

    def test_merge_trixel_bounds(self):
        """
        Test that the merge_trixel_bounds method works
        """
        input_bound = [(1, 7), (2, 4), (21, 35), (8, 11), (36, 42), (43, 43)]
        result = HalfSpace.merge_trixel_bounds(input_bound)
        shld_be = [(1, 11), (21, 43)]
        self.assertEqual(result, shld_be)

    def test_join_trixel_bound_sets(self):
        """
        Test that HalfSpace.join_trixel_bound_sets works
        """
        b1 = [(32, 47), (6, 8), (11, 19), (12, 14), (66, 73)]
        b2 = [(35, 41), (7, 15), (41, 44)]
        result = HalfSpace.join_trixel_bound_sets(b1, b2)
        shld_be = [(7, 8), (11, 15), (35, 44)]
        self.assertEqual(result, shld_be)

    def test_contains_many_pts(self):
        """
        Test that HalfSpace.contains_many_pts works
        """
        rng = np.random.RandomState(5142)
        n_pts = 100
        vv_list = np.zeros((n_pts, 3), dtype=float)
        for ii in range(n_pts):
            vv = rng.random_sample(3) - 0.5
            vv /= np.sqrt(np.sum(vv**2))
            vv_list[ii] = vv

        vv = rng.random_sample(3) - 0.5
        vv /= np.sqrt(np.sum(vv**2))
        hs = HalfSpace(vv, 0.3)
        results = hs.contains_many_pts(vv_list)
        is_true = np.where(results)[0]
        self.assertGreater(len(is_true), n_pts // 4)
        self.assertLess(len(is_true), 3 * n_pts // 4)
        for i_vv, vv in enumerate(vv_list):
            self.assertEqual(hs.contains_pt(vv), results[i_vv])


class TrixelFinderTest(unittest.TestCase):
    long_message = True

    def check_pt(self, pt, answer):
        """
        Take a Cartesian point (pt) and a known
        htmid for that point (answer).  Find the htmid
        for the point using find_htmid and verify that
        we get the expected answer.
        """
        ra, dec = spherical_from_cartesian(pt)
        ii = find_htmid(np.degrees(ra), np.degrees(dec), 3)
        binary = "{0:b}".format(ii)
        self.assertEqual(binary, answer)

    def test_against_fatboy(self):
        """
        Test find_htmid against a random selection of stars from fatboy
        """
        dtype = np.dtype([("htmid", int), ("ra", float), ("dec", float)])
        data = np.genfromtxt(
            os.path.join(get_data_dir(), "tests", "testData", "htmid_test_data.txt"),
            dtype=dtype,
        )
        self.assertGreater(len(data), 20)
        for i_pt in range(len(data)):
            htmid_test = find_htmid(data["ra"][i_pt], data["dec"][i_pt], 21)
            self.assertEqual(htmid_test, data["htmid"][i_pt])
            level_test = level_from_htmid(htmid_test)
            self.assertEqual(level_test, 21)

    def test_find_htmid_vectorized(self):
        """
        Test that find_htmid works correctly on vectors
        """
        rng = np.random.RandomState(81723122)
        n_samples = 1000
        ra = rng.random_sample(n_samples) * 360.0
        dec = rng.random_sample(n_samples) * 180.0 - 90.0
        level = 7
        htmid_vec = find_htmid(ra, dec, level)
        self.assertIsInstance(htmid_vec, np.ndarray)
        htmid_fast = _find_htmid_fast(ra, dec, level)
        self.assertIsInstance(htmid_fast, np.ndarray)
        np.testing.assert_array_equal(htmid_vec, htmid_fast)
        for ii in range(n_samples):
            htmid_slow = _find_htmid_slow(ra[ii], dec[ii], level)
            self.assertIsInstance(htmid_slow, numbers.Number)
            self.assertEqual(htmid_slow, htmid_vec[ii])
            htmid_single = find_htmid(ra[ii], dec[ii], level)
            self.assertIsInstance(htmid_single, numbers.Number)
            self.assertEqual(htmid_single, htmid_vec[ii])

    def test_level_from_htmid(self):
        """
        Test that level_from_htmid behaves as expected
        """
        for ii in range(8, 16):
            self.assertEqual(level_from_htmid(ii), 1)

        self.assertEqual(level_from_htmid(2**9 + 5), 4)
        self.assertEqual(level_from_htmid(2**15 + 88), 7)

        with self.assertRaises(RuntimeError) as context:
            level_from_htmid(2**10)
        self.assertIn("4+2n", context.exception.args[0])

        for ii in range(8):
            with self.assertRaises(RuntimeError) as context:
                level_from_htmid(2**10)
            self.assertIn("4+2n", context.exception.args[0])

    def test_trixel_finding(self):
        """
        Check that find_htmid works by passing in some
        points whose htmid are known because of their
        proximity to the corners of low-level Trixels.
        Use check_pt to verify that find_htmid gives
        the right answer.
        """
        epsilon = 1.0e-6
        dx = np.array([epsilon, 0.0, 0.0])
        dy = np.array([0.0, epsilon, 0.0])
        dz = np.array([0.0, 0.0, epsilon])

        xx = np.array([1.0, 0.0, 0.0])
        yy = np.array([0.0, 1.0, 0.0])
        zz = np.array([0.0, 0.0, 1.0])

        pt = xx + dy + dz
        # N320
        self.check_pt(pt, "11111000")

        pt = xx - dy + dz
        # N000
        self.check_pt(pt, "11000000")

        pt = xx - dy - dz
        # S320
        self.check_pt(pt, "10111000")

        pt = yy + dx + dz
        # N300
        self.check_pt(pt, "11110000")

        pt = yy - dx + dz
        # N220
        self.check_pt(pt, "11101000")

        pt = yy - dx - dz
        # S100
        self.check_pt(pt, "10010000")

        pt = zz + dy + dx
        # N310
        self.check_pt(pt, "11110100")

        pt = zz - dy + dx
        # N010
        self.check_pt(pt, "11000100")

        pt = zz - dy - dx
        # N110
        self.check_pt(pt, "11010100")

        pt = -xx + dz + dy
        # N200
        self.check_pt(pt, "11100000")

        pt = -xx - dz + dy
        # S120
        self.check_pt(pt, "10011000")

        pt = -xx - dz - dy
        # S200
        self.check_pt(pt, "10100000")

        pt = -yy + dx + dz
        # N020
        self.check_pt(pt, "11001000")

        pt = -yy - dx + dz
        # N100
        self.check_pt(pt, "11010000")

        pt = -yy - dx - dz
        # S220
        self.check_pt(pt, "10101000")

        pt = -zz + dx + dy
        # S010
        self.check_pt(pt, "10000100")

        pt = -zz - dx + dy
        # S110
        self.check_pt(pt, "10010100")

        pt = -zz - dx - dy
        # S210
        self.check_pt(pt, "10100100")

        pt = xx + yy + zz
        # N333
        self.check_pt(pt, "11111111")

    def test_trixel_from_htmid(self):
        """
        Check that trixel_from_htmid works by
        finding the htmid from an RA, Dec pair,
        instantiating the Trixel corresponding
        to that htmid, and verifying that that
        Trixel (and not its neighbors) contains
        the RA, Dec pair.
        """
        rng = np.random.RandomState(88)
        n_tests = 100
        for i_test in range(n_tests):
            pt = rng.normal(0.0, 1.0, 3)
            ra, dec = spherical_from_cartesian(pt)
            ra = np.degrees(ra)
            dec = np.degrees(dec)
            ii = find_htmid(ra, dec, 5)
            tt = trixel_from_htmid(ii)
            self.assertTrue(tt.contains(ra, dec))
            tt1 = trixel_from_htmid(ii - 1)
            self.assertFalse(tt1.contains(ra, dec))
            tt2 = trixel_from_htmid(ii + 1)
            self.assertFalse(tt2.contains(ra, dec))

    def test_trixel_eq_ne(self):
        """
        Test that the __eq__ and __ne__ operators on the Trixel class work
        """
        t1 = trixel_from_htmid(8 * 16 + 1)
        t2 = trixel_from_htmid(8 * 16 + 1)
        self.assertEqual(t1, t2)
        t3 = trixel_from_htmid(8 * 16 + 3)
        self.assertNotEqual(t1, t3)
        self.assertTrue(t1 == t2)
        self.assertFalse(t1 == t3)
        self.assertTrue(t1 != t3)
        self.assertFalse(t2 == t3)
        self.assertTrue(t2 != t3)

    def test_get_all_trixels(self):
        """
        Test method to get all trixels up to a certain level
        """
        max_level = 5
        n_trixel_per_level = {}
        n_trixel_per_level[0] = 0
        for level in range(1, max_level + 1):
            n_trixel_per_level[level] = 8 * (4 ** (level - 1))

        trixel_dict = get_all_trixels(max_level)
        n_found = {}
        for level in range(max_level + 1):
            n_found[level] = 0

        for htmid in trixel_dict:
            level = level_from_htmid(htmid)
            n_found[level] += 1

        # verify that the correct number of trixels were
        # found per level
        for level in n_found:
            msg = "failed on level %d" % level
            self.assertEqual(n_found[level], n_trixel_per_level[level], msg=msg)

        # make sure no trixels were duplicated
        self.assertEqual(len(np.unique(list(trixel_dict.keys()))), len(trixel_dict))

        for htmid in trixel_dict.keys():
            level = level_from_htmid(htmid)
            self.assertLessEqual(level, max_level)
            self.assertGreaterEqual(level, 1)
            t0 = trixel_from_htmid(htmid)
            self.assertEqual(t0, trixel_dict[htmid])

    def test_trixel_bounding_circle(self):
        """
        Verify that the trixel's bounding_circle method returns
        a circle that contains all of the corners of the
        trixel
        """
        rng = np.random.RandomState(142)
        n_test_cases = 5
        for i_test in range(n_test_cases):
            htmid = (13 << 6) + rng.randint(1, 2**6 - 1)
            trixel = trixel_from_htmid(htmid)
            bounding_circle = trixel.bounding_circle
            ra_0, dec_0 = spherical_from_cartesian(bounding_circle[0])
            ra_list = []
            dec_list = []
            for cc in trixel.corners:
                ra, dec = spherical_from_cartesian(cc)
                ra_list.append(ra)
                dec_list.append(dec)
            ra_list = np.array(ra_list)
            dec_list = np.array(dec_list)
            distance = _angular_separation(ra_0, dec_0, ra_list, dec_list)
            distance = arcsec_from_radians(distance)
            radius = arcsec_from_radians(bounding_circle[2])
            self.assertLessEqual(distance.max() - radius, 1.0e-8)
            self.assertLess(np.abs(distance.max() - radius), 1.0e-8)

    def test_trixel_contains_many(self):
        """
        Test that trixel.contains_pt and trixel.contains can
        work with numpy arrays of input
        """
        htmid = (15 << 6) + 45
        trixel = trixel_from_htmid(htmid)
        ra_0, dec_0 = trixel.get_center()
        radius = trixel.get_radius()
        rng = np.random.RandomState(44)
        n_pts = 100
        rr = radius * rng.random_sample(n_pts) * 1.1
        theta = rng.random_sample(n_pts) * 2.0 * np.pi
        ra_list = ra_0 + rr * np.cos(theta)
        dec_list = dec_0 + rr * np.sin(theta)
        contains_arr = trixel.contains(ra_list, dec_list)
        n_in = 0
        n_out = 0
        for i_pt in range(n_pts):
            single_contains = trixel.contains(ra_list[i_pt], dec_list[i_pt])
            self.assertEqual(single_contains, contains_arr[i_pt])
            if single_contains:
                n_in += 1
            else:
                n_out += 1

        self.assertGreater(n_in, 0)
        self.assertGreater(n_out, 0)

        xyz_list = cartesian_from_spherical(np.radians(ra_list), np.radians(dec_list))

        contains_xyz_arr = trixel.contains_pt(xyz_list)
        np.testing.assert_array_equal(contains_xyz_arr, contains_arr)


if __name__ == "__main__":
    unittest.main()
