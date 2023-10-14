import os
import unittest

import numpy as np

from rubin_sim.data import get_data_dir
from rubin_sim.scheduler.utils import EuclidOverlapFootprint, SkyAreaGenerator, SkyAreaGeneratorGalplane

datadir = os.path.join(get_data_dir(), "scheduler")


class TestSkyArea(unittest.TestCase):
    def setUp(self):
        self.nside = 32

    @unittest.skipUnless(os.path.isdir(datadir), "Test data not available.")
    def test_skyareagenerator(self):
        # Just test that it sets up and returns maps
        s = SkyAreaGenerator(nside=self.nside)
        footprints, labels = s.return_maps()
        expected_labels = ["", "LMC_SMC", "bulge", "dusty_plane", "lowdust", "nes", "scp", "virgo"]
        self.assertEqual(set(np.unique(labels)), set(expected_labels))
        # Check that ratios in the low-dust wfd in r band are 1
        # This doesn't always have to be the case, but should be with defaults
        lowdust = np.where(labels == "lowdust")
        self.assertTrue(np.all(footprints["r"][lowdust] == 1))

    @unittest.skipUnless(os.path.isdir(datadir), "Test data not available.")
    def test_skyareagenerator_nside(self):
        # Just check two other likely common nsides
        for nside in (16, 64):
            s = SkyAreaGenerator(nside=nside)
            footprints, labels = s.return_maps()
            lowdust = np.where(labels == "lowdust")
            self.assertTrue(np.all(footprints["r"][lowdust] == 1))

    @unittest.skipUnless(os.path.isdir(datadir), "Test data not available.")
    def test_skyareageneratorgalplane(self):
        # Just test that it sets up and returns maps
        s = SkyAreaGeneratorGalplane(nside=self.nside)
        footprints, labels = s.return_maps()
        expected_labels = ["", "LMC_SMC", "bulgy", "dusty_plane", "lowdust", "nes", "scp", "virgo"]
        self.assertEqual(set(np.unique(labels)), set(expected_labels))
        # Check that ratios in the low-dust wfd in r band are 1
        # This doesn't always have to be the case, but should be with defaults
        lowdust = np.where(labels == "lowdust")
        self.assertTrue(np.all(footprints["r"][lowdust] == 1))

    @unittest.skipUnless(os.path.isdir(datadir), "Test data not available.")
    def test_euclidoverlapfootprint(self):
        # Just test that it sets up and returns maps
        s = EuclidOverlapFootprint(nside=self.nside)
        footprints, labels = s.return_maps()
        expected_labels = [
            "",
            "LMC_SMC",
            "bulgy",
            "dusty_plane",
            "lowdust",
            "nes",
            "scp",
            "virgo",
            "euclid_overlap",
        ]
        self.assertEqual(set(np.unique(labels)), set(expected_labels))
        # Check that ratios in the low-dust wfd in r band are 1
        # This doesn't always have to be the case, but should be with defaults
        lowdust = np.where(labels == "lowdust")
        self.assertTrue(np.all(footprints["r"][lowdust] == 1))


if __name__ == "__main__":
    unittest.main()
