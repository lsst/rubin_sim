import numpy as np
import os
import unittest
from rubin_sim.data import get_data_dir


class MapSizeUnitTest(unittest.TestCase):
    """
    Check that the files in this package are as large as we would expect
    if git-lfs worked properly
    """

    longMessage = True

    def setUp(self):
        self.pkg_dir = os.path.join(get_data_dir(), 'maps')

    def testDustMaps(self):
        """
        Go through contents of DustMaps directory and check that files
        are the size we expect them to be.
        """
        mb = 1024*1024  # because os.path.getsize returns the size in bytes
        kb = 1024
        control_size_dict = {'SFD_dust_4096_ngp.fits': 64*mb,
                             'SFD_dust_4096_sgp.fits': 64*mb,
                             'dust_nside_1024.npz': 96*mb,
                             'dust_nside_128.npz': 1.5*mb,
                             'dust_nside_16.npz': 24*kb,
                             'dust_nside_2.npz': 582,
                             'dust_nside_256.npz': 6*mb,
                             'dust_nside_32.npz': 96*kb,
                             'dust_nside_4.npz': 1.7*kb,
                             'dust_nside_512.npz': 24*mb,
                             'dust_nside_64.npz': 384*kb,
                             'dust_nside_8.npz': 6.2*kb}

        for file_name in control_size_dict:
            full_name = os.path.join(self.pkg_dir, 'DustMaps', file_name)
            size = os.path.getsize(full_name)
            self.assertLess(np.abs(size-control_size_dict[file_name]),
                            0.1*control_size_dict[file_name],
                            msg='offending file is %s' % file_name)

        list_of_files = os.listdir(os.path.join(self.pkg_dir, 'DustMaps'))
        self.assertEqual(len(list_of_files), len(control_size_dict)+1,
                         msg='directory contents are: %s' % str(list_of_files))

    def testStarMaps(self):
        """
        Test that the files in the StarMaps directory are all several
        megabytes in size
        """
        root_dir = os.path.join(self.pkg_dir, 'StarMaps')
        list_of_files = os.listdir(root_dir)
        for file_name in list_of_files:
            if file_name != '.gitignore':
                full_name = os.path.join(root_dir, file_name)
                self.assertGreater(os.path.getsize(full_name), 1024*1024,
                                   msg='offending file is %s' % file_name)

        # verify that we got all of the expected StarMaps
        self.assertIn('starDensity_u_nside_64.npz', list_of_files)
        self.assertIn('starDensity_u_wdstars_nside_64.npz', list_of_files)
        self.assertIn('starDensity_g_nside_64.npz', list_of_files)
        self.assertIn('starDensity_g_wdstars_nside_64.npz', list_of_files)
        self.assertIn('starDensity_r_nside_64.npz', list_of_files)
        self.assertIn('starDensity_r_wdstars_nside_64.npz', list_of_files)
        self.assertIn('starDensity_i_nside_64.npz', list_of_files)
        self.assertIn('starDensity_i_wdstars_nside_64.npz', list_of_files)
        self.assertIn('starDensity_z_nside_64.npz', list_of_files)
        self.assertIn('starDensity_z_wdstars_nside_64.npz', list_of_files)
        self.assertIn('starDensity_y_nside_64.npz', list_of_files)
        self.assertIn('starDensity_y_wdstars_nside_64.npz', list_of_files)

        self.assertEqual(len(list_of_files), 13)


if __name__ == "__main__":
    unittest.main()
