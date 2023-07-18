import os
import unittest

from rubin_sim.utils import SpecMap, default_spec_map


class SpecMapTest(unittest.TestCase):
    def verify_file(self, file_name, dir_name, test_spec_map=default_spec_map):
        """
        Verify that test_spec_map[file_name] results in os.path.join(dir_name, file_name+'.gz')
        """
        test_name = test_spec_map[file_name]
        control_name = os.path.join(dir_name, file_name + ".gz")
        msg = "%s should map to %s; it actually maps to %s" % (
            file_name,
            control_name,
            test_name,
        )
        self.assertEqual(test_name, control_name, msg=msg)

        add_space = file_name + " "
        self.assertNotEqual(add_space, file_name)
        test_name = test_spec_map[add_space]
        msg = "%s should map to %s; it actually maps to %s" % (
            add_space,
            control_name,
            test_name,
        )
        self.assertEqual(test_name, control_name, msg=msg)

        add_space = " " + file_name
        self.assertNotEqual(add_space, file_name)
        test_name = test_spec_map[add_space]
        msg = "%s should map to %s; it actually maps to %s" % (
            add_space,
            control_name,
            test_name,
        )
        self.assertEqual(test_name, control_name, msg=msg)

        add_gz = file_name + ".gz"
        self.assertNotEqual(add_gz, file_name)
        test_name = test_spec_map[add_gz]
        msg = "%s should map to %s; it actually maps to %s" % (
            add_gz,
            control_name,
            test_name,
        )
        self.assertEqual(test_name, control_name, msg=msg)

    def test_mlt(self):
        """
        Test that default_spec_map correctly locates MLT dwarf spectra
        """
        self.verify_file("lte004-3.5-0.0a+0.0.BT-Settl.spec", "starSED/mlt")

    def test_m_spec(self):
        """
        Test that default_spec_map correctly finds old MLT dwarf spectra
        that begin with 'm'
        """
        self.verify_file("m5.1Full.dat", "starSED/old_mlt")

    def test_l4_spec(self):
        """
        Test that default_spec_map correctly finds l4Full.dat
        """
        self.verify_file("l4Full.dat", "starSED/old_mlt")

    def test_l_spec(self):
        """
        Test that default_spec_map correctly find the L#_# spectra
        """
        self.verify_file("L2_0Full.dat", "starSED/old_mlt")

    def test_burrows_spec(self):
        """
        Test that default_spec_map correctly find the burrows spectra
        """
        self.verify_file("burrows+2006c91.21_T1400_g5.5_cf_0.3X", "starSED/old_mlt")

    def test_bergeron(self):
        """
        Test that default_spec_map correctly locates the bergeron spectra
        """
        self.verify_file("bergeron_4750_85.dat_4900", "starSED/wDs")

    def test_kurucz(self):
        """
        Test that default_spec_map correctly locates the kurucz spectra
        """
        self.verify_file("km30_5000.fits_g10_5040", "starSED/kurucz")
        self.verify_file("kp10_9000.fits_g40_9100", "starSED/kurucz")

    def test_galaxy(self):
        """
        Test that default_spec_map correctly locates the galaxy SEDs
        """
        self.verify_file("Const.79E06.002Z.spec", "galaxySED")
        self.verify_file("Inst.79E06.02Z.spec", "galaxySED")
        self.verify_file("Exp.40E08.02Z.spec", "galaxySED")
        self.verify_file("Burst.40E08.002Z.spec", "galaxySED")

    def test_dir_dict(self):
        """
        Test a user-defined SpecMap with a dir_dict
        """
        dir_dict_test_map = SpecMap(dir_dict={"(^lte)": "silly_sub_dir"})
        self.verify_file("lte_11111.txt", "silly_sub_dir", test_spec_map=dir_dict_test_map)
        self.verify_file("Const.79E06.002Z.spec", "galaxySED", test_spec_map=dir_dict_test_map)
        self.verify_file("Inst.79E06.02Z.spec", "galaxySED", test_spec_map=dir_dict_test_map)
        self.verify_file("Exp.40E08.02Z.spec", "galaxySED", test_spec_map=dir_dict_test_map)
        self.verify_file("Burst.40E08.002Z.spec", "galaxySED", test_spec_map=dir_dict_test_map)
        self.verify_file("km30_5000.fits_g10_5040", "starSED/kurucz", test_spec_map=dir_dict_test_map)
        self.verify_file("kp10_9000.fits_g40_9100", "starSED/kurucz", test_spec_map=dir_dict_test_map)
        self.verify_file(
            "burrows+2006c91.21_T1400_g5.5_cf_0.3X",
            "starSED/old_mlt",
            test_spec_map=dir_dict_test_map,
        )
        self.verify_file("L2_0Full.dat", "starSED/old_mlt", test_spec_map=dir_dict_test_map)
        self.verify_file("m5.1Full.dat", "starSED/old_mlt", test_spec_map=dir_dict_test_map)

    def test_file_dict(self):
        """
        Test a user-defined SpecMap with a file_dict
        """
        file_dict_test_map = SpecMap(file_dict={"abcd.txt": "file_dict_test_dir/abcd.txt.gz"})

        self.assertEqual(file_dict_test_map["abcd.txt"], "file_dict_test_dir/abcd.txt.gz")
        self.verify_file("lte_11111.txt", "starSED/mlt", test_spec_map=file_dict_test_map)
        self.verify_file("Const.79E06.002Z.spec", "galaxySED", test_spec_map=file_dict_test_map)
        self.verify_file("Inst.79E06.02Z.spec", "galaxySED", test_spec_map=file_dict_test_map)
        self.verify_file("Exp.40E08.02Z.spec", "galaxySED", test_spec_map=file_dict_test_map)
        self.verify_file("Burst.40E08.002Z.spec", "galaxySED", test_spec_map=file_dict_test_map)
        self.verify_file(
            "km30_5000.fits_g10_5040",
            "starSED/kurucz",
            test_spec_map=file_dict_test_map,
        )
        self.verify_file(
            "kp10_9000.fits_g40_9100",
            "starSED/kurucz",
            test_spec_map=file_dict_test_map,
        )
        self.verify_file(
            "burrows+2006c91.21_T1400_g5.5_cf_0.3X",
            "starSED/old_mlt",
            test_spec_map=file_dict_test_map,
        )
        self.verify_file("L2_0Full.dat", "starSED/old_mlt", test_spec_map=file_dict_test_map)
        self.verify_file("m5.1Full.dat", "starSED/old_mlt", test_spec_map=file_dict_test_map)

    def test_file_and_dir_dict(self):
        """
        Test a user-defined SpecMap with both a file_dict and a dir_dict
        """
        test_map = SpecMap(
            file_dict={"abcd.txt": "file_dir/abcd.txt.gz"},
            dir_dict={"(^burrows)": "dir_dir"},
        )

        self.assertEqual(test_map["abcd.txt"], "file_dir/abcd.txt.gz")
        self.verify_file("lte_11111.txt", "starSED/mlt", test_spec_map=test_map)
        self.verify_file("Const.79E06.002Z.spec", "galaxySED", test_spec_map=test_map)
        self.verify_file("Inst.79E06.02Z.spec", "galaxySED", test_spec_map=test_map)
        self.verify_file("Exp.40E08.02Z.spec", "galaxySED", test_spec_map=test_map)
        self.verify_file("Burst.40E08.002Z.spec", "galaxySED", test_spec_map=test_map)
        self.verify_file("km30_5000.fits_g10_5040", "starSED/kurucz", test_spec_map=test_map)
        self.verify_file("kp10_9000.fits_g40_9100", "starSED/kurucz", test_spec_map=test_map)
        self.verify_file("burrows+2006c91.21_T1400_g5.5_cf_0.3X", "dir_dir", test_spec_map=test_map)
        self.verify_file("L2_0Full.dat", "starSED/old_mlt", test_spec_map=test_map)
        self.verify_file("m5.1Full.dat", "starSED/old_mlt", test_spec_map=test_map)

    def test_contains(self):
        """
        Test that 'k in SpecMap' works as it should
        """

        test_map = SpecMap(
            file_dict={"abcd.txt": "file_dir/abcd.txt.gz"},
            dir_dict={"(^burrows)": "dir_dir"},
        )

        self.assertFalse("banana" in test_map)
        self.assertTrue("abcd.txt" in test_map)
        self.assertTrue("burrows_123.txt" in test_map)


if __name__ == "__main__":
    unittest.main()
