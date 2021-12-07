import os
import unittest
from rubin_sim.utils import SpecMap, defaultSpecMap


class SpecMapTest(unittest.TestCase):
    def verifyFile(self, file_name, dir_name, testSpecMap=defaultSpecMap):
        """
        Verify that testSpecMap[file_name] results in os.path.join(dir_name, file_name+'.gz')
        """
        test_name = testSpecMap[file_name]
        control_name = os.path.join(dir_name, file_name + ".gz")
        msg = "%s should map to %s; it actually maps to %s" % (
            file_name,
            control_name,
            test_name,
        )
        self.assertEqual(test_name, control_name, msg=msg)

        add_space = file_name + " "
        self.assertNotEqual(add_space, file_name)
        test_name = testSpecMap[add_space]
        msg = "%s should map to %s; it actually maps to %s" % (
            add_space,
            control_name,
            test_name,
        )
        self.assertEqual(test_name, control_name, msg=msg)

        add_space = " " + file_name
        self.assertNotEqual(add_space, file_name)
        test_name = testSpecMap[add_space]
        msg = "%s should map to %s; it actually maps to %s" % (
            add_space,
            control_name,
            test_name,
        )
        self.assertEqual(test_name, control_name, msg=msg)

        add_gz = file_name + ".gz"
        self.assertNotEqual(add_gz, file_name)
        test_name = testSpecMap[add_gz]
        msg = "%s should map to %s; it actually maps to %s" % (
            add_gz,
            control_name,
            test_name,
        )
        self.assertEqual(test_name, control_name, msg=msg)

    def testMLT(self):
        """
        Test that defaultSpecMap correctly locates MLT dwarf spectra
        """
        self.verifyFile("lte004-3.5-0.0a+0.0.BT-Settl.spec", "starSED/mlt")

    def test_m_spec(self):
        """
        Test that defaultSpecMap correctly finds old MLT dwarf spectra
        that begin with 'm'
        """
        self.verifyFile("m5.1Full.dat", "starSED/old_mlt")

    def test_l4_spec(self):
        """
        Test that defaultSpecMap correctly finds l4Full.dat
        """
        self.verifyFile("l4Full.dat", "starSED/old_mlt")

    def test_L_spec(self):
        """
        Test that defaultSpecMap correctly find the L#_# spectra
        """
        self.verifyFile("L2_0Full.dat", "starSED/old_mlt")

    def test_burrows_spec(self):
        """
        Test that defaultSpecMap correctly find the burrows spectra
        """
        self.verifyFile("burrows+2006c91.21_T1400_g5.5_cf_0.3X", "starSED/old_mlt")

    def testBergeron(self):
        """
        Test that defaultSpecMap correctly locates the bergeron spectra
        """
        self.verifyFile("bergeron_4750_85.dat_4900", "starSED/wDs")

    def testKurucz(self):
        """
        Test that defaultSpecMap correctly locates the kurucz spectra
        """
        self.verifyFile("km30_5000.fits_g10_5040", "starSED/kurucz")
        self.verifyFile("kp10_9000.fits_g40_9100", "starSED/kurucz")

    def testGalaxy(self):
        """
        Test that defaultSpecMap correctly locates the galaxy SEDs
        """
        self.verifyFile("Const.79E06.002Z.spec", "galaxySED")
        self.verifyFile("Inst.79E06.02Z.spec", "galaxySED")
        self.verifyFile("Exp.40E08.02Z.spec", "galaxySED")
        self.verifyFile("Burst.40E08.002Z.spec", "galaxySED")

    def testDirDict(self):
        """
        Test a user-defined SpecMap with a dirDict
        """
        dirDictTestMap = SpecMap(dirDict={"(^lte)": "silly_sub_dir"})
        self.verifyFile("lte_11111.txt", "silly_sub_dir", testSpecMap=dirDictTestMap)
        self.verifyFile(
            "Const.79E06.002Z.spec", "galaxySED", testSpecMap=dirDictTestMap
        )
        self.verifyFile("Inst.79E06.02Z.spec", "galaxySED", testSpecMap=dirDictTestMap)
        self.verifyFile("Exp.40E08.02Z.spec", "galaxySED", testSpecMap=dirDictTestMap)
        self.verifyFile(
            "Burst.40E08.002Z.spec", "galaxySED", testSpecMap=dirDictTestMap
        )
        self.verifyFile(
            "km30_5000.fits_g10_5040", "starSED/kurucz", testSpecMap=dirDictTestMap
        )
        self.verifyFile(
            "kp10_9000.fits_g40_9100", "starSED/kurucz", testSpecMap=dirDictTestMap
        )
        self.verifyFile(
            "burrows+2006c91.21_T1400_g5.5_cf_0.3X",
            "starSED/old_mlt",
            testSpecMap=dirDictTestMap,
        )
        self.verifyFile("L2_0Full.dat", "starSED/old_mlt", testSpecMap=dirDictTestMap)
        self.verifyFile("m5.1Full.dat", "starSED/old_mlt", testSpecMap=dirDictTestMap)

    def testFileDict(self):
        """
        Test a user-defined SpecMap with a fileDict
        """
        fileDictTestMap = SpecMap(
            fileDict={"abcd.txt": "file_dict_test_dir/abcd.txt.gz"}
        )

        self.assertEqual(fileDictTestMap["abcd.txt"], "file_dict_test_dir/abcd.txt.gz")
        self.verifyFile("lte_11111.txt", "starSED/mlt", testSpecMap=fileDictTestMap)
        self.verifyFile(
            "Const.79E06.002Z.spec", "galaxySED", testSpecMap=fileDictTestMap
        )
        self.verifyFile("Inst.79E06.02Z.spec", "galaxySED", testSpecMap=fileDictTestMap)
        self.verifyFile("Exp.40E08.02Z.spec", "galaxySED", testSpecMap=fileDictTestMap)
        self.verifyFile(
            "Burst.40E08.002Z.spec", "galaxySED", testSpecMap=fileDictTestMap
        )
        self.verifyFile(
            "km30_5000.fits_g10_5040", "starSED/kurucz", testSpecMap=fileDictTestMap
        )
        self.verifyFile(
            "kp10_9000.fits_g40_9100", "starSED/kurucz", testSpecMap=fileDictTestMap
        )
        self.verifyFile(
            "burrows+2006c91.21_T1400_g5.5_cf_0.3X",
            "starSED/old_mlt",
            testSpecMap=fileDictTestMap,
        )
        self.verifyFile("L2_0Full.dat", "starSED/old_mlt", testSpecMap=fileDictTestMap)
        self.verifyFile("m5.1Full.dat", "starSED/old_mlt", testSpecMap=fileDictTestMap)

    def testFileAndDirDict(self):
        """
        Test a user-defined SpecMap with both a fileDict and a dirDict
        """
        testMap = SpecMap(
            fileDict={"abcd.txt": "file_dir/abcd.txt.gz"},
            dirDict={"(^burrows)": "dir_dir"},
        )

        self.assertEqual(testMap["abcd.txt"], "file_dir/abcd.txt.gz")
        self.verifyFile("lte_11111.txt", "starSED/mlt", testSpecMap=testMap)
        self.verifyFile("Const.79E06.002Z.spec", "galaxySED", testSpecMap=testMap)
        self.verifyFile("Inst.79E06.02Z.spec", "galaxySED", testSpecMap=testMap)
        self.verifyFile("Exp.40E08.02Z.spec", "galaxySED", testSpecMap=testMap)
        self.verifyFile("Burst.40E08.002Z.spec", "galaxySED", testSpecMap=testMap)
        self.verifyFile(
            "km30_5000.fits_g10_5040", "starSED/kurucz", testSpecMap=testMap
        )
        self.verifyFile(
            "kp10_9000.fits_g40_9100", "starSED/kurucz", testSpecMap=testMap
        )
        self.verifyFile(
            "burrows+2006c91.21_T1400_g5.5_cf_0.3X", "dir_dir", testSpecMap=testMap
        )
        self.verifyFile("L2_0Full.dat", "starSED/old_mlt", testSpecMap=testMap)
        self.verifyFile("m5.1Full.dat", "starSED/old_mlt", testSpecMap=testMap)

    def test_contains(self):
        """
        Test that 'k in SpecMap' works as it should
        """

        testMap = SpecMap(
            fileDict={"abcd.txt": "file_dir/abcd.txt.gz"},
            dirDict={"(^burrows)": "dir_dir"},
        )

        self.assertFalse("banana" in testMap)
        self.assertTrue("abcd.txt" in testMap)
        self.assertTrue("burrows_123.txt" in testMap)


if __name__ == "__main__":
    unittest.main()
