import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib

matplotlib.use("Agg")
import numpy.ma as ma
import unittest
from rubin_sim.maf.slicers.opsimFieldSlicer import OpsimFieldSlicer
from rubin_sim.maf.slicers.uniSlicer import UniSlicer
import warnings

warnings.simplefilter("always")


def makeFieldData():
    """Set up sample field data."""
    # These are a subset of the fields from opsim.
    fieldId = [
        2001,
        2002,
        2003,
        2004,
        2005,
        2006,
        2007,
        2008,
        2009,
        2010,
        2011,
        2012,
        2013,
        2014,
        2015,
        2016,
        2017,
        2018,
        2019,
        2020,
        2021,
        2022,
        2023,
        2024,
        2025,
        2026,
        2027,
        2028,
        2029,
        2030,
        2031,
        2032,
        2033,
        2034,
        2035,
        2036,
        2037,
        2038,
        2039,
        2040,
        2041,
        2042,
        2043,
        2044,
        2045,
        2046,
        2047,
        2048,
        2049,
        2050,
        2051,
        2052,
        2053,
        2054,
        2055,
        2056,
        2057,
        2058,
        2059,
        2060,
        2061,
        2062,
        2063,
        2064,
        2065,
        2066,
        2067,
        2068,
        2069,
        2070,
        2071,
        2072,
        2073,
        2074,
        2075,
        2076,
        2077,
        2078,
        2079,
        2080,
        2081,
        2082,
        2083,
        2084,
        2085,
        2086,
        2087,
        2088,
        2089,
        2090,
        2091,
        2092,
        2093,
        2094,
        2095,
        2096,
        2097,
        2098,
        2099,
        2100,
    ]
    ra_rad = [
        1.4961071750760884,
        4.009380232682723,
        2.2738050744968632,
        2.7527439701957053,
        6.043715459855715,
        0.23946974745438585,
        3.4768050063149119,
        2.8063803008646744,
        4.0630173623005916,
        2.2201678117208452,
        4.7334418014345294,
        1.5497433725869068,
        5.9900783302378473,
        0.29310704352081429,
        5.3196557553180082,
        0.96352968501972802,
        5.9359027094472925,
        0.34728263102270451,
        4.6792656480113752,
        1.6039197923263617,
        4.1171937820400464,
        2.1659915251395399,
        5.3738319087411623,
        0.90935339843842322,
        3.4226285865754575,
        2.8605567206041291,
        4.6243097210206079,
        1.658875319842678,
        5.4287873030993294,
        0.85439793750118176,
        5.8809462498239249,
        0.40223922380247534,
        4.1721494427145123,
        2.1110359976232238,
        3.3676726595846915,
        2.9155126475948951,
        3.3118314978483459,
        2.9713538093312404,
        4.2279900718200034,
        2.0551952353595833,
        4.5684690919151176,
        1.7147163484226187,
        5.4846279322030744,
        0.79855764129106599,
        5.8251056207201799,
        0.45807965317074106,
        5.769146214630454,
        0.51403882623283237,
        4.5125086205619365,
        1.7706764203013494,
        5.5405873382928013,
        0.74259816862400985,
        4.283950276856884,
        1.9992351634808523,
        3.2558718254423198,
        3.0273134817372669,
        5.0265477131110696,
        1.256637460910367,
        3.7699109179914521,
        2.5132743891881342,
        0.0,
        3.7165246581483231,
        2.5666606490312627,
        4.9731614532679407,
        1.3100238539116456,
        6.2297990473364582,
        0.053386447097758416,
        3.8232977104671799,
        2.4598875967124063,
        5.079933972955943,
        1.2032510679073434,
        4.4557055250397459,
        1.82747964898169,
        3.1990676646549288,
        3.084117642524657,
        5.7123425864756632,
        0.57084272070392295,
        5.5973914990801914,
        0.68579407441569451,
        4.3407541713279745,
        1.9424311358516118,
        3.6631498498956194,
        2.6200354572839668,
        6.1764234401365998,
        0.10676173388483716,
        4.9197863787006817,
        1.3633986621626044,
        3.8766735839833384,
        2.4065117231962478,
        5.1333101127866563,
        1.1498751943929302,
        3.1415926535897931,
        5.6548662438290274,
        0.62831906335055854,
        4.3982294487094107,
        1.884955858470176,
        3.9300611754079662,
        2.3531241317716196,
        4.8663995862232081,
        1.4167858541145277,
    ]
    dec_rad = [
        -0.25205231807872636,
        -0.25205228478831621,
        -0.25205228478831621,
        -0.25205228478831621,
        -0.25205145255075168,
        -0.25205145255075168,
        -0.24630904473998308,
        -0.24630904473998308,
        -0.24630894487049795,
        -0.24630894487049795,
        -0.24630801276519362,
        -0.24630801276519362,
        -0.24630799611998855,
        -0.24630799611998855,
        -0.24630796283132372,
        -0.24630796283132372,
        -0.24014665642446359,
        -0.24014665642446359,
        -0.24014665642446359,
        -0.24014665642446359,
        -0.24014655655672376,
        -0.24014655655672376,
        -0.24014653991151874,
        -0.24014653991151874,
        -0.24014648997764879,
        -0.24014648997764879,
        -0.23394214023110541,
        -0.23394214023110541,
        -0.23394209029549018,
        -0.23394209029549018,
        -0.23394204036162028,
        -0.23394204036162028,
        -0.23394204036162028,
        -0.23394204036162028,
        -0.23394204036162028,
        -0.23394204036162028,
        -0.22802912366201158,
        -0.22802912366201158,
        -0.22802899050386166,
        -0.22802899050386166,
        -0.22802897385865656,
        -0.22802897385865656,
        -0.22802897385865656,
        -0.22802897385865656,
        -0.22802804175335223,
        -0.22802804175335223,
        -0.22161461721627795,
        -0.22161461721627795,
        -0.22161461721627795,
        -0.22161461721627795,
        -0.22161456728240808,
        -0.22161456728240808,
        -0.2216145339919979,
        -0.2216145339919979,
        -0.22161446741292293,
        -0.22161446741292293,
        -0.21924843187676188,
        -0.21924843187676188,
        -0.21924838194289203,
        -0.21924838194289203,
        -0.2192483652976869,
        -0.2185302106194596,
        -0.2185302106194596,
        -0.2185292785141553,
        -0.2185292785141553,
        -0.2185292785141553,
        -0.2185292785141553,
        -0.21852919528987524,
        -0.21852919528987524,
        -0.21852916200121042,
        -0.21852916200121042,
        -0.21669173065121086,
        -0.21669173065121086,
        -0.21669169736254604,
        -0.21669169736254604,
        -0.21669083183457133,
        -0.21669083183457133,
        -0.21669076525549638,
        -0.21669076525549638,
        -0.2166907319650862,
        -0.2166907319650862,
        -0.21643082408833589,
        -0.21643082408833589,
        -0.21643080744313081,
        -0.21643080744313081,
        -0.21643077415446602,
        -0.21643077415446602,
        -0.21643074086405584,
        -0.21643074086405584,
        -0.21643062435111099,
        -0.21643062435111099,
        -0.21479804002474462,
        -0.21479794015525952,
        -0.21479794015525952,
        -0.21479784028751969,
        -0.21479784028751969,
        -0.21311147675042988,
        -0.21311147675042988,
        -0.2131105113547154,
        -0.2131105113547154,
    ]
    fieldId = np.array(fieldId, "int")
    ra_rad = np.array(ra_rad, "float")
    dec_rad = np.array(dec_rad, "float")
    fieldData = np.core.records.fromarrays(
        [fieldId, np.degrees(ra_rad), np.degrees(dec_rad)],
        names=["fieldId", "fieldRA", "fieldDec"],
    )
    return fieldData


def makeDataValues(fieldData, size=10000, min=0.0, max=1.0, random=None):
    """Generate a simple array of numbers, evenly arranged between min/max, but (optional) random order."""
    datavalues = np.arange(0, size, dtype="float")
    datavalues *= (float(max) - float(min)) / (datavalues.max() - datavalues.min())
    datavalues += min
    if random is None:
        raise RuntimeError("Must pass in random number seed as kwarg 'random'")

    rng = np.random.RandomState(random)
    randorder = rng.rand(size)
    randind = np.argsort(randorder)
    datavalues = datavalues[randind]
    # Add valid fieldId values to match data values
    fieldId = np.zeros(len(datavalues), "int")
    idxs = rng.rand(size) * len(fieldData["fieldId"])
    for i, d in enumerate(datavalues):
        fieldId[i] = fieldData[int(idxs[i])][0]
    simData = np.core.records.fromarrays(
        [fieldId, datavalues], names=["fieldId", "testdata"]
    )
    return simData


class TestOpsimFieldSlicerSetup(unittest.TestCase):
    def setUp(self):
        self.testslicer = OpsimFieldSlicer()
        self.fieldData = makeFieldData()
        self.simData = makeDataValues(self.fieldData, random=88)

    def tearDown(self):
        del self.testslicer
        self.testslicer = None

    def testSlicertype(self):
        """Test instantiation of slicer sets slicer type as expected."""
        self.assertEqual(self.testslicer.slicerName, self.testslicer.__class__.__name__)
        self.assertEqual(self.testslicer.slicerName, "OpsimFieldSlicer")

    def testSlicerNbins(self):
        """Test that generate expected number of bins for a given set of fields."""
        self.assertEqual(self.testslicer.nslice, None)
        self.testslicer.setupSlicer(self.simData, self.fieldData)
        self.assertEqual(self.testslicer.nslice, len(self.fieldData["fieldId"]))


class TestOpsimFieldSlicerEqual(unittest.TestCase):
    def setUp(self):
        self.testslicer = OpsimFieldSlicer()
        self.fieldData = makeFieldData()
        self.simData = makeDataValues(self.fieldData, random=56)
        self.testslicer.setupSlicer(self.simData, self.fieldData)

    def tearDown(self):
        del self.testslicer
        self.testslicer = None

    def testSlicerEquivalence(self):
        """Test that slicers are marked equal when appropriate, and unequal when appropriate."""
        # Note that opsimfield slicers are considered 'equal' when all fieldId's, RA and Decs match.
        testslicer2 = OpsimFieldSlicer()
        fieldData2 = np.copy(self.fieldData)
        testslicer2.setupSlicer(self.simData, fieldData2)
        # These slicers should be equal.
        self.assertTrue(self.testslicer == testslicer2)
        self.assertFalse(self.testslicer != testslicer2)
        # These slicers should not be equal.
        fieldData2["fieldId"] = fieldData2["fieldId"] + 1
        testslicer2.setupSlicer(self.simData, fieldData2)
        self.assertTrue(self.testslicer != testslicer2)
        self.assertFalse(self.testslicer == testslicer2)
        # Test a slicer that is not the same kind.
        testslicer2 = UniSlicer()
        self.assertNotEqual(self.testslicer, testslicer2)
        # Test slicers that haven't been setup
        ts1 = OpsimFieldSlicer()
        ts2 = OpsimFieldSlicer()

        self.assertTrue(ts1 == ts2)
        self.assertFalse(ts1 != ts2)
        # Set up one with an odd value.
        ts2 = OpsimFieldSlicer(fieldRaColName="WackyName")
        self.assertTrue(ts1 != ts2)
        self.assertFalse(ts1 == ts2)


@unittest.skip("Skipping because warning does not seem to trigger reliably on py2.")
class TestOpsimFieldSlicerWarning(unittest.TestCase):
    def setUp(self):
        self.testslicer = OpsimFieldSlicer()
        self.fieldData = makeFieldData()
        self.simData = makeDataValues(self.fieldData, random=4532)

    def tearDown(self):
        del self.testslicer
        self.testslicer = None

    def testWarning(self):
        self.testslicer.setupSlicer(self.simData, self.fieldData)
        with warnings.catch_warnings(record=True) as w:
            self.testslicer.setupSlicer(self.simData, self.fieldData)
            self.assertEqual(len(w), 1)
            self.assertIn("Re-setting up an OpsimFieldSlicer", str(w[-1].message))


class TestOpsimFieldSlicerIteration(unittest.TestCase):
    def setUp(self):
        self.testslicer = OpsimFieldSlicer(latLonDeg=True)
        self.fieldData = makeFieldData()
        self.simData = makeDataValues(self.fieldData, random=776221)
        self.testslicer.setupSlicer(self.simData, self.fieldData)

    def tearDown(self):
        del self.testslicer
        self.testslicer = None

    def testIteration(self):
        """Test iteration goes through expected range and ra/dec are in expected range (radians)."""
        for fid, ra, dec, s in zip(
            self.fieldData["fieldId"],
            np.radians(self.fieldData["fieldRA"]),
            np.radians(self.fieldData["fieldDec"]),
            self.testslicer,
        ):
            self.assertEqual(fid, s["slicePoint"]["fid"])
            self.assertEqual(ra, s["slicePoint"]["ra"])
            self.assertEqual(dec, s["slicePoint"]["dec"])
            self.assertGreaterEqual(s["slicePoint"]["fid"], 0)
            self.assertLessEqual(s["slicePoint"]["ra"], 2 * np.pi)
            self.assertGreaterEqual(s["slicePoint"]["dec"], -np.pi)
            self.assertLessEqual(s["slicePoint"]["dec"], np.pi)

    def testGetItem(self):
        """Test getting indexed value."""
        for i, s in enumerate(self.testslicer):
            dict1 = s
            dict2 = self.testslicer[i]
            np.testing.assert_array_equal(dict1["idxs"], dict2["idxs"])
            self.assertDictEqual(dict1["slicePoint"], dict2["slicePoint"])
        n = 0
        self.assertEqual(
            self.testslicer[n]["slicePoint"]["fid"], self.fieldData["fieldId"][n]
        )
        self.assertEqual(
            self.testslicer[n]["slicePoint"]["ra"],
            np.radians(self.fieldData["fieldRA"][n]),
        )
        self.assertEqual(
            self.testslicer[n]["slicePoint"]["dec"],
            np.radians(self.fieldData["fieldDec"][n]),
        )
        n = len(self.testslicer) - 1
        self.assertEqual(
            self.testslicer[n]["slicePoint"]["fid"], self.fieldData["fieldId"][n]
        )
        self.assertEqual(
            self.testslicer[n]["slicePoint"]["ra"],
            np.radians(self.fieldData["fieldRA"][n]),
        )
        self.assertEqual(
            self.testslicer[n]["slicePoint"]["dec"],
            np.radians(self.fieldData["fieldDec"][n]),
        )


class TestOpsimFieldSlicerSlicing(unittest.TestCase):
    # Note that this is really testing baseSpatialSlicer, as slicing is done there for healpix grid

    def setUp(self):
        self.testslicer = OpsimFieldSlicer()
        self.fieldData = makeFieldData()
        self.simData = makeDataValues(self.fieldData, random=98)

    def tearDown(self):
        del self.testslicer
        self.testslicer = None

    def testSlicing(self):
        """Test slicing returns (all) data points which match fieldId values."""
        # Test that slicing fails before setupBinner
        self.assertRaises(NotImplementedError, self.testslicer._sliceSimData, 0)
        # Set up slicer.
        self.testslicer.setupSlicer(self.simData, self.fieldData)
        for s in self.testslicer:
            didxs = np.where(self.simData["fieldId"] == s["slicePoint"]["fid"])
            binidxs = s["idxs"]
            self.assertEqual(len(binidxs), len(didxs[0]))
            if len(binidxs) > 0:
                didxs = np.sort(didxs[0])
                binidxs = np.sort(binidxs)
                np.testing.assert_equal(
                    self.simData["testdata"][didxs], self.simData["testdata"][binidxs]
                )


class TestOpsimFieldSlicerPlotting(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(65332)
        self.testslicer = OpsimFieldSlicer()
        self.fieldData = makeFieldData()
        self.simData = makeDataValues(self.fieldData, random=462)
        self.testslicer.setupSlicer(self.simData, self.fieldData)

        self.metricdata = ma.MaskedArray(
            data=np.zeros(len(self.testslicer), dtype="float"),
            mask=np.zeros(len(self.testslicer), "bool"),
            fill_value=self.testslicer.badval,
        )
        for i, s in enumerate(self.testslicer):
            idxs = s["idxs"]
            if len(idxs) > 0:
                self.metricdata.data[i] = np.mean(self.simData["testdata"][idxs])
            else:
                self.metricdata.mask[i] = True
        self.metricdata2 = ma.MaskedArray(
            data=rng.rand(len(self.testslicer)),
            mask=np.zeros(len(self.testslicer), "bool"),
            fill_value=self.testslicer.badval,
        )

    def tearDown(self):
        del self.testslicer
        self.testslicer = None


if __name__ == "__main__":
    unittest.main()
