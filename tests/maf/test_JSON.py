import numpy as np
import numpy.ma as ma
import unittest
import json
import matplotlib

matplotlib.use("Agg")
import rubin_sim.maf.slicers as slicers


def makeDataValues(size=100, min=0.0, max=1.0, random=-1):
    """Generate a simple array of numbers, evenly arranged between min/max, but (optional) random order."""
    datavalues = np.arange(0, size, dtype="float")
    datavalues *= (float(max) - float(min)) / (datavalues.max() - datavalues.min())
    datavalues += min
    if random > 0:
        rng = np.random.RandomState(random)
        randorder = rng.rand(size)
        randind = np.argsort(randorder)
        datavalues = datavalues[randind]
    datavalues = np.array(list(zip(datavalues)), dtype=[("testdata", "float")])
    return datavalues


def makeMetricData(slicer, dtype="float", seed=8800):
    rng = np.random.RandomState(seed)
    metricValues = rng.rand(len(slicer)).astype(dtype)
    metricValues = ma.MaskedArray(
        data=metricValues, mask=np.zeros(len(slicer), "bool"), fill_value=slicer.badval
    )
    return metricValues


def makeFieldData():
    """Set up sample field data."""
    # These are a subset of the fields from opsim.
    fieldId = [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010]
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
    ]
    fieldId = np.array(fieldId, "int")
    ra_rad = np.array(ra_rad, "float")
    dec_rad = np.array(dec_rad, "float")
    fieldData = np.core.records.fromarrays(
        [fieldId, ra_rad, dec_rad], names=["fieldId", "fieldRA", "fieldDec"]
    )
    return fieldData


def makeOpsimDataValues(fieldData, size=10000, min=0.0, max=1.0, random=88):
    """Generate a simple array of numbers, evenly arranged between min/max, but (optional) random order."""
    datavalues = np.arange(0, size, dtype="float")
    datavalues *= (float(max) - float(min)) / (datavalues.max() - datavalues.min())
    datavalues += min
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


class TestJSONoutUniSlicer(unittest.TestCase):
    def setUp(self):
        self.testslicer = slicers.UniSlicer()

    def tearDown(self):
        del self.testslicer

    def test(self):
        metricVal = makeMetricData(self.testslicer, "float", seed=88102231)
        io = self.testslicer.outputJSON(
            metricVal,
            metricName="testMetric",
            simDataName="testSim",
            info_label="testmeta",
        )
        jsn = json.loads(io.getvalue())
        jsn_header = jsn[0]
        jsn_data = jsn[1]
        self.assertEqual(jsn_header["metricName"], "testMetric")
        self.assertEqual(jsn_header["simDataName"], "testSim")
        self.assertEqual(jsn_header["info_label"], "testmeta")
        self.assertEqual(jsn_header["slicerName"], "UniSlicer")
        self.assertEqual(jsn_header["slicerLen"], 1)
        self.assertEqual(len(jsn_data), 1)


class TestJSONoutOneDSlicer2(unittest.TestCase):
    def setUp(self):
        # Set up a slicer and some metric data for that slicer.
        dv = makeDataValues(1000, random=40082)
        self.testslicer = slicers.OneDSlicer(sliceColName="testdata")
        self.testslicer.setupSlicer(dv)

    def tearDown(self):
        del self.testslicer

    def test(self):
        metricVal = makeMetricData(self.testslicer, "float", seed=18)
        io = self.testslicer.outputJSON(metricVal)
        jsn = json.loads(io.getvalue())
        jsn_header = jsn[0]
        jsn_data = jsn[1]
        self.assertEqual(jsn_header["slicerName"], "OneDSlicer")
        self.assertEqual(jsn_header["slicerLen"], len(self.testslicer))
        self.assertEqual(len(jsn_data), len(metricVal) + 1)
        for jsndat, binleft, mval in zip(
            jsn_data, self.testslicer.slicePoints["bins"], metricVal.data
        ):
            self.assertEqual(jsndat[0], binleft)
            self.assertEqual(jsndat[1], mval)


class TestJSONoutHealpixSlicer(unittest.TestCase):
    def setUp(self):
        # Set up a slicer and some metric data for that slicer.
        self.testslicer = slicers.HealpixSlicer(nside=4, verbose=False)

    def tearDown(self):
        del self.testslicer

    def test(self):
        metricVal = makeMetricData(self.testslicer, "float", seed=452)
        io = self.testslicer.outputJSON(metricVal)
        jsn = json.loads(io.getvalue())
        jsn_header = jsn[0]
        jsn_data = jsn[1]
        self.assertEqual(jsn_header["slicerName"], "HealpixSlicer")
        self.assertEqual(jsn_header["slicerLen"], len(self.testslicer))
        self.assertEqual(len(jsn_data), len(metricVal))
        for jsndat, ra, dec, mval in zip(
            jsn_data,
            self.testslicer.slicePoints["ra"],
            self.testslicer.slicePoints["dec"],
            metricVal.data,
        ):
            self.assertAlmostEqual(jsndat[0], ra / np.pi * 180.0)
            self.assertAlmostEqual(jsndat[1], dec / np.pi * 180.0)
            self.assertEqual(jsndat[2], mval)


if __name__ == "__main__":
    unittest.main()
