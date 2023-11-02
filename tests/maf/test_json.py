import json
import unittest

import numpy as np
import numpy.ma as ma

import rubin_sim.maf.slicers as slicers


def make_data_values(size=100, min=0.0, max=1.0, random=-1):
    """Generate a simple array of numbers, evenly arranged between min/max,
    but (optional) random order."""
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


def make_metric_data(slicer, dtype="float", seed=8800):
    rng = np.random.RandomState(seed)
    metric_values = rng.rand(len(slicer)).astype(dtype)
    metric_values = ma.MaskedArray(
        data=metric_values, mask=np.zeros(len(slicer), "bool"), fill_value=slicer.badval
    )
    return metric_values


def make_field_data():
    """Set up sample field data."""
    # These are a subset of the fields from opsim.
    field_id = [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010]
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
    field_id = np.array(field_id, "int")
    ra_rad = np.array(ra_rad, "float")
    dec_rad = np.array(dec_rad, "float")
    field_data = np.core.records.fromarrays(
        [field_id, ra_rad, dec_rad], names=["field_id", "fieldRA", "fieldDec"]
    )
    return field_data


def make_opsim_data_values(field_data, size=10000, min=0.0, max=1.0, random=88):
    """Generate a simple array of numbers, evenly arranged between min/max,
    but (optional) random order."""
    datavalues = np.arange(0, size, dtype="float")
    datavalues *= (float(max) - float(min)) / (datavalues.max() - datavalues.min())
    datavalues += min
    rng = np.random.RandomState(random)
    randorder = rng.rand(size)
    randind = np.argsort(randorder)
    datavalues = datavalues[randind]
    # Add valid field_id values to match data values
    field_id = np.zeros(len(datavalues), "int")
    idxs = rng.rand(size) * len(field_data["field_id"])
    for i, d in enumerate(datavalues):
        field_id[i] = field_data[int(idxs[i])][0]
    sim_data = np.core.records.fromarrays([field_id, datavalues], names=["field_id", "testdata"])
    return sim_data


class TestJSONoutUniSlicer(unittest.TestCase):
    def setUp(self):
        self.testslicer = slicers.UniSlicer()

    def tearDown(self):
        del self.testslicer

    def test(self):
        metric_val = make_metric_data(self.testslicer, "float", seed=88102231)
        io = self.testslicer.output_json(
            metric_val,
            metric_name="testMetric",
            sim_data_name="testSim",
            info_label="testmeta",
        )
        jsn = json.loads(io.getvalue())
        jsn_header = jsn[0]
        jsn_data = jsn[1]
        self.assertEqual(jsn_header["metric_name"], "testMetric")
        self.assertEqual(jsn_header["sim_data_name"], "testSim")
        self.assertEqual(jsn_header["info_label"], "testmeta")
        self.assertEqual(jsn_header["slicer_name"], "UniSlicer")
        self.assertEqual(jsn_header["slicer_len"], 1)
        self.assertEqual(len(jsn_data), 1)


class TestJSONoutOneDSlicer2(unittest.TestCase):
    def setUp(self):
        # Set up a slicer and some metric data for that slicer.
        dv = make_data_values(1000, random=40082)
        self.testslicer = slicers.OneDSlicer(slice_col_name="testdata")
        self.testslicer.setup_slicer(dv)

    def tearDown(self):
        del self.testslicer

    def test(self):
        metric_val = make_metric_data(self.testslicer, "float", seed=18)
        io = self.testslicer.output_json(metric_val)
        jsn = json.loads(io.getvalue())
        jsn_header = jsn[0]
        jsn_data = jsn[1]
        self.assertEqual(jsn_header["slicer_name"], "OneDSlicer")
        self.assertEqual(jsn_header["slicer_len"], len(self.testslicer))
        self.assertEqual(len(jsn_data), len(metric_val) + 1)
        for jsndat, bin_left, mval in zip(jsn_data, self.testslicer.slice_points["bins"], metric_val.data):
            self.assertEqual(jsndat[0], bin_left)
            self.assertEqual(jsndat[1], mval)


class TestJSONoutHealpixSlicer(unittest.TestCase):
    def setUp(self):
        # Set up a slicer and some metric data for that slicer.
        self.testslicer = slicers.HealpixSlicer(nside=4, verbose=False)

    def tearDown(self):
        del self.testslicer

    def test(self):
        metric_val = make_metric_data(self.testslicer, "float", seed=452)
        io = self.testslicer.output_json(metric_val)
        jsn = json.loads(io.getvalue())
        jsn_header = jsn[0]
        jsn_data = jsn[1]
        self.assertEqual(jsn_header["slicer_name"], "HealpixSlicer")
        self.assertEqual(jsn_header["slicer_len"], len(self.testslicer))
        self.assertEqual(len(jsn_data), len(metric_val))
        for jsndat, ra, dec, mval in zip(
            jsn_data,
            self.testslicer.slice_points["ra"],
            self.testslicer.slice_points["dec"],
            metric_val.data,
        ):
            self.assertAlmostEqual(jsndat[0], ra / np.pi * 180.0)
            self.assertAlmostEqual(jsndat[1], dec / np.pi * 180.0)
            self.assertEqual(jsndat[2], mval)


if __name__ == "__main__":
    unittest.main()
