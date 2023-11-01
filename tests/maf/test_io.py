import unittest
from tempfile import TemporaryFile

import healpy as hp
import numpy as np
import numpy.ma as ma

import rubin_sim.maf.slicers as slicers


class TestSlicers(unittest.TestCase):
    def setUp(self):
        self.filenames = []
        self.baseslicer = slicers.BaseSlicer()

    def test_healpix_slicer_obj(self):
        rng = np.random.RandomState(8121)
        nside = 32
        slicer = slicers.HealpixSlicer(nside=nside)
        metric_values = rng.rand(hp.nside2npix(nside)).astype("object")
        metric_values = ma.MaskedArray(
            data=metric_values,
            mask=np.where(metric_values < 0.1, True, False),
            fill_value=slicer.badval,
        )
        with TemporaryFile() as filename:
            info_label = "testdata"
            slicer.write_data(filename, metric_values, info_label=info_label)
            _ = filename.seek(0)
            metric_values_back, slicer_back, header = self.baseslicer.read_data(filename)
            np.testing.assert_almost_equal(metric_values_back, metric_values)
            assert slicer == slicer_back
            assert info_label == header["info_label"]
            attr2check = ["nside", "nslice", "columns_needed", "lon_col", "lat_col"]
            for att in attr2check:
                assert getattr(slicer, att) == getattr(slicer_back, att)

    def test_healpix_slicer_floats(self):
        rng = np.random.RandomState(71231)
        nside = 32
        slicer = slicers.HealpixSlicer(nside=nside)
        metric_values = rng.rand(hp.nside2npix(nside))
        with TemporaryFile() as filename:
            slicer.write_data(filename, metric_values, info_label="testdata")
            _ = filename.seek(0)
            metric_values_back, slicer_back, header = self.baseslicer.read_data(filename)
            np.testing.assert_almost_equal(metric_values_back, metric_values)
            assert slicer == slicer_back
            attr2check = ["nside", "nslice", "columns_needed", "lon_col", "lat_col"]
            for att in attr2check:
                assert getattr(slicer, att) == getattr(slicer_back, att)

    def test_healpix_slicer_masked(self):
        rng = np.random.RandomState(712551)
        nside = 32
        slicer = slicers.HealpixSlicer(nside=nside)
        metric_values = rng.rand(hp.nside2npix(nside))
        metric_values = ma.MaskedArray(
            data=metric_values,
            mask=np.where(metric_values < 0.1, True, False),
            fill_value=slicer.badval,
        )
        with TemporaryFile() as filename:
            slicer.write_data(filename, metric_values, info_label="testdata")
            _ = filename.seek(0)
            metric_values_back, slicer_back, header = self.baseslicer.read_data(filename)
            np.testing.assert_almost_equal(metric_values_back, metric_values)
            assert slicer == slicer_back
            attr2check = ["nside", "nslice", "columns_needed", "lon_col", "lat_col"]
            for att in attr2check:
                assert getattr(slicer, att) == getattr(slicer_back, att)

    def test_one_d_slicer(self):
        rng = np.random.RandomState(71111)
        slicer = slicers.OneDSlicer(slice_col_name="testdata")
        data_values = np.zeros(10000, dtype=[("testdata", "float")])
        data_values["testdata"] = rng.rand(10000)
        slicer.setup_slicer(data_values)
        with TemporaryFile() as filename:
            slicer.write_data(filename, data_values[:100])
            _ = filename.seek(0)
            data_back, slicer_back, header = self.baseslicer.read_data(filename)
            assert slicer == slicer_back
            # np.testing.assert_almost_equal(data_back,data_values[:100])
            attr2check = ["nslice", "columns_needed"]
            for att in attr2check:
                if type(getattr(slicer, att)).__module__ == "numpy":
                    np.testing.assert_almost_equal(getattr(slicer, att), getattr(slicer_back, att))
                else:
                    assert getattr(slicer, att) == getattr(slicer_back, att)

    def test_userpoints_slicer(self):
        rng = np.random.RandomState(7442)

        names = ["fieldRA", "fieldDec", "fieldId"]
        dt = ["float", "float", "int"]
        metric_values = rng.rand(100)
        field_data = np.zeros(100, dtype=list(zip(names, dt)))
        field_data["fieldRA"] = rng.rand(100)
        field_data["fieldDec"] = rng.rand(100)
        field_data["fieldId"] = np.arange(100)

        slicer = slicers.UserPointsSlicer(ra=field_data["fieldRA"], dec=field_data["fieldDec"])

        names = ["data1", "data2", "fieldId"]
        sim_data = np.zeros(100, dtype=list(zip(names, dt)))
        sim_data["data1"] = rng.rand(100)
        sim_data["fieldId"] = np.arange(100)
        with TemporaryFile() as filename:
            slicer.write_data(filename, metric_values)
            _ = filename.seek(0)
            metric_back, slicer_back, header = self.baseslicer.read_data(filename)
            assert slicer == slicer_back
            np.testing.assert_almost_equal(metric_back, metric_values)
            attr2check = [
                "nslice",
                "columns_needed",
                "lon_col",
                "lat_col",
                "shape",
            ]
            for att in attr2check:
                if type(getattr(slicer, att)).__name__ == "dict":
                    for key in getattr(slicer, att):
                        np.testing.assert_almost_equal(
                            getattr(slicer, att)[key], getattr(slicer_back, att)[key]
                        )
                else:
                    assert getattr(slicer, att) == getattr(slicer_back, att)

    def test_unislicer(self):
        rng = np.random.RandomState(34229)
        slicer = slicers.UniSlicer()
        data = np.zeros(1, dtype=[("testdata", "float")])
        data[:] = rng.rand(1)
        slicer.setup_slicer(data)
        with TemporaryFile() as filename:
            metric_value = np.array([25.0])
            slicer.write_data(filename, metric_value)
            _ = filename.seek(0)
            data_back, slicer_back, header = self.baseslicer.read_data(filename)
            assert slicer == slicer_back
            np.testing.assert_almost_equal(data_back, metric_value)
            attr2check = ["nslice", "columns_needed"]
            for att in attr2check:
                assert getattr(slicer, att) == getattr(slicer_back, att)

    def test_complex(self):
        # Test case where there is a complex metric
        rng = np.random.RandomState(5442)
        nside = 8
        slicer = slicers.HealpixSlicer(nside=nside)
        data = np.zeros(slicer.nslice, dtype="object")
        for i, ack in enumerate(data):
            n_el = rng.rand(1) * 4  # up to 4 elements
            data[i] = np.arange(n_el)
        with TemporaryFile() as filename:
            slicer.write_data(filename, data)
            _ = filename.seek(0)
            data_back, slicer_back, header = self.baseslicer.read_data(filename)
            assert slicer == slicer_back
            # This is a crazy slow loop!
            for i, ack in enumerate(data):
                np.testing.assert_almost_equal(data_back[i], data[i])

    def test_n_d_slicer(self):
        rng = np.random.RandomState(621)
        colnames = ["test1", "test2", "test3"]
        data = []
        for c in colnames:
            data.append(rng.rand(1000))
        dv = np.core.records.fromarrays(data, names=colnames)
        slicer = slicers.NDSlicer(colnames, bins_list=10)
        slicer.setup_slicer(dv)
        with TemporaryFile() as filename:
            metricdata = np.zeros(slicer.nslice, dtype="float")
            for i, s in enumerate(slicer):
                metricdata[i] = i
            slicer.write_data(filename, metricdata)
            _ = filename.seek(0)
            data_back, slicer_back, header = self.baseslicer.read_data(filename)
            assert slicer == slicer_back
            np.testing.assert_almost_equal(data_back, metricdata)


if __name__ == "__main__":
    unittest.main()
