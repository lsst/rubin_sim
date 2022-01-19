import numpy as np
import numpy.ma as ma
import matplotlib

matplotlib.use("Agg")
import healpy as hp
import unittest
import rubin_sim.maf.slicers as slicers
from tempfile import TemporaryFile


class TestSlicers(unittest.TestCase):
    def setUp(self):
        self.filenames = []
        self.baseslicer = slicers.BaseSlicer()

    def test_healpixSlicer_obj(self):
        rng = np.random.RandomState(8121)
        nside = 32
        slicer = slicers.HealpixSlicer(nside=nside)
        metricValues = rng.rand(hp.nside2npix(nside)).astype("object")
        metricValues = ma.MaskedArray(
            data=metricValues,
            mask=np.where(metricValues < 0.1, True, False),
            fill_value=slicer.badval,
        )
        with TemporaryFile() as filename:
            metadata = "testdata"
            slicer.writeData(filename, metricValues, metadata=metadata)
            _ = filename.seek(0)
            metricValuesBack, slicerBack, header = self.baseslicer.readData(filename)
            np.testing.assert_almost_equal(metricValuesBack, metricValues)
            assert slicer == slicerBack
            assert metadata == header["metadata"]
            attr2check = ["nside", "nslice", "columnsNeeded", "lonCol", "latCol"]
            for att in attr2check:
                assert getattr(slicer, att) == getattr(slicerBack, att)

    def test_healpixSlicer_floats(self):
        rng = np.random.RandomState(71231)
        nside = 32
        slicer = slicers.HealpixSlicer(nside=nside)
        metricValues = rng.rand(hp.nside2npix(nside))
        with TemporaryFile() as filename:
            slicer.writeData(filename, metricValues, metadata="testdata")
            _ = filename.seek(0)
            metricValuesBack, slicerBack, header = self.baseslicer.readData(filename)
            np.testing.assert_almost_equal(metricValuesBack, metricValues)
            assert slicer == slicerBack
            attr2check = ["nside", "nslice", "columnsNeeded", "lonCol", "latCol"]
            for att in attr2check:
                assert getattr(slicer, att) == getattr(slicerBack, att)

    def test_healpixSlicer_masked(self):
        rng = np.random.RandomState(712551)
        nside = 32
        slicer = slicers.HealpixSlicer(nside=nside)
        metricValues = rng.rand(hp.nside2npix(nside))
        metricValues = ma.MaskedArray(
            data=metricValues,
            mask=np.where(metricValues < 0.1, True, False),
            fill_value=slicer.badval,
        )
        with TemporaryFile() as filename:
            slicer.writeData(filename, metricValues, metadata="testdata")
            _ = filename.seek(0)
            metricValuesBack, slicerBack, header = self.baseslicer.readData(filename)
            np.testing.assert_almost_equal(metricValuesBack, metricValues)
            assert slicer == slicerBack
            attr2check = ["nside", "nslice", "columnsNeeded", "lonCol", "latCol"]
            for att in attr2check:
                assert getattr(slicer, att) == getattr(slicerBack, att)

    def test_oneDSlicer(self):
        rng = np.random.RandomState(71111)
        slicer = slicers.OneDSlicer(sliceColName="testdata")
        dataValues = np.zeros(10000, dtype=[("testdata", "float")])
        dataValues["testdata"] = rng.rand(10000)
        slicer.setupSlicer(dataValues)
        with TemporaryFile() as filename:
            slicer.writeData(filename, dataValues[:100])
            _ = filename.seek(0)
            dataBack, slicerBack, header = self.baseslicer.readData(filename)
            assert slicer == slicerBack
            # np.testing.assert_almost_equal(dataBack,dataValues[:100])
            attr2check = ["nslice", "columnsNeeded"]
            for att in attr2check:
                if type(getattr(slicer, att)).__module__ == "numpy":
                    np.testing.assert_almost_equal(
                        getattr(slicer, att), getattr(slicerBack, att)
                    )
                else:
                    assert getattr(slicer, att) == getattr(slicerBack, att)

    def test_userpointsSlicer(self):
        rng = np.random.RandomState(7442)

        names = ["fieldRA", "fieldDec", "fieldId"]
        dt = ["float", "float", "int"]
        metricValues = rng.rand(100)
        fieldData = np.zeros(100, dtype=list(zip(names, dt)))
        fieldData["fieldRA"] = rng.rand(100)
        fieldData["fieldDec"] = rng.rand(100)
        fieldData["fieldId"] = np.arange(100)

        slicer = slicers.UserPointsSlicer(
            ra=fieldData["fieldRA"], dec=fieldData["fieldDec"]
        )

        names = ["data1", "data2", "fieldId"]
        simData = np.zeros(100, dtype=list(zip(names, dt)))
        simData["data1"] = rng.rand(100)
        simData["fieldId"] = np.arange(100)
        with TemporaryFile() as filename:
            slicer.writeData(filename, metricValues)
            _ = filename.seek(0)
            metricBack, slicerBack, header = self.baseslicer.readData(filename)
            assert slicer == slicerBack
            np.testing.assert_almost_equal(metricBack, metricValues)
            attr2check = [
                "nslice",
                "columnsNeeded",
                "lonCol",
                "latCol",
                "shape",
                "spatialExtent",
            ]
            for att in attr2check:
                if type(getattr(slicer, att)).__name__ == "dict":
                    for key in getattr(slicer, att):
                        np.testing.assert_almost_equal(
                            getattr(slicer, att)[key], getattr(slicerBack, att)[key]
                        )
                else:
                    assert getattr(slicer, att) == getattr(slicerBack, att)

    def test_unislicer(self):
        rng = np.random.RandomState(34229)
        slicer = slicers.UniSlicer()
        data = np.zeros(1, dtype=[("testdata", "float")])
        data[:] = rng.rand(1)
        slicer.setupSlicer(data)
        with TemporaryFile() as filename:
            metricValue = np.array([25.0])
            slicer.writeData(filename, metricValue)
            _ = filename.seek(0)
            dataBack, slicerBack, header = self.baseslicer.readData(filename)
            assert slicer == slicerBack
            np.testing.assert_almost_equal(dataBack, metricValue)
            attr2check = ["nslice", "columnsNeeded"]
            for att in attr2check:
                assert getattr(slicer, att) == getattr(slicerBack, att)

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
            slicer.writeData(filename, data)
            _ = filename.seek(0)
            dataBack, slicerBack, header = self.baseslicer.readData(filename)
            assert slicer == slicerBack
            # This is a crazy slow loop!
            for i, ack in enumerate(data):
                np.testing.assert_almost_equal(dataBack[i], data[i])

    def test_nDSlicer(self):
        rng = np.random.RandomState(621)
        colnames = ["test1", "test2", "test3"]
        data = []
        for c in colnames:
            data.append(rng.rand(1000))
        dv = np.core.records.fromarrays(data, names=colnames)
        slicer = slicers.NDSlicer(colnames, binsList=10)
        slicer.setupSlicer(dv)
        with TemporaryFile() as filename:
            metricdata = np.zeros(slicer.nslice, dtype="float")
            for i, s in enumerate(slicer):
                metricdata[i] = i
            slicer.writeData(filename, metricdata)
            _ = filename.seek(0)
            dataBack, slicerBack, header = self.baseslicer.readData(filename)
            assert slicer == slicerBack
            np.testing.assert_almost_equal(dataBack, metricdata)


if __name__ == "__main__":
    unittest.main()
