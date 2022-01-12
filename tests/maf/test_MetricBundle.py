import unittest
import matplotlib

matplotlib.use("Agg")

import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.stackers as stackers
import rubin_sim.maf.maps as maps
import rubin_sim.maf.metricBundles as metricBundles
import rubin_sim.maf.db as db
import glob
import os
import tempfile
import shutil
from rubin_sim.utils.CodeUtilities import sims_clean_up
from rubin_sim.data import get_data_dir


class TestMetricBundle(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        sims_clean_up()

    def setUp(self):
        self.outDir = tempfile.mkdtemp(prefix="TMB")

    def testOut(self):
        """
        Check that the metric bundle can generate the expected output
        """
        nside = 8
        slicer = slicers.HealpixSlicer(nside=nside)
        metric = metrics.MeanMetric(col="airmass")
        sql = 'filter="r"'
        stacker1 = stackers.RandomDitherFieldPerVisitStacker()
        stacker2 = stackers.GalacticStacker()
        map1 = maps.GalCoordsMap()
        map2 = maps.StellarDensityMap()

        metricB = metricBundles.MetricBundle(
            metric, slicer, sql, stackerList=[stacker1, stacker2], mapsList=[map1, map2]
        )
        database = os.path.join(get_data_dir(), "tests", "example_dbv1.7_0yrs.db")

        opsdb = db.OpsimDatabase(database=database)
        resultsDb = db.ResultsDb(outDir=self.outDir)

        bgroup = metricBundles.MetricBundleGroup(
            {0: metricB}, opsdb, outDir=self.outDir, resultsDb=resultsDb
        )
        bgroup.runAll()
        bgroup.plotAll()
        bgroup.writeAll()

        outThumbs = glob.glob(os.path.join(self.outDir, "thumb*"))
        outNpz = glob.glob(os.path.join(self.outDir, "*.npz"))
        outPdf = glob.glob(os.path.join(self.outDir, "*.pdf"))

        # By default, make 2 plots for healpix
        assert len(outThumbs) == 2
        assert len(outPdf) == 2
        assert len(outNpz) == 1

    def tearDown(self):
        if os.path.isdir(self.outDir):
            shutil.rmtree(self.outDir)


if __name__ == "__main__":
    unittest.main()
