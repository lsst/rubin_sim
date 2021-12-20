import unittest
import os
import tempfile
import shutil
from rubin_sim.data import get_data_dir
from rubin_sim.utils.CodeUtilities import sims_clean_up
import rubin_sim.maf as maf


class Test3x2(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        sims_clean_up()

    def setUp(self):
        self.outDir = tempfile.mkdtemp(prefix="TMB")

    @unittest.skipUnless(
        os.path.isdir(os.path.join(get_data_dir(), "maps")),
        "Skipping 3x3 metric test because no dust maps.",
    )
    def test_3x2(self):
        # Only testing that the metric successfully runs, not checking that the
        # output values are valid.
        bundleList = []
        nside = 64
        colmap = maf.batches.colMapDict.ColMapDict("fbs")
        nfilters_needed = 6
        lim_ebv = 0.2
        ptsrc_lim_mag_i_band = 25.9
        m = maf.metrics.ExgalM5_with_cuts(
            m5Col=colmap["fiveSigmaDepth"],
            filterCol=colmap["filter"],
            lsstFilter="i",
            nFilters=nfilters_needed,
            extinction_cut=lim_ebv,
            depth_cut=ptsrc_lim_mag_i_band,
        )
        s = maf.slicers.HealpixSlicer(nside=nside, useCache=False)
        sql = 'note not like "DD%" and night < 365'
        ThreebyTwoSummary_simple = maf.metrics.StaticProbesFoMEmulatorMetricSimple(
            nside=nside, metricName="3x2ptFoM_simple"
        )
        ThreebyTwoSummary = maf.mafContrib.StaticProbesFoMEmulatorMetric(
            nside=nside, metricName="3x2ptFoM"
        )
        bundleList.append(
            maf.metricBundles.MetricBundle(
                m, s, sql, summaryMetrics=[ThreebyTwoSummary, ThreebyTwoSummary_simple]
            )
        )

        database = os.path.join(get_data_dir(), "tests", "example_dbv1.7_0yrs.db")
        conn = maf.db.OpsimDatabase(database=database)
        resultsDb = maf.db.ResultsDb(outDir=self.outDir)
        bd = maf.metricBundles.makeBundlesDictFromList(bundleList)
        bg = maf.metricBundles.MetricBundleGroup(
            bd, conn, outDir=self.outDir, resultsDb=resultsDb
        )
        bg.runAll()

    def tearDown(self):
        if os.path.isdir(self.outDir):
            shutil.rmtree(self.outDir)


if __name__ == "__main__":
    unittest.main()
