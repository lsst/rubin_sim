import unittest
import os
import tempfile
import shutil
from rubin_sim.data import get_data_dir
from rubin_sim.utils.code_utilities import sims_clean_up
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
        colmap = maf.batches.col_map_dict("fbs")
        nfilters_needed = 6
        lim_ebv = 0.2
        ptsrc_lim_mag_i_band = 25.9
        m = maf.metrics.ExgalM5WithCuts(
            m5_col=colmap["fiveSigmaDepth"],
            filter_col=colmap["filter"],
            lsst_filter="i",
            n_filters=nfilters_needed,
            extinction_cut=lim_ebv,
            depth_cut=ptsrc_lim_mag_i_band,
        )
        s = maf.slicers.HealpixSlicer(nside=nside, use_cache=False)
        sql = 'note not like "DD%" and night < 365'
        ThreebyTwoSummary_simple = maf.metrics.StaticProbesFoMEmulatorMetricSimple(
            nside=nside, metric_name="3x2ptFoM_simple"
        )
        ThreebyTwoSummary = maf.maf_contrib.StaticProbesFoMEmulatorMetric(
            nside=nside, metric_name="3x2ptFoM"
        )
        bundleList.append(
            maf.metric_bundles.MetricBundle(
                m, s, sql, summary_metrics=[ThreebyTwoSummary, ThreebyTwoSummary_simple]
            )
        )

        database = os.path.join(get_data_dir(), "tests", "example_dbv1.7_0yrs.db")
        resultsDb = maf.db.ResultsDb(out_dir=self.outDir)
        bd = maf.metric_bundles.make_bundles_dict_from_list(bundleList)
        bg = maf.metric_bundles.MetricBundleGroup(
            bd, database, out_dir=self.outDir, results_db=resultsDb
        )
        bg.run_all()

    def tearDown(self):
        if os.path.isdir(self.outDir):
            shutil.rmtree(self.outDir)


if __name__ == "__main__":
    unittest.main()
