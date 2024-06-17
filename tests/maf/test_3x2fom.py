import os
import shutil
import tempfile
import unittest

from rubin_scheduler.data import get_data_dir
from rubin_scheduler.utils.code_utilities import sims_clean_up

import rubin_sim.maf as maf

TEST_DB = "example_v3.4_0yrs.db"


class Test3x2(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        sims_clean_up()

    def setUp(self):
        self.out_dir = tempfile.mkdtemp(prefix="TMB")

    @unittest.skipUnless(
        os.path.isdir(os.path.join(get_data_dir(), "maps")),
        "Skipping 3x3 metric test because no dust maps.",
    )
    def test_3x2(self):
        # Only testing that the metric successfully runs, not checking that
        # the output values are valid.
        bundle_list = []
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
        threeby_two_summary_simple = maf.metrics.StaticProbesFoMEmulatorMetricSimple(
            nside=nside, metric_name="3x2ptFoM_simple"
        )
        threeby_two_summary = maf.maf_contrib.StaticProbesFoMEmulatorMetric(
            nside=nside, metric_name="3x2ptFoM"
        )
        bundle_list.append(
            maf.metric_bundles.MetricBundle(
                m,
                s,
                sql,
                summary_metrics=[threeby_two_summary, threeby_two_summary_simple],
            )
        )

        database = os.path.join(get_data_dir(), "tests", TEST_DB)
        results_db = maf.db.ResultsDb(out_dir=self.out_dir)
        bd = maf.metric_bundles.make_bundles_dict_from_list(bundle_list)
        bg = maf.metric_bundles.MetricBundleGroup(bd, database, out_dir=self.out_dir, results_db=results_db)
        bg.run_all()

    def tearDown(self):
        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)


if __name__ == "__main__":
    unittest.main()
