import glob
import os
import shutil
import tempfile
import unittest

from rubin_scheduler.data import get_data_dir
from rubin_scheduler.utils.code_utilities import sims_clean_up

import rubin_sim.maf.db as db
import rubin_sim.maf.maps as maps
import rubin_sim.maf.metric_bundles as metric_bundles
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.stackers as stackers
from rubin_sim.maf.metric_bundles.metric_bundle import _cols_from_pdconstraint

TEST_DB = "example_v3.4_0yrs.db"


class TestMetricBundle(unittest.TestCase):
    @classmethod
    def tearDown_class(cls):
        sims_clean_up()

    def setUp(self):
        self.out_dir = tempfile.mkdtemp(prefix="TMB")

    def test_out(self):
        """
        Check that the metric bundle can generate the expected output
        """
        nside = 8
        slicer = slicers.HealpixSlicer(nside=nside)
        metric = metrics.MeanMetric(col="airmass")
        sql = "filter='r'"
        stacker1 = stackers.HourAngleStacker()
        stacker2 = stackers.GalacticStacker()
        map = maps.GalCoordsMap()

        metric_b = metric_bundles.MetricBundle(
            metric, slicer, sql, stacker_list=[stacker1, stacker2], maps_list=[map]
        )
        database = os.path.join(get_data_dir(), "tests", TEST_DB)

        results_db = db.ResultsDb(out_dir=self.out_dir)

        bgroup = metric_bundles.MetricBundleGroup(
            {0: metric_b}, database, out_dir=self.out_dir, results_db=results_db
        )
        bgroup.run_all()
        bgroup.plot_all()
        bgroup.write_all()

        out_thumbs = glob.glob(os.path.join(self.out_dir, "thumb*"))
        out_npz = glob.glob(os.path.join(self.out_dir, "*.npz"))
        out_pdf = glob.glob(os.path.join(self.out_dir, "*.pdf"))

        # By default, make 2 plots for healpix
        assert len(out_thumbs) == 2
        assert len(out_pdf) == 2
        assert len(out_npz) == 1

    def test_pdconstraint_required_columns_ignore_literals(self):
        cases = [
            ("", set()),
            ("band == 'r'", {"band"}),
            ('band == "g"', {"band"}),
            ("band in ('r', 'i', 'z')", {"band"}),
            ("`filter name` == 'r'", {"filter name"}),
            ("night < 5 and band == 'r' and not simulated", {"night", "band", "simulated"}),
            ("band == 'it\\'s g'", {"band"}),
            ('band == "say \\"g\\""', {"band"}),
            ("@np.isfinite(fiveSigmaDepth)", {"fiveSigmaDepth"}),
            ("@threshold < fiveSigmaDepth and band == 'r'", {"fiveSigmaDepth", "band"}),
        ]
        for pdconstraint, expected in cases:
            with self.subTest(pdconstraint=pdconstraint):
                self.assertEqual(_cols_from_pdconstraint(pdconstraint), expected)

    def test_pdconstraint_db_cols(self):
        """Columns named in a pdconstraint are fetched; literals are not."""
        metric = metrics.MeanMetric(col="airmass")
        slicer = slicers.UniSlicer()
        self.assertEqual(metric_bundles.MetricBundle(metric, slicer, "").pdconstraint, "")

        pdconstraint = "@np.isfinite(fiveSigmaDepth) and band == 'r'"
        bundle = metric_bundles.MetricBundle(metric, slicer, "", pdconstraint=pdconstraint)
        self.assertEqual(bundle.pdconstraint, pdconstraint)
        group = metric_bundles.MetricBundleGroup({"band": bundle}, None, out_dir=self.out_dir)
        group.set_current("", pdconstraint=pdconstraint)
        for db_cols in (bundle.db_cols, group.db_cols):
            self.assertIn("fiveSigmaDepth", db_cols)
            self.assertIn("band", db_cols)
            for not_a_column in ("np", "isfinite", "r"):
                self.assertNotIn(not_a_column, db_cols)

    def test_pdconstraint_grouping(self):
        """Bundles are grouped, and made incompatible, by pdconstraint."""
        metric = metrics.MeanMetric(col="airmass")
        slicer = slicers.UniSlicer()
        database = os.path.join(get_data_dir(), "tests", TEST_DB)

        b_none = metric_bundles.MetricBundle(metric, slicer, "night < 100")
        b_early = metric_bundles.MetricBundle(metric, slicer, "night < 100", pdconstraint="night < 50")
        b_late = metric_bundles.MetricBundle(metric, slicer, "night < 100", pdconstraint="night > 50")
        b_other = metric_bundles.MetricBundle(metric, slicer, "")
        bundles = {"none": b_none, "early": b_early, "late": b_late, "other": b_other}
        bg = metric_bundles.MetricBundleGroup(bundles, database, out_dir=self.out_dir)

        self.assertEqual(set(bg.constraints), {"night < 100", ""})
        self.assertEqual(set(bg.pdconstraints["night < 100"]), {"", "night < 50", "night > 50"})
        self.assertEqual(bg.pdconstraints[""], [""])
        self.assertFalse(bg._check_compatible(b_early, b_late))
        self.assertFalse(bg._check_compatible(b_none, b_early))

    def test_pdconstraint_end_to_end(self):
        """A pandas constraint selects the same visits as the SQL one."""
        metric = metrics.CountMetric(col="observationId")
        slicer = slicers.UniSlicer()
        database = os.path.join(get_data_dir(), "tests", TEST_DB)

        b_all = metric_bundles.MetricBundle(metric, slicer, "night < 10")
        b_sql = metric_bundles.MetricBundle(metric, slicer, "night < 5")
        b_pd = metric_bundles.MetricBundle(metric, slicer, "night < 10", pdconstraint="night < 5")
        for name, bundle in (("all", b_all), ("sql", b_sql), ("pd", b_pd)):
            metric_bundles.MetricBundleGroup({name: bundle}, database, out_dir=self.out_dir).run_all()

        count_all = b_all.metric_values.data[0]
        count_pd = b_pd.metric_values.data[0]
        assert 0 < count_pd < count_all
        assert count_pd == b_sql.metric_values.data[0]

    def tearDown(self):
        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)


if __name__ == "__main__":
    unittest.main()
