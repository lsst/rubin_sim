import glob
import os
import shutil
import tempfile
import unittest

import numpy as np
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

    def test_pdconstraint_stored(self):
        """Test that pdconstraint is stored correctly on MetricBundle.
        """
        metric = metrics.MeanMetric(col="airmass")
        slicer = slicers.UniSlicer()

        b_none = metric_bundles.MetricBundle(metric, slicer, "")
        assert b_none.pdconstraint == ""

        b_pd = metric_bundles.MetricBundle(metric, slicer, "", pdconstraint="night < 5")
        assert b_pd.pdconstraint == "night < 5"

    def test_pdconstraint_required_columns_ignore_literals(self):
        cases = [
            ("band == 'r'", {"band"}),
            ('band == "g"', {"band"}),
            ("band in ('r', 'i', 'z')", {"band"}),
            ("`filter name` == 'r'", {"filter name"}),
            ("night < 5 and band == 'r' and not simulated", {"night", "band", "simulated"}),
            ("band == 'it\\'s g'", {"band"}),
            ('band == "say \\"g\\""', {"band"}),
        ]
        for pdconstraint, expected in cases:
            with self.subTest(pdconstraint=pdconstraint):
                self.assertEqual(_cols_from_pdconstraint(pdconstraint), expected)

    def test_pdconstraint_band_value_not_in_group_db_cols(self):
        pdconstraint = "band == 'r' and not simulated"
        bundle = metric_bundles.MetricBundle(
            metrics.MeanMetric(col="airmass"), slicers.UniSlicer(), "", pdconstraint=pdconstraint
        )
        self.assertIn("band", bundle.db_cols)
        self.assertIn("simulated", bundle.db_cols)
        self.assertNotIn("r", bundle.db_cols)

        group = metric_bundles.MetricBundleGroup({"band": bundle}, None, out_dir=self.out_dir)
        group.set_current("", pdconstraint=pdconstraint)
        self.assertIn("band", group.db_cols)
        self.assertNotIn("r", group.db_cols)

    def test_pdconstraint_incompatible(self):
        """Bundles with different pdconstraints are not compatible.
        """
        metric = metrics.MeanMetric(col="airmass")
        slicer = slicers.UniSlicer()
        database = os.path.join(get_data_dir(), "tests", TEST_DB)

        b1 = metric_bundles.MetricBundle(metric, slicer, "", pdconstraint="night < 5")
        b2 = metric_bundles.MetricBundle(metric, slicer, "", pdconstraint="night > 5")

        bg = metric_bundles.MetricBundleGroup(
            {"b1": b1, "b2": b2}, database, out_dir=self.out_dir
        )
        assert bg.constraints == [""]
        assert set(bg.pdconstraints[""]) == {"night < 5", "night > 5"}
        assert not bg._check_compatible(b1, b2)

    def test_pdconstraint_group_structure(self):
        """Check structure of MetricBundleGroup.pdconstraints.
        """
        metric = metrics.MeanMetric(col="airmass")
        slicer = slicers.UniSlicer()
        database = os.path.join(get_data_dir(), "tests", TEST_DB)

        b_no_pd = metric_bundles.MetricBundle(metric, slicer, "night < 100")
        b_pd = metric_bundles.MetricBundle(
            metric, slicer, "night < 100", pdconstraint="night < 50"
        )
        b_other = metric_bundles.MetricBundle(metric, slicer, "")

        bg = metric_bundles.MetricBundleGroup(
            {"a": b_no_pd, "b": b_pd, "c": b_other},
            database,
            out_dir=self.out_dir,
        )
        assert set(bg.constraints) == {"night < 100", ""}
        assert set(bg.pdconstraints["night < 100"]) == {"", "night < 50"}
        assert bg.pdconstraints[""] == [""]

    def test_pdconstraint_end_to_end(self):
        """run_all with a pdconstraint should yield fewer visits than without.
        """
        metric = metrics.CountMetric(col="observationId")
        slicer = slicers.UniSlicer()
        database = os.path.join(get_data_dir(), "tests", TEST_DB)

        b_all = metric_bundles.MetricBundle(metric, slicer, "night < 10")
        b_pd = metric_bundles.MetricBundle(
            metric, slicer, "night < 10", pdconstraint="night < 5"
        )

        bg_all = metric_bundles.MetricBundleGroup(
            {"all": b_all}, database, out_dir=self.out_dir
        )
        bg_all.run_all()

        bg_pd = metric_bundles.MetricBundleGroup(
            {"pd": b_pd}, database, out_dir=self.out_dir
        )
        bg_pd.run_all()

        count_all = b_all.metric_values.data[0]
        count_pd = b_pd.metric_values.data[0]
        assert count_pd > 0
        assert count_pd < count_all
        assert np.isfinite(count_pd)

    def tearDown(self):
        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)


if __name__ == "__main__":
    unittest.main()
