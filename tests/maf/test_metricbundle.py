import glob
import os
import shutil
import sqlite3
import tempfile
import unittest

import pandas as pd
from rubin_scheduler.data import get_data_dir
from rubin_scheduler.utils.code_utilities import sims_clean_up

import rubin_sim.maf.db as db
import rubin_sim.maf.maps as maps
import rubin_sim.maf.metric_bundles as metric_bundles
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.stackers as stackers

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

    def test_pdconstraint_db_cols(self):
        """Predicate columns must be explicitly requested, without parsing."""
        visits = pd.DataFrame(
            {
                "observationId": [1, 2, 3],
                "night": [1, 2, 3],
                "band": ["g", "r", "g"],
                "dayObs": [20230225, 20230226, 20230227],
            }
        )
        with sqlite3.connect(":memory:") as connection:
            visits.to_sql("observations", connection, index=False)
            for predicate, columns in (
                ("abs(night) < 3", {"night"}),
                ('band.str.startswith("g")', {"band"}),
                ("dayObs < 20230227", {"dayObs"}),
            ):
                with self.subTest(predicate=predicate):
                    bundle = metric_bundles.MetricBundle(
                        metrics.CountMetric(col="observationId"),
                        slicers.UniSlicer(),
                        pdconstraint=predicate,
                    )
                    self.assertEqual(bundle.db_cols, {"observationId"})
                    group = metric_bundles.MetricBundleGroup(
                        {"count": bundle},
                        connection,
                        out_dir=self.out_dir,
                        save_early=False,
                    )
                    with self.assertRaises(pd.errors.UndefinedVariableError):
                        group.run_all()
                    bundle.db_cols.update(columns)
                    group.run_all()
                    self.assertEqual(set(group.db_cols), {"observationId"} | columns)
                    self.assertEqual(bundle.metric_values.data[0], len(visits.query(predicate)))

    def test_pdconstraint_grouping(self):
        """Bundles are grouped, and made incompatible, by pdconstraint."""
        metric = metrics.MeanMetric(col="airmass")
        slicer = slicers.UniSlicer()
        database = os.path.join(get_data_dir(), "tests", TEST_DB)

        b_none = metric_bundles.MetricBundle(metric, slicer, "night < 100")
        b_early = metric_bundles.MetricBundle(
            metric, slicer, "night < 100", pdconstraint="night < 50", info_label="early"
        )
        b_late = metric_bundles.MetricBundle(
            metric, slicer, "night < 100", pdconstraint="night > 50", info_label="late"
        )
        b_other = metric_bundles.MetricBundle(metric, slicer, "")
        bundles = {"none": b_none, "early": b_early, "late": b_late, "other": b_other}
        bg = metric_bundles.MetricBundleGroup(bundles, database, out_dir=self.out_dir)

        self.assertEqual(set(bg.constraints), {"night < 100", ""})
        self.assertEqual(set(bg.pdconstraints["night < 100"]), {"", "night < 50", "night > 50"})
        self.assertEqual(bg.pdconstraints[""], [""])
        self.assertFalse(bg._check_compatible(b_early, b_late))
        self.assertFalse(bg._check_compatible(b_none, b_early))

    def test_pdconstraint_identity_collision(self):
        """Different pandas selections require distinct ResultsDb labels."""
        for first_constraint in (None, "night < 5"):
            for custom_file_roots in (False, True):
                for as_list in (False, True):
                    with self.subTest(
                        first_constraint=first_constraint,
                        custom_file_roots=custom_file_roots,
                        as_list=as_list,
                    ):
                        bundles = [
                            metric_bundles.MetricBundle(
                                metrics.CountMetric(col="observationId"),
                                slicers.UniSlicer(),
                                pdconstraint=constraint,
                                file_root=f"bundle_{i}" if custom_file_roots else None,
                            )
                            for i, constraint in enumerate((first_constraint, "night > 5"))
                        ]
                        bundle_dict = bundles if as_list else dict(enumerate(bundles))
                        if as_list and not custom_file_roots:
                            with self.assertRaisesRegex(NameError, "same file_root"):
                                metric_bundles.MetricBundleGroup(bundle_dict, None, out_dir=self.out_dir)
                            continue
                        with self.assertRaisesRegex(ValueError, "Use distinct info_label values"):
                            metric_bundles.MetricBundleGroup(bundle_dict, None, out_dir=self.out_dir)

        bundles[0].info_label = "early"
        bundles[1].info_label = "late"
        metric_bundles.MetricBundleGroup(dict(enumerate(bundles)), None, out_dir=self.out_dir)

    def test_pdconstraint_end_to_end(self):
        """A pandas constraint selects the same visits as the SQL one."""
        metric = metrics.CountMetric(col="observationId")
        slicer = slicers.UniSlicer()
        database = os.path.join(get_data_dir(), "tests", TEST_DB)

        b_all = metric_bundles.MetricBundle(metric, slicer, "night < 10")
        b_sql = metric_bundles.MetricBundle(metric, slicer, "night < 5")
        b_pd = metric_bundles.MetricBundle(metric, slicer, "night < 10", pdconstraint="night < 5")
        b_pd.db_cols.add("night")
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
