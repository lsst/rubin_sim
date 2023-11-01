import os
import shutil
import tempfile
import unittest
import warnings

import numpy as np

import rubin_sim.maf.db as db


class TestResultsDb(unittest.TestCase):
    def setUp(self):
        self.out_dir = "Out"
        self.metric_name = "Count ExpMJD"
        self.slicer_name = "OneDSlicer"
        self.run_name = "fakeopsim"
        self.sql_constraint = ""
        self.info_label = "Dithered"
        self.metric_data_file = "testmetricdatafile.npz"
        self.plot_type = "BinnedData"
        self.plot_name = "testmetricplot_BinnedData.png"
        self.summary_stat_name1 = "Mean"
        self.summary_stat_value1 = 20
        self.summary_stat_name2 = "Median"
        self.summary_stat_value2 = 18
        self.summary_stat_name3 = "TableFrac"
        self.summary_stat_value3 = np.empty(10, dtype=[("name", "|S12"), ("value", float)])
        for i in range(10):
            self.summary_stat_value3["name"] = "test%d" % (i)
            self.summary_stat_value3["value"] = i
        self.display_dict = {
            "group": "seeing",
            "subgroup": "all",
            "order": 1,
            "caption": "lalalalal",
        }

    def test_db_creation(self):
        # Test default sqlite file created.
        tempdir = tempfile.mkdtemp(prefix="resDb")
        resultsdb = db.ResultsDb(out_dir=tempdir)
        self.assertTrue(os.path.isfile(os.path.join(tempdir, "resultsDb_sqlite.db")))
        resultsdb.close()
        # Test that get appropriate exception if directory doesn't exist.
        sqlitefilename = os.path.join(self.out_dir + "test", "testDb_sqlite.db")
        self.assertRaises(ValueError, db.ResultsDb, database=sqlitefilename)
        shutil.rmtree(tempdir)

    def test_add_data(self):
        tempdir = tempfile.mkdtemp(prefix="resDb")
        results_db = db.ResultsDb(out_dir=tempdir)
        # Add metric.
        metric_id = results_db.update_metric(
            self.metric_name,
            self.slicer_name,
            self.run_name,
            self.sql_constraint,
            self.info_label,
            self.metric_data_file,
        )
        # Try to re-add metric
        # (should get back same metric id as previous, with no add).
        metric_id2 = results_db.update_metric(
            self.metric_name,
            self.slicer_name,
            self.run_name,
            self.sql_constraint,
            self.info_label,
            self.metric_data_file,
        )
        self.assertEqual(metric_id, metric_id2)
        run1 = results_db.session.query(db.MetricRow).filter_by(metric_id=metric_id).all()
        self.assertEqual(len(run1), 1)
        # Add plot.
        results_db.update_plot(metric_id, self.plot_type, self.plot_name)
        # Add normal summary statistics.
        results_db.update_summary_stat(metric_id, self.summary_stat_name1, self.summary_stat_value1)
        results_db.update_summary_stat(metric_id, self.summary_stat_name2, self.summary_stat_value2)
        # Add something like tableFrac summary statistic.
        results_db.update_summary_stat(metric_id, self.summary_stat_name3, self.summary_stat_value3)
        # Test get warning when try to add a non-conforming summary stat
        # (not 'name' & 'value' cols).
        teststat = np.empty(10, dtype=[("col", "|S12"), ("value", float)])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results_db.update_summary_stat(metric_id, "testfail", teststat)
            self.assertIn("not save", str(w[-1].message))
        # Test we get warning when try to add a string (non-conforming)
        # summary stat.
        teststat = "teststring"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results_db.update_summary_stat(metric_id, "testfail", teststat)
            self.assertIn("not save", str(w[-1].message))
        shutil.rmtree(tempdir)


class TestUseResultsDb(unittest.TestCase):
    def setUp(self):
        self.out_dir = "Out"
        self.metric_name = "Count ExpMJD"
        self.slicer_name = "OneDSlicer"
        self.run_name = "fakeopsim"
        self.constraint = ""
        self.info_label = "Dithered"
        self.metric_data_file = "testmetricdatafile.npz"
        self.plot_type = "BinnedData"
        self.plot_name = "testmetricplot_BinnedData.png"
        self.summary_stat_name1 = "Mean"
        self.summary_stat_value1 = 20
        self.summary_stat_name2 = "Median"
        self.summary_stat_value2 = 18
        self.tempdir = tempfile.mkdtemp(prefix="resDb")
        self.results_db = db.ResultsDb(self.tempdir)
        self.metric_id = self.results_db.update_metric(
            self.metric_name,
            self.slicer_name,
            self.run_name,
            self.constraint,
            self.info_label,
            self.metric_data_file,
        )
        self.results_db.update_plot(self.metric_id, self.plot_type, self.plot_name)
        self.results_db.update_summary_stat(self.metric_id, self.summary_stat_name1, self.summary_stat_value1)
        self.results_db.update_summary_stat(self.metric_id, self.summary_stat_name2, self.summary_stat_value2)

    def testget_ids(self):
        mids = self.results_db.get_all_metric_ids()
        self.assertEqual(mids[0], self.metric_id)
        mid = self.results_db.get_metric_id(self.metric_name)
        self.assertEqual(mid[0], self.metric_id)
        mid = self.results_db.get_metric_id("notreal")
        self.assertEqual(len(mid), 0)

    def testshow_summary(self):
        self.results_db.get_summary_stats()

    def tearDown(self):
        self.results_db.close()
        shutil.rmtree(self.tempdir)


if __name__ == "__main__":
    unittest.main()
