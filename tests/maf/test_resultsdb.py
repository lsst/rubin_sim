import matplotlib

matplotlib.use("Agg")
import os
import warnings
import unittest
import numpy as np
import rubin_sim.maf.db as db
import shutil
import tempfile


class TestResultsDb(unittest.TestCase):
    def setUp(self):
        self.outDir = "Out"
        self.metric_name = "Count ExpMJD"
        self.slicer_name = "OneDSlicer"
        self.runName = "fakeopsim"
        self.constraint = ""
        self.info_label = "Dithered"
        self.metricDataFile = "testmetricdatafile.npz"
        self.plotType = "BinnedData"
        self.plotName = "testmetricplot_BinnedData.png"
        self.summaryStatName1 = "Mean"
        self.summaryStatValue1 = 20
        self.summaryStatName2 = "Median"
        self.summaryStatValue2 = 18
        self.summaryStatName3 = "TableFrac"
        self.summaryStatValue3 = np.empty(
            10, dtype=[("name", "|S12"), ("value", float)]
        )
        for i in range(10):
            self.summaryStatValue3["name"] = "test%d" % (i)
            self.summaryStatValue3["value"] = i
        self.displayDict = {
            "group": "seeing",
            "subgroup": "all",
            "order": 1,
            "caption": "lalalalal",
        }

    def testDbCreation(self):
        # Test default sqlite file created.
        tempdir = tempfile.mkdtemp(prefix="resDb")
        resultsdb = db.ResultsDb(out_dir=tempdir)
        self.assertTrue(os.path.isfile(os.path.join(tempdir, "resultsDb_sqlite.db")))
        resultsdb.close()
        # Test that get appropriate exception if directory doesn't exist.
        sqlitefilename = os.path.join(self.outDir + "test", "testDb_sqlite.db")
        self.assertRaises(ValueError, db.ResultsDb, database=sqlitefilename)
        shutil.rmtree(tempdir)

    def testAddData(self):
        tempdir = tempfile.mkdtemp(prefix="resDb")
        resultsDb = db.ResultsDb(out_dir=tempdir)
        # Add metric.
        metric_id = resultsDb.update_metric(
            self.metric_name,
            self.slicer_name,
            self.runName,
            self.constraint,
            self.info_label,
            self.metricDataFile,
        )
        # Try to re-add metric (should get back same metric id as previous, with no add).
        metric_id2 = resultsDb.update_metric(
            self.metric_name,
            self.slicer_name,
            self.runName,
            self.constraint,
            self.info_label,
            self.metricDataFile,
        )
        self.assertEqual(metric_id, metric_id2)
        run1 = (
            resultsDb.session.query(db.MetricRow).filter_by(metric_id=metric_id).all()
        )
        self.assertEqual(len(run1), 1)
        # Add plot.
        resultsDb.update_plot(metric_id, self.plotType, self.plotName)
        # Add normal summary statistics.
        resultsDb.update_summary_stat(
            metric_id, self.summaryStatName1, self.summaryStatValue1
        )
        resultsDb.update_summary_stat(
            metric_id, self.summaryStatName2, self.summaryStatValue2
        )
        # Add something like tableFrac summary statistic.
        resultsDb.update_summary_stat(
            metric_id, self.summaryStatName3, self.summaryStatValue3
        )
        # Test get warning when try to add a non-conforming summary stat (not 'name' & 'value' cols).
        teststat = np.empty(10, dtype=[("col", "|S12"), ("value", float)])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            resultsDb.update_summary_stat(metric_id, "testfail", teststat)
            self.assertIn("not save", str(w[-1].message))
        # Test get warning when try to add a string (non-conforming) summary stat.
        teststat = "teststring"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            resultsDb.update_summary_stat(metric_id, "testfail", teststat)
            self.assertIn("not save", str(w[-1].message))
        shutil.rmtree(tempdir)


class TestUseResultsDb(unittest.TestCase):
    def setUp(self):
        self.outDir = "Out"
        self.metric_name = "Count ExpMJD"
        self.slicer_name = "OneDSlicer"
        self.runName = "fakeopsim"
        self.constraint = ""
        self.info_label = "Dithered"
        self.metricDataFile = "testmetricdatafile.npz"
        self.plotType = "BinnedData"
        self.plotName = "testmetricplot_BinnedData.png"
        self.summaryStatName1 = "Mean"
        self.summaryStatValue1 = 20
        self.summaryStatName2 = "Median"
        self.summaryStatValue2 = 18
        self.tempdir = tempfile.mkdtemp(prefix="resDb")
        self.resultsDb = db.ResultsDb(self.tempdir)
        self.metric_id = self.resultsDb.update_metric(
            self.metric_name,
            self.slicer_name,
            self.runName,
            self.constraint,
            self.info_label,
            self.metricDataFile,
        )
        self.resultsDb.update_plot(self.metric_id, self.plotType, self.plotName)
        self.resultsDb.update_summary_stat(
            self.metric_id, self.summaryStatName1, self.summaryStatValue1
        )
        self.resultsDb.update_summary_stat(
            self.metric_id, self.summaryStatName2, self.summaryStatValue2
        )

    def testgetIds(self):
        mids = self.resultsDb.getAllMetricIds()
        self.assertEqual(mids[0], self.metric_id)
        mid = self.resultsDb.get_metric_id(self.metric_name)
        self.assertEqual(mid[0], self.metric_id)
        mid = self.resultsDb.get_metric_id("notreal")
        self.assertEqual(len(mid), 0)

    def testshowSummary(self):
        self.resultsDb.getSummaryStats()

    def tearDown(self):
        self.resultsDb.close()
        shutil.rmtree(self.tempdir)


if __name__ == "__main__":
    unittest.main()
