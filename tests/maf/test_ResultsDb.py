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
        self.outDir = 'Out'
        self.metricName = 'Count ExpMJD'
        self.slicerName = 'OneDSlicer'
        self.runName = 'fakeopsim'
        self.constraint = ''
        self.metadata = 'Dithered'
        self.metricDataFile = 'testmetricdatafile.npz'
        self.plotType = 'BinnedData'
        self.plotName = 'testmetricplot_BinnedData.png'
        self.summaryStatName1 = 'Mean'
        self.summaryStatValue1 = 20
        self.summaryStatName2 = 'Median'
        self.summaryStatValue2 = 18
        self.summaryStatName3 = 'TableFrac'
        self.summaryStatValue3 = np.empty(10, dtype=[('name', '|S12'), ('value', float)])
        for i in range(10):
            self.summaryStatValue3['name'] = 'test%d' % (i)
            self.summaryStatValue3['value'] = i
        self.displayDict = {'group': 'seeing', 'subgroup': 'all', 'order': 1, 'caption': 'lalalalal'}

    def testDbCreation(self):
        # Test default sqlite file created.
        tempdir = tempfile.mkdtemp(prefix='resDb')
        resultsdb = db.ResultsDb(outDir=tempdir)
        self.assertTrue(os.path.isfile(os.path.join(tempdir, 'resultsDb_sqlite.db')))
        resultsdb.close()
        # Test that get appropriate exception if directory doesn't exist.
        sqlitefilename = os.path.join(self.outDir + 'test', 'testDb_sqlite.db')
        self.assertRaises(ValueError, db.ResultsDb, database=sqlitefilename)
        shutil.rmtree(tempdir)

    def testAddData(self):
        tempdir = tempfile.mkdtemp(prefix='resDb')
        resultsDb = db.ResultsDb(outDir=tempdir)
        # Add metric.
        metricId = resultsDb.updateMetric(self.metricName, self.slicerName,
                                          self.runName, self.constraint,
                                          self.metadata, self.metricDataFile)
        # Try to re-add metric (should get back same metric id as previous, with no add).
        metricId2 = resultsDb.updateMetric(self.metricName, self.slicerName,
                                           self.runName, self.constraint,
                                           self.metadata, self.metricDataFile)
        self.assertEqual(metricId, metricId2)
        run1 = resultsDb.session.query(db.MetricRow).filter_by(metricId=metricId).all()
        self.assertEqual(len(run1), 1)
        # Add plot.
        resultsDb.updatePlot(metricId, self.plotType, self.plotName)
        # Add normal summary statistics.
        resultsDb.updateSummaryStat(metricId, self.summaryStatName1, self.summaryStatValue1)
        resultsDb.updateSummaryStat(metricId, self.summaryStatName2, self.summaryStatValue2)
        # Add something like tableFrac summary statistic.
        resultsDb.updateSummaryStat(metricId, self.summaryStatName3, self.summaryStatValue3)
        # Test get warning when try to add a non-conforming summary stat (not 'name' & 'value' cols).
        teststat = np.empty(10, dtype=[('col', '|S12'), ('value', float)])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            resultsDb.updateSummaryStat(metricId, 'testfail', teststat)
            self.assertIn("not save", str(w[-1].message))
        # Test get warning when try to add a string (non-conforming) summary stat.
        teststat = 'teststring'
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            resultsDb.updateSummaryStat(metricId, 'testfail', teststat)
            self.assertIn("not save", str(w[-1].message))
        shutil.rmtree(tempdir)


class TestUseResultsDb(unittest.TestCase):

    def setUp(self):
        self.outDir = 'Out'
        self.metricName = 'Count ExpMJD'
        self.slicerName = 'OneDSlicer'
        self.runName = 'fakeopsim'
        self.constraint = ''
        self.metadata = 'Dithered'
        self.metricDataFile = 'testmetricdatafile.npz'
        self.plotType = 'BinnedData'
        self.plotName = 'testmetricplot_BinnedData.png'
        self.summaryStatName1 = 'Mean'
        self.summaryStatValue1 = 20
        self.summaryStatName2 = 'Median'
        self.summaryStatValue2 = 18
        self.tempdir = tempfile.mkdtemp(prefix='resDb')
        self.resultsDb = db.ResultsDb(self.tempdir)
        self.metricId = self.resultsDb.updateMetric(self.metricName, self.slicerName,
                                                    self.runName, self.constraint,
                                                    self.metadata, self.metricDataFile)
        self.resultsDb.updatePlot(self.metricId, self.plotType, self.plotName)
        self.resultsDb.updateSummaryStat(self.metricId, self.summaryStatName1, self.summaryStatValue1)
        self.resultsDb.updateSummaryStat(self.metricId, self.summaryStatName2, self.summaryStatValue2)

    def testgetIds(self):
        mids = self.resultsDb.getAllMetricIds()
        self.assertEqual(mids[0], self.metricId)
        mid = self.resultsDb.getMetricId(self.metricName)
        self.assertEqual(mid[0], self.metricId)
        mid = self.resultsDb.getMetricId('notreal')
        self.assertEqual(len(mid), 0)

    def testshowSummary(self):
        self.resultsDb.getSummaryStats()

    def tearDown(self):
        self.resultsDb.close()
        shutil.rmtree(self.tempdir)


if __name__ == "__main__":
    unittest.main()
