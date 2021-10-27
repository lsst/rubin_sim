import unittest
import os
import tempfile
import shutil
from rubin_sim.data import get_data_dir
import rubin_sim.maf.batches as batches
from rubin_sim.utils.CodeUtilities import sims_clean_up
import rubin_sim.maf.db as db
import rubin_sim.maf.metricBundles as metricBundles


class TestBatches(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        sims_clean_up()

    def setUp(self):
        self.outDir = tempfile.mkdtemp(prefix='TMB')

    def testload_them_all(self):
        ack = batches.altazHealpix()
        ack = batches.altazLambert()
        ack = batches.standardSummary()
        ack = batches.standardMetrics('night')
        ack = batches.descWFDBatch()
        ack = batches.tdcBatch()
        ack = batches.filtersPerNight()
        ack = batches.filtersWholeSurvey()
        ack = batches.glanceBatch()
        ack = batches.hourglassPlots()
        ack = batches.slewBasics()
        ack = batches.fOBatch()
        ack = batches.astrometryBatch()
        ack = batches.rapidRevisitBatch()
        ack = batches.agnBatch()
        ack = batches.timeGaps()
        ack = batches.metadataBasics('airmass')
        ack = batches.metadataBasicsAngle('rotskyPos')
        ack = batches.metadataMaps('fiveSigmaDepth')

    def test_movingObjectsBatches(self):
        ack = batches.quickDiscoveryBatch()
        ack = batches.discoveryBatch()
        ack = batches.characterizationInnerBatch()
        ack = batches.characterizationOuterBatch()

    @unittest.skipUnless(os.path.isdir(os.path.join(get_data_dir(), 'maf')),
                     "Skipping scienceRadarBatch test because operating without full MAF test data")
    def test_scienceRadar(self):
        # Loading the science radar batch requires reading a significant set of input files
        # This test is skipped if running with the lighter set of test data.
        # batch requires reading a lot of input files for lightcurves
        ack = batches.scienceRadarBatch()

    @unittest.skipUnless(os.path.isdir(os.path.join(get_data_dir(), 'maf')),
                     "Skipping glance test because operating without full MAF test data")
    def test_glance(self):
        ack = batches.glanceBatch()

        database = os.path.join(get_data_dir(), 'tests', 'example_dbv1.7_0yrs.db')
        opsdb = db.OpsimDatabase(database=database)
        resultsDb = db.ResultsDb(outDir=self.outDir)
        bgroup = metricBundles.MetricBundleGroup(ack, opsdb, outDir=self.outDir, resultsDb=resultsDb)
        bgroup.runAll()

    def tearDown(self):
        if os.path.isdir(self.outDir):
            shutil.rmtree(self.outDir)


if __name__ == "__main__":
    unittest.main()
