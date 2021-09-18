import unittest
import os
from rubin_sim.data import get_data_dir
import rubin_sim.maf.batches as batches


class TestBatches(unittest.TestCase):

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

    @unittest.skipUnless(os.path.isdir(os.path.join(get_data_dir(), 'maf')),
                     "Skipping scienceRadarBatch test because operating without full MAF test data")
    def test_scienceRadar(self):
        # Loading the science radar batch requires reading a significant set of input files
        # This test is skipped if running with the lighter set of test data.
        # batch requires reading a lot of input files for lightcurves
        ack = batches.scienceRadarBatch()

if __name__ == "__main__":
    unittest.main()
