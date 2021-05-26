import unittest
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
        ack = batches.scienceRadarBatch()
        ack = batches.slewBasics()
        ack = batches.fOBatch()
        ack = batches.astrometryBatch()
        ack = batches.rapidRevisitBatch()


if __name__ == "__main__":
    unittest.main()
