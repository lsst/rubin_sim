import matplotlib
matplotlib.use("Agg")
import os
import unittest
import rubin_sim.maf.db as db
from rubin_sim.utils.CodeUtilities import sims_clean_up
from rubin_sim.data import get_data_dir


class TestDb(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        sims_clean_up()

    def setUp(self):
        self.database = os.path.join(getPackageDir('sims_data'),
                                     'OpSimData', 'astro-lsst-01_2014.db')
        self.driver = 'sqlite'

    def tearDown(self):
        del self.driver
        del self.database

    def testBaseDatabase(self):
        """Test base database class."""
        # Test instantiation connects to expected tables.
        basedb = db.Database(database=self.database, driver=self.driver)
        expectedTables = ['Config', 'ScheduledDowntime', 'SlewMaxSpeeds',
                          'Field', 'Session', 'SummaryAllProps', 'ObsExposures',
                          'SlewActivities', 'TargetExposures', 'ObsHistory',
                          'SlewFinalState', 'TargetHistory', 'ObsProposalHistory',
                          'SlewHistory', 'TargetProposalHistory', 'Proposal',
                          'ProposalField',
                          'SlewInitialState', 'UnscheduledDowntime']
        self.assertEqual(set(basedb.tableNames),
                         set(expectedTables))
        # Test general query with a simple query.
        query = 'select fieldId, ra, dec from Field where dec>0 limit 3'
        data = basedb.query_arbitrary(query)
        self.assertEqual(len(data), 3)
        # Test query columns with a simple query.
        data = basedb.query_columns('Field', colnames=['fieldId', 'ra', 'dec'], numLimit=3)
        self.assertEqual(data.dtype.names, ('fieldId', 'ra', 'dec'))
        self.assertEqual(len(data), 3)

    def testSqliteFileNotExists(self):
        """Test that db gives useful error message if db file doesn't exist."""
        self.assertRaises(IOError, db.Database, 'thisdatabasedoesntexist_sqlite.db')


if __name__ == "__main__":
    unittest.main()
