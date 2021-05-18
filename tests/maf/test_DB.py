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
        self.database = os.path.join(get_data_dir(), 'tests', 'example_dbv1.7_0yrs.db')
        self.driver = 'sqlite'

    def tearDown(self):
        del self.driver
        del self.database

    def testBaseDatabase(self):
        """Test base database class."""
        # Test instantiation connects to expected tables.
        basedb = db.Database(database=self.database, driver=self.driver)
        expectedTables = ['info', 'observations']
        self.assertEqual(set(basedb.tableNames),
                         set(expectedTables))
        # Test general query with a simple query.
        query = 'select airmass, fieldra, fielddec from observations where fielddec>0 limit 3'
        data = basedb.query_arbitrary(query)
        self.assertEqual(len(data), 3)

    def testSqliteFileNotExists(self):
        """Test that db gives useful error message if db file doesn't exist."""
        self.assertRaises(IOError, db.Database, 'thisdatabasedoesntexist_sqlite.db')


if __name__ == "__main__":
    unittest.main()
