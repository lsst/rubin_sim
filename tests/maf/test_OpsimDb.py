import matplotlib
matplotlib.use("Agg")
import os
import unittest
import rubin_sim.maf.db as db
from rubin_sim.utils.CodeUtilities import sims_clean_up
from rubin_sim.data import get_data_dir


class TestOpsimDb(unittest.TestCase):
    """Test opsim specific database class."""

    @classmethod
    def tearDownClass(cls):
        sims_clean_up()

    def setUp(self):
        self.database = os.path.join(get_data_dir(), 'tests','example_dbv1.7_0yrs.db')
        self.oo = db.OpsimDatabase(database=self.database)

    def tearDown(self):
        self.oo.close()
        del self.oo
        del self.database
        self.oo = None

    def testOpsimDbSetup(self):
        """Test opsim specific database class setup/instantiation."""
        # Test tables were connected to.
        self.assertIn('observations', self.oo.tableNames)
        self.assertEqual(self.oo.defaultTable, 'observations')

    def testOpsimDbMetricData(self):
        """Test queries for sim data. """
        data = self.oo.fetchMetricData(['seeingFwhmEff', ], 'filter="r" and seeingFwhmEff<1.0')
        self.assertEqual(data.dtype.names, ('seeingFwhmEff',))
        self.assertLessEqual(data['seeingFwhmEff'].max(), 1.0)

    def testCreateSqlWhere(self):
        """
        Test that the createSQLWhere method handles expected cases.
        """
        # propTags is a dictionary of lists returned by OpsimDatabase
        propTags = {'WFD': [1, 2, 3], 'DD': [4], 'Rolling': [2]}
        # If tag is in dictionary with one value, returned sql where clause
        # is simply 'propId = 4'
        tag = 'DD'
        sqlWhere = self.oo.createSQLWhere(tag, propTags)
        self.assertEqual(sqlWhere, 'proposalId = 4')
        # if multiple proposals with the same tag, all should be in list.
        tag = 'WFD'
        sqlWhere = self.oo.createSQLWhere(tag, propTags)
        self.assertEqual(sqlWhere.split()[0], '(proposalId')
        for id_val in propTags['WFD']:
            self.assertIn('%s' % (id_val), sqlWhere)
        # And the same id can be in multiple proposals.
        tag = 'Rolling'
        sqlWhere = self.oo.createSQLWhere(tag, propTags)
        self.assertEqual(sqlWhere, 'proposalId = 2')
        # And tags not in propTags are handled.
        badprop = 'proposalId like "NO PROP"'
        tag = 'nogo'
        sqlWhere = self.oo.createSQLWhere(tag, propTags)
        self.assertEqual(sqlWhere, badprop)
        # And tags which identify no proposal ID are handled.
        propTags['Rolling'] = []
        sqlWhere = self.oo.createSQLWhere(tag, propTags)
        self.assertEqual(sqlWhere, badprop)

    def test_getConfig(self):
        summary, details = self.oo.fetchConfig()
        self.assertTrue('Version' in summary.keys())
        self.assertTrue('RunInfo' in summary.keys())



if __name__ == "__main__":
    unittest.main()
