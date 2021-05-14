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
        self.database = os.path.join(get_data_dir(), ('sims_data'), 'OpSimData',
                                     'astro-lsst-01_2014.db')
        self.oo = db.OpsimDatabaseV4(database=self.database)

    def tearDown(self):
        del self.oo
        del self.database
        self.oo = None

    def testOpsimDbSetup(self):
        """Test opsim specific database class setup/instantiation."""
        # Test tables were connected to.
        self.assertIn('SummaryAllProps', self.oo.tableNames)
        self.assertEqual(self.oo.defaultTable, 'SummaryAllProps')

    def testOpsimDbMetricData(self):
        """Test queries for sim data. """
        data = self.oo.fetchMetricData(['seeingFwhmEff', ], 'filter="r" and seeingFwhmEff<1.0')
        self.assertEqual(data.dtype.names, ('seeingFwhmEff',))
        self.assertLessEqual(data['seeingFwhmEff'].max(), 1.0)

    def testOpsimDbPropID(self):
        """Test queries for prop ID"""
        propids, propTags = self.oo.fetchPropInfo()
        self.assertGreater(len(list(propids.keys())), 0)
        self.assertGreater(len(propTags['WFD']), 0)
        self.assertGreaterEqual(len(propTags['DD']), 0)
        for w in propTags['WFD']:
            self.assertIn(w, propids)
        for d in propTags['DD']:
            self.assertIn(d, propids)

    def testOpsimDbFields(self):
        """Test queries for field data."""
        # Fetch field data for all fields.
        dataAll = self.oo.fetchFieldsFromFieldTable()
        self.assertEqual(dataAll.dtype.names, ('fieldId', 'fieldRA', 'fieldDec'))
        # Fetch field data for all fields requested by a particular propid.
        # Need to reinstate this capability.

    def testOpsimDbRunLength(self):
        """Test query for length of opsim run."""
        nrun = self.oo.fetchRunLength()
        self.assertEqual(nrun, 0.04)

    def testOpsimDbSimName(self):
        """Test query for opsim name."""
        simname = self.oo.fetchOpsimRunName()
        self.assertIsInstance(simname, str)
        self.assertEqual(simname, 'astro-lsst-01_2014')

    def testOpsimDbConfig(self):
        """Test generation of config data. """
        configsummary, configdetails = self.oo.fetchConfig()
        self.assertIsInstance(configsummary, dict)
        self.assertIsInstance(configdetails, dict)
        #  self.assertEqual(set(configsummary.keys()), set(['Version', 'RunInfo', 'Proposals', 'keyorder']))
        propids, proptags = self.oo.fetchPropInfo()
        propidsSummary = []
        for propname in configsummary['Proposals']:
            if propname != 'keyorder':
                propidsSummary.append(configsummary['Proposals'][propname]['PropId'])
        self.assertEqual(set(propidsSummary), set(propids))

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


if __name__ == "__main__":
    unittest.main()
