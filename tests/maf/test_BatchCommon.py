import unittest
import rubin_sim.maf.batches as batches
from rubin_sim.maf.db import OpsimDatabaseV4


class OpsTestDb(OpsimDatabaseV4):
    def __init__(self):
        # Override init so we don't connect to anything.
        self._colNames()


class TestCommon(unittest.TestCase):

    def testColMap(self):
        colmap = batches.ColMapDict('opsimv4')
        self.assertEqual(colmap['raDecDeg'], True)
        self.assertEqual(colmap['ra'], 'fieldRA')
        opsdb = OpsTestDb()
        colmap = batches.getColMap(opsdb)
        self.assertEqual(colmap['raDecDeg'], True)
        self.assertEqual(colmap['ra'], 'fieldRA')

    def testFilterList(self):
        filterlist, colors, orders, sqls, metadata = batches.common.filterList(all=False, extraSql=None)
        self.assertEqual(len(filterlist), 6)
        self.assertEqual(len(colors), 6)
        self.assertEqual(sqls['u'], 'filter = "u"')
        filterlist, colors, orders, sqls, metadata = batches.common.filterList(all=True, extraSql=None)
        self.assertIn('all', filterlist)
        self.assertEqual(sqls['all'], '')
        filterlist, colors, orders, sqls, metadata = batches.common.filterList(all=True, extraSql='night=3')
        self.assertEqual(sqls['all'], 'night=3')
        self.assertEqual(sqls['u'], '(night=3) and (filter = "u")')
        self.assertEqual(metadata['u'], 'night=3 u band')
        filterlist, colors, orders, sqls, metadata = batches.common.filterList(all=True, extraSql='night=3',
                                                                               extraMetadata='night 3')
        self.assertEqual(metadata['u'], 'night 3 u band')


if __name__ == "__main__":
    unittest.main()
