import os
import unittest
import rubin_sim.maf.db as db
import tempfile
import shutil


class TestTrackingDb(unittest.TestCase):

    def setUp(self):
        self.opsimRun = 'testopsim'
        self.opsimGroup = 'test'
        self.opsimComment = 'opsimcomment'
        self.mafComment = 'mafcomment'
        self.mafDir = 'mafdir'
        self.mafVersion = '1.0'
        self.mafDate = '2017-01-01'
        self.opsimVersion = '4.0'
        self.opsimDate = '2017-02-01'
        self.dbFile = None

    def test_testTrackingDbCreation(self):
        """Test tracking database creation."""
        tempdir = tempfile.mkdtemp(prefix='trackDb')
        trackingDbFile = os.path.join(tempdir, 'tracking.db')
        trackingdb = db.TrackingDb(database=trackingDbFile)
        self.assertTrue(os.path.isfile(trackingDbFile))
        trackingdb.close()
        shutil.rmtree(tempdir)

    def test_testAddRun(self):
        """Test adding a run to the tracking database."""
        tempdir = tempfile.mkdtemp(prefix='trackDb')
        trackingDbFile = os.path.join(tempdir, 'tracking.db')
        trackingdb = db.TrackingDb(database=trackingDbFile)
        trackId = trackingdb.addRun(opsimGroup=self.opsimGroup, opsimRun=self.opsimRun,
                                    opsimComment=self.opsimComment,
                                    opsimVersion=self.opsimVersion, opsimDate=self.opsimDate,
                                    mafComment=self.mafComment, mafDir=self.mafDir,
                                    mafVersion=self.mafVersion, mafDate=self.mafDate,
                                    dbFile=self.dbFile)
        tdb = db.Database(database=trackingDbFile)
        res = tdb.query_arbitrary('select * from runs')
        self.assertEqual(res['mafRunId'][0], trackId)
        # Try adding this run again. Should return previous trackId.
        trackId2 = trackingdb.addRun(mafDir=self.mafDir)
        self.assertEqual(trackId, trackId2)
        # Test will add additional run, with new trackId.
        trackId3 = trackingdb.addRun(mafDir='test2')
        self.assertNotEqual(trackId, trackId3)
        trackingdb.close()
        tdb.close()
        shutil.rmtree(tempdir)

    def test_testDelRun(self):
        """Test removing a run from the tracking database."""
        tempdir = tempfile.mkdtemp(prefix='trackDb')
        trackingDbFile = os.path.join(tempdir, 'tracking.db')
        trackingdb = db.TrackingDb(database=trackingDbFile)
        tdb = db.Database(database=trackingDbFile)
        trackId = trackingdb.addRun(mafDir=self.mafDir)
        trackId2 = trackingdb.addRun(mafDir=self.mafDir + 'test2')
        res = tdb.query_arbitrary('select * from runs')
        self.assertEqual(res['mafRunId'][0], trackId)
        # Test removal works.
        trackingdb.delRun(trackId)
        res = tdb.query_arbitrary('select * from runs')
        # The run returned here is trackId2
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0][0], trackId2)
        # Test cannot remove run which does not exist.
        self.assertRaises(Exception, trackingdb.delRun, trackId)
        trackingdb.close()
        tdb.close()
        shutil.rmtree(tempdir)


if __name__ == "__main__":
    unittest.main()
