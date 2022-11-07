import os
import unittest
import rubin_sim.maf.db as db
import tempfile
import shutil
import sqlite3
import pandas as pd


class TestTrackingDb(unittest.TestCase):
    def setUp(self):
        self.opsimRun = "testopsim"
        self.opsimGroup = "test"
        self.opsimComment = "opsimcomment"
        self.mafComment = "mafcomment"
        self.mafDir = "mafdir"
        self.mafVersion = "1.0"
        self.mafDate = "2017-01-01"
        self.opsimVersion = "4.0"
        self.opsimDate = "2017-02-01"
        self.dbFile = None

    def test_testTrackingDbCreation(self):
        """Test tracking database creation."""
        tempdir = tempfile.mkdtemp(prefix="trackDb")
        trackingDbFile = os.path.join(tempdir, "tracking.db")
        trackingdb = db.TrackingDb(database=trackingDbFile)
        self.assertTrue(os.path.isfile(trackingDbFile))
        trackingdb.close()
        shutil.rmtree(tempdir)

    def test_testAddRun(self):
        """Test adding a run to the tracking database."""
        tempdir = tempfile.mkdtemp(prefix="trackDb")
        trackingDbFile = os.path.join(tempdir, "tracking.db")
        trackingdb = db.TrackingDb(database=trackingDbFile)
        trackId = trackingdb.add_run(
            run_group=self.opsimGroup,
            run_name=self.opsimRun,
            run_comment=self.opsimComment,
            run_version=self.opsimVersion,
            run_date=self.opsimDate,
            maf_comment=self.mafComment,
            maf_dir=self.mafDir,
            maf_version=self.mafVersion,
            maf_date=self.mafDate,
            db_file=self.dbFile,
        )
        con = sqlite3.connect(trackingDbFile)
        res = pd.read_sql("select * from runs", con).to_records()
        self.assertEqual(res["maf_run_id"][0], trackId)
        # Try adding this run again. Should return previous trackId.
        trackId2 = trackingdb.add_run(maf_dir=self.mafDir)
        self.assertEqual(trackId, trackId2)
        # Test will add additional run, with new trackId.
        trackId3 = trackingdb.add_run(maf_dir="test2")
        self.assertNotEqual(trackId, trackId3)
        trackingdb.close()
        con.close()
        shutil.rmtree(tempdir)

    def test_testDelRun(self):
        """Test removing a run from the tracking database."""
        tempdir = tempfile.mkdtemp(prefix="trackDb")
        trackingDbFile = os.path.join(tempdir, "tracking.db")
        trackingdb = db.TrackingDb(database=trackingDbFile)
        trackId = trackingdb.add_run(maf_dir=self.mafDir)
        trackId2 = trackingdb.add_run(maf_dir=self.mafDir + "test2")
        con = sqlite3.connect(trackingDbFile)
        res = pd.read_sql("select * from runs", con).to_records(index=False)
        self.assertEqual(res["maf_run_id"][0], trackId)
        # Test removal works.
        trackingdb.delRun(trackId)
        res = pd.read_sql("select * from runs", con).to_records(index=False)
        # The run returned here is trackId2
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0][0], trackId2)
        # Test cannot remove run which does not exist.
        self.assertRaises(Exception, trackingdb.delRun, trackId)
        trackingdb.close()
        con.close()
        shutil.rmtree(tempdir)


if __name__ == "__main__":
    unittest.main()
