import os
import shutil
import sqlite3
import tempfile
import unittest

import pandas as pd

import rubin_sim.maf.db as db


class TestTrackingDb(unittest.TestCase):
    def setUp(self):
        self.run_name = "test)_run"
        self.run_group = "test"
        self.opsim_comment = "runcomment"
        self.maf_comment = "mafcomment"
        self.maf_dir = "mafdir"
        self.maf_version = "1.0"
        self.maf_date = "2017-01-01"
        self.run_version = "4.0"
        self.run_date = "2017-02-01"
        self.db_file = None

    def test_test_tracking_db_creation(self):
        """Test tracking database creation."""
        tempdir = tempfile.mkdtemp(prefix="trackDb")
        tracking_db_file = os.path.join(tempdir, "tracking.db")
        trackingdb = db.TrackingDb(database=tracking_db_file)
        self.assertTrue(os.path.isfile(tracking_db_file))
        trackingdb.close()
        shutil.rmtree(tempdir)

    def test_test_add_run(self):
        """Test adding a run to the tracking database."""
        tempdir = tempfile.mkdtemp(prefix="trackDb")
        tracking_db_file = os.path.join(tempdir, "tracking.db")
        trackingdb = db.TrackingDb(database=tracking_db_file)
        track_id = trackingdb.add_run(
            run_group=self.run_group,
            run_name=self.run_name,
            run_comment=self.opsim_comment,
            run_version=self.run_version,
            run_date=self.run_date,
            maf_comment=self.maf_comment,
            maf_dir=self.maf_dir,
            maf_version=self.maf_version,
            maf_date=self.maf_date,
            db_file=self.db_file,
        )
        con = sqlite3.connect(tracking_db_file)
        res = pd.read_sql("select * from runs", con).to_records()
        self.assertEqual(res["maf_run_id"][0], track_id)
        # Try adding this run again. Should return previous track_id.
        track_id2 = trackingdb.add_run(maf_dir=self.maf_dir)
        self.assertEqual(track_id, track_id2)
        # Test will add additional run, with new track_id.
        track_id3 = trackingdb.add_run(maf_dir="test2")
        self.assertNotEqual(track_id, track_id3)
        trackingdb.close()
        con.close()
        shutil.rmtree(tempdir)

    def test_test_del_run(self):
        """Test removing a run from the tracking database."""
        tempdir = tempfile.mkdtemp(prefix="trackDb")
        tracking_db_file = os.path.join(tempdir, "tracking.db")
        trackingdb = db.TrackingDb(database=tracking_db_file)
        track_id = trackingdb.add_run(maf_dir=self.maf_dir)
        track_id2 = trackingdb.add_run(maf_dir=self.maf_dir + "test2")
        con = sqlite3.connect(tracking_db_file)
        res = pd.read_sql("select * from runs", con).to_records(index=False)
        self.assertEqual(res["maf_run_id"][0], track_id)
        # Test removal works.
        trackingdb.delRun(track_id)
        res = pd.read_sql("select * from runs", con).to_records(index=False)
        # The run returned here is track_id2
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0][0], track_id2)
        # Test cannot remove run which does not exist.
        self.assertRaises(Exception, trackingdb.delRun, track_id)
        trackingdb.close()
        con.close()
        shutil.rmtree(tempdir)


if __name__ == "__main__":
    unittest.main()
