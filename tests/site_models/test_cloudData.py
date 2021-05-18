import os
import sqlite3
import unittest
from astropy.time import Time, TimeDelta
from rubin_sim.site_models import CloudData
from rubin_sim.data import get_data_dir


class TestCloudModel(unittest.TestCase):

    def setUp(self):
        self.th = Time('2020-01-01', format='isot', scale='tai')
        self.cloud_db = os.path.join(get_data_dir(), 'site_models', 'cloud.db')
        self.num_original_values = 29201

    def test_basic_information_after_creation(self):
        cloudData = CloudData(self.th, cloud_db=self.cloud_db)
        self.assertEqual(cloudData.start_time, self.th)
        cloudData = CloudData(self.th, cloud_db=self.cloud_db, offset_year = 1)
        self.assertEqual(cloudData.start_time, Time('2021-01-01', format='isot', scale='tai'))

    def test_information_after_initialization(self):
        # Test setting cloud_db explicitly.
        cloudData = CloudData(self.th, cloud_db=self.cloud_db)
        cloudData.read_data()
        self.assertEqual(cloudData.cloud_values.size, self.num_original_values)
        self.assertEqual(cloudData.cloud_dates.size, self.num_original_values)
        # Test that find built-in module automatically.
        cloudData = CloudData(self.th)
        cloudData.read_data()
        self.assertEqual(cloudData.cloud_dates.size, self.num_original_values)

    def test_get_clouds(self):
        cloudData = CloudData(self.th, cloud_db=self.cloud_db)
        cloudData.read_data()
        dt = TimeDelta(700000, format='sec')
        self.assertEqual(cloudData(self.th + dt), 0.5)
        dt = TimeDelta(701500, format='sec')
        self.assertEqual(cloudData(self.th + dt), 0.5)
        dt = TimeDelta(705000, format='sec')
        self.assertEqual(cloudData(self.th + dt), 0.375)
        dt = TimeDelta(6306840, format='sec')
        self.assertEqual(cloudData(self.th + dt), 0.0)

    def test_get_clouds_using_different_start_month(self):
        # Just changing the starting month
        t2 = Time('2020-05-24', format='isot', scale='tai')
        cloud1 = CloudData(t2)
        self.assertEqual(cloud1.start_time, self.th)
        cloud1.read_data()
        dt = TimeDelta(700000, format='sec')
        self.assertEqual(cloud1(t2 + dt), 0.0)
        dt = TimeDelta(6306840, format='sec')
        self.assertEqual(cloud1(t2 + dt), 0.25)


if __name__ == "__main__":
    unittest.main()
