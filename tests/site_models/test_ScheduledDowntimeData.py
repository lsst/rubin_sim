import os
import unittest
from astropy.time import Time, TimeDelta
from rubin_sim.site_models import ScheduledDowntimeData
from rubin_sim.data import get_data_dir


class ScheduledDowntimeDataTest(unittest.TestCase):

    def setUp(self):
        self.th = Time('2020-01-01', format='isot', scale='tai')
        self.startofnight = -0.34
        self.downtime_db = os.path.join(get_data_dir(), 'site_models', 'scheduled_downtime.db')

    def test_basic_information_after_creation(self):
        downtimeData = ScheduledDowntimeData(self.th, start_of_night_offset=self.startofnight)
        self.assertEqual(downtimeData.scheduled_downtime_db, self.downtime_db)
        self.assertEqual(self.th + TimeDelta(self.startofnight, format='jd'), downtimeData.night0)
        downtimeData = ScheduledDowntimeData(self.th, start_of_night_offset=0)
        self.assertEqual(downtimeData.night0, self.th)

    def test_information_after_initialization(self):
        downtimeData = ScheduledDowntimeData(self.th, start_of_night_offset=self.startofnight)
        downtimeData.read_data()
        self.assertEqual(len(downtimeData.downtime), 31)
        # Check some of the downtime values.
        dnight = downtimeData.downtime['end'] - downtimeData.downtime['start']
        self.assertEqual(dnight[0].jd, 7)
        self.assertEqual(downtimeData.downtime['activity'][0], 'general maintenance')
        self.assertEqual(dnight[4].jd, 14)
        self.assertEqual(downtimeData.downtime['activity'][4], 'recoat mirror')

    def test_call(self):
        downtimeData = ScheduledDowntimeData(self.th, start_of_night_offset=self.startofnight)
        downtimeData.read_data()
        downtimes = downtimeData()
        self.assertEqual(downtimes['activity'][4], 'recoat mirror')


if __name__ == "__main__":
    unittest.main()
