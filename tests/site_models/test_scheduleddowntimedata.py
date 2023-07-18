import os
import unittest

from astropy.time import Time, TimeDelta

from rubin_sim.data import get_data_dir
from rubin_sim.site_models import ScheduledDowntimeData


class ScheduledDowntimeDataTest(unittest.TestCase):
    def setUp(self):
        self.th = Time("2020-01-01", format="isot", scale="tai")
        self.startofnight = -0.34
        self.downtime_db = os.path.join(get_data_dir(), "site_models", "scheduled_downtime.db")

    def test_basic_information_after_creation(self):
        downtime_data = ScheduledDowntimeData(self.th, start_of_night_offset=self.startofnight)
        self.assertEqual(downtime_data.scheduled_downtime_db, self.downtime_db)
        self.assertEqual(self.th + TimeDelta(self.startofnight, format="jd"), downtime_data.night0)
        downtime_data = ScheduledDowntimeData(self.th, start_of_night_offset=0)
        self.assertEqual(downtime_data.night0, self.th)

    def test_information_after_initialization(self):
        downtime_data = ScheduledDowntimeData(self.th, start_of_night_offset=self.startofnight)
        downtime_data.read_data()
        self.assertEqual(len(downtime_data.downtime), 31)
        # Check some of the downtime values.
        dnight = downtime_data.downtime["end"] - downtime_data.downtime["start"]
        self.assertEqual(dnight[0].jd, 7)
        self.assertEqual(downtime_data.downtime["activity"][0], "general maintenance")
        self.assertEqual(dnight[4].jd, 14)
        self.assertEqual(downtime_data.downtime["activity"][4], "recoat mirror")

    def test_call(self):
        downtime_data = ScheduledDowntimeData(self.th, start_of_night_offset=self.startofnight)
        downtime_data.read_data()
        downtimes = downtime_data()
        self.assertEqual(downtimes["activity"][4], "recoat mirror")


if __name__ == "__main__":
    unittest.main()
