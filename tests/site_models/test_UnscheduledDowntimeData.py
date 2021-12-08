import unittest
from astropy.time import Time, TimeDelta

from rubin_sim.site_models import UnscheduledDowntimeData


class UnscheduledDowntimeDataTest(unittest.TestCase):
    def setUp(self):
        self.th = Time("2020-01-01", format="isot", scale="tai")
        self.startofnight = -0.34
        self.seed = 1516231120
        self.survey_length = 3650 * 2

    def test_basic_information_after_creation(self):
        downtimeData = UnscheduledDowntimeData(
            self.th,
            start_of_night_offset=self.startofnight,
            survey_length=self.survey_length,
            seed=self.seed,
        )
        self.assertEqual(downtimeData.seed, self.seed)
        self.assertEqual(downtimeData.survey_length, self.survey_length)
        self.assertEqual(
            self.th + TimeDelta(self.startofnight, format="jd"), downtimeData.night0
        )
        downtimeData = UnscheduledDowntimeData(
            self.th,
            start_of_night_offset=0,
            survey_length=self.survey_length,
            seed=self.seed,
        )
        self.assertEqual(downtimeData.night0, self.th)

    def test_information_after_initialization(self):
        downtimeData = UnscheduledDowntimeData(
            self.th,
            start_of_night_offset=self.startofnight,
            survey_length=self.survey_length,
            seed=self.seed,
        )
        downtimeData.make_data()
        self.assertEqual(len(downtimeData.downtime), 155)
        # Check some of the downtime values.
        dnight = downtimeData.downtime["end"] - downtimeData.downtime["start"]
        self.assertEqual(dnight[0].jd, 1)
        self.assertEqual(downtimeData.downtime["activity"][0], "minor event")
        self.assertEqual(dnight[2].jd, 7)
        self.assertEqual(downtimeData.downtime["activity"][2], "major event")

    def test_alternate_seed(self):
        downtimeData = UnscheduledDowntimeData(
            self.th,
            start_of_night_offset=self.startofnight,
            survey_length=self.survey_length,
            seed=3,
        )
        downtimeData.make_data()
        self.assertEqual(len(downtimeData.downtime), 145)

    def test_call(self):
        downtimeData = UnscheduledDowntimeData(
            self.th,
            start_of_night_offset=self.startofnight,
            survey_length=self.survey_length,
            seed=self.seed,
        )
        downtimeData.make_data()
        downtimes = downtimeData()
        self.assertEqual(downtimes["activity"][2], "major event")


if __name__ == "__main__":
    unittest.main()
