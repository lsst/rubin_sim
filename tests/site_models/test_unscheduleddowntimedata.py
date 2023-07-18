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
        downtime_data = UnscheduledDowntimeData(
            self.th,
            start_of_night_offset=self.startofnight,
            survey_length=self.survey_length,
            seed=self.seed,
        )
        self.assertEqual(downtime_data.seed, self.seed)
        self.assertEqual(downtime_data.survey_length, self.survey_length)
        self.assertEqual(self.th + TimeDelta(self.startofnight, format="jd"), downtime_data.night0)
        downtime_data = UnscheduledDowntimeData(
            self.th,
            start_of_night_offset=0,
            survey_length=self.survey_length,
            seed=self.seed,
        )
        self.assertEqual(downtime_data.night0, self.th)

    def test_information_after_initialization(self):
        downtime_data = UnscheduledDowntimeData(
            self.th,
            start_of_night_offset=self.startofnight,
            survey_length=self.survey_length,
            seed=self.seed,
        )
        downtime_data.make_data()
        self.assertEqual(len(downtime_data.downtime), 155)
        # Check some of the downtime values.
        dnight = downtime_data.downtime["end"] - downtime_data.downtime["start"]
        self.assertEqual(dnight[0].jd, 1)
        self.assertEqual(downtime_data.downtime["activity"][0], "minor event")
        self.assertEqual(dnight[2].jd, 7)
        self.assertEqual(downtime_data.downtime["activity"][2], "major event")

    def test_alternate_seed(self):
        downtime_data = UnscheduledDowntimeData(
            self.th,
            start_of_night_offset=self.startofnight,
            survey_length=self.survey_length,
            seed=3,
        )
        downtime_data.make_data()
        self.assertEqual(len(downtime_data.downtime), 145)

    def test_call(self):
        downtime_data = UnscheduledDowntimeData(
            self.th,
            start_of_night_offset=self.startofnight,
            survey_length=self.survey_length,
            seed=self.seed,
        )
        downtime_data.make_data()
        downtimes = downtime_data()
        self.assertEqual(downtimes["activity"][2], "major event")


if __name__ == "__main__":
    unittest.main()
