import unittest
from datetime import datetime

from rubin_sim.utils import TimeHandler

SECONDS_IN_DAY = 60.0 * 60.0 * 24.0


class TimeHandlerTest(unittest.TestCase):
    def setUp(self):
        self.start_date = "2020-05-24"
        self.th = TimeHandler(self.start_date)

    def test_basic_information_after_creation(self):
        self.assertEqual(self.th.initial_dt, datetime(2020, 5, 24))

    def test_bad_date_string(self):
        with self.assertRaises(ValueError):
            TimeHandler("18-09-15")

    def test_return_initial_timestamp(self):
        truth_timestamp = (datetime(2020, 5, 24) - datetime(1970, 1, 1)).total_seconds()
        self.assertEqual(self.th.initial_timestamp, truth_timestamp)

    def test_time_adjustment_seconds(self):
        self.th.update_time(30.0, "seconds")
        self.assertEqual(self.th.current_dt, datetime(2020, 5, 24, 0, 0, 30))

    def test_time_adjustment_hours(self):
        self.th.update_time(3.5, "hours")
        self.assertEqual(self.th.current_dt, datetime(2020, 5, 24, 3, 30, 0))

    def test_time_adjustment_days(self):
        self.th.update_time(4, "days")
        self.assertEqual(self.th.current_dt, datetime(2020, 5, 28))

    def test_multiple_time_adjustments(self):
        for i in range(3):
            self.th.update_time(30.0, "seconds")
        self.assertEqual(self.th.current_dt, datetime(2020, 5, 24, 0, 1, 30))

    def test_timestamp_after_time_adjustment(self):
        truth_timestamp = (datetime(2020, 5, 24, 0, 0, 30) - datetime(1970, 1, 1)).total_seconds()
        self.th.update_time(30.0, "seconds")
        self.assertEqual(self.th.current_timestamp, truth_timestamp)
        self.assertNotEqual(self.th.current_timestamp, self.th.initial_timestamp)

    def test_current_timestamp_string(self):
        self.assertEqual(self.th.current_timestring, "2020-05-24T00:00:00")

    def test_time_span_less_than_time_elapsed(self):
        self.th.update_time(10, "days")
        self.assertFalse(self.th.has_time_elapsed(9 * SECONDS_IN_DAY))

    def test_time_span_is_greater_than_time_elapsed(self):
        self.th.update_time(10, "days")
        self.assertTrue(self.th.has_time_elapsed(11 * SECONDS_IN_DAY))

    def test_future_timestring(self):
        self.assertEqual(self.th.future_timestring(19.0, "hours"), "2020-05-24T19:00:00")

    def test_midnight_timestamp(self):
        self.th.update_time(15, "hours")
        truth_timestamp = (datetime(2020, 5, 24) - datetime(1970, 1, 1)).total_seconds()
        self.assertEqual(self.th.current_midnight_timestamp, truth_timestamp)

    def test_next_midnight_timestamp(self):
        self.th.update_time(15, "hours")
        truth_timestamp = (datetime(2020, 5, 25) - datetime(1970, 1, 1)).total_seconds()
        self.assertEqual(self.th.next_midnight_timestamp, truth_timestamp)

    def test_future_timestamp(self):
        truth_timestamp = 1590418800.0
        self.assertEqual(self.th.future_timestamp(39, "hours"), truth_timestamp)

    def test_future_datetime(self):
        truth_datetime = datetime(2020, 5, 25, 15)
        self.assertEqual(self.th.future_datetime(39, "hours"), truth_datetime)

    def test_future_datetime_alternate_timestamp(self):
        alternate_timestamp = 1590537600.0
        truth_datetime = datetime(2020, 5, 28, 15)
        self.assertEqual(
            self.th.future_datetime(39, "hours", timestamp=alternate_timestamp),
            truth_datetime,
        )

    def test_time_since_start(self):
        self.th.update_time(10, "days")
        self.assertEqual(self.th.time_since_start, 864000)

    def test_time_since_given(self):
        self.assertEqual(self.th.time_since_given(1590364800), 86400)

    def test_time_since_given_datetime(self):
        future_given_date = datetime(self.th.initial_dt.year, 6, 10)
        self.assertEqual(self.th.time_since_given_datetime(future_given_date), 1468800)
        past_given_date = datetime(self.th.initial_dt.year, 4, 20)
        self.assertEqual(self.th.time_since_given_datetime(past_given_date, reverse=True), 2937600)
        same_given_date = datetime(self.th.initial_dt.year, 5, 24)
        self.assertEqual(self.th.time_since_given_datetime(same_given_date, reverse=True), 0)


if __name__ == "__main__":
    unittest.main()
