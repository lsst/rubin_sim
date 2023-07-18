import os
import unittest

from astropy.time import Time, TimeDelta

from rubin_sim.data import get_data_dir
from rubin_sim.site_models import SeeingData

# Unit test here uses oldest/original opsim seeing database, "Seeing.db".
# Could be updated to use a new DB, but that would require changing some of these hard-coded numbers.


class TestSeeingData(unittest.TestCase):
    def setUp(self):
        self.time = Time("2020-01-01", format="isot", scale="tai")
        self.seeing_db = os.path.join(get_data_dir(), "tests", "seeing.db")
        print(self.seeing_db)

    def test_information_after_read(self):
        seeing_data = SeeingData(self.time, seeing_db=self.seeing_db)
        seeing_data.read_data()
        self.assertTrue(len(seeing_data.seeing_values) > 0)
        self.assertTrue(len(seeing_data.seeing_dates) > 0)
        self.assertEqual(seeing_data.start_time, self.time)
        self.assertEqual(seeing_data.seeing_db, self.seeing_db)

    def test_fwhm500_at_time(self):
        seeing_data = SeeingData(self.time, self.seeing_db, offset_year=0)
        seeing_data.read_data()
        dt = TimeDelta(75400, format="sec")
        self.assertEqual(seeing_data(self.time + dt), 0.859431982040405)
        dt = TimeDelta(76700, format="sec")
        self.assertEqual(seeing_data(self.time + dt), 0.646009027957916)
        dt = TimeDelta(63190400, format="sec")
        self.assertEqual(seeing_data(self.time + dt), 0.64860999584198)
        dt = TimeDelta(189424900, format="sec")
        self.assertEqual(seeing_data(self.time + dt), 0.699440002441406)
        # Test time selection from seeing data.
        dt = TimeDelta(800, format="sec")
        fwhm500 = seeing_data(self.time + dt)
        # Hack seeing data to remove first date, thus db does not start at zero.
        seeing_data.seeing_dates = seeing_data.seeing_dates[:-1]
        seeing_data.seeing_values = seeing_data.seeing_values[:-1]
        seeing_data.time_range = seeing_data.seeing_dates[-1] - seeing_data.seeing_dates[0]
        seeing_data.min_time = seeing_data.seeing_dates[0]
        self.assertEqual(fwhm500, seeing_data(self.time + dt))

    def test_using_different_start_month(self):
        t2 = Time("2020-05-24", format="isot", scale="tai")
        seeing_data = SeeingData(t2, self.seeing_db, offset_year=0)
        self.assertEqual(seeing_data.start_time, self.time)
        seeing_data.read_data()
        dt = TimeDelta(75400, format="sec")
        self.assertEqual(seeing_data(t2 + dt), 0.437314003705978)
        dt = TimeDelta(63190400, format="sec")
        self.assertEqual(seeing_data(t2 + dt), 0.453994989395142)


if __name__ == "__main__":
    unittest.main()
