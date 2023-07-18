import unittest

import numpy as np
from astropy.time import Time, TimeDelta

from rubin_sim.site_models import DowntimeModel, ScheduledDowntimeData, UnscheduledDowntimeData


class TestDowntimeModel(unittest.TestCase):
    def test_call(self):
        # Check the calculation from fwhm_500 to fwhm_eff/fwhm_geom.
        # Use simple effective wavelengths and airmass values.
        downtime_model = DowntimeModel()
        t = Time("2022-10-01")
        sched = ScheduledDowntimeData(t)
        sched.read_data()
        unsched = UnscheduledDowntimeData(t)
        unsched.make_data()
        efd_data = {"unscheduled_downtimes": unsched(), "scheduled_downtimes": sched()}
        # Set time to within first scheduled downtime.
        t_now = sched.downtime[0]["start"] + TimeDelta(0.5, format="jd")
        target_dict = {"time": t_now}
        dt_status = downtime_model(efd_data, target_dict)
        # Expect return dict of : {'status': status, 'end': end_down, 'next': next_sched['start']}
        # Check keys
        for k in ("status", "end", "next"):
            self.assertTrue(k in dt_status)
        # downtime status is "True" if system is down.
        self.assertEqual(True, dt_status["status"])
        self.assertEqual(dt_status["end"], sched.downtime[0]["end"])
        self.assertEqual(dt_status["next"], sched.downtime[1]["start"])


if __name__ == "__main__":
    unittest.main()
