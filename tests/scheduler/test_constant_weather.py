import unittest

import numpy as np

import rubin_sim.site_models
from rubin_sim.scheduler.model_observatory import ModelObservatory


class TestConstantWeather(unittest.TestCase):
    def test_all_constant_weather(self):
        test_seeing = 1.234
        test_clouds = 0.11
        test_wind_speed = 10.0
        test_wind_direction = np.radians(30.0)

        seeing_data = rubin_sim.site_models.ConstantSeeingData(test_seeing)
        cloud_data = rubin_sim.site_models.ConstantCloudData(test_clouds)
        wind_data = rubin_sim.site_models.ConstantWindData(
            wind_speed=test_wind_speed,
            wind_direction=test_wind_direction,
        )

        model_observatory = ModelObservatory(
            seeing_data=seeing_data, cloud_data=cloud_data, wind_data=wind_data, no_sky=True
        )
        conditions = model_observatory.return_conditions()
        self.assertEqual(conditions.wind_direction, test_wind_direction)
        self.assertEqual(conditions.wind_speed, test_wind_speed)
        self.assertEqual(conditions.bulk_cloud, test_clouds)
        assert np.nanmin(conditions.fwhm_eff["g"]) > test_seeing


if __name__ == "__main__":
    unittest.main()
