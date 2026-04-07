import unittest
from functools import partial

import numpy as np
from scipy import stats

from rubin_sim.sim_archive.prenight import AnomalousOverheadFunc


class TestAnomalousOverheadFunc(unittest.TestCase):
    def test_scatter_overhead_normal_dist(self):
        sample_size = 20
        exptime = 30.0
        for scale in [1.75, 2]:
            for loc in [0.0, 3.0]:
                dist_params = {"scale": scale, "loc": loc}
                cdf = partial(stats.norm.cdf, loc=loc, scale=scale)
                func = AnomalousOverheadFunc(seed=6563, scatter_kwargs=dist_params)
                for slewtime in [0.0, 100.0]:
                    for visittime in [40.0, 50.0]:
                        overheads = np.array(
                            list(func(visittime, slewtime, exptime=exptime) for i in range(sample_size))
                        )
                        _, p_value = stats.kstest(overheads, cdf)
                        # Depending on the random number seed, there's a
                        # 0.1% chance per iteration of failing this test even
                        # if everything is okay.
                        assert p_value > 0.001

    def test_scatter_overhead_uniform_dist(self):
        sample_size = 20
        exptime = 30.0
        for scale in [1.75, 2]:
            for loc in [0.0, 3.0]:
                low = loc - scale / 2
                high = loc + scale / 2
                dist_params = {"low": low, "high": high}
                cdf = partial(stats.uniform.cdf, loc=low, scale=scale)
                func = AnomalousOverheadFunc(
                    seed=6563, scatter_distribution="uniform", scatter_kwargs=dist_params
                )
                for slewtime in [0.0, 100.0]:
                    for visittime in [60.0, 40.0]:
                        overheads = np.array(
                            list(func(visittime, slewtime, exptime) for i in range(sample_size))
                        )
                        _, p_value = stats.kstest(overheads, cdf)
                        # Depending on the random number seed, there's a
                        # 0.1% chance per iteration of failing this test even
                        # if everything is okay.
                        assert p_value > 0.001

    def test_no_negative_overhead(self):
        sample_size = 20
        slewtime = 2.0
        visittime = 35.0
        exptime = 30.0
        min_overhead = visittime + slewtime
        func = AnomalousOverheadFunc(
            seed=6563, scatter_kwargs={"scale": 100.0, "loc": 0.0}, min_overhead=min_overhead
        )
        offsets = np.array(list(func(visittime, slewtime, exptime) for i in range(sample_size)))
        assert np.all(offsets + visittime + slewtime - exptime >= min_overhead)

    def test_slew_scale(self):
        sample_size = 20
        slewtime = 20.0
        visittime = 36.0
        exptime = 30.0
        slew_scale = 0.1
        func = AnomalousOverheadFunc(seed=6563, slew_scale=slew_scale)
        overheads = np.array(list(func(visittime, slewtime, exptime) for i in range(sample_size)))
        fractional_overheads = overheads / slewtime
        cdf = partial(stats.norm.cdf, loc=0.0, scale=slew_scale)
        _, p_value = stats.kstest(fractional_overheads, cdf)
        # Depending on the random number seed, there's a
        # 0.1% chance per iteration of failing this test even
        # if everything is okay.
        assert p_value > 0.001

    def test_visit_scale(self):
        sample_size = 20
        slewtime = 20.0
        exptime = 30.0
        visittime = 35.0
        visit_scale = 0.1
        func = AnomalousOverheadFunc(seed=6563, visit_scale=visit_scale)
        overheads = np.array(list(func(visittime, slewtime, exptime) for i in range(sample_size)))
        fractional_overheads = overheads / (visittime - exptime)

        assert np.all(fractional_overheads >= 0.0)

        cdf = partial(stats.halfnorm.cdf, loc=0.0, scale=visit_scale)
        _, p_value = stats.kstest(fractional_overheads[fractional_overheads > 0], cdf)
        # Depending on the random number seed, there's a
        # 0.1% chance per iteration of failing this test even
        # if everything is okay.
        assert p_value > 0.001


if __name__ == "__main__":
    unittest.main()
