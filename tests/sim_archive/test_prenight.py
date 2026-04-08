import unittest
from functools import partial

import numpy as np
from scipy import stats

from rubin_sim.sim_archive.prenight import AnomalousOverheadFunc


class TestAnomalousOverheadFunc(unittest.TestCase):
    def test_scatter_overhead_normal_dist(self) -> None:
        sample_size = 20
        for scale in [1.75, 2]:
            for loc in [0.0, 3.0]:
                dist_params = {"scale": scale, "loc": loc}
                cdf = partial(stats.norm.cdf, loc=loc, scale=scale)
                func = AnomalousOverheadFunc(
                    seed=6563, scatter_distribution="normal", scatter_kwargs=dist_params
                )
                for slewtime in [0.0, 100.0]:
                    obs = {"slewtime": slewtime}
                    overheads = np.array(list(func(obs=obs) for i in range(sample_size)))
                    _, p_value = stats.kstest(overheads, cdf)
                    # Depending on the random number seed, there's a
                    # 0.1% chance per iteration of failing this test even
                    # if everything is okay.
                    assert p_value > 0.001

    def test_scatter_overhead_halfnormal_dist(self) -> None:
        sample_size = 20
        for scale in [1.75, 2]:
            for loc in [0.0, 3.0]:
                dist_params = {"scale": scale, "loc": loc}
                cdf = partial(stats.halfnorm.cdf, loc=loc, scale=scale)
                func = AnomalousOverheadFunc(seed=6563, scatter_kwargs=dist_params)
                for slewtime in [0.0, 100.0]:
                    obs = {"slewtime": slewtime}
                    overheads = np.array(list(func(obs=obs) for i in range(sample_size)))
                    _, p_value = stats.kstest(overheads, cdf)
                    # Depending on the random number seed, there's a
                    # 0.1% chance per iteration of failing this test even
                    # if everything is okay.
                    assert p_value > 0.001

    def test_no_negative_overhead(self) -> None:
        sample_size = 20
        obs = dict(slewtime=2.0, visittime=35.0, exptime=30.0)
        min_overhead = 7.0
        func = AnomalousOverheadFunc(
            seed=6563, scatter_kwargs={"scale": 100.0, "loc": 0.0}, min_overhead=min_overhead
        )
        offsets = np.array(list(func(obs=obs) for i in range(sample_size)))
        assert np.all(offsets + obs["visittime"] + obs["slewtime"] - obs["exptime"] >= min_overhead)

    def test_slew_scale(self) -> None:
        sample_size = 20
        obs = dict(
            slewtime=20.0,
            visittime=36.0,
            exptime=30.0,
        )
        slew_scale = 0.1
        func = AnomalousOverheadFunc(seed=6563, slew_scale=slew_scale)
        overheads = np.array(list(func(obs=obs) for i in range(sample_size)))
        fractional_overheads = overheads / obs["slewtime"]
        cdf = partial(stats.norm.cdf, loc=0.0, scale=slew_scale)
        _, p_value = stats.kstest(fractional_overheads, cdf)
        # Depending on the random number seed, there's a
        # 0.1% chance per iteration of failing this test even
        # if everything is okay.
        assert p_value > 0.001


if __name__ == "__main__":
    unittest.main()
