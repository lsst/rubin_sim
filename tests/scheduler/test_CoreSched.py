import numpy as np
import unittest
from rubin_sim.scheduler.schedulers import Core_scheduler
import rubin_sim.scheduler.basis_functions as basis_functions
import rubin_sim.scheduler.surveys as surveys
from rubin_sim.scheduler.utils import standard_goals
from rubin_sim.scheduler.modelObservatory import Model_observatory


class TestCoreSched(unittest.TestCase):

    def testsched(self):
        target_map = standard_goals()['r']

        bfs = []
        bfs.append(basis_functions.M5_diff_basis_function())
        bfs.append(basis_functions.Target_map_basis_function(target_map=target_map))
        weights = np.array([1., 1])
        survey = surveys.Greedy_survey(bfs, weights)
        scheduler = Core_scheduler([survey])

        observatory = Model_observatory(mjd_start=59853.5)

        # Check that we can update conditions
        scheduler.update_conditions(observatory.return_conditions())

        # Check that we can get an observation out
        obs = scheduler.request_observation()
        assert(obs is not None)

        # Check that we can flush the Queue
        scheduler.flush_queue()
        assert(len(scheduler.queue) == 0)

        # Check that we can add an observation
        scheduler.add_observation(obs)


if __name__ == "__main__":
    unittest.main()
