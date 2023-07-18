import numpy as np

import rubin_sim.scheduler as fs
import rubin_sim.scheduler.Training as optional
from rubin_sim.speedObservatory import Speed_observatory


class BlackTraining:
    def __init__(self, preferences=[1], gray_train=False, custom_period=1):
        self.pref = preferences

        self.survey_length = 0.2  # days
        self.surveys = []
        survey_filters = ["r"]
        for f in survey_filters:
            self.bfs = []
            self.bfs.append(fs.Slewtime_basis_function_cost(filtername=f))
            self.bfs.append(fs.Visit_repeat_basis_function_cost(filtername=f, survey_filters=survey_filters))
            self.bfs.append(fs.Target_map_basis_function_cost(filtername=f, survey_filters=survey_filters))
            self.bfs.append(fs.Normalized_alt_basis_function_cost(filtername=f))
            self.bfs.append(fs.Hour_angle_basis_function_cost())
            self.bfs.append(fs.Depth_percentile_basis_function_cost())
            weights = np.array([5, 2, 1, 1, 2, 1])
            self.surveys.append(
                fs.Simple_greedy_survey_fields_cost(self.bfs, weights, filtername=f, block_size=10)
            )

    def de_opt(
        self,
        n_p,
        F,
        cr,
        max_iter,
        D,
        domain,
        load_candidate_solution,
        gray_trianing=False,
    ):
        self.D = D
        self.domain = domain
        self.optimizer = optional.DE_optimizer(
            self,
            n_p,
            F,
            cr,
            max_iter,
            gray_training=gray_trianing,
            load_candidate_solution=load_candidate_solution,
        )

    def target(self, x):
        x[0] = 5  # reduce redundant solutions
        for survey in self.surveys:
            survey.basis_weights = x
        scheduler = fs.Core_scheduler_cost(self.surveys)
        observatory = Speed_observatory()
        observatory, scheduler, observations = fs.sim_runner(
            observatory, scheduler, survey_length=self.survey_length
        )
        return -1 * fs.simple_performance_measure(observations, self.pref)

    def refined_individual(self):
        return np.zeros(self.D)


n_p = 50  # number of candidate solutions that are supposed to explore the space of solution in each iteration, rule of thumb: ~10*D
F = 0.8  # algorithm meta parameter (mutation factor that determines the amount of change for the derivation of candidate solutions of the next iteration)
cr = 0.8  # algorithm meta parameter (crossover rate that determines the rate of mixing of previous candidates to make new candidates)
max_iter = 100  # maximum number of iterations. maximum number of function evaluations = n_p * max_iter,
domain = np.array(
    [[0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10]]
)  # Final solution would lie in this domain
D = 6  # weights dimension


train = BlackTraining()
train.de_opt(n_p, F, cr, max_iter, D, domain, load_candidate_solution=False)
