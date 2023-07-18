__author__ = "Elahe"

import numpy as np


class DeOptimizer:
    def __init__(
        self,
        evaluator,
        population_size,
        f,
        cr,
        max_iter,
        strategy=9,
        vtr=-1e99,
        eps=0,
        show_progress=1,
        monitor_cycle=np.inf,
        gray_training=False,
        load_candidate_solution=False,
    ):
        self.show_progress = show_progress
        self.evaluator = evaluator
        self.population_size = population_size
        self.f = f
        self.cr = cr
        self.max_iter = max_iter
        self.strategy = strategy

        if strategy > 5:
            self.st = strategy - 5
        else:
            self.st = strategy

        self.monitor_cycle = monitor_cycle
        self.D = evaluator.D
        self.eps = eps
        self.vtr = vtr

        # if population slightly changes while optimization or not
        self.gray_training = gray_training
        self.load_candidate_solution = load_candidate_solution

        self.optimize()

    def optimize(self):
        # initialize optimization
        self.initialize_optimization()
        # initialize the population
        if self.load_candidate_solution:
            self.load_generation()
        if not self.load_candidate_solution:
            self.make_random_population()

        # score initial population
        # self.score_population()
        # update progress parameters
        self.init_progress()

        # iterations
        while not self.terminate():
            self.evolve()
            self.update_progress()
            self.print_status()
            self.monitor()
            if self.gray_training:
                self.swap_population()
                self.update_progress()

    def initialize_optimization(self):
        # Initial pop features
        self.population = np.zeros((self.population_size, self.D))
        self.scores = 1e99 * np.ones(self.population_size)
        self.count = 0
        self.nfeval = 0
        self.after_score_pop = np.zeros((self.population_size, self.D))

    def make_random_population(self):
        for indiv in range(self.population_size):
            self.population[indiv, :] = self.make_random_individual()

    def make_random_individual(self):
        delta = self.evaluator.domain[:, 1] - self.evaluator.domain[:, 0]
        offset = self.evaluator.domain[:, 0]
        ind = np.multiply(np.random.rand(1, self.D), delta) + offset
        return ind

    def score_population(self):
        for indiv in range(self.population_size):
            self.scores[indiv] = self.evaluator.target(self.population[indiv, :])
            self.nfeval += 1
            """change individuals for the refined values by gray training """
            self.after_score_pop[indiv, :] = self.evaluator.refined_individual()
            self.print_ind(
                indiv,
                self.scores[indiv],
                self.population[indiv, :],
                self.after_score_pop[indiv, :],
            )

    def init_progress(self):
        self.best_index = np.argmin(self.scores)
        self.best_val = self.scores[self.best_index]
        self.best_ind = self.population[self.best_index, :]
        self.mean_val = np.mean(self.scores)

    def update_progress(self):
        # bests
        temp_best_index = np.argmin(self.scores)
        temp_best_score = self.scores[temp_best_index]
        # preserving best ever individual
        if temp_best_score < self.best_val:
            print(temp_best_score, self.best_val)
            self.best_index = temp_best_index
            self.best_val = temp_best_score
            self.best_ind = self.population[self.best_index, :]
            new_mean_val = np.mean(self.scores)
            self.obj_prog = self.mean_val - new_mean_val
            self.mean_val = new_mean_val
        # algorithm parameters
        self.count += 1

    def evolve(self):
        # Candidate
        self.ui = self.cal_trials()
        self.after_score_ui = np.zeros(np.shape(self.ui))
        # Selection
        for trial_indx in range(self.population_size):
            trial_score = self.score_trial(trial_indx)

            if trial_score < self.scores[trial_indx]:
                self.population[trial_indx, :] = self.ui[trial_indx, :]
                self.scores[trial_indx] = trial_score
                self.after_score_pop[trial_indx, :] = self.after_score_ui[trial_indx, :]

            self.print_ind(
                trial_indx,
                self.scores[trial_indx],
                self.population[trial_indx, :],
                self.after_score_pop[trial_indx, :],
            )
        self.save_last_generation()  # most recent generation

    def score_trial(self, trial_indx):
        tempval = self.evaluator.target(self.ui[trial_indx, :])
        self.nfeval += 1
        if self.gray_training:
            self.after_score_ui[trial_indx, :] = self.evaluator.refined_individual()
        return tempval

    def monitor(self):
        if self.count != 0 and self.count % self.monitor_cycle == 0:
            self.monitor_score = self.best_val
            self.monitor_indiv = self.best_ind
            self.monitor_mean = self.mean_val

    def swap_population(self):
        # regularize refined population before swap (gene-wise swap)
        for gene in range(self.D):
            self_val = self.after_score_pop[:, gene]
            lower_alt_val = self.evaluator.domain[gene, 0]
            upper_alt_val = self.evaluator.domain[gene, 1]
            lower_bound = self_val >= lower_alt_val
            upper_bound = self_val <= upper_alt_val
            self_val = np.where(lower_bound, self_val, lower_alt_val)
            self_val = np.where(upper_bound, self_val, upper_alt_val)
            # swap individuals:
            self.population[:, gene] = self_val
        # modify scores
        self.score_population()

    def cal_trials(self):
        popold = self.population

        rot = np.arange(0, self.population_size)
        rotd = np.arange(0, self.D)
        ind = np.random.permutation(4)

        a1 = np.random.permutation(self.population_size)
        rt = (rot + ind[0]) % self.population_size
        a2 = a1[rt]
        rt = (rot + ind[1]) % self.population_size
        a3 = a2[rt]
        rt = (rot + ind[2]) % self.population_size
        a4 = a3[rt]
        rt = (rot + ind[3]) % self.population_size
        a5 = a4[rt]

        pm1 = self.population[a1, :]
        pm2 = self.population[a2, :]
        pm3 = self.population[a3, :]
        pm4 = self.population[a4, :]
        pm5 = self.population[a5, :]

        pop_of_best_ind = np.zeros((self.population_size, self.D))
        for i in range(0, self.population_size):
            pop_of_best_ind[i] = self.best_ind

        cr_decision = np.random.rand(self.population_size, self.D) < self.cr

        if self.strategy > 5:
            cr_decision = np.sort(np.transpose(cr_decision))
            for i in range(0, self.population_size):
                n = np.floor(np.random.rand(1) * self.D)
                if n > 0:
                    rtd = (rotd + n) % self.D
                    rtd = rtd.astype(int)
                    cr_decision[:, i] = cr_decision[rtd, i]
            cr_decision = np.transpose(cr_decision)
        mpo = cr_decision < 0.5
        ui = 0
        if self.st == 1:
            dif = self.f * (pm1 - pm2)
            ui = pop_of_best_ind + dif
            ui = self.regularize_candidate(ui, pop_of_best_ind)
            ui = np.multiply(popold, mpo) + np.multiply(ui, cr_decision)
        elif self.st == 2:
            dif = self.f * (pm1 - pm2)
            ui = pm3 + dif
            ui = self.regularize_candidate(ui, pop_of_best_ind)
            ui = np.multiply(popold, mpo) + np.multiply(ui, cr_decision)
        elif self.st == 3:
            dif = self.f * (pop_of_best_ind - popold + pm1 - pm2)
            ui = popold + dif
            ui = self.regularize_candidate(ui, pop_of_best_ind)
            ui = np.multiply(popold, mpo) + np.multiply(ui, cr_decision)
        elif self.st == 4:
            dif = self.f * (pm1 - pm2 + pm3 - pm4)
            ui = pop_of_best_ind + dif
            ui = self.regularize_candidate(ui, pop_of_best_ind)
            ui = np.multiply(popold, mpo) + np.multiply(ui, cr_decision)
        elif self.st == 5:
            dif = self.f * (pm1 - pm2 + pm3 - pm4)
            ui = pm5 + dif
            ui = self.regularize_candidate(ui, pop_of_best_ind)
            ui = np.multiply(popold, mpo) + np.multiply(ui, cr_decision)
        return ui

    def regularize_candidate(self, candidate, alternative):
        lower_bound = self.evaluator.domain[:, 0]
        upper_bound = self.evaluator.domain[:, 1]
        candidate = np.where(candidate >= lower_bound, candidate, alternative)
        regularized_cand = np.where(candidate <= upper_bound, candidate, alternative)
        return regularized_cand

    def terminate(self):
        # Termination 1 : By maxiter
        if self.count >= self.max_iter:
            self.termination = 1
            return True
        """
    #Termination 2 : By Value to reach
        if self.best_val < self.vtr :
            self.termination = 2
            return True
    #Termination 3 : By monitor cycle change in the objective
        if  self.monitor_obj_change < self.eps:
            self.termination = 3
            return True
            """
        return False

    def save_last_generation(self):
        np.save("last_gen_pop", self.population)
        np.save("last_gen_scr", self.scores)

    def load_generation(self):
        try:
            temp_population = np.load("last_gen_pop.npy")
            if np.shape(temp_population)[0] != self.population_size:
                print(
                    "Previous generation is not of the same size of new setting, DE starts with a random initialization"
                )
                self.load_candidate_solution = False
            elif np.shape(temp_population)[1] != self.D:
                print(
                    "Previous solution is not of the same size of new solution, DE starts with a random initialization"
                )
                self.load_candidate_solution = False
            else:
                self.population = temp_population
                self.scores = np.load("last_gen_scr.npy")
                print("Warm start: DE starts with a previously evolved population")

        except:
            print("No previous generation is available, DE starts with a random initialization")
            self.load_candidate_solution = False

    def print_ind(self, ind_index, score, indiv, refined_indiv):
        # print("{}: Objective: {},\nCandidate: {},\nRefined candidate: {}".format(ind_index +1, score, indiv, refined_indiv))
        print("{}: Objective: {},\nCandidate: {}".format(ind_index + 1, -1.0 * score, indiv))
        with open("Output/Output.txt", "a") as text_file:
            text_file.write("{}: Objective: {},\nCandidate: {}\n".format(ind_index + 1, -1.0 * score, indiv))
            text_file.close()

    def print_status(self):
        print("********************************************************************************************")
        print(
            "iter {}:best performance: {}, best candidate no.: {}\nbest candidate: {}".format(
                self.count, self.best_val * -1, self.best_index + 1, self.best_ind
            )
        )
        print("")
        with open("Output/Output.txt", "a") as text_file:
            text_file.write(
                "********************************************************************************************"
            )
            text_file.write(
                "iter {}:best performance: {}, best candidate no.: {}\nbest candidate: {}\n".format(
                    self.count, self.best_val * -1, self.best_index + 1, self.best_ind
                )
            )
            text_file.close()

    def final_print(self):
        print("")
        print("* Problem Specifications")
        print("Date           : {}".format(self.evaluator.scheduler.Date))
        print("Preferences    : {}".format(self.evaluator.pref))
        print("")
        print("* DE parameters")
        print("F          : {}".format(self.f))
        print("Cr         : {}".format(self.cr))
        print("Pop Size   : {}".format(self.population_size))
        print("No. of eval: {}".format(self.nfeval))
        print("No. of Iter: {}".format(self.count))
        print("Termination: {}".format(self.termination))
        print("eps        : {}".format(self.eps))
        print("vtr        : {}".format(self.vtr))
        print("")
        with open("Output/Output.txt", "a") as text_file:
            text_file.write("\n* Problem Specifications\n")
            text_file.close()
            text_file.write("Date           : {}\n".format(self.evaluator.scheduler.Date))
            text_file.close()
            text_file.write("Preferences    : {}\n".format(self.evaluator.pref))
            text_file.close()
            text_file.write("\n\n")
            text_file.close()
            text_file.write("* DE parameters\n")
            text_file.write("F          : {}\n".format(self.f))
            text_file.close()
            text_file.write("Cr         : {}\n".format(self.cr))
            text_file.close()
            text_file.write("Pop Size   : {}\n".format(self.population_size))
            text_file.close()
            text_file.write("No. of eval: {}\n".format(self.nfeval))
            text_file.close()
            text_file.write("No. of Iter: {}\n".format(self.count))
            text_file.close()
            text_file.write("Termination: {}\n".format(self.termination))
            text_file.close()
            text_file.write("eps        : {}\n".format(self.eps))
            text_file.close()
            text_file.write("vtr        : {}\n".format(self.vtr))
            text_file.close()
