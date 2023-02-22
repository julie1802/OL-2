import concurrent.futures
import os
from typing import Union, List
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

import util


####################################################
# START OF: Single Approval Voting Experiments #####
####################################################

class SingleExperimentPOJO:

    def __init__(self, n: int, m: int, delta_p: float, homogeneous: bool, complete: bool, pi_value: float,
                 x_o_omega_star: int = 0, p_bar_omega_star: Union[float, None] = None, maxed_alternatives: bool = False):
        # base values
        self.n = n
        self.m = m
        self.delta_p = delta_p
        # environment settings
        self.homogeneous = homogeneous
        self.complete = complete
        # OL values
        self.pi_value = pi_value
        self.x_o_omega_star = x_o_omega_star
        # fixed value for p-bar_omega-star
        self.p_bar_omega_star = p_bar_omega_star
        # value for esuring worst-case
        self.maxed_alternatives = maxed_alternatives

    def __str__(self):
        return f"SingleExperimentPOJO(n={self.n}, m={self.m}, delta_p={self.delta_p}, hom={self.homogeneous}, " \
               f"com={self.complete}, pi={self.pi_value}, X_o\u005E\u03C9\u204E={self.x_o_omega_star}, " \
               f"p_bar={self.p_bar_omega_star}, max_alt={self.maxed_alternatives})"


def experiment_function(se_pojo: SingleExperimentPOJO, num_votings: int = 200):
    # get sample distribution from parameter space
    distribution = util.create_probabilities_optimized(se_pojo.n, se_pojo.m, se_pojo.delta_p, se_pojo.homogeneous,
                                                       se_pojo.complete, se_pojo.p_bar_omega_star, se_pojo.maxed_alternatives)

    # run votings
    amount_won = 0.0
    for i in range(num_votings):
        votes = util.apply_voting(distribution, se_pojo.pi_value, se_pojo.x_o_omega_star, se_pojo.complete)
        if util.truth_won(votes):
            amount_won += 1

    return amount_won / num_votings

##################################################
# END OF: Single Approval Voting Experiments #####
##################################################


#######################################
# START OF: Experiment Parameters #####
#######################################

# number of active CPU-Workers
__num_workers_used__ = max(1, os.cpu_count() - 2)

# line for py-plots
__p_min__ = 0.5


def __set_p_min_for_pyplot__(p_min_value: float):
    global __p_min__
    __p_min__ = p_min_value


# distance between distributed sample points in parameter space for p-bar_omega-star
__data_point_distance__ = 0.05


def __set_data_point_distance__(data_point_distance: float):
    global __data_point_distance__
    __data_point_distance__ = data_point_distance


# number of sample points in parameter space
__num_samples__ = 200


def __set_num_samples__(num_samples_value: int):
    global __num_samples__
    __num_samples__ = num_samples_value


# number of votings for each sampled distribution
__num_votings__ = 1000


def __set_num_votings__(num_votings_value: int):
    global __num_votings__
    __num_votings__ = num_votings_value

#####################################
# END OF: Experiment Parameters #####
#####################################


###################################################
# START OF: Multi Approval Voting Experiments #####
###################################################

def run_experiments(n: int = 3, m: int = 2, delta_p: float = 0.4, homogeneous: bool = True, complete: bool = True,
                    pi_value: float = 0.2, x_o_omega_star: int = 0) -> float:
    # setup data points for p-bar_omega-star in the allowed range [delta p, 1]
    num_data_points = int((1.0 - delta_p) / __data_point_distance__ + 1)
    data_points = [max(0.0, min(1.0, delta_p + i * __data_point_distance__)) for i in range(num_data_points)]

    global __num_workers_used__

    # create Single Approval Voting Experiments (guaranteed Worst-Case)
    experiments_worst_case = [SingleExperimentPOJO(
        n=n,
        m=m,
        delta_p=delta_p,
        homogeneous=homogeneous,
        complete=complete,
        pi_value=pi_value,
        x_o_omega_star=x_o_omega_star,
        p_bar_omega_star=data_point,
        maxed_alternatives=True
    ) for _ in range(int(__num_samples__ / 2)) for data_point in data_points]
    # create Single Approval Voting Experiments (random samples)
    experiments_random = [SingleExperimentPOJO(
        n=n,
        m=m,
        delta_p=delta_p,
        homogeneous=homogeneous,
        complete=complete,
        pi_value=pi_value,
        x_o_omega_star=x_o_omega_star,
        p_bar_omega_star=data_point,
        maxed_alternatives=False
    ) for _ in range(int(__num_samples__ / 2)) for data_point in data_points]
    
    experiments = experiments_worst_case + experiments_random

    # setup py plot
    plt.plot(data_points, [__p_min__] * len(data_points), label="_reference", color="red")
    plt.title(
        ('Plot for win rates in a{com}complete{hom}homogeneous\n\u0394p={delta_p}' +
         ' environment with n={n}, m={m}, \u03C0={pi} and X_o\u005E\u03C9\u204E={x_o_omega_star}').format(
            com=complete and ' ' or ' non-', hom=homogeneous and ' ' or ' non-',
            delta_p=delta_p, n=n, m=m, pi=pi_value, x_o_omega_star=x_o_omega_star
        ))
    plt.xlabel('Value for p\u0305\u005E\u03C9\u204E')
    plt.ylabel('Minimal win rate of \u03C9\u204E')

    # sync with progress bar
    total_number_of_tasks = len(experiments)

    # setup storage for resulting win rates on each data point
    data_point_values = {x: 1.0 for x in data_points}

    # calculate experiments with multiprocessing
    exceptions = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=__num_workers_used__) as executor:
        with tqdm(total=total_number_of_tasks) as progress:
            work_load = {executor.submit(experiment_function,
                                         experiments[i],
                                         __num_votings__): i for i in range(total_number_of_tasks)}
            for future in work_load:
                future.add_done_callback(lambda p: progress.update())
            for future in concurrent.futures.as_completed(work_load):
                index = work_load[future]
                try:
                    result_probability = future.result()
                except Exception as exc:
                    # catch thrown exceptions and store for printing later on
                    exceptions.append('%d %s generated an exception: %s' % (index, experiments[index], exc))
                else:
                    se_pojo = experiments[index]
                    data_point = se_pojo.p_bar_omega_star
                    if data_point_values[data_point] > result_probability:
                        # only save Worst-Case win rates
                        data_point_values[data_point] = result_probability
    # print caught exceptions
    for exception in exceptions:
        print(exception)

    y_values = list(data_point_values.values())
    # create pyplot
    plt.plot(data_points, y_values, color="blue", label="Worst-Case win rates")

    # show and save plot
    # plt.legend(loc=2)  # legend position set to upper left corner
    plt.ylim(top=1.0)
    plt.ylim(bottom=0.0)
    plt.xlim(right=1.0)
    # plt.show()
    if not os.path.isdir('plots'):
        os.mkdir('plots')
    plt.savefig('plots/plot_n{n}_m{m}_dp{delta_p}_{com}_{hom}_{pi}-{x_o_omega_star}.png'.format(
        com=complete and 'complete' or 'non-complete',
        hom=homogeneous and 'homogeneous' or 'non-homogeneous',
        delta_p=delta_p, n=n, m=m, pi=pi_value, x_o_omega_star=x_o_omega_star
    ))
    plt.clf()

    return np.min(y_values)

#################################################
# END OF: Multi Approval Voting Experiments #####
#################################################
