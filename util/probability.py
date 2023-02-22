import math
import random
from typing import Union

import numpy as np


###########################################
# START OF: Approval Voting Functions #####
###########################################

def create_probabilities_optimized(n: int, m: int, delta_p: float, homogeneous: bool = True, complete: bool = True,
                                   enforced: Union[float, None] = None, maxed_alternatives: bool = False, uniform_dist: bool = True):
    # create values for each p bar
    p_bar_values = np.zeros(m)
    if delta_p > 0.9999999999:
        p_bar_values[0] = 1.0

    elif not complete:
        # dist is not complete

        # first value from uniform/log distribution over [delta_p, 1]
        p_bar_values[0] = uniform_dist and __my_random__(delta_p, 1.0) or __my_log_random__(delta_p, 1.0)
        if enforced is not None:
            p_bar_values[0] = enforced

        # maximum for other choices from distribution over [0, p-bar_omega-star - delta-p]
        max_value = max(0.0, p_bar_values[0] - delta_p)
        for j in range(1, m):
            if maxed_alternatives:
                p_bar_values[j] = max_value
            else:
                p_bar_values[j] = __my_random__(0, max_value)

    else:
        # dist is complete
        avg_space = (1.0 - delta_p) / m
        avg_reducer = avg_space * (m - 1)

        # first value from uniform/log distribution over [delta_p, 1]
        p_bar_values[0] = uniform_dist and __my_random__(1.0 - avg_reducer, 1.0) or __my_log_random__(1.0 - avg_reducer,
                                                                                                      1.0)
        if enforced is not None and enforced >= 1.0 - avg_reducer:
            p_bar_values[0] = enforced

        remaining = 1.0 - p_bar_values[0]
        upper_bound = p_bar_values[0] - delta_p
        for j in range(1, m - 1):
            if maxed_alternatives:
                value = min(upper_bound, remaining)
            else:
                min_probs = max(0.0, remaining - (m - j - 1) * avg_space)
                max_probs = min(avg_space, remaining)
                value = __my_random__(min_probs, max_probs)
            p_bar_values[j] = value
            remaining -= value
        p_bar_values[m - 1] = remaining

    # create actual values for each agent
    p_values = np.zeros([n, m])

    # copy p_bar_values for each agent
    for i in range(n):
        p_values[i] = p_bar_values

    if not homogeneous:
        p_bar_values *= n
        for i in range(n - 1):
            for j in range(m):
                min_probs = max(0, p_bar_values[j] - (n - i - 1))
                max_probs = min(1, p_bar_values[j])
                p_values[i][j] = __my_random__(min_probs, max_probs)
                p_bar_values[j] -= p_values[i][j]
        # assign the remaining probabilities to the last agent
        p_values[n - 1] = p_bar_values

    return p_values


def apply_voting(p_values, pi_value: float, x_o_omega_star: int = 0, complete: bool = True):
    n = len(p_values)
    m = len(p_values[0])

    votes = np.zeros([n, m])

    if complete:
        # complete voting
        x_o_omega_2 = x_o_omega_star == 0 and 1 or 0
        ol_votes = np.array([x_o_omega_star, x_o_omega_2] + [0] * (m - 2))

        for i in range(n):
            if __my_random__(0.0, 1.0) <= pi_value:
                # agent follows OL
                votes[i] = ol_votes
            else:
                # agent does not follow OL
                cumulative = np.cumsum(p_values[i])
                prob = __my_random__(0, cumulative[m - 1])
                vote_index = [x for x in range(m) if cumulative[x] >= prob][0]
                votes[i][vote_index] = 1
    else:
        # non-complete voting
        ol_votes = np.array([x_o_omega_star] + [1] * (m - 1))

        for i in range(n):
            if __my_random__(0.0, 1.0) <= pi_value:
                # agent follows OL
                votes[i] = ol_votes
            else:
                # agent does not follow OL
                for j in range(m):
                    if random.uniform(0, 1) <= p_values[i][j]:
                        votes[i][j] = 1

    return votes


def truth_won(votes) -> bool:
    sums = np.sum(votes, axis=0)
    return sums[0] > sums[np.argmax(sums[1:len(sums)]) + 1]


#########################################
# END OF: Approval Voting Functions #####
#########################################


################################
# START OF: Help Functions #####
################################

def __my_random__(a, b) -> float:
    if b <= 0.0:
        return 0.0
    if a == b:
        return a
    candidate = random.uniform(a, b)

    return candidate


def __my_log_random__(a, b) -> float:
    if b <= 0.0:
        return 0.0
    if a == b:
        return a
    base = __my_log_func__()
    candidate = base * (b - a) + a
    return candidate


def __my_log_func__() -> float:
    # return 2**((-math.log(random.random() + 1)) / math.log(2) + 1) - 1
    return math.exp(-2.5 * random.random())

##############################
# END OF: Help Functions #####
##############################
