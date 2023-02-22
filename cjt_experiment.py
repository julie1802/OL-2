import sys
import os
import getopt
import json
from typing import Union, Dict, List

from refined_experiment import (run_experiments, __set_num_votings__, __set_num_samples__, __set_data_point_distance__)
from util import (derived_calculation)

help_string = '\nUsage:  cjt_experiment.py [OPTIONS]' \
              '\n' \
              '\nRun experiment for a set of world constraints' \
              '\n' \
              '\nOptions:' \
              '\n  -d, --delta-p float        Set the \u0394p value' \
              '\n  -h, --help                 Display help text' \
              '\n  -m, --m-options int        The number of options for the agents' \
              '\n  -o, --ol-competence float  The competence of the Opinion Leader (OL)' \
              '\n  -p, --p-min float          Minimum success threshold P_min' \
              '\n      --pi float             Probability of agents to follow the OL' \
              '\n  -s, --step-maximum int     Maximum number of steps while calculating win rates' \
              '\n                             for new numbers of agents (default: 1000)'


################################
# START OF: Help Functions #####
################################

def __float_to_str__(value: float, rounding: int = 2) -> str:
    return str(round(value, rounding))


def __str_to_float__(value: str) -> float:
    return float(value)


def __int_to_str__(value: int) -> str:
    return str(value)


def __str_to_int__(value: str) -> int:
    return int(value)


##############################
# END OF: Help Functions #####
##############################


############################################################
# START OF: Hyper-Cube for Worst-Case win-rate storage #####
############################################################

class DataCube(object):

    def __init__(self, cube: Dict[str, Dict[str, Dict[str, Dict[str, float]]]]):
        # cube[delta p][m][n] = p_success
        self.cube = cube

    def get_row(self, delta_p: str, m: int, pi_value: str) -> Union[Dict[str, float], None]:
        m_str = __int_to_str__(m)
        if delta_p not in self.cube.keys():
            return None
        if m_str not in self.cube[delta_p].keys():
            return None
        if pi_value not in self.cube[delta_p][m_str].keys():
            return None
        return self.cube[delta_p][m_str][pi_value]

    def set_value(self, delta_p: str, m: int, pi_value: str, n: int, p_success: float):
        m_str = __int_to_str__(m)
        n_str = __int_to_str__(n)
        if delta_p not in self.cube.keys():
            self.cube[delta_p] = {}
        if m_str not in self.cube[delta_p].keys():
            self.cube[delta_p][m_str] = {}
        if pi_value not in self.cube[delta_p][m_str].keys():
            self.cube[delta_p][m_str][pi_value] = {}
        self.cube[delta_p][m_str][pi_value][n_str] = p_success


def __load_data_cube__(filename: str) -> DataCube:
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            json_object = json.load(f)
        return DataCube(json_object['cube'])
    else:
        return DataCube(cube={})


def load_data_cubes(x_o_omega_star: int = 0) -> List[DataCube]:
    if x_o_omega_star == 0:
        # OL does not vote for omega_star
        return [__load_data_cube__('cubes/data_cube_o0_h0_c0.json'),  # non-homogeneous, non-complete
            __load_data_cube__('cubes/data_cube_o0_h0_c1.json'),  # non-homogeneous, complete
            __load_data_cube__('cubes/data_cube_o0_h1_c0.json'),  # homogeneous, non-complete
            __load_data_cube__('cubes/data_cube_o0_h1_c1.json')  # homogeneous, complete
        ]
    else:
        # OL votes for omega_star
        return [__load_data_cube__('cubes/data_cube_o1_h0_c0.json'),  # non-homogeneous, non-complete
            __load_data_cube__('cubes/data_cube_o1_h0_c1.json'),  # non-homogeneous, complete
            __load_data_cube__('cubes/data_cube_o1_h1_c0.json'),  # homogeneous, non-complete
            __load_data_cube__('cubes/data_cube_o1_h1_c1.json')  # homogeneous, complete
        ]


def __save_data_cube__(filename: str, data_cube):
    json_object = {'cube': data_cube.cube}
    with open(filename, 'w') as f:
        json.dump(json_object, f)


def save_data_cubes(data_cubes: List[DataCube], x_o_omega_star: int = 0):
    if not os.path.isdir('cubes'):
        os.mkdir('cubes')
    if x_o_omega_star == 0:
        __save_data_cube__('cubes/data_cube_o0_h0_c0.json', data_cubes[0])
        __save_data_cube__('cubes/data_cube_o0_h0_c1.json', data_cubes[1])
        __save_data_cube__('cubes/data_cube_o0_h1_c0.json', data_cubes[2])
        __save_data_cube__('cubes/data_cube_o0_h1_c1.json', data_cubes[3])
    else:
        __save_data_cube__('cubes/data_cube_o1_h0_c0.json', data_cubes[0])
        __save_data_cube__('cubes/data_cube_o1_h0_c1.json', data_cubes[1])
        __save_data_cube__('cubes/data_cube_o1_h1_c0.json', data_cubes[2])
        __save_data_cube__('cubes/data_cube_o1_h1_c1.json', data_cubes[3])


def get_values_above_p_min(dict_list: List[Dict[str, float]], p_min: float) -> List[int]:
    return [__str_to_int__(k) for k in dict_list[0].keys() if
            dict_list[0][k] >= p_min and dict_list[1][k] >= p_min and dict_list[2][k] >= p_min and dict_list[3][
                k] >= p_min]


def previous_n_value(dictionary: Dict[str, float], n: int) -> int:
    n_list = [__str_to_int__(k) for k in dictionary.keys() if __str_to_int__(k) < n]
    n_list.append(0)  # safety addition for empty list
    return max(n_list)


##########################################################
# END OF: Hyper-Cube for Worst-Case win-rate storage #####
##########################################################


######################################################
# START OF: Actual Main Function for Experiments #####
######################################################

def main(argv) -> Union[int, None]:
    m = None
    delta_p = None
    ol_competence = 0
    x_o_omega_star = None
    p_min = None
    pi_str = None

    step_maximum = 1000

    # load command line arguments
    try:
        opts, args = getopt.gnu_getopt(argv, "d:m:o:p:s:h",
                                       ["delta-p=", "m-options=", "ol-competence", "p-min=", "pi=", "step-maximum",
                                        "help"])
    except getopt.GetoptError:
        print(help_string)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(help_string)
            sys.exit()
        elif opt in ('-d', '--delta-p'):
            try:
                delta_p_value = __str_to_float__(arg)
                if not (0.0 <= delta_p_value <= 1.0):
                    print("Pi has to be between 0.0 and 1.0")
                    sys.exit(2)
                delta_p = __float_to_str__(delta_p_value)
            except ValueError:
                print(f"Delta p is expected to be a float, instead got '{arg}'")
                sys.exit(2)
        elif opt in ('-m', '--m-options'):
            try:
                m = int(arg)
                if m < 2:
                    print("m is expected to be greater than or equal to 2")
                    sys.exit(2)
            except ValueError:
                print(f"m is expected to be an integer, instead got '{arg}'")
                sys.exit(2)
        elif opt in ('-o', '--ol-competence'):
            try:
                ol_competence = __str_to_float__(arg)
                if not (0.0 <= ol_competence <= 1.0):
                    print("OL competence has to be between 0.0 and 1.0")
                    sys.exit(2)
                x_o_omega_star = ol_competence == 1.0 and 1 or 0
            except ValueError:
                print(f"OL competence is expected to be a float, instead got '{arg}'")
                sys.exit(2)
        elif opt in ('-p', '--p-min'):
            try:
                p_min = float(arg)
                if not (0.0 <= p_min <= 1.0):
                    print("P_min has to be between 0.0 and 1.0")
                    sys.exit(2)
            except ValueError:
                print(f"P_min is expected to be a float, instead got '{arg}'")
                sys.exit(2)
        elif opt in '--pi':
            try:
                pi_value = __str_to_float__(arg)
                pi_str = __float_to_str__(pi_value)
                if not (0.0 <= pi_value <= 1.0):
                    print("Pi has to be between 0.0 and 1.0")
                    sys.exit(2)
            except ValueError:
                print(f"Pi is expected to be a float, instead got '{arg}'")
                sys.exit(2)
        elif opt in ('-s', '--step-maximum'):
            try:
                step_maximum = int(arg)
                if step_maximum < 1:
                    print("Step maximum needs to be at least 1")
                    sys.exit(2)
            except ValueError:
                print(f"Step maximum is expected to be an integer, instead got '{arg}'")
                sys.exit(2)

    if m is None or delta_p is None or x_o_omega_star is None or p_min is None or pi_str is None:
        print('The values for m, delta_p, P_min, OL-competence and Pi are mandatory!')
        sys.exit(2)

    # print expected n value from formula in the paper
    print('The expected n_min is: {formula:.3f}\n'.format(
        formula=derived_calculation(delta_p=__str_to_float__(delta_p), m=m, p_min=p_min,
                                    pi_value=__str_to_float__(pi_str), ol_competence=ol_competence)
    ))

    # load data cubes
    data_cubes = load_data_cubes(x_o_omega_star=x_o_omega_star)

    start_n = 1
    target_n = -1

    # test if row exists
    row_h0_c0 = data_cubes[0].get_row(delta_p=delta_p, m=m, pi_value=pi_str)
    if row_h0_c0 is None:
        # no values registered for this combination
        start_n = 1
    else:
        row_h0_c1 = data_cubes[1].get_row(delta_p=delta_p, m=m, pi_value=pi_str)
        row_h1_c0 = data_cubes[2].get_row(delta_p=delta_p, m=m, pi_value=pi_str)
        row_h1_c1 = data_cubes[3].get_row(delta_p=delta_p, m=m, pi_value=pi_str)

        filtered_keys = get_values_above_p_min([row_h0_c0, row_h0_c1, row_h1_c0, row_h1_c1], p_min)
        if len(filtered_keys) == 0:
            # no registered n values above p_min
            start_n = max([__str_to_int__(x) for x in row_h0_c0.keys()]) + 1
        else:
            target_n = min(filtered_keys)
            previous_n = previous_n_value(row_h0_c0, target_n)
            if previous_n + 1 == target_n:
                # checked perimeter of p_min
                return target_n
            else:
                start_n = previous_n + 1

    i = 0
    while i < step_maximum and (target_n == -1 or start_n + i < target_n):
        delta_p_value = __str_to_float__(delta_p)
        pi_value = __str_to_float__(pi_str)
        n = start_n + i
        print(f"Searching n={n}")
        print("\nNon-homogeneous | Non-complete")
        value_h0_c0 = run_experiments(n=n, m=m, delta_p=delta_p_value, homogeneous=False, complete=False,
            pi_value=pi_value, x_o_omega_star=x_o_omega_star)
        print("\nNon-homogeneous |     Complete")
        value_h0_c1 = run_experiments(n=n, m=m, delta_p=delta_p_value, homogeneous=False, complete=True,
            pi_value=pi_value, x_o_omega_star=x_o_omega_star)
        print("\n    Homogeneous | Non-complete")
        value_h1_c0 = run_experiments(n=n, m=m, delta_p=delta_p_value, homogeneous=True, complete=False,
            pi_value=pi_value, x_o_omega_star=x_o_omega_star)
        print("\n    Homogeneous |     Complete")
        value_h1_c1 = run_experiments(n=n, m=m, delta_p=delta_p_value, homogeneous=True, complete=True,
            pi_value=pi_value, x_o_omega_star=x_o_omega_star)
        worst_case_value = min(value_h0_c0, value_h0_c1, value_h1_c0, value_h1_c1)

        data_cubes[0].set_value(delta_p=delta_p, m=m, pi_value=pi_str, n=n, p_success=value_h0_c0)
        data_cubes[1].set_value(delta_p=delta_p, m=m, pi_value=pi_str, n=n, p_success=value_h0_c1)
        data_cubes[2].set_value(delta_p=delta_p, m=m, pi_value=pi_str, n=n, p_success=value_h1_c0)
        data_cubes[3].set_value(delta_p=delta_p, m=m, pi_value=pi_str, n=n, p_success=value_h1_c1)

        # save data cubes after each calculated n
        save_data_cubes(data_cubes)

        print(f'\nWorst-Case success probability for n={n} is: {worst_case_value:.3f}\n')

        if worst_case_value >= p_min:
            target_n = n
            break

        # increase step counter
        i += 1

    if target_n == -1 or start_n + step_maximum < target_n:
        print('Ran out of steps. Printing closest n value.')

    return target_n


####################################################
# END OF: Actual Main Function for Experiments #####
####################################################


if __name__ == '__main__':
    __set_data_point_distance__(0.05)
    __set_num_samples__(200)
    __set_num_votings__(1000)
    n_value = main(sys.argv[1:])
    print(f'The calculated n_exp: {n_value}')
