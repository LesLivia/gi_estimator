import multiprocessing
import time
import traceback
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyNetLogo
import seaborn as sns
import statsmodels.api as sm
from pathlib import Path
from scipy.stats import mannwhitneyu
from typing import List, Tuple, Dict, Optional

PLOT_STYLE = 'seaborn-darkgrid'

NETLOGO_PROJECT_DIRECTORY = "/home/cgc87/github/robot-assisted-evacuation/"  # type:str
MODEL_FILE = NETLOGO_PROJECT_DIRECTORY + "/impact2.10.7/v2.11.0.nlogo"  # type:str
NETLOGO_HOME = "/home/cgc87/netlogo-5.3.1-64"  # type:str
NETLOGO_VERSION = "5"  # type:str

TURTLE_PRESENT_REPORTER = "count turtles"  # type:str
EVACUATED_REPORTER = "number_passengers - count agents + 1"  # type:str
DEAD_REPORTER = "count agents with [ st_dead = 1 ]"  # type:str

SET_SIMULATION_ID_COMMAND = "set SIMULATION_ID {}"  # type:str
SEED_SIMULATION_REPORTER = "seed-simulation"

RESULTS_CSV_FILE = "data/experiment_results.csv"  # type:str
NO_SUPPORT_COLUMN = "no-support"  # type:str

ENABLE_STAFF_COMMAND = "set REQUEST_STAFF_SUPPORT TRUE"  # type:str
ENABLE_PASSENGER_COMMAND = "set REQUEST_BYSTANDER_SUPPORT TRUE"

ONLY_STAFF_SUPPORT_COLUMN = "staff-support"  # type:str
ONLY_PASSENGER_SUPPORT_COLUMN = "passenger-support"  # type:str
ADAPTIVE_SUPPORT_COLUMN = "adaptive-support"

SIMULATION_SCENARIOS = {NO_SUPPORT_COLUMN: [],
                        ONLY_STAFF_SUPPORT_COLUMN: [ENABLE_STAFF_COMMAND],
                        ONLY_PASSENGER_SUPPORT_COLUMN: [ENABLE_PASSENGER_COMMAND],
                        ADAPTIVE_SUPPORT_COLUMN: [ENABLE_PASSENGER_COMMAND,
                                                  ENABLE_STAFF_COMMAND]}  # type: Dict[str, List[str]]

SAMPLES = 100  # type:int


# Using https://www.stat.ubc.ca/~rollin/stats/ssize/n2.html
# And https://www.statology.org/pooled-standard-deviation-calculator/


# function to calculate Cohen's d for independent samples
# Inspired by: https://machinelearningmastery.com/effect-size-measures-in-python/

def cohen_d_from_metrics(mean_1, mean_2, std_dev_1, std_dev_2):
    # type: (float, float, float, float) -> float
    pooled_std_dev = np.sqrt((std_dev_1 ** 2 + std_dev_2 ** 2) / 2)
    return (mean_1 - mean_2) / pooled_std_dev


def calculate_sample_size(mean_1, mean_2, std_dev_1, std_dev_2, alpha=0.05, power=0.8):
    # type: (float, float, float, float, float, float) -> float
    analysis = sm.stats.TTestIndPower()  # type: sm.stats.TTestIndPower
    effect_size = cohen_d_from_metrics(mean_1, mean_2, std_dev_1, std_dev_2)
    result = analysis.solve_power(effect_size=effect_size,
                                  alpha=alpha,
                                  power=power,
                                  alternative="two-sided")
    return result


def run_simulation(simulation_id, post_setup_commands):
    # type: (int, List[str]) -> Optional[float]
    try:
        current_seed = netlogo_link.report(SEED_SIMULATION_REPORTER)  # type:str
        netlogo_link.command("setup")
        netlogo_link.command(SET_SIMULATION_ID_COMMAND.format(simulation_id))

        if len(post_setup_commands) > 0:
            for post_setup_command in post_setup_commands:
                netlogo_link.command(post_setup_command)
                print("id:{} seed:{} {} executed".format(simulation_id, current_seed, post_setup_command))
        else:
            print("id:{} seed:{} no post-setup commands".format(simulation_id, current_seed))

        metrics_dataframe = netlogo_link.repeat_report(
            netlogo_reporter=[TURTLE_PRESENT_REPORTER, EVACUATED_REPORTER, DEAD_REPORTER],
            reps=2000)  # type: pd.DataFrame

        evacuation_finished = metrics_dataframe[
            metrics_dataframe[TURTLE_PRESENT_REPORTER] == metrics_dataframe[DEAD_REPORTER]]

        evacuation_time = evacuation_finished.index.min()  # type: float
        print("id:{} seed:{} evacuation time {}".format(simulation_id, current_seed, evacuation_time))

        return evacuation_time
    except Exception:
        traceback.print_exc()

    return None


def initialize(gui):
    # type: (bool) -> None
    global netlogo_link

    netlogo_link = pyNetLogo.NetLogoLink(netlogo_home=NETLOGO_HOME,
                                         netlogo_version=NETLOGO_VERSION,
                                         gui=gui)  # type: pyNetLogo.NetLogoLink
    netlogo_link.load_model(MODEL_FILE)


def start_experiments(experiment_configurations):
    # type: (Dict[str, List[str]]) -> None

    start_time = time.time()  # type: float

    experiment_data = {}  # type: Dict[str, List[float]]
    for experiment_name, experiment_commands in experiment_configurations.items():
        scenario_times = run_parallel_simulations(SAMPLES,
                                                  post_setup_commands=experiment_commands)  # type:List[float]
        experiment_data[experiment_name] = scenario_times

    end_time = time.time()  # type: float
    print("Simulation finished after {} seconds".format(end_time - start_time))

    experiment_results = pd.DataFrame(experiment_data)  # type:pd.DataFrame
    experiment_results.to_csv(RESULTS_CSV_FILE)

    print("Data written to {}".format(RESULTS_CSV_FILE))


def run_simulation_with_dict(dict_parameters):
    # type: (Dict) -> float
    return run_simulation(**dict_parameters)


def run_parallel_simulations(samples, post_setup_commands, gui=False):
    # type: (int, List[str], bool) -> List[float]

    initialise_arguments = (gui,)  # type: Tuple
    simulation_parameters = [{"simulation_id": simulation_id, "post_setup_commands": post_setup_commands}
                             for simulation_id in range(samples)]  # type: List[Dict]

    results = []  # type: List[float]
    executor = Pool(initializer=initialize,
                    initargs=initialise_arguments)  # type: multiprocessing.pool.Pool

    for simulation_output in executor.map(func=run_simulation_with_dict,
                                          iterable=simulation_parameters):
        if simulation_output:
            results.append(simulation_output)

    executor.close()
    executor.join()

    return results


def get_dataframe(csv_file):
    # type: (str) -> pd.DataFrame
    results_dataframe = pd.read_csv(csv_file, index_col=[0])  # type: pd.DataFrame
    results_dataframe = results_dataframe.dropna()

    return results_dataframe


def plot_results(csv_file, samples_in_title=False):
    # type: (str) -> None
    file_description = Path(csv_file).stem  # type: str
    results_dataframe = get_dataframe(csv_file)  # type: pd.DataFrame

    print(results_dataframe.describe())

    title = ""
    if samples_in_title:
        title = "{} samples".format(len(results_dataframe))
    _ = sns.violinplot(data=results_dataframe).set_title(title)
    plt.savefig("img/" + file_description + "_violin_plot.png")
    plt.show()

    _ = sns.stripplot(data=results_dataframe, jitter=True).set_title(title)
    plt.savefig("img/" + file_description + "_strip_plot.png")
    plt.show()


def test_hypothesis(first_scenario_column, second_scenario_column, csv_file, alternative_hypothesis="two-sided"):
    # type: (str, str, str) -> None
    print("CURRENT ANALYSIS: Analysing file {}".format(csv_file))
    results_dataframe = get_dataframe(csv_file)  # type: pd.DataFrame

    first_scenario_data = results_dataframe[first_scenario_column].values  # type: List[float]
    first_scenario_mean = np.mean(first_scenario_data).item()  # type:float
    first_scenario_stddev = np.std(first_scenario_data).item()  # type:float

    second_scenario_data = results_dataframe[second_scenario_column].values  # type: List[float]
    second_scenario_mean = np.mean(second_scenario_data).item()  # type:float
    second_scenario_stddev = np.std(second_scenario_data).item()  # type:float

    print("{}-> mean = {} std = {} len={}".format(first_scenario_column, first_scenario_mean, first_scenario_stddev,
                                                  len(first_scenario_data)))
    print("{}-> mean = {} std = {} len={}".format(second_scenario_column, second_scenario_mean, second_scenario_stddev,
                                                  len(second_scenario_data)))
    print("Sample size: {}".format(
        calculate_sample_size(first_scenario_mean, second_scenario_mean, first_scenario_stddev,
                              second_scenario_stddev)))

    null_hypothesis = "MANN-WHITNEY RANK TEST: " + \
                      "The distribution of {} times is THE SAME as the distribution of {} times".format(
                          first_scenario_column, second_scenario_column)  # type: str
    alternative_hypothesis = "ALTERNATIVE HYPOTHESIS: the distribution underlying {} is stochastically {} than the " \
                             "distribution underlying {}".format(first_scenario_column, alternative_hypothesis,
                                                                 second_scenario_column)  # type:str

    threshold = 0.05  # type:float
    u, p_value = mannwhitneyu(x=first_scenario_data, y=second_scenario_data)
    print("U={} , p={}".format(u, p_value))
    if p_value > threshold:
        print("FAILS TO REJECT NULL HYPOTHESIS: {}".format(null_hypothesis))
    else:
        print("REJECT NULL HYPOTHESIS: {}".format(null_hypothesis))
        print(alternative_hypothesis)


if __name__ == "__main__":
    # start_experiments(SIMULATION_SCENARIOS)

    fall_lengths = [360, 420, 480, 540, 600]  # type: List[int]
    for fall_length in fall_lengths:
        current_file = "data/{}_fall_100_samples_experiment_results.csv".format(fall_length)  # type:str
        plt.style.use(PLOT_STYLE)
        plot_results(csv_file=current_file)

        for alternative_scenario in SIMULATION_SCENARIOS.keys():
            if alternative_scenario != ADAPTIVE_SUPPORT_COLUMN:
                test_hypothesis(first_scenario_column=ADAPTIVE_SUPPORT_COLUMN,
                                second_scenario_column=alternative_scenario,
                                alternative_hypothesis="less",
                                csv_file=current_file)
