import logging
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
from typing import Dict

import abm_gamemodel
from analyser import SyntheticTypeAnalyser
from controller import AutonomicManagerController
from environment import NetlogoEvacuationEnvironment
from synthetic_runner import MODEL_FILE, ENCODER_FILE

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Running inference on CPU

PROJECT_DIRECTORY = "/home/cgc87/github/wdywfm-adaptive-robot/"  # type:str


def run_scenario(robot_controller, emergency_environment):
    # type: ( AutonomicManagerController,  NetlogoEvacuationEnvironment) -> str

    current_sensor_data = emergency_environment.reset()  # type: np.ndarray

    model_filename = "efg/simulation_{}_game_model.efg".format(emergency_environment.simulation_id)  # type:str

    robot_controller.measure_distance(emergency_environment)
    robot_action = robot_controller.sensor_data_callback(current_sensor_data, model_filename)  # type:str

    logging.debug("robot_action {}".format(robot_action))

    return robot_action


def main():
    parser = ArgumentParser("Get a robot action from the adaptive controller",
                            formatter_class=ArgumentDefaultsHelpFormatter)  # type: ArgumentParser
    parser.add_argument("simulation_id")

    parser.add_argument("helper_gender")
    parser.add_argument("helper_culture")
    parser.add_argument("helper_age")
    parser.add_argument("fallen_gender")
    parser.add_argument("fallen_culture")
    parser.add_argument("fallen_age")
    parser.add_argument("helper_fallen_distance")
    parser.add_argument("staff_fallen_distance")

    arguments = parser.parse_args()
    configuration = vars(arguments)  # type:Dict

    type_analyser = SyntheticTypeAnalyser(model_file=PROJECT_DIRECTORY + MODEL_FILE)  # type: SyntheticTypeAnalyser
    robot_controller = AutonomicManagerController(type_analyser,
                                                  abm_gamemodel.generate_game_model)

    emergency_environment = \
        NetlogoEvacuationEnvironment(configuration,
                                     PROJECT_DIRECTORY + ENCODER_FILE)  # type: NetlogoEvacuationEnvironment

    robot_action = run_scenario(robot_controller, emergency_environment)  # type:str
    print(robot_action)


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    # logging.basicConfig(level=logging.INFO)

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    main()
