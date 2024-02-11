import os

import numpy as np

import analyser

MODEL_PATH = 'submodules/gi_estimator/model/trained_model.h5'
PROJECT_PATH = 'stratego_generator/' + MODEL_PATH


class AutonomicManagerController:
    def __init__(self):
        curr_path = os.getcwd()
        if 'impact' in curr_path:
            model_path = os.getcwd().replace('impact2.10.7', PROJECT_PATH)
        else:
            model_path = os.getcwd().replace('resources/robot_controllers', MODEL_PATH)

        self.type_analyser: analyser.SyntheticTypeAnalyser = analyser.SyntheticTypeAnalyser(model_file=model_path)

    def get_shared_identity_probability(self, sensor_data):
        # type: (np.ndarray) -> float

        type_probabilities = self.type_analyser.obtain_probabilities(sensor_data)  # type: np.ndarray
        shared_identity_prob = type_probabilities.item()  # type: float

        return shared_identity_prob
