import numpy as np

import analyser


class AutonomicManagerController():

    def __init__(self, type_analyser):
        self.type_analyser = type_analyser  # type: analyser.SyntheticTypeAnalyser

    def get_shared_identity_probability(self, sensor_data):
        # type: (np.ndarray) -> float

        type_probabilities = self.type_analyser.obtain_probabilities(sensor_data)  # type: np.ndarray
        shared_identity_prob = type_probabilities.item()  # type: float

        return shared_identity_prob


def main():
    manager = AutonomicManagerController(analyser.SyntheticTypeAnalyser(model_file="trained_model.h5"))
    sample_sensor_reading = np.zeros(shape=(1, 31))  # type: np.ndarray
    robot_action = manager.get_shared_identity_probability(sample_sensor_reading)
    print(robot_action)
