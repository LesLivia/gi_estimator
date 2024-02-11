import os
import pickle

from sklearn.preprocessing import OneHotEncoder

import analyser

MODEL_PATH = 'submodules/gi_estimator/model/trained_model.h5'
ENCODER_PATH = 'submodules/gi_estimator/model/encoder.pickle'
PROJECT_PATH = 'stratego_generator/'


class AutonomicManagerController:
    def __init__(self):
        curr_path = os.getcwd()
        if 'impact' in curr_path:
            model_path = os.getcwd().replace('impact2.10.7', PROJECT_PATH + MODEL_PATH)
            encoder_path = os.getcwd().replace('impact2.10.7', PROJECT_PATH + ENCODER_PATH)
        else:
            model_path = os.getcwd().replace('resources/robot_controllers', MODEL_PATH)
            encoder_path = os.getcwd().replace('resources/robot_controllers', ENCODER_PATH)

        self.type_analyser: analyser.SyntheticTypeAnalyser = analyser.SyntheticTypeAnalyser(model_file=model_path)

        # with open(encoder_path, "rb") as encoder_file:
        #    self.encoder: OneHotEncoder = pickle.load(encoder_file, encoding='latin1')

    def get_shared_identity_probability(self, sensor_data):

        type_probabilities = self.type_analyser.obtain_probabilities(sensor_data)
        shared_identity_prob = type_probabilities.item()

        return shared_identity_prob
