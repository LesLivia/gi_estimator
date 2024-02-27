import logging
import os
import pickle

import numpy as np
from sklearn.preprocessing import OneHotEncoder

ENCODER_FILE = os.getcwd() + "/model/encoder.pickle"  # type:str


def encode_training_data(sensor_data_train):  # type: (np.ndarray) -> np.ndarray

    logging.info("Encoding categorical features...")
    encoder = OneHotEncoder(sparse_output=False)  # type: OneHotEncoder
    encoder.fit(sensor_data_train)

    with open(ENCODER_FILE, "wb") as encoder_file:
        pickle.dump(encoder, encoder_file)
        logging.info("Encoder saved at {}".format(ENCODER_FILE))

    sensor_data_train = encoder.transform(sensor_data_train)

    return sensor_data_train
