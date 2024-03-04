import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from synthetic_runner import encode_training_data

MAX_EPOCHS = 500  # type: int

EARLY_STOPPING_PATIENCE = int(MAX_EPOCHS * 0.10)  # type: int
TRAINING_BATCH_SIZE = 2048  # type: int
LEARNING_RATE = 0.001  # type: float
# UNITS_PER_LAYER = [16, 16]  # type: List[int]
UNITS_PER_LAYER = None  # For plain Logistic Regression

TRAINING_DATA_DIRECTORY = os.getcwd() + "/data/training"
CALIBRATION_SENSOR_DATA_FILE = "{}/sensor_data_validation.npy".format(TRAINING_DATA_DIRECTORY)
CALIBRATION_PERSON_TYPE_FILE = "{}/person_type_validation.npy".format(TRAINING_DATA_DIRECTORY)
NETLOGO_DATA_FILE_PREFIX = "request-for-help-results"  # type:str
REQUEST_RESULT_COLUMN = "offer-help"  # type:str


def get_netlogo_dataset():
    # type:() -> Tuple[np.ndarray, np.ndarray]

    dataframes = [pd.read_csv("{}/{}_{}.csv".format(TRAINING_DATA_DIRECTORY,
                                                    dataframe_index,
                                                    NETLOGO_DATA_FILE_PREFIX))
                  for dataframe_index in range(0, 97)]  # type: List[pd.DataFrame]

    netlogo_dataframe = pd.concat(dataframes, axis=0)  # type: pd.DataFrame

    netlogo_sensor_data = netlogo_dataframe.drop(REQUEST_RESULT_COLUMN, axis=1)  # type: pd.DataFrame
    netlogo_person_type = netlogo_dataframe[REQUEST_RESULT_COLUMN]  # type: pd.DataFrame

    return netlogo_sensor_data.values, netlogo_person_type.values


def start_training():
    sensor_data, person_type = get_netlogo_dataset()  # type: Tuple[np.ndarray, np.ndarray]
    results = train_test_split(sensor_data, person_type, test_size=0.33, stratify=person_type, random_state=0)
    encode_training_data(results[0])  # type:np.ndarray


if __name__ == "__main__":
    start_training()
