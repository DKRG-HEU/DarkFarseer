import numpy as np
import pandas as pd
import os


def load_PEMSBAY_data():
    assert os.path.isfile('datasets/bay/pems-bay.h5')
    assert os.path.isfile('datasets/bay/distances_bay_2017.csv')


    df = pd.read_hdf('datasets/bay/pems-bay.h5')
    transfer_set = df.values
    distance_df = pd.read_csv('datasets/bay/distances_bay_2017.csv', dtype={'from': 'str', 'to': 'str'})
    normalized_k = 0.1

    dist_mx = np.zeros((325, 325), dtype=np.float32)

    dist_mx[:] = np.inf

    sensor_ids = df.columns.values.tolist()

    sensor_id_to_ind = {}

    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i
        
    for row in distance_df.values:
            if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
                continue
            dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))

    A_new = adj_mx
    return A_new, transfer_set