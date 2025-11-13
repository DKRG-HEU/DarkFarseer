import numpy as np
import pandas as pd


def load_PEMS04_data():
    path = "datasets/PEMS04/PEMS04.npz"
    dist_path = "datasets/PEMS04/PEMS04.csv"
    data = np.load(path)["data"]
    ids = [i for i in range(0, data.shape[1])]

    distances = pd.read_csv(dist_path)
    # print(distances)
    num_sensors = len(ids)
    dist = np.ones((num_sensors, num_sensors), dtype=np.float32) * np.inf
    sensor_id_to_ind = {int(sensor_id): i for i, sensor_id in enumerate(ids)}

    for row in distances.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    finite_dist = dist.reshape(-1)
    finite_dist = finite_dist[~np.isinf(finite_dist)]
    sigma = finite_dist.std()
    adj = np.exp(-np.square(dist / sigma))

    return adj, data