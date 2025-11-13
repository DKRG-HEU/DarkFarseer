import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances


def load_air36_data():
    path = "datasets/air/small36.h5"
    df = pd.DataFrame(pd.read_hdf(path, 'pm25'))
    # print(df)
    stations = pd.DataFrame(pd.read_hdf(path, 'stations'))
    # print(stations)
    df = df.fillna(compute_mean(df))

    st_coord = stations.loc[:, ['latitude', 'longitude']]
    dist = geographical_distance(st_coord, to_rad=True).values
    adj = get_similarity(dist, thr=0.1)
    # np.fill_diagonal(adj, 0)
    return adj, df.values.transpose()


def compute_mean(x, index=None):
    """Compute the mean values for each datetime. The mean is first computed hourly over the week of the year.
    Further NaN values are computed using hourly mean over the same month through the years. If other NaN are present,
    they are removed using the mean of the sole hours. Hoping reasonably that there is at least a non-NaN entry of the
    same hour of the NaN datetime in all the dataset."""
    if isinstance(x, np.ndarray) and index is not None:
        shape = x.shape
        x = x.reshape((shape[0], -1))
        df_mean = pd.DataFrame(x, index=index)
    else:
        df_mean = x.copy()
    cond0 = [df_mean.index.year, df_mean.index.isocalendar().week, df_mean.index.hour]
    cond1 = [df_mean.index.year, df_mean.index.month, df_mean.index.hour]
    conditions = [cond0, cond1, cond1[1:], cond1[2:]]
    while df_mean.isna().values.sum() and len(conditions):
        nan_mean = df_mean.groupby(conditions[0]).transform(np.nanmean)
        df_mean = df_mean.fillna(nan_mean)
        conditions = conditions[1:]
    if df_mean.isna().values.sum():
        df_mean = df_mean.fillna(method='ffill')
        df_mean = df_mean.fillna(method='bfill')
    if isinstance(x, np.ndarray):
        df_mean = df_mean.values.reshape(shape)
    return df_mean

def geographical_distance(x=None, to_rad=True):
    """
    Compute the as-the-crow-flies distance between every pair of samples in `x`. The first dimension of each point is
    assumed to be the latitude, the second is the longitude. The inputs is assumed to be in degrees. If it is not the
    case, `to_rad` must be set to False. The dimension of the data must be 2.

    Parameters
    ----------
    x : pd.DataFrame or np.ndarray
        array_like structure of shape (n_samples_2, 2).
    to_rad : bool
        whether to convert inputs to radians (provided that they are in degrees).

    Returns
    -------
    distances :
        The distance between the points in kilometers.
    """
    _AVG_EARTH_RADIUS_KM = 6371.0088

    # Extract values of X if it is a DataFrame, else assume it is 2-dim array of lat-lon pairs
    latlon_pairs = x.values if isinstance(x, pd.DataFrame) else x

    # If the input values are in degrees, convert them in radians
    if to_rad:
        latlon_pairs = np.vectorize(np.radians)(latlon_pairs)

    distances = haversine_distances(latlon_pairs) * _AVG_EARTH_RADIUS_KM

    # Cast response
    if isinstance(x, pd.DataFrame):
        res = pd.DataFrame(distances, x.index, x.index)
    else:
        res = distances

    return res

def thresholded_gaussian_kernel(x, theta=None, threshold=None, threshold_on_input=False):
    if theta is None:
        theta = np.std(x)
    weights = np.exp(-np.square(x / theta))
    if threshold is not None:
        mask = x > threshold if threshold_on_input else weights < threshold
        weights[mask] = 0.
    return weights

def get_similarity(dist, thr=0.1, small=True):
    if small:
        theta = np.std(dist[:36, :36])  # use same theta for both air and air36
    else:
        theta = np.std(dist[:437, :437])
    adj = thresholded_gaussian_kernel(dist, theta=theta, threshold=thr)
    # if not include_self:
    #     adj[np.diag_indices_from(adj)] = 0.
    # if force_symmetric:
    #     adj = np.maximum.reduce([adj, adj.T])
    # if sparse:
    #     import scipy.sparse as sps
    #     adj = sps.coo_matrix(adj)
    return adj