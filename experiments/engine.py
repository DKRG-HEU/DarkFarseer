import torch
from torch import nn
import numpy as np
import random

from experiments.params import get_args

from utils.utils import *
from networks.Kriging_model import Darkfarseer
from utils.numpy_metrics import *
from utils.BCC_subgraph import get_BBC

from utils.utils import StandardScaler

import time
import copy
from tqdm import tqdm



def get_missing_node_one_HOP(khop_list, Adj, missing_mask, dataset):
    """
    - PEMS0*/METR-LA/PEMS-BAY: direction="both";
    - AIR36/AIR437/NREL*/SEA-LOOP: direction="in";
    """
    if dataset in ["METR-LA", "PEMS-BAY", "PEMS04", "PEMS03"]:
        direction = "both"
    else:
        direction = "in"

    khop = torch.tensor(khop_list, requires_grad=False)
    missing_mask = torch.tensor(list(missing_mask), requires_grad=False)
    Adj = torch.tensor(Adj, requires_grad=False)
    
    if direction == "out":
        rows = Adj[missing_mask]
    elif direction == "in":
        rows = Adj[:, missing_mask].T 
    elif direction == "both":
        rows_out = Adj[missing_mask]
        rows_in = Adj[:, missing_mask].T
        rows = rows_out + rows_in
    
    non_zero_indices = rows.nonzero(as_tuple=True)
    khop[non_zero_indices[1]] = 1
    khop[missing_mask] = 0

    return khop


def test_kriging(KrigModel, unknow_set, test_set, A_s, Missing0, device, 
               scaler_test, dataset, BBC_full, notMap_full, positive_select_full, negative_select_full):

    test_data = test_set
    unknow_set = set(unknow_set)
    time_dim = KrigModel.time_window
    test_omask = np.ones(test_data.shape)

    if Missing0 == True:
        test_omask[test_data == 0] = 0

    test_inputs = (test_data * test_omask).astype('float32')
    test_inputs_s = test_inputs
   
    missing_index = np.ones(np.shape(test_data))
    missing_index[:, list(unknow_set)] = 0
    missing_index_s = missing_index
    
    o = np.zeros([test_data.shape[0]//time_dim*time_dim, test_inputs_s.shape[1]])
    
    for i in range(0, test_data.shape[0]//time_dim*time_dim, time_dim):
        inputs = test_inputs_s[i:i+time_dim, :]
        missing_inputs = missing_index_s[i:i+time_dim, :]
        T_inputs = inputs*missing_inputs

        T_inputs = T_inputs
        T_inputs = np.expand_dims(T_inputs, axis = 0)
        T_inputs = torch.from_numpy(T_inputs.astype('float32')).to(device)

        A_q2 = torch.from_numpy((calculate_random_walk_matrix(A_s).T).astype('float32')).to(device)
        A_h2 = torch.from_numpy((calculate_random_walk_matrix(A_s.T).T).astype('float32')).to(device)

        khop_list = [0 for i in range(0, A_s.shape[0])]

        khop = get_missing_node_one_HOP(khop_list, A_s, unknow_set, dataset)
        khop_index = khop
        khop = khop.bool()

        unobserved_nodes_BBC = BBC_full[list(unknow_set)]
        unobserved_nodes_BBC = torch.from_numpy(unobserved_nodes_BBC)

        local_node = torch.tensor(T_inputs).to(device)
        local_node = local_node[:, :, khop]

        local_node = local_node.to(device)

        imputation, _ = KrigModel(T_inputs, local_node, A_q2, A_h2, A_s,
                                khop_index, list(unknow_set), notMap_full,
                                positive_select_full, negative_select_full)  #Obtain the reconstruction
            
        imputation = imputation.cuda().data.cpu().numpy()
        o[i:i+time_dim, :] = imputation[0, :, :]
    
        o = o

    truth = test_inputs_s[0:test_set.shape[0]//time_dim*time_dim]
    o[missing_index_s[0:test_set.shape[0]//time_dim*time_dim] == 1] =\
          truth[missing_index_s[0:test_set.shape[0]//time_dim*time_dim] == 1]
    
    truth = scaler_test.inverse_transform(truth)
    o = scaler_test.inverse_transform(o)

    test_mask =  1 - missing_index_s[0:test_set.shape[0]//time_dim*time_dim]

    if Missing0 == True:
        test_mask[truth == 0] = 0
        o[truth == 0] = 0
    

    MAE = masked_mae(o, truth, test_mask)
    RMSE = masked_rmse(o, truth, test_mask)
    MAPE = masked_mape(o, truth, test_mask)
    MSE = masked_mse(o, truth, test_mask)
    R2 = masked_r2(o, truth, test_mask)
    MRE = masked_mre(o, truth, test_mask)
    CORR = masked_corr_coefficient(o, truth, test_mask)

    return MAE, RMSE, MAPE, MSE, CORR, MRE, R2