from __future__ import division

import torch
from torch import nn
import numpy as np
import random

from experiments.params import get_args

from utils.utils import *
from networks.Kriging_model import Darkfarseer
from utils.numpy_metrics import *
from utils.BCC_subgraph import get_BBC

from utils.utils import StandardScaler, MinMaxScaler
from experiments.engine import test_kriging, get_missing_node_one_HOP

import time
import copy
from tqdm import tqdm


# -------------------------------------- Parameter Settings -----------------------------------
args = get_args()
# random seed
rand = np.random.RandomState(args.seed)

# dataset & model
dataset = args.dataset # [METR-LA, PEMS-BAY, SEA-LOOP, PEMS07, PEMS08, PEMS04, AIR36, AIR437, NREL]
model_name = "Darkfarseer"

# training parameters
partition = args.partition
device = torch.device(args.device)
batch_size = args.batch_size
missing_rate = args.virtual_node_rate

max_iter = args.epochs
learning_rate = args.learning_rate
early_stop = 300
clip = args.clip
num_run = args.num_run

# model parameters
eta = args.eta


def calculate_degree_statistics(adj_matrix):
    n = len(adj_matrix)
    
    # remove self-loops if they exist
    np.fill_diagonal(adj_matrix, 0)

    is_symmetric = np.allclose(adj_matrix, adj_matrix.T, atol=1e-8)
    
    if is_symmetric:
        degrees = np.sum(adj_matrix != 0, axis=1)
    else:
        out_degrees = np.sum(adj_matrix != 0, axis=1)
        in_degrees = np.sum(adj_matrix != 0, axis=0)
        degrees = out_degrees + in_degrees
    
    avg_degree = np.mean(degrees)
    max_degree = np.max(degrees)
    min_degree = np.min(degrees)
    degree_variance = np.var(degrees)
    degree_std = np.std(degrees)

    e_div_nlogn = sum(degrees) / (n*np.log(n))
    
    return avg_degree, max_degree, min_degree, degree_variance, degree_std, e_div_nlogn


def load_data(dataset, edge_threshold):
    temp_file = None
    if dataset == 'METR-LA':
        from preprocessing.process_LA import load_METRLA_data
        A, X = load_METRLA_data()
        temp_file = "METR_LA"
    elif dataset == "PEMS-BAY":
        from preprocessing.process_BAY import load_PEMSBAY_data
        A, X = load_PEMSBAY_data()
        X = X.transpose()
        temp_file = "PEMS_BAY"
    elif dataset == 'PEMS03':
        from preprocessing.process_PEMS03 import load_PEMS03_data
        A, X = load_PEMS03_data()
        X = X[:, :, 0]
        X = X.transpose()
        temp_file = "PEMS03"
    elif dataset == 'PEMS04':
        from preprocessing.process_PEMS04 import load_PEMS04_data
        A, X = load_PEMS04_data()
        X = X[:, :, 0]
        X = X.transpose()
        temp_file = "PEMS04"
    elif dataset == "AIR36":
        from preprocessing.process_AIR36 import load_air36_data
        A, X = load_air36_data()
        temp_file = "AIR36"
    elif dataset == "NREL-PA":
        from preprocessing.process_NREL_PA import load_nrel_pa_data
        A, X = load_nrel_pa_data()
        temp_file = "NREL_PA"
    elif dataset == "USHCN":
        from preprocessing.process_USHCN import load_udata
        # XXX: follow the code of IGNNK
        A, X, OM = load_udata()
        # X = X*OM
        X = X[:,:,:,0]
        X = X.reshape(1218,120*12)
        X = X/100

        temp_file = "USHCN"
    else:
        raise FileExistsError("Dataset does not exist")

    train_ratio, val_ratio, test_ratio = partition.split("/")

    # Split train/validation/test set
    train_len = int(X.shape[1] * 0.1*int(train_ratio))
    val_len = int(X.shape[1] * 0.1*int(val_ratio))
    test_len = X.shape[1] - train_len - val_len

    training_set = X[:, :train_len].transpose()
    val_set = X[:, train_len: train_len + val_len].transpose()
    test_set = X[:, -test_len:].transpose()


    num_sample = int(X.shape[0]*missing_rate)
    unknow_set = rand.choice(list(range(0,X.shape[0])), num_sample, replace=False)
    unknow_set = set(unknow_set)

    full_set = set(range(0,X.shape[0]))        
    know_set = full_set - unknow_set
    training_set_s = training_set[:, list(know_set)]  # get the training data in the sample time period
    val_set_s = val_set[:, list(know_set)]  # get the valing data in the sample time period

    A_s = A[:, list(know_set)][list(know_set), :]  # get the observed adjacent matrix from the full adjacent matrix

    train_min_global = np.min(training_set_s)
    train_max_global = np.max(training_set_s)

    train_mean = np.mean(training_set_s)
    train_std = np.std(training_set_s)

    if dataset == "PEMS03" or dataset == "PEMS04":
        scaler = StandardScaler(mean=train_mean, std=train_std)
    else:
        scaler = MinMaxScaler(min_val=train_min_global, max_val=train_max_global)
        

    training_set_s = scaler.transform(training_set_s)
    val_set_s = scaler.transform(val_set_s)
    test_set = scaler.transform(test_set)

    scaler_observed_nodes = scaler
    scaler_test = scaler

    BBC_full, notMap_full, positive_select_full, negative_select_full, negative_art_full = get_BBC(A, dataset, edge_threshold)
    BBC_sub, notMap_sub, positive_select_sub, negative_select_sub, negative_art_sub = get_BBC(A_s, dataset, edge_threshold)

    
    print("**"*15)
    print("Test Adj mat info:")
    avg_degree_full, max_degree_full, min_degree_full, degree_variance_full, degree_std_full, sp_degree_full = calculate_degree_statistics(A)
    print(f"    * avg_degree: {avg_degree_full:.2f}")
    print(f"    * max_degree: {max_degree_full}")
    print(f"    * min_degree: {min_degree_full}")
    print(f"    * degree_variance: {degree_variance_full:.2f}")
    print(f"    * degree_std: {degree_std_full:.2f}")
    print(f"    * sp_degree: {sp_degree_full:.2f}")
    print("**"*15)
    print("**"*15)
    print("Train Adj mat info:")
    avg_degree, max_degree, min_degree, degree_variance, degree_std, sp_degree = calculate_degree_statistics(A_s)
    print(f"    * avg_degree: {avg_degree:.2f}")
    print(f"    * max_degree: {max_degree}")
    print(f"    * min_degree: {min_degree}")
    print(f"    * degree_variance: {degree_variance:.2f}")
    print(f"    * degree_std: {degree_std:.2f}")
    print(f"    * sp_degree: {sp_degree:.2f}")
    print("**"*15)

    return A, X, BBC_full, BBC_sub, positive_select_full, negative_select_full, negative_art_full, \
        positive_select_sub, negative_select_sub, negative_art_sub, notMap_full, notMap_sub, \
        training_set, val_set, test_set, unknow_set, full_set, know_set, training_set_s, val_set_s, A_s,\
              scaler_observed_nodes, scaler_test, sp_degree, sp_degree_full, temp_file


if __name__ == "__main__":
    """
    Model training
    """

    A, X, BBC_full, BBC_sub, positive_select_full, negative_select_full, negative_art_full, \
        positive_select_sub, negative_select_sub, negative_art_sub, notMap_full, notMap_sub,\
        training_set, val_set, test_set, unknow_set, full_set, know_set, training_set_s, \
        val_set_s, A_s, scaler_observed_nodes, scaler_test, sp_degree, sp_degree_full, temp_file = load_data(args.dataset, args.mu)

    n_o_n_m = X.shape[0] - int(X.shape[0]*missing_rate)  # sampled space dimension
    h = args.time_window
    n_m = int(X.shape[0]*missing_rate)  # number of mask node during training
    n_u = int(X.shape[0]*missing_rate)  # target locations, n_u locations will be deleted from the training data

    print('##################################    start training    ##################################')
    
    avg_MAE = []
    avg_RMSE = []
    avg_R2 = []
    avg_CORR = []
    avg_MRE = []

    for _ in range(num_run):
        # init
        KrigModel = Darkfarseer(sp_degree, args.time_window,
                                 args.hidden_size, args.beta, testing=False)

        def init_weights(m):
            if hasattr(m, 'weight') and hasattr(m.weight, 'data'):
                nn.init.xavier_uniform_(m.weight.data)

            if hasattr(m, 'bias') and hasattr(m.bias, 'data') and m.bias.data is not None:
                nn.init.zeros_(m.bias.data)
    
        # init weights
        KrigModel.apply(init_weights)

        num_params = 0
        for param in KrigModel.parameters():
            num_params += param.numel()
        print("---"*20)
        print()
        print('[Network %s] Total number of parameters : %.3f M' % ("Darkfarseer", num_params / 1e6))
        print()
        print("---"*20)

        KrigModel.to(device)
        criterion = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(KrigModel.parameters(), lr = learning_rate)

        RMSE_list = []
        MAE_list = []
        MAPE_list = []
        pred = []
        truth = []
        min_val_score = 1e6
        best_epoch = None
        stop_epoch = 0

        A_s_train = A_s

        for epoch in tqdm(range(max_iter)):
            KrigModel.train()
            for i in range(training_set.shape[0]//(h * batch_size)):  #  using time_length as reference to record test_error
                t_random = np.random.randint(0, high=(training_set_s.shape[0] - h), size=batch_size, dtype='l')
                know_mask = set(random.sample(range(0, training_set_s.shape[1]), n_o_n_m))  # sample nodes
                feed_batch = []
                for j in range(batch_size):
                    feed_batch.append(training_set_s[t_random[j]: t_random[j] + h, :][:, list(know_mask)])  # generate batches
                
                inputs = np.array(feed_batch)
                inputs_omask = np.ones(np.shape(inputs))

                if not dataset == 'NREL-PA': 
                    inputs_omask[inputs == 0] = 0           
                                                        
                missing_index = np.ones((inputs.shape))
                missing_mask = random.sample(range(0,n_o_n_m), n_m)  # Masked locations
                unobserved_nodes_BBC = BBC_sub[missing_mask]
                unobserved_nodes_BBC = torch.from_numpy(unobserved_nodes_BBC)

                missing_index[:, :, missing_mask] = 0

                mask_predict = np.zeros((inputs.shape))
                mask_predict[:, :, missing_mask] = 1
                mask_predict = torch.from_numpy(mask_predict).to(device)                

                Mf_inputs = inputs * inputs_omask * missing_index


                Mf_inputs = torch.from_numpy(Mf_inputs.astype('float32')).to(device)
                mask = torch.from_numpy(inputs_omask.astype('float32')).to(device)  # The reconstruction errors on irregular 0s are not used for training

                A_dynamic = A_s_train[list(know_mask), :][:, list(know_mask)]  # Obtain the dynamic adjacent matrix


                A_q = torch.from_numpy((calculate_random_walk_matrix(A_dynamic).T).astype('float32')).to(device)
                A_h = torch.from_numpy((calculate_random_walk_matrix(A_dynamic.T).T).astype('float32')).to(device)


                khop_list = [0 for i in range(0, n_o_n_m)]
                khop = get_missing_node_one_HOP(khop_list, A_dynamic, missing_mask, dataset)
                khop_index = khop.detach()
                khop = khop.bool()

                Mf_inputs_local_node = torch.tensor(inputs.astype('float32') * inputs_omask.astype('float32'))
                Mf_inputs_local_node = Mf_inputs_local_node[:, :, khop]
                Mf_inputs_local_node = Mf_inputs_local_node.to(device)

                outputs = torch.from_numpy(inputs.astype('float32')).to(device)  # The label
                
                optimizer.zero_grad()

                X_res, spatial_loss = KrigModel(Mf_inputs, Mf_inputs_local_node, A_q, A_h, A_dynamic,
                                khop_index, missing_mask, notMap_sub,
                                positive_select_sub, negative_select_sub)  # Obtain the reconstruction

                loss = criterion(X_res*mask*mask_predict, outputs*mask*mask_predict) + eta*spatial_loss

                # print(f"spatial_loss: {spatial_loss*eta}")
                # print(f"preidction_loss: {criterion(X_res*mask*mask_predict, outputs*mask*mask_predict)}")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(KrigModel.parameters(), max_norm=clip)
                optimizer.step()

            # -------------------------------------- validation ------------------------------------------
            KrigModel.eval()
            val_loss = list()
            num_val = val_set.shape[0]//(h * batch_size)
            

            for j in range(num_val):
                t_random = np.random.randint(0, high=(val_set_s.shape[0] - h), size=batch_size, dtype='l')
                know_mask = set(random.sample(range(0, val_set_s.shape[1]), n_o_n_m)) # sample nodes
                feed_batch = []
                for j in range(batch_size):
                    feed_batch.append(val_set_s[t_random[j]: t_random[j] + h, :][:, list(know_mask)]) 
                
                inputs = np.array(feed_batch)
                inputs_omask = np.ones(np.shape(inputs))

                if not dataset == 'NREL-PA': 
                    inputs_omask[inputs == 0] = 0   
                                                    
                missing_index = np.ones((inputs.shape))
                missing_mask = random.sample(range(0,n_o_n_m), n_m) # Masked nodes
                unobserved_nodes_BBC = BBC_sub[missing_mask]
                unobserved_nodes_BBC = torch.from_numpy(unobserved_nodes_BBC)

                missing_index[:, :, missing_mask] = 0

                mask_predict = np.zeros((inputs.shape))
                mask_predict[:, :, missing_mask] = 1
                mask_predict = torch.from_numpy(mask_predict).to(device)

                Mf_inputs = inputs * inputs_omask * missing_index
                Mf_inputs = torch.from_numpy(Mf_inputs.astype('float32')).to(device)
                mask = torch.from_numpy(inputs_omask.astype('float32')).to(device)
                A_dynamic = A_s[list(know_mask), :][:, list(know_mask)]

                A_q = torch.from_numpy((calculate_random_walk_matrix(A_dynamic).T).astype('float32')).to(device)
                A_h = torch.from_numpy((calculate_random_walk_matrix(A_dynamic.T).T).astype('float32')).to(device)

                khop_list = [0 for i in range(0, n_o_n_m)]
                khop = get_missing_node_one_HOP(khop_list, A_dynamic, missing_mask, dataset)
                khop_index = khop.detach()
                khop = khop.bool()
                Mf_inputs_local_node = torch.tensor(inputs.astype('float32') * inputs_omask.astype('float32'))
                Mf_inputs_local_node = Mf_inputs_local_node[:, :, khop]
                Mf_inputs_local_node = Mf_inputs_local_node.to(device)

                outputs = torch.from_numpy(inputs).to(device)
                
                X_res, _ = KrigModel(Mf_inputs, Mf_inputs_local_node, A_q, A_h, A_dynamic,
                                khop_index, missing_mask, notMap_sub,
                                positive_select_sub, negative_select_sub)  #Obtain the reconstruction


                val_mse = criterion(X_res*mask*mask_predict, outputs*mask*mask_predict)
                val_loss.append(val_mse.detach().cpu())

            mean_val = np.mean(val_loss)
            print("epoch: ", epoch, "val_MSE: ", mean_val)
            if mean_val < min_val_score:
                min_val_score = mean_val
                stop_epoch = 0

                print("**********## The best model updated! -> Epoch:{} ##**********".format(epoch))
                best_model = copy.deepcopy(KrigModel.state_dict())
                best_epoch = epoch
                
                 # Save the model
                torch.save(best_model, f'checkpoints/{temp_file}/Darkfarseer_{dataset}_epoch_' + str(epoch) + '.pth')
            else:
                stop_epoch += 1
                if stop_epoch > early_stop:
                    break

        #---------------------------------------- testing -----------------------------------------------
        model_path = f'checkpoints/{temp_file}/Darkfarseer_{dataset}_epoch_' + str(best_epoch) + '.pth'
        KrigModel = Darkfarseer(sp_degree_full, args.time_window,
                                args.hidden_size, args.beta, testing=True)

        # load model
        KrigModel.load_state_dict(torch.load(model_path))
        KrigModel = KrigModel.to(device)
        KrigModel.eval()

        if not dataset == 'NREL-PA':
            MAE, RMSE, MAPE, MSE, CORR, MRE, R2 = test_kriging(KrigModel, unknow_set, test_set,
                                                                A, True, device, scaler_test, args.dataset,
                                                                  BBC_full, notMap_full, positive_select_full, negative_select_full)
        else:
            MAE, RMSE, MAPE, MSE, CORR, MRE, R2 = test_kriging(KrigModel, unknow_set, test_set,
                                                                A, False, device, scaler_test, args.dataset,
                                                                  BBC_full, notMap_full, positive_select_full, negative_select_full)
            
        print("##############  Results ##############")
        print("MAE = ", MAE, "RMSE = ", RMSE, "MRE = ", MRE)

        avg_MAE.append(MAE)
        avg_RMSE.append(RMSE)
        avg_R2.append(R2)
        avg_CORR.append(CORR)
        avg_MRE.append(MRE)

    print('-'*20)
    print("Number of runs:", num_run)
    print('-'*20)
    print("#### Result:")
    print("MAE: ", np.mean(avg_MAE), "std:", np.std(avg_MAE))
    print("RMSE: ", np.mean(avg_RMSE), "std:", np.std(avg_RMSE))
    print("MRE: ", np.mean(avg_MRE), "std:", np.std(avg_MRE))