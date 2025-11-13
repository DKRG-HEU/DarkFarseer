import torch
from torch import nn
import torch.nn.functional as F
from networks.layers.spatial_conv import SpatialConvOrderK
from networks.layers.SeTS import SeTS
from einops import rearrange
import random
from copy import deepcopy
    

class Darkfarseer(nn.Module):
    def __init__(self, sp_degree, time_window, hidden_size, drop_percent, testing=False):
        super(Darkfarseer, self).__init__()
        self.testing = testing
        self.time_window = time_window

        self.sp_degree = sp_degree
        self.drop_percent = drop_percent

        self.d_hidden = hidden_size  # hidden dim
        self.fc_start = nn.Linear(1, self.d_hidden)

        self.GNN1 = SpatialConvOrderK(c_in=self.d_hidden, c_out=self.d_hidden,
                                       support_len=2, order=1, include_self=True)
        self.GNN2 = SpatialConvOrderK(c_in=self.d_hidden, c_out=self.d_hidden,
                                       support_len=2, order=1, include_self=True)    

        self.relu = nn.ReLU(inplace=True) 
        self.leaky_relu1 = nn.LeakyReLU()

        # SeTS
        self.time_enhance_1 = SeTS()
        self.time_enhance_2 = SeTS()

        self.reduce_dim1 = nn.Linear(24, 1)
        self.reduce_dim2 = nn.Linear(24, 1)

        # output layer
        self.final_linear1 = nn.Linear(self.d_hidden, self.d_hidden//2)
        self.final_linear2 = nn.Linear(self.d_hidden//2, 1)

    @staticmethod
    def contrastive_loss(positive_pairs, negative_pairs, temperature=0.5):
        """
        InfoNCE Loss
        """
        num_negatives = negative_pairs.shape[-1]

        pos1 = positive_pairs[0].permute(0, 2, 3, 1).squeeze(-2)
        pos2 = positive_pairs[1].permute(0, 2, 3, 1).squeeze(-2)

        positive_sim = F.cosine_similarity(pos1, pos2, dim=-1)
        negative_pairs_perm = negative_pairs.permute(0, 2, 3, 1)
        pos1_expanded = pos1.unsqueeze(2).expand(-1, -1, num_negatives, -1)

        negative_sim = F.cosine_similarity(pos1_expanded, negative_pairs_perm, dim=-1)
        positive_sim = positive_sim.unsqueeze(-1)

        logits = torch.cat([positive_sim, negative_sim], dim=-1)
        logits = logits / temperature
        logits = logits.view(-1, 1 + num_negatives)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)

        return loss
        
    @staticmethod
    def time_aug(x_global, p=0.2):
        batch_size, time_window, num_nodes = x_global.size()

        num_to_mask = int(time_window * p)

        mask_indices = torch.randperm(time_window)[:num_to_mask]
        mask = torch.ones_like(x_global)
        mask[:, mask_indices, :] = 0
        
        x_global_masked = x_global * mask
        
        return x_global_masked

    @staticmethod
    def add_random_edges(A_q, p=0.003):
        num_nodes = A_q.size(0)
        A_q = A_q.detach().clone()
        
        zero_indices = torch.where(A_q == 0)
        
        mask = torch.rand(zero_indices[0].size()) < p
        chosen_indices = torch.where(mask)[0]

        A_q[zero_indices[0][chosen_indices], zero_indices[1][chosen_indices]] = 1
        
        return A_q
    
    def cal_edge_weight(self, sim_h, sim_s):
        edge_weight = (1-1/self.sp_degree)* sim_h + 1/self.sp_degree *sim_s

        return edge_weight

    def calculate_random_walk_matrix(self, adj_mx):
        """
        calculate random walk matrix
        """
        degrees = adj_mx.sum(dim=1)  # Shape: [N]
        d_inv = torch.pow(degrees, -1)
        d_inv[torch.isinf(d_inv)] = 0.0
        d_mat_inv = torch.diag(d_inv)  # Shape: [N, N]
        random_walk_mx = torch.matmul(d_mat_inv, adj_mx)  # Shape: [N, N]

        return random_walk_mx
    
    def RSCL(self, missing_mask, imputation_ori_view, imputation_aug_view, positive_select,\
              negative_select, notMap, device='cuda:0'):
        contrast_loss = 0
        pos_neg_counter = 0

        if self.training:
            for missing_index in missing_mask:
                missing_node_series = imputation_ori_view[:, :, :, missing_index].unsqueeze(-1)
  
                positive_list = positive_select[missing_index]
                if len(positive_list) == 1 and positive_list[0] == missing_index:
                    continue
                
                positive_list_imputation = []
                for posi in positive_list:
                    if posi == missing_index:
                        continue

                    node_indices = torch.tensor(posi).to(device=device)
                    # calculate Prototypes
                    positive_list_imputation.append(imputation_aug_view[:, :, :,\
                                                                         node_indices].mean(dim=-1, keepdim=True))  

                if missing_index in notMap:
                    continue
                else:
                    try:
                        negative_list = negative_select[missing_index]
                    except:
                        print(f"Negative list not found for missing index {missing_index}")
                        continue
                    try:
                        negative_list = random.sample(negative_list, k=20)
                    except:
                        pass

                negative_samples = negative_list

                ll = []
                for node_indices in negative_samples:
                    if isinstance(node_indices, list):
                        node_indices = torch.tensor(node_indices).to(device=device)
                        # calculate Prototypes
                        ll.append(imputation_aug_view[:, :, :, node_indices].mean(dim=-1, keepdim=True))
                    else:
                        node_indices = torch.tensor(node_indices).to(device=device)
                        ll.append(imputation_aug_view[:, :, :, node_indices].unsqueeze(-1))

                topology_prototypes_negative = torch.cat(ll, dim=-1)

                temp_loss = 0

                for topology_prototypes_positive in positive_list_imputation:
                    positive_pairs_spatial = (missing_node_series, topology_prototypes_positive)
                    negative_pairs_spatial = topology_prototypes_negative
                    # RSCL
                    spatial_loss = self.contrastive_loss(positive_pairs_spatial, negative_pairs_spatial)
                    temp_loss += spatial_loss

                temp_loss = temp_loss / len(positive_list)
                contrast_loss = contrast_loss + temp_loss
                pos_neg_counter += 1

        try:
            return contrast_loss/pos_neg_counter
        except:
            return 0

    def SDGS(self, A_ori, missing_mask, imputation, positive_select, notMap, n, device="cuda:0"):
        missing_node_spatial_prototype_dict = dict()
        A_ori = torch.from_numpy(A_ori).float().to(device)

        if self.sp_degree > 1:
            for missing_index in missing_mask:
                positive_list = positive_select[missing_index]

                used = 0

                if isinstance(positive_list[0], list):
                    positive_list = positive_list[0]

                for posi in positive_list:
                    if posi == missing_index:
                        continue
                    
                    node_indices = torch.tensor(posi).to(device=device)
                    if not used:
                        if [posi].__len__() == 1:
                            missing_node_spatial_prototype_dict[missing_index] =\
                                  imputation[:, :, :, node_indices].unsqueeze(-1)
                        else:
                            missing_node_spatial_prototype_dict[missing_index] =\
                                  imputation[:, :, :, node_indices].mean(dim=-1, keepdim=True)

            A_temp = A_ori.detach()

            # drop more edges during the inference phase than during the training phase
            if self.testing:
                current_drop_percentage = self.drop_percent*1.2
            else:
                current_drop_percentage = self.drop_percent

            for missing_index in missing_mask:
                if missing_index not in notMap:  # ensure that the Prototype corresponding to the virtual node exists
                    # node embedding similarity
                    imputation_h_i = imputation[:, :, :, missing_index].unsqueeze(-1).expand(-1,-1,-1,n)
                    imputation_h_j = imputation
                    # calculate cosine similarity based on hidden dimensions
                    cos_sim_imputation = F.cosine_similarity(imputation_h_i, imputation_h_j)  # [b s n]
                    cos_sim_imputation = self.reduce_dim1(cos_sim_imputation.permute(0, 2, 1)).squeeze(-1)  # [b n]

                    # spatial prototype similarity
                    spatial_prototype_i = missing_node_spatial_prototype_dict[missing_index].expand(-1,-1,-1,n)
                    other_node_imputation = imputation
                    cos_sim_space = F.cosine_similarity(spatial_prototype_i, other_node_imputation)  # [b s n]
                    cos_sim_space = self.reduce_dim2(cos_sim_space.permute(0, 2, 1)).squeeze(-1)  # [b n]

                    # compute weight
                    edge_weight = self.cal_edge_weight(cos_sim_imputation.detach(), cos_sim_space.detach())  # [b n]
                    edge_weight = torch.mean(edge_weight, dim=0, keepdim=True)  # [1 n]

                    # avoid altering self-loops when processing adj_mat elements, as self-loops are beneficial for the model
                    non_self_loop = [True for i in range(n)]
                    non_self_loop[missing_index] = False
                    non_self_loop = torch.tensor(non_self_loop).to(device=device)

                    # because the dataset contains directed-graphs (e.g., PEMS03/PEMS04/PEMS-BAY/METR-LA)
                    # we handle the in-neighbors and out-neighbors of virtual nodes separately

                    # select virtual nodes with non-zero in-degree and out-degree
                    out_degree_index_mask = (A_temp[missing_index] > 0) & non_self_loop  # neighbors: True
                    in_degree_index_mask = (A_temp[:, missing_index] > 0) & non_self_loop  # neighbors: True

                    out_degree_flag = False
                    in_degree_flag = False

                    out_degree = (A_temp[missing_index] > 0).sum()
                    drop_out_num = int(out_degree*current_drop_percentage)
                    if drop_out_num > 0:
                        out_degree_flag = True

                    in_degree = (A_temp[:, missing_index] > 0).sum()
                    drop_in_num = int(in_degree*current_drop_percentage)
                    if drop_in_num > 0:
                        in_degree_flag = True

                    if not out_degree_flag and not in_degree_flag:
                        continue

                    if out_degree_flag:
                        # find the indices corresponding to bottom-k
                        out_degree_masked_edge_weight = edge_weight.masked_fill(~out_degree_index_mask, float('inf'))
                        _, out_degree_bottom_k_indices = torch.topk(-out_degree_masked_edge_weight, drop_in_num, dim=1)

                        out_degree_add_values = torch.zeros((n)).to(device=device)
                        out_degree_add_values[out_degree_bottom_k_indices] = -torch.inf

                    if in_degree_flag:
                        # find the indices corresponding to bottom-k
                        in_degree_masked_edge_weight = edge_weight.masked_fill(~in_degree_index_mask, float('inf'))
                        _, in_degree_bottom_k_indices = torch.topk(-in_degree_masked_edge_weight, drop_out_num, dim=1)

                        in_degree_add_values = torch.zeros((n)).to(device=device)
                        in_degree_add_values[in_degree_bottom_k_indices] = -torch.inf


                    A_edge_judge = torch.zeros((n, n)).to(device=device)
                    if in_degree_flag:
                        A_edge_judge[:, missing_index] = in_degree_add_values
                    if out_degree_flag:
                        A_edge_judge[missing_index, :] = out_degree_add_values

                    # replace the elements of the edges to be dropped with '-inf'
                    A_ori = A_ori + A_edge_judge

            # replace the elements of the edges to be dropped with low intensity values
            A_ori = torch.where(A_ori == -torch.inf, torch.tensor(1e-5).to(device=device), A_ori)

            # norm
            with torch.no_grad():
                A_q_2 = self.calculate_random_walk_matrix(A_ori)
                A_h_2 = self.calculate_random_walk_matrix(A_ori)

            return A_q_2, A_h_2
        else:
            return None, None

        
    def forward(self, x_global, x_local, A_q, A_h, A_ori, khop_index,
                 missing_mask, notMap, positive_select, negative_select):

        b, s, n = x_global.size()
        # Temporal-level Aug
        masked_x_global = self.time_aug(x_global, p=0.2)

        _x_global = x_global.unsqueeze(-1)  # [batch_size, time_window, num_nodes, 1]
        masked_x_global = masked_x_global.unsqueeze(-1)

        # start MLP
        imputation = self.relu(self.fc_start(_x_global))
        imputation_aug = self.relu(self.fc_start(masked_x_global))

        imputation = rearrange(imputation, 'b s n d -> b d s n')
        imputation_aug = rearrange(imputation_aug, 'b s n d -> b d s n')

        # SeTS
        imputation = self.time_enhance_1(imputation, x_local, khop_index)
        imputation_aug = self.time_enhance_1(imputation_aug, x_local, khop_index)

        # Topology-level Aug
        with torch.no_grad():
            A_q_aug = self.add_random_edges(A_q, p=0.003)
            A_h_aug = self.add_random_edges(A_h, p=0.003)


        imputation = rearrange(imputation, 'b d s n -> b d n s')
        imputation_aug = rearrange(imputation_aug, 'b d s n -> b d n s')

        # MPNN
        imputation =  self.GNN1(imputation, [A_q, A_h]) # b d n s
        imputation_aug =  self.GNN1(imputation_aug, [A_q_aug, A_h_aug]) # b d n s

        imputation = imputation.permute(0, 1, 3, 2)
        imputation_aug = imputation_aug.permute(0, 1, 3, 2)

        imputation_drop_edge = imputation
        aug_view = imputation_aug
        
        # SeTS
        imputation = self.time_enhance_2(imputation, x_local, khop_index)

        # RSCL
        Loss_vc = self.RSCL(missing_mask, imputation_drop_edge, aug_view,
                             positive_select, negative_select, notMap, device=x_global.device)

        # SDGS
        A_q_2, A_h_2 = self.SDGS(A_ori, missing_mask, imputation_drop_edge,
                                  positive_select, notMap, n, device=x_global.device)
        
        # MPNN
        imputation = rearrange(imputation, 'b d s n -> b d n s')

        if A_q_2 is None:
            imputation = self.leaky_relu1(self.GNN2(imputation, [A_q, A_h])) # b d n s
        else:
            imputation = self.leaky_relu1(self.GNN2(imputation, [A_q_2, A_h_2])) # b d n s

        imputation = imputation.permute(0, 2, 3, 1)
        
        # MLP readout
        result = self.final_linear2(self.final_linear1(imputation)).squeeze(-1).permute(0, 2, 1)

        if self.training:
            return result, Loss_vc
        else:
            return result, 0
