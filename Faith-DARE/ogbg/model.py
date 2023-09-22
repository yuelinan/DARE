import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import reset
import torch.nn as nn
from conv import GNN_node, GNN_node_Virtualnode
import numpy as np
import random
from torch_geometric.nn import ARMAConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
nn_act = torch.nn.ReLU()
F_act = F.relu

#######
class Graph_DARE_DA_MI(torch.nn.Module):

    def __init__(self, num_tasks, num_layer = 5, emb_dim = 300, gnn_type = 'gin', drop_ratio = 0.5, gamma = 0.4, use_linear_predictor=False):
        '''
            num_tasks (int): number of labels to be predicted
        '''

        super(Graph_DARE_DA_MI, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.gamma  = gamma
        self.infonce = InfoNCE(emb_dim)
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        gnn_name = gnn_type.split('-')[0]
        emb_dim_rat = emb_dim
        if 'virtual' in gnn_type: 
            rationale_gnn_node = GNN_node_Virtualnode(2, emb_dim_rat, JK = "last", drop_ratio = drop_ratio, residual = True, gnn_name = gnn_name)
            self.graph_encoder = GNN_node_Virtualnode(num_layer, emb_dim, JK = "last", drop_ratio = drop_ratio, residual = True, gnn_name = gnn_name)
        else:
            rationale_gnn_node = GNN_node(2, emb_dim_rat, JK = "last", drop_ratio = drop_ratio, residual = True, gnn_name = gnn_name)
            self.graph_encoder = GNN_node(num_layer, emb_dim, JK = "last", drop_ratio = drop_ratio, residual = True, gnn_name = gnn_name)
        self.separator = separator(
            rationale_gnn_node=rationale_gnn_node, 
            gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim_rat, 2*emb_dim_rat), torch.nn.BatchNorm1d(2*emb_dim_rat), nn_act, torch.nn.Dropout(), torch.nn.Linear(2*emb_dim_rat, 1)),
            nn=None
            )
        rep_dim = emb_dim
        if use_linear_predictor:
            self.predictor = torch.nn.Linear(rep_dim, self.num_tasks)
        else:
            self.predictor = torch.nn.Sequential(torch.nn.Linear(rep_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), nn_act, torch.nn.Dropout(), torch.nn.Linear(2*emb_dim, self.num_tasks))

    def shuffle_batch(self, xc):
        num = xc.shape[0]
        l = [i for i in range(num)]
        random.shuffle(l)
        random_idx = torch.tensor(l)
        x = xc[random_idx]
        return x
    def forward(self, batched_data,cluster_one_hot=None,second_cluster_one_hot=None,cluster_centers=None):
        # print(batched_data)
        # print(batched_data.size())

        h_node = self.graph_encoder(batched_data)
        h_r, h_env, r_node_num, env_node_num = self.separator(batched_data, h_node)

        shuffle_env = self.shuffle_batch(h_env)
        combine_rationale = h_r + 0.8*shuffle_env
        loss_infonce = self.infonce(h_r,combine_rationale)

        self.x_samples = h_r
        self.y_samples = h_env
        pred_rem = self.predictor(h_r)
        pred_rep = self.predictor(combine_rationale)
        loss_reg =  torch.abs(r_node_num / (r_node_num + env_node_num) - self.gamma  * torch.ones_like(r_node_num)).mean()
        output = {'pred_rep': pred_rep, 'pred_rem': pred_rem, 'loss_reg':loss_reg,'loss_infonce':loss_infonce}
        return output
    
    def eval_forward(self, batched_data):
        h_node = self.graph_encoder(batched_data)
        h_r, _, _, _ = self.separator(batched_data, h_node)
        pred_rem = self.predictor(h_r)
        return pred_rem 



class separator(torch.nn.Module):
    def __init__(self, rationale_gnn_node, gate_nn, nn=None):
        super(separator, self).__init__()
        self.rationale_gnn_node = rationale_gnn_node
        self.gate_nn = gate_nn
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.rationale_gnn_node)
        reset(self.gate_nn)
        reset(self.nn)

    def forward(self, batched_data, h_node, size=None):
        x = self.rationale_gnn_node(batched_data)
        
        batch = batched_data.batch
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size

        gate = self.gate_nn(x).view(-1, 1)

        h_node = self.nn(h_node) if self.nn is not None else h_node
        assert gate.dim() == h_node.dim() and gate.size(0) == h_node.size(0)
        gate = torch.sigmoid(gate)
        
        h_out = scatter_add(gate * h_node, batch, dim=0, dim_size=size)
        
        c_out = scatter_add((1 - gate) * h_node, batch, dim=0, dim_size=size)
        
        r_node_num = scatter_add(gate, batch, dim=0, dim_size=size)
        
        
        env_node_num = scatter_add((1 - gate), batch, dim=0, dim_size=size)
        
        return h_out, c_out, r_node_num + 1e-8 , env_node_num + 1e-8 

    def eval_forward(self, batched_data, h_node, size=None):
        x = self.rationale_gnn_node(batched_data)
        
        batch = batched_data.batch
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size
        
        gate = self.gate_nn(x).view(-1, 1)
        
        h_node = self.nn(h_node) if self.nn is not None else h_node
        assert gate.dim() == h_node.dim() and gate.size(0) == h_node.size(0)
        gate = torch.sigmoid(gate)

        return gate

class CLUB_NCE(nn.Module):
    def __init__(self, emb_dim = 300):
        super(CLUB_NCE, self).__init__()
        lstm_hidden_dim = emb_dim//2
        
        self.F_func = nn.Sequential(nn.Linear(lstm_hidden_dim*4, lstm_hidden_dim*2),
                                    #nn.Dropout(p=0.2),
                                    nn.ReLU(),
                                    nn.Linear(lstm_hidden_dim*2, 1),
                                    nn.Softplus())
    
    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        # print(y_samples.size())
        # print(x_samples.size())
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))#

        T0 = self.F_func(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim = -1))  #[sample_size, sample_size, 1]

        lower_bound = T0.mean() - (T1.logsumexp(dim = 1).mean() - np.log(sample_size)) 
        upper_bound = T0.mean() - T1.mean()
        
        return lower_bound, upper_bound

class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    
    def __init__(self, emb_dim = 300):
        super(CLUB, self).__init__()
        lstm_hidden_dim = emb_dim

        self.p_mu = nn.Sequential(nn.Linear(lstm_hidden_dim, lstm_hidden_dim//2),
                                nn.ReLU(),
                                nn.Linear(lstm_hidden_dim//2, lstm_hidden_dim))
       
        self.p_logvar = nn.Sequential(nn.Linear(lstm_hidden_dim, lstm_hidden_dim//2),
                                       nn.ReLU(),
                                       nn.Linear(lstm_hidden_dim//2, lstm_hidden_dim),
                                       nn.Tanh())
        

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def forward(self, x_samples, y_samples): 
        
        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2./logvar.exp()  
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 
        
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        lld = (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)

        return lld , upper_bound



        
class L1OutUB(nn.Module):  # naive upper bound
    def __init__(self,emb_dim = 300):
        super(L1OutUB, self).__init__()
        lstm_hidden_dim = emb_dim
        self.p_mu = nn.Sequential(nn.Linear(lstm_hidden_dim, lstm_hidden_dim//2),
                                nn.ReLU(),
                                nn.Linear(lstm_hidden_dim//2, lstm_hidden_dim))
       
        self.p_logvar = nn.Sequential(nn.Linear(lstm_hidden_dim, lstm_hidden_dim//2),
                                       nn.ReLU(),
                                       nn.Linear(lstm_hidden_dim//2, lstm_hidden_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def log_sum_exp(self,value):
        
        m, _ = torch.max(value, dim=0, keepdim=True)
        value0 = value - m
        
        m = m.squeeze(0)
        return m + torch.log(torch.sum(torch.exp(value0),
                                        dim=0, keepdim=False))
      
    def forward(self, x_samples, y_samples): 
        batch_size = y_samples.shape[0]
        mu, logvar = self.get_mu_logvar(x_samples)

        positive = (- (mu - y_samples)**2 /2./logvar.exp() - logvar/2.).sum(dim = -1) #[nsample]

        mu_1 = mu.unsqueeze(1)          # [nsample,1,dim]
        logvar_1 = logvar.unsqueeze(1)
        y_samples_1 = y_samples.unsqueeze(0)            # [1,nsample,dim]
        all_probs =  (- (y_samples_1 - mu_1)**2/2./logvar_1.exp()- logvar_1/2.).sum(dim = -1)  #[nsample, nsample]

        diag_mask =  torch.ones([batch_size]).diag().unsqueeze(-1) * (-20.)
        device = x_samples.device
        diag_mask = diag_mask.to(device)
        negative = self.log_sum_exp(all_probs + diag_mask) - np.log(batch_size-1.) #[nsample]
      
        upper  = (positive - negative).mean()
        loglikeli = (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
        # return loglikeli, upper
        return loglikeli, upper




class InfoNCE(nn.Module):
    def __init__(self, emb_dim = 300):
        super(InfoNCE, self).__init__()
        print('InfoNCE')
        lstm_hidden_dim = emb_dim//2
        
        self.F_func = nn.Sequential(nn.Linear(lstm_hidden_dim*4, lstm_hidden_dim*2),
                                    #nn.Dropout(p=0.2),
                                    nn.ReLU(),
                                    nn.Linear(lstm_hidden_dim*2, 1),
                                    nn.Softplus())
                                    
    def forward(self, x_samples, y_samples): 
        sample_size = y_samples.shape[0]
        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))#
        T0 = self.F_func(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim = -1))  #[sample_size, sample_size, 1]
        lower_bound = T0.mean() - (T1.logsumexp(dim = 1).mean() - np.log(sample_size))    
        return -lower_bound

