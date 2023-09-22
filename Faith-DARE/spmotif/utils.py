import torch
import argparse
from sklearn.metrics import r2_score
from kmeans_pytorch import kmeans
from sklearn.cluster import KMeans
import torch_geometric
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from sklearn.metrics import roc_auc_score
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import itertools
def get_args():
    parser = argparse.ArgumentParser(description='Graph rationalization with Environment-based Augmentation')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    # model
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='dimensionality of hidden units in GNNs (default: 128)')
    parser.add_argument('--use_linear_predictor', default=False, action='store_true',
                        help='Use Linear predictor')
    parser.add_argument('--gamma', type=float, default=0.4,
                        help='size ratio to regularize the rationale subgraph (default: 0.4)')

    # training
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--patience', type=int, default=50,
                        help='patience for early stop (default: 50)')
    parser.add_argument('--beta_dare', type=float, default=1.0,)                    
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='Learning rate (default: 1e-2)')
    parser.add_argument('--pred_rep', type=float, default=1.0)
    parser.add_argument('--loss_infonce', type=float, default=1.0)
    parser.add_argument('--l2reg', type=float, default=5e-6,
                        help='L2 norm (default: 5e-6)')
    parser.add_argument('--bias', type=float, default=0.333,
                        help='L2 norm (default: 5e-6)')
    parser.add_argument('--use_lr_scheduler', default=False, action='store_true',
                        help='Use learning rate scheduler CosineAnnealingLR')
    parser.add_argument('--use_clip_norm', default=False, action='store_true',
                        help='Use learning rate clip norm')
    parser.add_argument('--path_list', nargs="+", default=[1,4],
                        help='path for alternative optimization')
    parser.add_argument('--initw_name', type=str, default='default',
                        choices=['default','orthogonal','normal','xavier','kaiming'],
                        help='method name to initialize neural weights')

    parser.add_argument('--dataset', type=str, default="ogbg-molbbbp",
                        help='dataset name (default: ogbg-molhiv)')
    parser.add_argument('--trails', type=int, default=5,
                        help='numer of experiments (default: 5)')
    parser.add_argument('--by_default', default=False, action='store_true',
                        help='use default configuration for hyperparameters')


    parser.add_argument('--beta_infonce', type=float, default=0.001)
    parser.add_argument('--beta_club', type=float, default=0.001)
    parser.add_argument('--date', type=str, default='0402')
    parser.add_argument('--model_name', type=str, default='Graph_Student', help='model name')
    parser.add_argument('--train_type', type=str, default='student', help='model name')

    args = parser.parse_args()
    
    return args

cls_criterion = torch.nn.BCEWithLogitsLoss()
CELoss = torch.nn.CrossEntropyLoss(reduction="mean")
reg_criterion = torch.nn.MSELoss()


def train_dare_da(args, model, device, loader, optimizers, task_type, optimizer_name,loss_logger,Dmodel,D_optimizer):
    optimizer = optimizers[optimizer_name]
    Dmodel = Dmodel.to(device)
    model.train()
    Dmodel.train()
    # if optimizer_name == 'predictor':
    #     set_requires_grad([model.graph_encoder, model.predictor,model.infonce], requires_grad=True)
    #     set_requires_grad([model.separator], requires_grad=False)
    # if optimizer_name == 'separator':
    #     set_requires_grad([model.separator], requires_grad=True)
    #     set_requires_grad([model.graph_encoder,model.predictor,model.infonce], requires_grad=False)
        
    for step, batch in enumerate(loader):
        # if step>10:
        #     break
        batch = batch.to(device)
        second_batch = copy.deepcopy(batch)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            
            optimizer.zero_grad()
            pred = model(batch)
            lower_bound, upper_bound = Dmodel(model.x_samples.detach(),model.y_samples.detach())
            Dloss = -lower_bound

            D_optimizer.zero_grad()
            Dloss.backward()
            D_optimizer.step()

            pred = model(second_batch)
            lower_bound, upper_bound = Dmodel(model.x_samples,model.y_samples)
            
            loss =  CELoss(pred['pred_rem'], batch.y)
            loss +=  args.pred_rep*CELoss(pred['pred_rep'], batch.y)

            loss += 0.001*upper_bound
            loss += args.loss_infonce*pred['loss_infonce']
            loss_logger.append(loss.cpu().detach().numpy().tolist())
            if optimizer_name == 'separator': 
                loss += pred['loss_reg']

            loss.backward()
            if args.use_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

def eval_spmotif(args, model, device, loader):
    model.eval()
    y_true = []
    y_pred = []
    acc = 0
    for step, batch in enumerate(loader):
        
        batch = batch.to(device)
        
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model.eval_forward(batch)
    
                acc += torch.sum(pred.argmax(-1).view(-1) == batch.y.view(-1))
    acc = float(acc) / len(loader.dataset)

    return  [acc]

def precision_at_k(y_true, y_pred, k):

    sorted_indices = np.argsort(y_pred)[::-1]

    top_k_indices = sorted_indices[:k]

    num_correct = np.sum(np.array(y_true)[top_k_indices])

    precision_at_k = num_correct / k
    return precision_at_k

def eval_spmotif_explain(args, model, device, loader):
    model.eval()
    y_true = []
    y_pred = []
    acc = 0
    auc_all = []
    precision_at_5_all = []
    for step, batch in enumerate(loader):

        batch = batch.to(device)
        edge_index = batch.edge_index.cpu()
        true_labels = [0 for i in range(batch.x.size(0))]
        for index, data in enumerate(batch.edge_gt_att):
            if data==1:
                true_labels[edge_index[0][index]] = 1
                true_labels[edge_index[1][index]] = 1
                
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred,gate = model.eval_explain_forward(batch)
                gate = gate.squeeze(1)

                y_true.extend(true_labels)
                y_pred.extend(gate.cpu().numpy().tolist())
                auc_all.append(  roc_auc_score(true_labels, gate.cpu().numpy().tolist())   )
                precision_at_5_all.append( precision_at_k(true_labels, gate.cpu().numpy().tolist() ,5)   )

    # auc = roc_auc_score(y_true, y_pred)
    return  np.mean(precision_at_5_all), np.mean(auc_all)

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'default':
                pass
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad