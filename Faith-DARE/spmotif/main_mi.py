
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os.path as osp
import random
## dataset
from sklearn.model_selection import train_test_split
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from spmotif_dataset import SPMotif
## training
from model import Graph_DARE_DA_MI,CLUB_NCE
from utils import init_weights, get_args, eval_spmotif,train_dare_da,eval_spmotif_explain

import logging

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


args_first = get_args()



logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO,)

logger = logging.getLogger(__name__)

def main(args):
    print(args)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")


    datadir = './data/'
    bias = args.bias
    train_dataset = SPMotif(osp.join(datadir, f'SPMotif-{bias}/'), mode='train')
    val_dataset = SPMotif(osp.join(datadir, f'SPMotif-{bias}/'), mode='val')
    test_dataset = SPMotif(osp.join(datadir, f'SPMotif-{bias}/'), mode='test')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    explain_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    n_train_data, n_val_data, n_test_data = len(train_loader.dataset), len(valid_loader.dataset), float(len(test_loader.dataset))
    logger.info(f"# Train: {n_train_data}  #Test: {n_test_data} #Val: {n_val_data}")

    model = Graph_DARE_DA_MI(args, gnn_type = args.gnn, num_tasks = 3, num_layer = args.num_layer,
                         emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, gamma=args.gamma, use_linear_predictor = args.use_linear_predictor).to(device)    
    init_weights(model, args.initw_name, init_gain=0.02)

    Dmodel = CLUB_NCE(args.emb_dim)
    
    opt_separator = optim.Adam(model.separator.parameters(), lr=args.lr, weight_decay=args.l2reg)
    
    opt_predictor = optim.Adam(list(model.graph_encoder.parameters())+list(model.predictor.parameters()) +list(model.infonce.parameters()) + list(model.node_enoder.parameters()), lr=args.lr, weight_decay=args.l2reg)

    D_optimizer = optim.Adam(Dmodel.parameters(), lr=args.lr, weight_decay=args.l2reg)

    optimizers = {'separator': opt_separator, 'predictor': opt_predictor}
    if args.use_lr_scheduler:
        schedulers = {}
        for opt_name, opt in optimizers.items():
            schedulers[opt_name] = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-4)
    else:
        schedulers = None
    cnt_wait = 0
    best_epoch = 0
    loss_logger = []
    valid_logger = []
    test_logger = []
    task_type = ["classification"]
    for epoch in range(args.epochs):
        # if epoch>2:break
        # print("=====Epoch {}".format(epoch))
        path = epoch % int(args.path_list[-1])
        if path in list(range(int(args.path_list[0]))):
            optimizer_name = 'separator' 
        elif path in list(range(int(args.path_list[0]), int(args.path_list[1]))):
            optimizer_name = 'predictor'
        if args.train_type == 'student':
            train_dare_da(args, model, device, train_loader, optimizers,task_type, optimizer_name,loss_logger,Dmodel,D_optimizer)

        if schedulers != None:
            schedulers[optimizer_name].step()
        train_perf = eval_spmotif(args, model, device, train_loader)[0]
        valid_perf = eval_spmotif(args, model, device, valid_loader)[0]
        test_logger_perfs = eval_spmotif(args, model, device, test_loader)[0]
        valid_logger.append(valid_perf)
        test_logger.append(test_logger_perfs)
        update_test = False
        if epoch != 0:
            if valid_perf == 1.0:
                update_test = False
            elif 'classification' in task_type and valid_perf >  best_valid_perf:
                update_test = True
            elif 'classification' not in task_type and valid_perf <  best_valid_perf:
                update_test = True
        if update_test or epoch == 0:
            best_valid_perf = valid_perf
            cnt_wait = 0
            best_epoch = epoch
            test_perfs = eval_spmotif(args, model, device, test_loader)
            
            
            test_auc  = test_perfs[0]
            # logger.info("=====Epoch {}, Metric: {}, Validation: {}, Test: {}, precision_at_5:{}, explain_auc:{}".format(epoch, 'AUC', valid_perf, test_auc,precision_at_5, explain_auc))
            logger.info("=====Epoch {}, Metric: {}, Validation: {}, Test: {}".format(epoch, 'AUC', valid_perf, test_auc))
        
        else:
            # print({'Train': train_perf, 'Validation': valid_perf})
            cnt_wait += 1
            if cnt_wait > args.patience:
                break
    precision_at_5, explain_auc = eval_spmotif_explain(args, model, device, explain_loader)       


    print('Test auc: {}'.format(test_auc))
    return [best_valid_perf, test_auc,precision_at_5, explain_auc]

    

def config_and_run(args):
    
    if args.by_default:
        if args.dataset == 'spmotif':
            if args.gnn == 'gin-virtual' or args.gnn == 'gin':
                args.gnn = 'gin'
                args.lr = 0.01
                args.l2reg = 1e-3
                args.gamma = 0.5
                args.num_layer = 2  
                args.batch_size = 32
                args.emb_dim = 64
                args.use_lr_scheduler = True
                args.patience = 40
                args.drop_ratio = 0.3
                args.initw_name = 'orthogonal' 
            if args.gnn == 'gcn-virtual' or args.gnn == 'gcn':
                args.gnn = 'gcn'
                args.lr = 0.01
                args.gamma = 0.5
                args.patience = 40
                args.initw_name = 'orthogonal' 
                args.num_layer = 2
                args.emb_dim = 64
                args.batch_size = 32

    for k, v in vars(args).items():
        logger.info("{:20} : {:10}".format(k, str(v)))

    args.plym_prop = 'none' 
    
    results = {'valid_auc': [], 'test_auc': [], 'precision_at_5':[],'explain_auc':[]}
    
    for seed in range(args.trails):
        if args.dataset.startswith('plym'):
            valid_rmse, test_rmse, test_r2 = main(args)
            results['test_r2'].append(test_r2)
            results['test_rmse'].append(test_rmse)
            results['valid_rmse'].append(valid_rmse)
        else:
            set_seed(seed)
            valid_auc, test_auc,precision_at_5, explain_auc = main(args)
            results['valid_auc'].append(valid_auc)
            results['test_auc'].append(test_auc)
            results['precision_at_5'].append(precision_at_5)
            results['explain_auc'].append(explain_auc)
    for mode, nums in results.items():
        logger.info('{}: {:.4f}+-{:.4f} {}'.format(
            mode, np.mean(nums), np.std(nums), nums))
        print('{}: {:.4f}+-{:.4f} {}'.format(
            mode, np.mean(nums), np.std(nums), nums))
    
if __name__ == "__main__":
    args = get_args()
    args_first = get_args()                   
    filename='/data/lnyue/GREA-main/spmotif_codes/logger_hyp_mi/'+str(args_first.date) +args_first.dataset+'_beta_dare_'+str(args_first.beta_dare) +str(args_first.bias)+ '_'+args_first.gnn+'__pred_rep_'+str(args_first.pred_rep)+'_loss_infonce_'+str(args_first.loss_infonce)+ '.log'
    config_and_run(args)
    print(filename)
    




