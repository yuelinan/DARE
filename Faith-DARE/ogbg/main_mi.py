
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import datetime
## dataset
from sklearn.model_selection import train_test_split
from dataset import PolymerRegDataset
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import random
## training
from model import InfoNCE, L1OutUB,Graph_DARE_DA_MI,CLUB_NCE
from utils_mi import init_weights, get_args, eval_test,train_dare_da

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
                    level = logging.INFO,
                    )

logger = logging.getLogger(__name__)

def main(args):
    print(args)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if args.dataset.startswith('ogbg'):
        dataset = PygGraphPropPredDataset(name = args.dataset, root='data')
        
        split_idx = dataset.get_idx_split()
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = 0)
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = 0)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = 0)
        evaluator = Evaluator(args.dataset)

    elif args.dataset.startswith('plym'):
        dataset = PolymerRegDataset(name = args.dataset.split('-')[1], root='data') # PolymerRegDataset
        full_idx = list(range(len(dataset)))
        train_ratio = 0.6
        valid_ratio = 0.1
        test_ratio = 0.3
        train_index, test_index, _, _ = train_test_split(full_idx, full_idx, test_size=test_ratio, random_state=42)
        train_index, val_index, _, _ = train_test_split(train_index, train_index, test_size=valid_ratio/(valid_ratio+train_ratio), random_state=42)

        train_index = torch.LongTensor(train_index)
        val_index = torch.LongTensor(val_index)
        test_index = torch.LongTensor(test_index)

        train_loader = DataLoader(dataset[train_index], batch_size=args.batch_size, shuffle=True, num_workers = 0)
        valid_loader = DataLoader(dataset[val_index], batch_size=args.batch_size, shuffle=False, num_workers = 0)
        test_loader = DataLoader(dataset[test_index], batch_size=args.batch_size, shuffle=False, num_workers = 0)
        evaluator = Evaluator('ogbg-molesol') # RMSE metric
    n_train_data, n_val_data, n_test_data = len(train_loader.dataset), len(valid_loader.dataset), float(len(test_loader.dataset))
    logger.info(f"# Train: {n_train_data}  #Test: {n_test_data} #Val: {n_val_data}")
    print(dataset.num_tasks)
    model = Graph_DARE_DA_MI( gnn_type = args.gnn, num_tasks = dataset.num_tasks, num_layer = args.num_layer,
                         emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, gamma=args.gamma, use_linear_predictor = args.use_linear_predictor).to(device)   
                          
    init_weights(model, args.initw_name, init_gain=0.02)

    Dmodel = CLUB_NCE(args.emb_dim)
    # Dmodel = CLUB(args.emb_dim)
    opt_separator = optim.Adam(model.separator.parameters(), lr=args.lr, weight_decay=args.l2reg)
    opt_predictor = optim.Adam(list(model.graph_encoder.parameters())+list(model.predictor.parameters()) +list(model.infonce.parameters()), lr=args.lr, weight_decay=args.l2reg)

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
    for epoch in range(args.epochs):
        # if epoch>2:break
        # print("=====Epoch {}".format(epoch))
        path = epoch % int(args.path_list[-1])
        if path in list(range(int(args.path_list[0]))):
            optimizer_name = 'separator' 
        elif path in list(range(int(args.path_list[0]), int(args.path_list[1]))):
            optimizer_name = 'predictor'
        model.train()
        train_dare_da(args, model, device, train_loader, optimizers, dataset.task_type, optimizer_name,loss_logger,Dmodel,D_optimizer)

        if schedulers != None:
            schedulers[optimizer_name].step()
        # train_perf = eval(args, model, device, train_loader, evaluator)[0]
        model.eval()
        valid_perf = eval_test(args, model, device, valid_loader, evaluator)[0]
        test_logger_perfs = eval_test(args, model, device, test_loader, evaluator)[0]
        valid_logger.append(valid_perf)
        test_logger.append(test_logger_perfs)
        update_test = False
        if epoch != 0:
            if 'classification' in dataset.task_type and valid_perf >  best_valid_perf:
                update_test = True
            elif 'classification' not in dataset.task_type and valid_perf <  best_valid_perf:
                update_test = True
        if update_test or epoch == 0:
            best_valid_perf = valid_perf
            cnt_wait = 0
            best_epoch = epoch
            test_perfs = eval_test(args, model, device, test_loader, evaluator)
            if args.dataset.startswith('ogbg'):
                test_auc  = test_perfs[0]
                logger.info("=====Epoch {}, Metric: {}, Validation: {}, Test: {}".format(epoch, 'AUC', valid_perf, test_auc))
                # logger.info("=====Epoch {}".format(epoch))
                # logger.info({'Metric': 'AUC', 'Validation': valid_perf, 'Test': test_auc})
            else:
                test_rmse, test_r2 = test_perfs[0], test_perfs[1]
                logger.info({'Metric': 'RMSE', 'Validation': valid_perf, 'Test': test_rmse, 'Test R2': test_r2})
        else:
            # print({'Train': train_perf, 'Validation': valid_perf})
            cnt_wait += 1
            if cnt_wait > args.patience:
                break
    logger.info('Finished training! Results from epoch {} with best validation {}.'.format(best_epoch, best_valid_perf))



    if args.dataset.startswith('ogbg'):
        logger.info('Test auc: {}'.format(test_auc))
        return [best_valid_perf, test_auc]
    if args.dataset.startswith('plym'):
        logger.info('Test rmse: {}, Test r2: {} \n'.format(test_rmse, test_r2))
        return [best_valid_perf, test_rmse, test_r2]

    

def config_and_run(args):
    
    if args.by_default:
        if args.dataset == 'ogbg-molhiv':
            args.gamma = 0.1
            args.batch_size = 512
            args.initw_name = 'orthogonal'

            args.lr = 1e-3
            args.l2reg = 1e-5
            args.epochs = 100
            args.num_layer = 3
            args.use_clip_norm = True
            args.path_list=[2, 4]

        if args.dataset == 'ogbg-molbbbp':
            args.l2reg = 5e-6
            args.lr = 1e-3
            args.initw_name = 'orthogonal'
            args.num_layer = 2
            args.emb_dim = 128
            args.batch_size = 256 
            args.use_lr_scheduler = True 
            args.gamma = 0.5

        if args.dataset == 'ogbg-molsider':
            args.l2reg = 1e-4
            args.lr = 1e-3
            args.patience = 100
            args.gamma = 0.5
            args.num_layer =  5
            args.epochs = 400

        if args.dataset == 'ogbg-moltoxcast':
            args.patience = 50
            args.lr = 1e-3
            args.epochs = 150
            args.l2reg = 1e-5
            args.gamma = 0.5
            args.num_layer = 2

    for k, v in vars(args).items():
        logger.info("{:20} : {:10}".format(k, str(v)))

    args.plym_prop = 'none' if args.dataset.startswith('ogbg') else args.dataset.split('-')[1].split('_')[0]
    if args.dataset.startswith('ogbg'):
        results = {'valid_auc': [], 'test_auc': []}
    else:
        results = {'valid_rmse': [], 'test_rmse': [], 'test_r2':[]}
    for seed in range(args.trails):
        if args.dataset.startswith('plym'):
            valid_rmse, test_rmse, test_r2 = main(args)
            results['test_r2'].append(test_r2)
            results['test_rmse'].append(test_rmse)
            results['valid_rmse'].append(valid_rmse)
        else:
            set_seed(seed)
            valid_auc, test_auc = main(args)
            results['valid_auc'].append(valid_auc)
            results['test_auc'].append(test_auc)
    for mode, nums in results.items():
        logger.info('{}: {:.4f}+-{:.4f} {}'.format(
            mode, np.mean(nums), np.std(nums), nums))
        print('{}: {:.4f}+-{:.4f} {}'.format(mode, np.mean(nums), np.std(nums), nums))

if __name__ == "__main__":
    args = get_args()
    config_and_run(args)
    




