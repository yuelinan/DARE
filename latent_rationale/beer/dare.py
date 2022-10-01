import os
import time
import datetime
import json
import numpy as np
import random
import torch
import torch.optim
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, \
    ExponentialLR
from torch.utils.data import DataLoader,Dataset
from latent_rationale.beer.constants import PAD_TOKEN
from latent_rationale.beer.models.model_helpers import build_model
from latent_rationale.beer.vocabulary import Vocabulary
from latent_rationale.beer.models.rl import CLUB_NCE
from latent_rationale.beer.util import \
    get_args, prepare_minibatch, get_minibatch, \
    print_parameters, beer_reader, beer_annotations_reader, load_embeddings, \
    initialize_model_
from latent_rationale.beer.evaluate import \
    evaluate_rationale, get_examples, evaluate_loss,evaluate_dev_percent
from latent_rationale.common.util import make_kv_string
import logging
from tqdm import tqdm

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO,
                    filemode='w')
logger = logging.getLogger(__name__)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def random_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return
def train():
    
    cfg = get_args()
    cfg = vars(cfg)
    
    print("device:", device)

    for k, v in cfg.items():
        logger.info("{:20} : {:10}".format(k, str(v)))

    num_iterations = cfg["num_iterations"]
    print_every = cfg["print_every"]
    eval_every = cfg["eval_every"]
    batch_size = cfg["batch_size"]
    eval_batch_size = cfg.get("eval_batch_size", batch_size)
    aspect = cfg["aspect"]

    if aspect > -1:
        assert "aspect"+str(aspect) in cfg["train_path"], \
            "chosen aspect does not match train file"
        assert "aspect"+str(aspect) in cfg["dev_path"], \
            "chosen aspect does not match dev file"

    print("Loading data")

    train_data = list(beer_reader(
        cfg["train_path"], aspect=cfg["aspect"], max_len=cfg["max_len"]))
    dev_data = list(beer_reader(
        cfg["dev_path"], aspect=cfg["aspect"], max_len=cfg["max_len"]))
    test_data = beer_annotations_reader(cfg["test_path"], aspect=cfg["aspect"])

    print("train", len(train_data))
    print("dev", len(dev_data))
    print("test", len(test_data))

    iters_per_epoch = len(train_data) // batch_size

    if eval_every == -1:
        eval_every = iters_per_epoch
        print("eval_every set to 1 epoch = %d iters" % eval_every)

    if num_iterations < 0:
        num_iterations = -num_iterations * iters_per_epoch
        print("num_iterations set to %d iters" % num_iterations)

    example = dev_data[0]
    print("First train example tokens:", example.tokens)
    print("First train example scores:", example.scores)

    print("Loading pre-trained word embeddings")
    vocab = Vocabulary()
    vectors = load_embeddings(cfg["embeddings"], vocab)

    Gmodel = build_model(cfg["model"], vocab, cfg=cfg)
    Dmodel = CLUB_NCE()

    initialize_model_(Gmodel)
    initialize_model_(Dmodel)

    with torch.no_grad():
        Gmodel.embed.weight.data.copy_(torch.from_numpy(vectors))
        print("Embeddings fixed: {}".format(cfg["fix_emb"]))
        Gmodel.embed.weight.requires_grad = not cfg["fix_emb"]

    Gmodel = Gmodel.to(device)
    Dmodel = Dmodel.to(device)
    optimizer = Adam(Gmodel.parameters(), lr=cfg["lr"],
                     weight_decay=cfg["weight_decay"])
    D_optimizer = Adam(Dmodel.parameters(), lr=cfg["lr"],
                     weight_decay=cfg["weight_decay"])

    if cfg["scheduler"] == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=cfg["lr_decay"],
            patience=cfg["patience"],
            threshold=cfg["threshold"], threshold_mode='rel',
            cooldown=cfg["cooldown"], verbose=True, min_lr=cfg["min_lr"])
    elif cfg["scheduler"] == "exponential":
        scheduler = ExponentialLR(optimizer, gamma=cfg["lr_decay"])
    elif cfg["scheduler"] == "multistep":
        milestones = cfg["milestones"]
        print("milestones (epoch):", milestones)
        scheduler = MultiStepLR(
            optimizer, milestones=milestones, gamma=cfg["lr_decay"])
    else:
        raise ValueError("Unknown scheduler")

    
    start = time.time()
    iter_i = 0
    epoch = 0
    best_eval = 1e12
    best_iter = 0
    pad_idx = vocab.w2i[PAD_TOKEN]

    if cfg.get("ckpt", ""):
        print("Resuming from ckpt: {}".format(cfg["ckpt"]))
        ckpt = torch.load(cfg["ckpt"])
        Gmodel.load_state_dict(ckpt["state_dict"])
        best_iter = ckpt["best_iter"]
        best_eval = ckpt["best_eval"]
        iter_i = ckpt["best_iter"]
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        cur_lr = scheduler.optimizer.param_groups[0]["lr"]
        print("# lr = ", cur_lr)

    for epoch in range(cfg['epochs']):  
        logger.info('epcch：'+str(epoch))
        for batch in get_minibatch(train_data, batch_size=batch_size,
                                    shuffle=True):
            Gmodel.train()
            Dmodel.train()

            x, targets, _ = prepare_minibatch(batch, Gmodel.vocab, device=device)

            output = Gmodel(x)
            lower_bound, upper_bound = Dmodel(Gmodel.x_samples.detach(),Gmodel.y_samples.detach())
            Dloss = -lower_bound

            D_optimizer.zero_grad()
            Dloss.backward()
            D_optimizer.step()
            
            output = Gmodel(x)
            lower_bound, upper_bound = Dmodel(Gmodel.x_samples,Gmodel.y_samples)
            mask = (x != pad_idx)
            Gloss, loss_optional = Gmodel.get_loss(output, targets, mask=mask)
            
            Gloss = Gloss + cfg['upper_bound']*upper_bound
            optimizer.zero_grad()
            Gloss.backward()
            torch.nn.utils.clip_grad_norm_(Gmodel.parameters(),
                                        max_norm=cfg["max_grad_norm"])
            optimizer.step()

            iter_i += 1

        cur_lr = scheduler.optimizer.param_groups[0]["lr"]
        if cur_lr > cfg["min_lr"]:
            if isinstance(scheduler, MultiStepLR):
                scheduler.step()
            elif isinstance(scheduler, ExponentialLR):
                scheduler.step()

        cur_lr = scheduler.optimizer.param_groups[0]["lr"]
        print("#lr", cur_lr)
        scheduler.optimizer.param_groups[0]["lr"] = max(cfg["min_lr"],cur_lr)

        Gmodel.eval()
        Dmodel.eval()
        print("Evaluating..", str(datetime.datetime.now()))

        dev_eval = evaluate_loss(
            Gmodel, dev_data, batch_size=eval_batch_size,
            device=device, cfg=cfg)

        if hasattr(Gmodel, "z"):
            path = os.path.join(
                cfg["save_path"],
                "rationales_i{:08d}_e{:03d}.txt".format(iter_i, epoch))
            test_precision, test_macro_prec, recall, macro_recall, percents = evaluate_rationale(
                Gmodel, test_data, aspect=aspect,
                device=device, path=path, batch_size=eval_batch_size)
            
            percents_dev = evaluate_dev_percent(Gmodel, dev_data,device=device, path=path, batch_size=eval_batch_size)
            logger.info('percents_dev'+str(percents_dev))

            test_eval = {}
            test_eval["precision"] = test_precision
            test_eval["macro_precision"] = test_macro_prec

            test_eval["recall"] = recall
            test_eval["macro_recall"] = macro_recall

        
            test_s = make_kv_string(test_eval)

            logger.info("best model  test {}".format(test_s))
        else:
            logger.info('hasattr is not')
            test_eval["precision"] = 0.
            test_eval["macro_precision"] = 0.

        compare_obj = dev_eval["obj"] if "obj" in dev_eval \
            else dev_eval["loss"]
        dynamic_threshold = best_eval * (1 - cfg["threshold"])

        if compare_obj < dynamic_threshold and iter_i > 5 * iters_per_epoch and percents_dev<0.16:
            print("new highscore", compare_obj)
            best_eval = compare_obj
            best_iter = iter_i
            if not os.path.exists(cfg["save_path"]):
                os.makedirs(cfg["save_path"])

            
            ckpt = {
                "state_dict": Gmodel.state_dict(),
                "cfg": cfg,
                "best_eval": best_eval,
                "best_iter": best_iter,
                "optimizer_state_dict": optimizer.state_dict()
            }

            path = os.path.join(cfg["save_path"], "Gmodel.pt")
            torch.save(ckpt, path)

        if isinstance(scheduler, ReduceLROnPlateau):
            if iter_i > 5 * iters_per_epoch:
                scheduler.step(compare_obj)


        

if __name__ == "__main__":
    random_seed()
    train()
