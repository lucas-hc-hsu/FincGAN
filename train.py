#!/usr/bin/env python
# coding: utf-8


import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" -> GPU ID: 1 Quadro RTX A6000
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" -> GPU ID: 5 Quadro RTX A6000
# os.environ["CUDA_VISIBLE_DEVICES"] = "2" -> GPU ID: 6 Quadro RTX A6000
# os.environ["CUDA_VISIBLE_DEVICES"] = "3" -> GPU ID: 2 RTX 3090
# os.environ["CUDA_VISIBLE_DEVICES"] = "4" -> GPU ID: 3 RTX 3090
# os.environ["CUDA_VISIBLE_DEVICES"] = "5" -> GPU ID: 0 RTX 2080 Ti
# os.environ["CUDA_VISIBLE_DEVICES"] = "6" -> GPU ID: 4 RTX 2080 Ti
# os.environ["CUDA_VISIBLE_DEVICES"] = "7" -> GPU ID: 7 RTX 2080 Ti
import torch
import torch.nn as nn
import torch.nn.functional as F

from hgt_model import HGT, Generator, latent_dim, emb_dim
from logger import get_logger
import argparse
import logging

import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve, auc
from dgl.data.utils import load_graphs, save_graphs

import utils
import sys


parser = argparse.ArgumentParser(description='Training GNN on ogbn-products benchmark')
parser.add_argument('--gpu_id', type=int, default=0)
# parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--seed', nargs="+", type=int, default=[0, 30])
parser.add_argument('--data_dir', type=str, default="graph/music_instrument_25.bin", help="directory of original grpah data")
parser.add_argument('--graph_dir', type=str, default="graph/", help="directory to save generated graph during training")
# parser.add_argument('--out_dir', type=str, default="model/")
parser.add_argument('--tmp_dir', type=str, default="tmp/")
parser.add_argument('--emb_dir', type=str, default="embed/")
parser.add_argument('--target', type=str, default='user')
parser.add_argument('--n_epoch', type=int, default=500)
parser.add_argument('--patient', type=int, default=50)
parser.add_argument('--clip',    type=int, default=1.0)
parser.add_argument('--max_lr',  type=float, default=1e-3)
parser.add_argument('--setting', type=str, default="origin",choices=['origin', 'embedding', 'oversampling', 'reweight', 'smote', 'noise', 'graphsmote', 'gan'])
parser.add_argument('--up', type=float, default=0.99, help="threshold of user-product edge generator")
parser.add_argument('--uu', type=float, default=0.91, help="threshold of user-user edge genertor")
parser.add_argument('--k', type=int, default=1)
# parser.add_argument('--ratio', type=float, default=0.5)
parser.add_argument('--ratio', nargs='+', type=float, default=[0.1007, 0.17, 0.23, 0.29, 0.34, 0.375, 0.41, 0.45, 0.47, 0.5, 0.52, 0.55]) # for test different threshold
parser.add_argument('--verbose', type=int, default=0, help="set 1 to show training details as generating new graph, otherwise 0")
parser.add_argument('--result_dir', type=str, default="results/", help="directory to save training result of each method during training")

''' Use one of the following two lines when parser throws error '''
args = parser.parse_args()
# args = parser.parse_args(args = [])

# Setup logger
logger = get_logger('FincGAN.train', level=logging.DEBUG if args.verbose else logging.INFO)

if args.verbose:
    logger.info(f"Arguments:\n{args}")

if not os.path.exists(args.graph_dir):
    logger.info(f"Directory {args.graph_dir} does not exist, creating...")
    os.makedirs(args.graph_dir)
    logger.info("Directory created successfully")

if not os.path.exists(args.result_dir):
    logger.info(f"Directory {args.result_dir} does not exist, creating...")
    os.makedirs(args.result_dir)
    logger.info("Directory created successfully")

if not os.path.exists(args.emb_dir):
    logger.info(f"Directory {args.emb_dir} does not exist, creating...")
    os.makedirs(args.emb_dir)
    logger.info("Directory created successfully")


# Configure device (GPU if available, otherwise CPU)
cuda = torch.cuda.is_available() and args.gpu_id >= 0
if cuda:
    device = torch.device(f"cuda:{args.gpu_id}")
    logger.info(f"Using GPU: {torch.cuda.get_device_name(args.gpu_id)} (cuda:{args.gpu_id})")
else:
    device = torch.device("cpu")
    logger.info(f"Using CPU for training")


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def train(model, G):
    patient = 0
    best_val_score = torch.tensor(0)
    best_val_acc = torch.tensor(0)
    best_test_acc = torch.tensor(0)
    for epoch in np.arange(args.n_epoch) + 1:
        if patient > 50 and epoch > 150:
            break

        model.train()
        logits = model(G, args.target)

        # The loss is computed only for labeled nodes.
        if args.setting == 'reweight':
            weight = (1 / torch.unique(labels[train_idx], return_counts=True)[1]) * torch.tensor([1, ratio * 10 / 5])
            loss = F.cross_entropy(logits[train_idx], labels[train_idx].to(device), weight=weight.to(device))
        else:
            loss = F.cross_entropy(logits[train_idx], labels[train_idx].to(device))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        scheduler.step()


        # evaluate
        model.eval()
        logits = model(G, args.target)
        pred_score, pred = logits.max(1)

        # transform to score
        logits = torch.sigmoid(logits)
        pred_score[pred == 0] = pred_score[pred == 0]*(-1)
        pred_score = (pred_score + 1)/2
        pred = pred.cpu()

        # calculate metrics
        train_acc = (pred[train_idx] == labels[train_idx]).float().mean()
        val_acc   = (pred[val_idx]   == labels[val_idx]).float().mean()
        test_acc  = (pred[test_idx]  == labels[test_idx]).float().mean()

        # metric for evaluating model on validation set
        val_metric = roc_auc_score(labels[val_idx], pred_score[val_idx].cpu().detach().numpy()) if args.setting == 'embedding' else (pred[val_idx]   == labels[val_idx]).float().mean()
        if best_val_score < val_metric:
            best_val_score = val_metric
            best_val_acc = val_acc
            best_test_acc = test_acc
            patient = 0
            torch.save(model.state_dict(), args.tmp_dir + 'music_hgt_model_' + args.setting + '_ratio_' + str(ratio) + '_seed_' + str(seed) + '.pt')
        else:
            patient = patient + 1

        if args.verbose:
            logger.debug('Epoch: %d LR: %.5f Loss %.4f, Train Acc %.4f, Val Score %.4f (Best %.4f), Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)' % (
                epoch,
                optimizer.param_groups[0]['lr'],
                loss.item(),
                train_acc.item(),
                val_metric.item(),
                best_val_score.item(),
                val_acc.item(),
                best_val_acc.item(),
                test_acc.item(),
                best_test_acc.item(),
            ))


txt_path = args.result_dir + "music_hgt_model_" + args.setting + ".txt"
logger.info(f"Training results will be saved to: {txt_path}")
f = open(txt_path, "w")
f.write(",".join(["method", "seed", "ratio", "prc", "roc", "f1", "precision", "recall", "acc\n"]))
f.close()


percent = [i/10 for i in range(1, 13)]
if args.setting in ['origin', 'embedding']:
    args.ratio = [0.1007]
for j in range(len(args.ratio)):
    ratio = args.ratio[j]

    for seed in range(args.seed[0], args.seed[1]):
        logger.info(f"Training with ratio {args.ratio[j]:.4f}, seed {seed}, method: {args.setting}")
        best_emb_score = 0
        torch.manual_seed(seed)


        # %%
#         device = torch.device("cuda:{}".format(args.gpu_id))
        [G], _ = load_graphs(args.data_dir)
        # [G], _ = load_graphs("./graph/"+'music_graphsmote_0.5.bin')
        # [G], _ = load_graphs("./graph/"+'music_graphsmote_0.5_pretrained.bin')
        # [G], _ = load_graphs("./graph/"+'music_gan_0.5.bin')
        # [G], _ = load_graphs("./graph/"+'music_gan_0.5_pretrained.bin')
        if args.verbose:
            logger.debug(f'Original Graph Information:\n{G}')

        # generate dataset based on setting
        if args.setting in ['origin', 'embedding']:
            pass
        elif args.setting in ["oversampling","smote", "noise", 'graphsmote']:
            if os.path.exists(args.graph_dir + "music_instrument_" + args.setting + '_ratio_' + str(ratio) + '_seed_' + str(seed) + ".bin"):
                [G], _ = load_graphs(args.graph_dir + "music_instrument_" + args.setting + '_ratio_' + str(ratio) + '_seed_' + str(seed) + ".bin")

            elif args.setting == "oversampling":
                graph_name = f"music_instrument_{args.setting}_ratio_{ratio}_seed_{seed}.bin"
                logger.info(f"Graph {graph_name} does not exist, creating...")
                G = utils.oversampling(G, seed, ratio, verbose=args.verbose)
                logger.info(f"Graph {graph_name} created successfully")
                save_graphs(args.graph_dir + graph_name, [G])
                logger.info(f"Graph saved to: {args.graph_dir}")

            elif args.setting == "smote":
                graph_name = f"music_instrument_{args.setting}_ratio_{ratio}_seed_{seed}.bin"
                logger.info(f"Graph {graph_name} does not exist, creating...")
                G = utils.smote(G, seed, ratio, verbose=args.verbose)
                logger.info(f"Graph {graph_name} created successfully")
                save_graphs(args.graph_dir + graph_name, [G])
                logger.info(f"Graph saved to: {args.graph_dir}")

            elif args.setting == "noise":
                graph_name = f"music_instrument_{args.setting}_ratio_{ratio}_seed_{seed}.bin"
                logger.info(f"Graph {graph_name} does not exist, creating...")
                G = utils.noise(G, seed, ratio, verbose=args.verbose)
                logger.info(f"Graph {graph_name} created successfully")
                save_graphs(args.graph_dir + graph_name, [G])
                logger.info(f"Graph saved to: {args.graph_dir}")

            elif args.setting == "graphsmote":
                graph_name = f"music_instrument_{args.setting}_ratio_{ratio}_seed_{seed}.bin"
                logger.info(f"Graph {graph_name} does not exist, creating...")
                G = utils.graphsmote(G, seed, ratio, args.uu, verbose=args.verbose)
                logger.info(f"Graph {graph_name} created successfully")
                save_graphs(args.graph_dir + graph_name, [G])
                logger.info(f"Graph saved to: {args.graph_dir}")

            if args.verbose:
                logger.debug(f'Transformed Graph Information:\n{G}')

        elif args.setting == "gan":
            graph_name = f"music_instrument_{args.setting}_ratio_{ratio}_seed_0.bin"
            if os.path.exists(args.graph_dir + graph_name):
                [G], _ = load_graphs(args.graph_dir + graph_name)
            else:
                logger.info(f"Graph {graph_name} does not exist, creating...")
                G = utils.gan(G, 0, ratio, args.uu, args.up, verbose=args.verbose)
                logger.info(f"Graph {graph_name} created successfully")
                save_graphs(args.graph_dir + graph_name, [G])
                logger.info(f"Graph saved to: {args.graph_dir}")

                if args.verbose:
                    logger.debug(f'Transformed Graph Information:\n{G}')

        # labels
        labels = G.nodes[args.target].data.pop('label')

        # generate train/val/test split
        train_mask = G.nodes[args.target].data.pop('train_mask')
        val_mask = G.nodes[args.target].data.pop('val_mask')
        test_mask = G.nodes[args.target].data.pop('test_mask')
        train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
        val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze()
        test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()

        # extract features
        features = G.nodes[args.target].data['feature']

        node_dict = {}
        edge_dict = {}
        for ntype in G.ntypes:
            node_dict[ntype] = len(node_dict)
        for etype in G.etypes:
            edge_dict[etype] = len(edge_dict)
            G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]

        #  input feature
        vote_idx = [] if args.setting in ['noise', "gan", "graphsmote"] else [19, 20, 21, 22, 23, 24] # invalid features
        use_idx = [i for i in range(256) if i not in vote_idx] if args.setting in ['noise', "gan", "graphsmote"] else [i for i in range(29) if i not in vote_idx]


        for ntype in G.ntypes:
            emb = G.nodes[ntype].data.pop('feature')
            emb = nn.Parameter(emb[:, use_idx], requires_grad = False)
            G.nodes[ntype].data['inp'] = emb
        # Keep graph on CPU for DGL CPU version, only move tensors to GPU
        # G = G.to(device)


        # %%
        n_inp = 256 if args.setting in ['noise', "gan", "graphsmote"] else 23
        n_hid = 512 if args.setting in ['noise', "gan", "graphsmote"] else 256


        model = HGT(node_dict, edge_dict,
                    n_inp=n_inp,
                    n_hid=n_hid,
                    n_out=labels.max().item()+1,
                    n_layers=2,
                    n_heads=4,
                    use_norm = True).to(device)
        # model.load_state_dict(torch.load(args.out_dir + 'music_hgt_model_pretrained.pt'))
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.n_epoch, max_lr = args.max_lr)
        if args.verbose:
            logger.debug('Training HGT with #param: %d' % (get_n_params(model)))


        # logistic
        # x = G.nodes['user'].data['inp']
        # x = x.cpu().numpy()
        # y = labels.cpu().numpy()

        # regr = linear_model.LogisticRegression(max_iter = 5000)
        # regr.fit(x[train_idx], y[train_idx])
        # y_pred = regr.predict(x[test_idx])

        # print("AUC: {:.4f}".format(roc_auc_score(y[test_idx], y_pred)))
        # print("Precision: {:.4f}".format(precision_score(y[test_idx], y_pred, average='binary')))
        # print("Recall: {:.4f}".format(recall_score(y[test_idx], y_pred, average='binary')))


        # %%
        if args.verbose:
            logger.debug('='*70)
            logger.debug('Training HGT with #param: %d' % (get_n_params(model)))
            logger.debug('='*70)
        train(model, G)


        # %%
        if args.verbose:
            logger.debug('='*70)
            logger.debug('Testing')
            logger.debug('='*70)
        model.load_state_dict(torch.load(args.tmp_dir + 'music_hgt_model_' + args.setting + '_ratio_' + str(ratio) + '_seed_' + str(seed) + '.pt'))
        model.eval()
        logits = model(G, args.target)
        logits = torch.sigmoid(logits)
        pred_score, pred = logits.max(1)
        pred_score[pred == 0] = pred_score[pred == 0]*(-1)
        pred_score = (pred_score + 1)/2


        pred_score = pred_score.cpu().detach().numpy()
        pred = logits.argmax(1).cpu()
        precision_, recall, th = precision_recall_curve(labels[test_idx], pred_score[test_idx])


        prc = auc(recall, precision_)
        roc = roc_auc_score(labels[test_idx], pred_score[test_idx])
        f1 = f1_score(labels[test_idx], pred[test_idx], average='binary')
        precision = precision_score(labels[test_idx], pred[test_idx], average='binary')
        rc = recall_score(labels[test_idx], pred[test_idx], average='binary')
        accuracy = (pred[test_idx]  == labels[test_idx]).float().mean().item()
        print("AUC-PRC: {:.4f}".format(prc))
        print("AUC-ROC: {:.4f}".format(roc))
        print("F1: {:.4f}".format(f1))
        print("Precision: {:.4f}".format(precision))
        print("Recall: {:.4f}".format(rc))
        print("ACC: {:.4f}".format(accuracy))


        f = open(args.result_dir + "music_hgt_model_" + args.setting + ".txt", "a")
        # f.write(",".join([args.setting, str(seed), str(percent[j]), str(prc), str(roc), str(f1), str(precision), str(rc), str(accuracy)])+'\n')
        f.write("{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(args.setting, seed, percent[j], prc, roc, f1, precision, rc, accuracy))
        f.close()

        if (args.setting == 'embedding') and best_emb_score < roc:
            emb = model.get_emb(G)
            torch.save(emb['user'], args.emb_dir + 'music_hgt_user_emb.pt')
            torch.save(emb['product'], args.emb_dir + 'music_hgt_product_emb.pt')
            best_emb_score = roc
    if (args.setting == 'embedding'):
        logger.info(f"User embeddings saved to: {args.emb_dir}music_hgt_user_emb.pt")
        logger.info(f"Product embeddings saved to: {args.emb_dir}music_hgt_product_emb.pt")
    logger.info("Training completed")


