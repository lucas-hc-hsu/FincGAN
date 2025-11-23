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


from hgt_model import HGT, Generator, latent_dim, emb_dim
from logger import get_logger
import argparse
import numpy as np
from dgl.data.utils import load_graphs, save_graphs
import utils
import sys
import random
from tqdm import tqdm
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='FincGAN Multi Graph Generator respect to different ratio')
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--data_dir', type=str, default="graph/music_instrument_25.bin")
parser.add_argument('--graph_dir', type=str, default="graph/", help="directory to save generated graph during training")
parser.add_argument('--ratio', nargs='+', type=float, default=[0.1007, 0.17, 0.23, 0.29, 0.34, 0.375, 0.41, 0.45, 0.47, 0.5, 0.52, 0.55]) # for test different threshold
parser.add_argument('--up', type=float, default=0.99, help="threshold of user-product edge generator")
parser.add_argument('--uu', type=float, default=0.91, help="threshold of user-user edge genertor")
parser.add_argument('--verbose', type=int, default=0, help="set 1 to show training details as generating new graph, otherwise 0")

''' If the error messages pop out during parser declaration, try another one.'''
# args = parser.parse_args(args = [])
args = parser.parse_args()

# Setup logger
logger = get_logger('FincGAN.graph_generator')
print(args)

# cuda settings
cuda = False  # Use CPU mode for DGL CPU version compatibility
# torch.cuda.set_device(args.gpu_id)  # Commented out for CPU mode
device = torch.device("cpu")
logger.info(f"Using CPU for generation (DGL CPU version)\n")


# load graph
[G], _ = load_graphs(args.data_dir)
logger.info(f"Graph Information of Amazon music instrument dataset\n")
print(G)

# set random seed
if args.verbose:
    # Logging handled by surrounding context
    print("------Generating Synthetic nodes-------")
random.seed(args.seed)
np.random.seed(args.seed)


original_train_mask = G.nodes['user'].data.pop('train_mask')
original_train_idx = torch.nonzero(original_train_mask, as_tuple=False).squeeze()
original_val_mask = G.nodes['user'].data.pop('val_mask')
original_test_mask = G.nodes['user'].data.pop('test_mask')
original_labels = G.nodes['user'].data.pop('label')


user_emb = torch.load('embed/music_hgt_user_emb.pt', map_location=torch.device('cpu'))
product_emb = torch.load('embed/music_hgt_product_emb.pt', map_location=torch.device('cpu'))

user_size = user_emb.shape[0]
product_size = product_emb.shape[0]
emb_dim = user_emb.shape[1]


# define and load model
node_generator = Generator()
uu_generator = Decoder(emb_dim=emb_dim, matrix_dim=256)
up_generator = MLP(emb_dim*2, 1, 512, 4)

node_generator.load_state_dict(torch.load("generator/" + 'music_G.pt', map_location="cpu"))
uu_generator.load_state_dict(torch.load("generator/" + 'uu_generator.pt', map_location="cpu"))
up_generator.load_state_dict(torch.load("generator/" + 'up_generator.pt', map_location="cpu"))


minor_idx = original_train_idx[(original_labels==1)[original_train_idx]]
minor_embed = user_emb[minor_idx,:]
minor_label = original_labels[minor_idx]


k_list = []
for ratio in args.ratio:
    k_list.append(int((ratio*len(original_train_idx) - len(minor_idx))/(1-ratio)))
# print(k_list)


k_max = max(k_list)
# print(f"k_max: {k_max}")
new_emb = torch.zeros((k_max, emb_dim), dtype=torch.float)
new_uu = torch.zeros((k_max, user_size), dtype=torch.float)
new_up = torch.zeros((k_max, product_size), dtype=torch.float)


finish_cnt = 0
cnt = 0

with torch.no_grad():
    for i in tqdm(range(k_max)):
        while finish_cnt == i:
            cnt = cnt + 1
            select_idx = random.choices(range(minor_idx.size(0)), k = 1)
            noise = Variable(torch.Tensor(np.random.normal(0, 1, (1, latent_dim))))
            z = Variable(torch.cat((minor_embed[select_idx, :], noise), dim = 1))
            embed = node_generator(z, minor_label[select_idx])

            up_emb = torch.empty((product_size, emb_dim*2))
            up_emb[:, :emb_dim] = embed
            up_emb[:, emb_dim:] = product_emb

            uu_emb = torch.empty((user_size, emb_dim*2))
            uu_emb[:, :emb_dim] = embed
            uu_emb[:, emb_dim:] = user_emb

            uu_edge = uu_generator(embed, user_emb)
            up_edge = up_generator(up_emb).view(-1)

            if ((uu_edge >= args.uu).sum().item() >= 1) & ((up_edge >= args.up).sum().item() >= 1):
                new_emb[i, :] = embed[0]
                new_uu[i, :] = uu_edge[0]
                new_up[i, :] = up_edge
                finish_cnt = finish_cnt + 1


new_uu[new_uu >= args.uu] = 1
new_uu[new_uu < args.uu] = 0
new_up[new_up >= args.up] = 1
new_up[new_up < args.up] = 0

for ratio, amount_of_new_nodes in zip(args.ratio, k_list):
    print(f"\namount_of_new_nodes: {amount_of_new_nodes}")

    local_new_uu = new_uu[:amount_of_new_nodes, :]
    local_new_up = new_up[:amount_of_new_nodes, :]
    local_new_emb = new_emb[:amount_of_new_nodes, :]

    u1 = G.edges(etype='u-u')[0].tolist()
    u2 = G.edges(etype='u-u')[1].tolist()
    u_u_edge = list(zip(u1, u2))


    (u1, u2) = np.where(local_new_uu == 1)
    u1 = u1 + user_size
    u_u_edge.extend(list(zip(u1, u2)))
    u_u_edge.extend(list(zip(u2, u1)))
    u_u_edge = list(set(u_u_edge))
    del u1, u2


    U = G.edges(etype='u-p')[0].tolist()
    P = G.edges(etype='u-p')[1].tolist()
    u_p_edge = list(zip(U, P))
    p_u_edge = list(zip(P, U))

    (U, P) = np.where(local_new_up == 1)
    U = U + user_size
    u_p_edge.extend(list(zip(U, P)))
    p_u_edge.extend(list(zip(P, U)))
    u_p_edge = list(set(u_p_edge))
    p_u_edge = list(set(p_u_edge))
    del U, P

    p1 = G.edges(etype='p-p')[0].tolist()
    p2 = G.edges(etype='p-p')[1].tolist()
    p_p_edge = list(zip(p1, p2))
    p_p_edge = list(set(p_p_edge))
    del p1, p2

    feat_user = torch.cat((user_emb, local_new_emb), dim = 0)
    feat_product = product_emb
    train_mask = torch.cat((original_train_mask, torch.ones(amount_of_new_nodes)), dim = 0)
    val_mask = torch.cat((original_val_mask, torch.zeros(amount_of_new_nodes)), dim = 0)
    test_mask = torch.cat((original_test_mask, torch.zeros(amount_of_new_nodes)), dim = 0)
    labels = torch.cat((original_labels, torch.ones(amount_of_new_nodes)), dim = 0)


    data_dict = {
        ('user', 'u-u', 'user'): u_u_edge,
        ('product', 'p-p', 'product'): p_p_edge,
        ('user', 'u-p', 'product'): u_p_edge,
        ('product', 'p-u', 'user'): p_u_edge,
    }

    g = dgl.heterograph(data_dict)
    g.nodes['user'].data['feature'] = feat_user.float()
    g.nodes['user'].data['train_mask'] = train_mask
    g.nodes['user'].data['val_mask'] = val_mask
    g.nodes['user'].data['test_mask'] = test_mask
    g.nodes['user'].data['label'] = labels.long()
    g.nodes['product'].data['feature'] = feat_product.float()

    print(g)
    graph_path = args.graph_dir + "music_instrument_gan_ratio_" + str(ratio) + '_seed_'+str(args.seed)+'.bin'
    save_graphs(graph_path, [g])
    logger.info(f"created graph successfully saved to : {graph_path}")
logger.info(f"program ends.")