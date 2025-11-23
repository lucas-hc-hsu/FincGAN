#!/usr/bin/env python
# coding: utf-8


import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" -> GPU ID: 1 Quadro RTX A6000
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" -> GPU ID: 5 Quadro RTX A6000
# os.environ["CUDA_VISIBLE_DEVICES"] = "2" -> GPU ID: 6 Quadro RTX A6000
# os.environ["CUDA_VISIBLE_DEVICES"] = "3" -> GPU ID: 2 RTX 3090
# os.environ["CUDA_VISIBLE_DEVICES"] = "4" -> GPU ID: 3 RTX 3090
# os.environ["CUDA_VISIBLE_DEVICES"] = "5" -> GPU ID: 0 RTX 2080 Ti
# os.environ["CUDA_VISIBLE_DEVICES"] = "6" -> GPU ID: 4 RTX 2080 Ti
# os.environ["CUDA_VISIBLE_DEVICES"] = "7" -> GPU ID: 7 RTX 2080 Ti
import torch


from fincgan.hgt_model import HGT, Generator, MLP, latent_dim, emb_dim
from fincgan.logger import get_logger
import random
import argparse
from tqdm import tqdm
from torch.autograd import Variable
from dgl.data.utils import load_graphs

import numpy as np


parser = argparse.ArgumentParser(description='Training U-P Edge Generator')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu_id', type=int, default=0, help="to train on which GPU")
parser.add_argument('--data_dir', type=str, default="./graph/music_instrument_25.bin")
parser.add_argument('--emb_dir', type=str, default="./embed/")
parser.add_argument('--tmp_dir', type=str, default="./tmp/")
parser.add_argument('--edge_dir', type=str, default="./generator/")
parser.add_argument('--n_epoch', type=int, default=700)
parser.add_argument('--batch_size', type=int, default=9192)
parser.add_argument('--emb_dim', type=int, default=256)
parser.add_argument('--max_lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--edge_generator_verbose', type=int, default=1, help="set 1 to show training details of user-product edge generator, otherwise 0")

''' Use one of the following two lines when parser throws error '''
# args = parser.parse_args(args = [])
args = parser.parse_args()

# Setup logger
logger = get_logger('FincGAN.up_generator')
logger.info(f"args:\n{args}")


cuda = False  # Use CPU mode for DGL CPU version compatibility
# torch.cuda.set_device(args.gpu_id)  # Commented out for CPU mode
device = torch.device("cpu")
logger.info(f"Using CPU for training (DGL CPU version)\n")

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)


# load graph

[G], _ = load_graphs(args.data_dir)
logger.info(f"Graph Information:")
print(G)
print("\n")

# load embedding
user_emb = torch.load(args.emb_dir + 'music_hgt_user_emb.pt', map_location = 'cpu')
logger.info(f"completed.")

product_emb = torch.load(args.emb_dir + 'music_hgt_product_emb.pt', map_location='cpu')
logger.info(f"completed.")


logger.info("Collecting user-product edges, then load it to device, it may take some times... ")
user_size = G.num_nodes('user')
product_size = G.num_nodes('product')
up_size = G.num_edges('u-p')
noedge_size = user_size*product_size - up_size

pair_indice = []
U = G.edges(etype='u-p')[0]
P = G.edges(etype='u-p')[1]
for u, p in list(zip(U, P)):
    pair_indice.append((u.item(), p.item()))
del U, P

unpair_indice = [(i, j) for i in tqdm(range(user_size)) for j in range(product_size)]
logger.info(f"not yet...")
unpair_indice = list(set(unpair_indice) - set(pair_indice))
unpair_u = np.array([p[0] for p in unpair_indice])
unpair_p = np.array([p[1] for p in unpair_indice])
del unpair_indice


pair_emb = torch.empty((len(pair_indice), args.emb_dim*2), dtype=torch.float)
for i, (u, p) in enumerate(pair_indice):
    pair_emb[i, :args.emb_dim] = user_emb[u, :]
    pair_emb[i, args.emb_dim:] = product_emb[p, :]

labels = torch.ones((len(pair_indice)))


torch_dataset = torch.utils.data.TensorDataset(pair_emb[:int(len(pair_indice)*.8), :], labels[:int(len(pair_indice)*.8)])
dataloader = torch.utils.data.DataLoader(
    torch_dataset,
    batch_size=args.batch_size,
    shuffle = True,
)
logger.info(f"completed.\n")


logger.info(f"Training user-product edge generator... ")
adversarial_loss = torch.nn.BCELoss(reduction='none')
decoder_up = MLP(args.emb_dim*2, 1, 512, 4)
optimizer_de_up = torch.optim.Adam(decoder_up.parameters())
scheduler_de_up = torch.optim.lr_scheduler.OneCycleLR(optimizer_de_up, total_steps=args.n_epoch*2*int(labels.shape[0]/args.batch_size), max_lr = args.max_lr)

if cuda:
    decoder_up.cuda()
    adversarial_loss.cuda()

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


best_loss = 100
for epoch in range(args.n_epoch):
    epoch_loss = 0

    for i, (imgs, lab) in enumerate(dataloader):
        optimizer_de_up.zero_grad()

        select_indice = random.sample(range(noedge_size), k = args.batch_size * 12)
        unpair_emb = torch.cat((user_emb[unpair_u[select_indice]], product_emb[unpair_p[select_indice]]), dim = 1)
        imgs = torch.cat((imgs, unpair_emb), dim = 0)
        imgs = Variable(imgs.type(Tensor))
        lab = torch.cat((lab, torch.zeros(args.batch_size * 12)), dim = 0)

        y_pred = decoder_up(imgs)
        loss_de_up = adversarial_loss(y_pred.float().view(-1), lab.type(Tensor).float())
        loss_de_up[lab == 1] = loss_de_up[lab == 1] * 3
        loss_de_up = torch.mean(loss_de_up)

        loss_de_up.backward()
        optimizer_de_up.step()
        scheduler_de_up.step()
        epoch_loss = epoch_loss + loss_de_up.item()

    logits = decoder_up(pair_emb[int(len(pair_indice)*.8):int(len(pair_indice)*.9), :]).float().view(-1)
    lab = torch.ones((logits.size(0)), dtype = torch.float)
    loss_de_up = adversarial_loss(logits, lab)
    loss_de_up[lab == 1] = loss_de_up[lab == 1] * 3
    loss_de_up = torch.mean(loss_de_up).item()
    if args.edge_generator_verbose:

        print("[Epoch {}/{}], [Train Loss: {:4f}], [Val Loss: {:4f}] ".format(epoch, args.n_epoch, epoch_loss, loss_de_up))


logger.info("Evaluating user-product edge generator... ")
decoder_up = decoder_up.cpu()
select_indice = random.sample(range(noedge_size), k = up_size*10)
unpair_emb = torch.cat((user_emb[unpair_u[select_indice]], product_emb[unpair_p[select_indice]]), dim = 1).cpu()
labels = torch.cat((torch.zeros(up_size*10), labels[int(len(pair_indice)*.9):])).cpu()


for i in range(100, 50, -1):
    r = i/100
    try:
        adj_g = decoder_up(torch.cat((unpair_emb.cpu(), pair_emb[int(len(pair_indice)*.9):, :].cpu()), dim = 0).cpu())

        adj_g[adj_g > r] = 1
        adj_g[adj_g <= r] = 0
        adj_g = adj_g.view(-1).cpu()

        if args.edge_generator_verbose:

            print("r: {}, ACC: {:.4f}, Precision: {:4f}, cnt: {}".format(
                r,
                ((adj_g == labels).sum()/labels.shape[0]).item(),
                (adj_g[adj_g == 1] == labels[adj_g == 1]).sum().item()/(adj_g == 1).sum().item(),
                (adj_g == 1).sum().item())
                )
    except:
        pass
logger.info(f"user-product edge generator training completed.\n")


if not os.path.exists(args.edge_dir):
        os.makedirs(args.edge_dir)
        logger.info(f"completed.")

torch.save(decoder_up.state_dict(), args.edge_dir + 'up_generator.pt')
logger.info(f"As up_generator.pt, state dictionary of user-product edge generator successfully saved to: {args.edge_dir}.")
logger.info(f"program ends.")
