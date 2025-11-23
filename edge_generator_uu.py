#!/usr/bin/env python
# coding: utf-8


import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" -> GPU ID: 1 Quadro RTX A6000
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" -> GPU ID: 5 Quadro RTX A6000
# os.environ["CUDA_VISIBLE_DEVICES"] = "2" -> GPU ID: 6 Quadro RTX A6000
# os.environ["CUDA_VISIBLE_DEVICES"] = "3" -> GPU ID: 2 RTX 3090
# os.environ["CUDA_VISIBLE_DEVICES"] = "4" -> GPU ID: 3 RTX 3090
# os.environ["CUDA_VISIBLE_DEVICES"] = "5" -> GPU ID: 0 RTX 2080 Ti
# os.environ["CUDA_VISIBLE_DEVICES"] = "6" -> GPU ID: 4 RTX 2080 Ti
# os.environ["CUDA_VISIBLE_DEVICES"] = "7" -> GPU ID: 7 RTX 2080 Ti
import torch


from hgt_model import HGT, Generator, Decoder, adj_loss, latent_dim, emb_dim
from logger import get_logger
import argparse
from dgl.data.utils import load_graphs


parser = argparse.ArgumentParser(description='Training U-U Edge Generator')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu_id', type=int, default=0, help="to train on which GPU")
parser.add_argument('--data_dir', type=str, default="./graph/music_instrument_25.bin")
parser.add_argument('--emb_dir', type=str, default="./embed/")
parser.add_argument('--edge_dir', type=str, default="./generator/")
parser.add_argument('--n_epoch', type=int, default=1000)
parser.add_argument('--emb_dim', type=int, default=256)
parser.add_argument('--matrix_dim', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--max_lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--clip', type=int, default=1.0)
parser.add_argument('--edge_generator_verbose', type=int, default=1, help="set 1 to show training details of user-user edge generator, otherwise 0")

''' Use one of the following two lines when parser throws error '''
# args = parser.parse_args(args = [])
args = parser.parse_args()

# Setup logger
logger = get_logger('FincGAN.uu_generator')
logger.info(f"args:\n{args}")


# torch.cuda.set_device(args.gpu_id)  # Commented out for CPU mode
device = torch.device("cpu")  # Use CPU for DGL CPU version compatibility
logger.info(f"Using CPU for training (DGL CPU version)\n")

torch.manual_seed(args.seed)


# load graph

[G], _ = load_graphs(args.data_dir)
logger.info(f"Graph Information:")
print(G)
print("\n")

# load embedding
user_emb = torch.load(args.emb_dir + "music_hgt_user_emb.pt", map_location=torch.device('cpu'))
logger.info(f"completed.")


user_size = G.num_nodes('user')
U1 = G.edges(etype='u-u')[0]
U2 = G.edges(etype='u-u')[1]
adj_uu = torch.zeros((user_size, user_size), dtype=torch.float)
for u1, u2 in list(zip(U1, U2)):
    adj_uu[u1.item(), u2.item()] = 1

adj_uu = adj_uu.to(device)
logger.info(f"completed.\n")


logger.info(f"Training user-user edge generator... ")
user_emb = user_emb.to(device)
decoder_uu = Decoder(emb_dim=args.emb_dim, matrix_dim=args.matrix_dim).to(device)
optimizer_de_uu = torch.optim.Adam(decoder_uu.parameters())
scheduler_de_uu = torch.optim.lr_scheduler.OneCycleLR(optimizer_de_uu, total_steps=args.n_epoch, max_lr = args.max_lr)


weight_uu = .03*((adj_uu == 0).sum().item()/(adj_uu == 1).sum().item()) # for reweight (connected pairs are rare)
for epoch in range(args.n_epoch):
    optimizer_de_uu.zero_grad()
    adj_g = decoder_uu(user_emb, user_emb)
    loss_de_uu = adj_loss(adj_g, adj_uu, weight_uu, device)
    loss_de_uu.backward()
    optimizer_de_uu.step()
    scheduler_de_uu.step()

    if args.edge_generator_verbose:
        if epoch % (args.n_epoch // 100) == 0:

            print(
                "[Epoch]: {}, [LR]: {:.6f}, [Loss]: {:.4f}, [ACC]: {:.4f}".format(
                    epoch,
                    optimizer_de_uu.param_groups[0]['lr'],
                    loss_de_uu.item(),
                    evaluate(adj_g, adj_uu))
            )


logger.info("Evaluating user-user edge generator... ")
for i in range(100, 49, -1):
    r = i/100
    try:
        adj_g = decoder_uu(user_emb, user_emb)
        adj_g[adj_g > r] = 1
        adj_g[adj_g <= r] = 0
        adj_uu = adj_uu.cpu()
        adj_g = adj_g.cpu()

        if args.edge_generator_verbose:

            print("r: {:.2f}, ACC: {:4f}, Precision {:.4f}, cnt: {}".format(
                i/100,
                evaluate(adj_g, adj_uu, weight=i/100),
                (adj_g[adj_g == 1].view(-1) == adj_uu[adj_g == 1].view(-1)).sum().item()/(adj_g == 1).sum().item(),
                (adj_g == 1).sum().item())
                )
    except:
        pass

logger.info(f"user-user edge generator training completed.\n")


if not os.path.exists(args.edge_dir):
        os.makedirs(args.edge_dir)
        logger.info(f"completed.")
torch.save(decoder_uu.state_dict(), args.edge_dir + 'uu_generator.pt')
logger.info(f"As uu_generator.pt, state dictionary of user-user edge generator successfully saved to: {args.edge_dir}.")
logger.info(f"program ends.")
