from fincgan.hgt_model import HGT, Generator, latent_dim, emb_dim
from fincgan.logger import get_logger
import random
import torch
from torch.autograd import Variable
import dgl
import numpy as np
from scipy.spatial.distance import pdist,squareform
from tqdm import tqdm

# Setup logger for utils module
logger = get_logger('FincGAN.utils')

def oversampling(G, seed, ratio, verbose):
    if verbose:
        logger.info("Starting graph transformation (oversampling)...")
    # retrieve nodes and edges
    u_u_edge = [(u0.item(), u1.item())for u0, u1 in zip(G.edges(etype='u-u')[0], G.edges(etype='u-u')[1])]
    p_p_edge = [(p0.item(), p1.item()) for p0, p1 in zip(G.edges(etype='p-p')[0], G.edges(etype='p-p')[1])]

    u_p_edge = []
    p_u_edge = []
    for n0, n1 in zip(G.edges(etype='u-p')[0], G.edges(etype='u-p')[1]):
        u_p_edge.append((n0.item(), n1.item()))
        p_u_edge.append((n1.item(), n0.item()))

    feat_user = G.nodes['user'].data['feature']
    train_mask = G.nodes['user'].data['train_mask']
    label = G.nodes['user'].data['label']

    # extract minorities' indices in training dataset
    train_id = (train_mask == 1).nonzero().flatten().tolist()
    minor_id = (label == 1).nonzero().flatten().tolist()
    train_minor = list(set(train_id).intersection(minor_id))
    train_final_num = int((len(train_id)-len(train_minor)) / (1 - ratio))

    # random select
    random.seed(seed)
    selected_idx = random.choices(train_minor, k=train_final_num-len(train_id))

    # add new edges and features
    if verbose:
        logger.info(f"Appending {len(selected_idx)} nodes")

    new_u_u_edge = []
    for i, u_id in tqdm(enumerate(selected_idx)):

        new_id = i + G.num_nodes('user')

        # (using early stop could make these loops faster)
        for uu in u_u_edge:
            if uu[0] == u_id:
                new_u_u_edge.append((new_id, uu[1]))
                new_u_u_edge.append((uu[0], new_id))
        u_p_edge = u_p_edge + [(new_id, up[1]) for up in u_p_edge if up[0] == u_id]
        p_u_edge = p_u_edge + [(pu[0], new_id) for pu in p_u_edge if pu[1] == u_id]

        new_feat = feat_user[u_id]
        feat_user = torch.cat((feat_user, new_feat.unsqueeze(0)),0)

    u_u_edge = u_u_edge + new_u_u_edge
    # create new graph
    data_dict = {
        ('user', 'u-u', 'user'): u_u_edge,
        ('product', 'p-p', 'product'): p_p_edge,
        ('user', 'u-p', 'product'): u_p_edge,
        ('product', 'p_u', 'user'): p_u_edge
    }

    G_new = dgl.heterograph(data_dict)
    G_new.nodes['user'].data['feature'] = feat_user
    G_new.nodes['user'].data['train_mask'] = torch.cat((train_mask, torch.tensor([1] * (train_final_num-len(train_id)))),0)
    G_new.nodes['user'].data['val_mask'] = torch.cat((G.nodes['user'].data['val_mask'], torch.tensor([0] * (train_final_num-len(train_id)))),0)
    G_new.nodes['user'].data['test_mask'] = torch.cat((G.nodes['user'].data['test_mask'], torch.tensor([0] * (train_final_num-len(train_id)))),0)
    G_new.nodes['user'].data['label'] = torch.cat((label, torch.tensor([1] * (train_final_num-len(train_id)))),0).long()
    G_new.nodes['product'].data['feature'] = G.nodes['product'].data['feature']

    return G_new

def smote(G, seed, ratio, verbose):
    logger.info("Starting graph transformation (SMOTE)...")
    # retrieve nodes and edges
    u_u_edge = [(u0.item(), u1.item())for u0, u1 in zip(G.edges(etype='u-u')[0], G.edges(etype='u-u')[1])]
    p_p_edge = [(p0.item(), p1.item()) for p0, p1 in zip(G.edges(etype='p-p')[0], G.edges(etype='p-p')[1])]

    u_p_edge = []
    p_u_edge = []
    for n0, n1 in zip(G.edges(etype='u-p')[0], G.edges(etype='u-p')[1]):
        u_p_edge.append((n0.item(), n1.item()))
        p_u_edge.append((n1.item(), n0.item()))

    feat_user = G.nodes['user'].data['feature']
    train_mask = G.nodes['user'].data['train_mask']
    label = G.nodes['user'].data['label']

    # extract minorities' indices in training dataset
    train_id = (train_mask == 1).nonzero().flatten().tolist()
    minor_id = (label == 1).nonzero().flatten().tolist()
    train_minor = list(set(train_id).intersection(minor_id))
    train_final_num = int((len(train_id)-len(train_minor)) / (1 - ratio))

    # distance, k = 1 to select neighbors
    feat_user_minor = feat_user[torch.tensor(train_minor)]
    distance = squareform(pdist(feat_user_minor.detach()))
    np.fill_diagonal(distance, distance.max()+100)
    nearest_neigh = distance.argmin(axis=-1)

    # interpolation
    random.seed(seed)
    selected_idx = random.choices(train_minor, k=train_final_num-len(train_id))
    interps = [random.uniform(0, 1) for i in range(train_final_num-len(train_id))]
    if verbose:
        logger.info(f"Appending {len(selected_idx)} nodes")

    # add nodes and edges
    new_u_u_edge = []
    for i, (u_id, interp) in tqdm(enumerate(zip(selected_idx, interps))):
        new_id = i + G.num_nodes("user")

        # get new feature using interpolation
        user_emb = feat_user[u_id]
        neigh_id = nearest_neigh[train_minor.index(u_id)]
        neigh_emb = feat_user[nearest_neigh[neigh_id]]
        new_feat = user_emb + (neigh_emb - user_emb) * interp

        for uu in u_u_edge:
            if uu[0] == neigh_id:
                new_u_u_edge.append((new_id, uu[1]))
                new_u_u_edge.append((uu[0], new_id))
        u_p_edge = u_p_edge + [(new_id, up[1]) for up in u_p_edge if up[0] == neigh_id]
        p_u_edge = p_u_edge + [(pu[0], new_id) for pu in p_u_edge if pu[1] == neigh_id]
        feat_user = torch.cat((feat_user, new_feat.unsqueeze(0)),0)

    u_u_edge = u_u_edge + new_u_u_edge

   # create new graph
    data_dict = {
        ('user', 'u-u', 'user'): u_u_edge,
        ('product', 'p-p', 'product'): p_p_edge,
        ('user', 'u-p', 'product'): u_p_edge,
        ('product', 'p_u', 'user'): p_u_edge
    }

    G_new = dgl.heterograph(data_dict)
    G_new.nodes['user'].data['feature'] = feat_user.float()
    G_new.nodes['user'].data['train_mask'] = torch.cat((train_mask, torch.tensor([1] * (train_final_num-len(train_id)))),0)
    G_new.nodes['user'].data['val_mask'] = torch.cat((G.nodes['user'].data['val_mask'], torch.tensor([0] * (train_final_num-len(train_id)))),0)
    G_new.nodes['user'].data['test_mask'] = torch.cat((G.nodes['user'].data['test_mask'], torch.tensor([0] * (train_final_num-len(train_id)))),0)
    G_new.nodes['user'].data['label'] = torch.cat((label, torch.tensor([1] * (train_final_num-len(train_id)))),0).long()
    G_new.nodes['product'].data['feature'] = G.nodes['product'].data['feature']

    return G_new


def noise(G, seed, ratio, verbose, dataset_name='amazon'):
    if verbose:
        logger.info("Starting graph transformation (noise)...")
    # load embedding
    feat_product = torch.load(f'embed/{dataset_name}_hgt_product_emb.pt', map_location=torch.device('cpu'))

    # retrieve nodes and edges
    u_u_edge = [(u0.item(), u1.item()) for u0, u1 in zip(G.edges(etype='u-u')[0], G.edges(etype='u-u')[1])]
    p_p_edge = [(p0.item(), p1.item()) for p0, p1 in zip(G.edges(etype='p-p')[0], G.edges(etype='p-p')[1])]

    u_p_edge = []
    p_u_edge = []
    for n0, n1 in zip(G.edges(etype='u-p')[0], G.edges(etype='u-p')[1]):
        u_p_edge.append((n0.item(), n1.item()))
        p_u_edge.append((n1.item(), n0.item()))

    feat_user = torch.load(f'embed/{dataset_name}_hgt_user_emb.pt', map_location=torch.device('cpu'))
    train_mask = G.nodes['user'].data['train_mask']
    label = G.nodes['user'].data['label']

    # extract minorities' indices in training dataset
    train_id = (train_mask == 1).nonzero().flatten().tolist()
    minor_id = (label == 1).nonzero().flatten().tolist()
    train_minor = list(set(train_id).intersection(minor_id))
    train_final_num = int((len(train_id)-len(train_minor)) / (1 - ratio))

    # random select
    random.seed(seed)
    selected_idx = random.choices(train_minor, k=train_final_num-len(train_id))

    # add new edges and features
    if verbose:
        logger.info(f"Appending {len(selected_idx)} nodes")

    new_u_u_edge = []
    for i, u_id in tqdm(enumerate(selected_idx)):

        new_id = i + G.num_nodes('user')

        # (using early stop could make these loops faster)
        for uu in u_u_edge:
            if uu[0] == u_id:
                new_u_u_edge.append((new_id, uu[1]))
                new_u_u_edge.append((uu[0], new_id))
        u_p_edge = u_p_edge + [(new_id, up[1]) for up in u_p_edge if up[0] == u_id]
        p_u_edge = p_u_edge + [(pu[0], new_id) for pu in p_u_edge if pu[1] == u_id]

        new_feat = feat_user[u_id]
        feat_user = torch.cat((feat_user, new_feat.unsqueeze(0)),0)

    u_u_edge = u_u_edge + new_u_u_edge
    # create new graph
    data_dict = {
        ('user', 'u-u', 'user'): u_u_edge,
        ('product', 'p-p', 'product'): p_p_edge,
        ('user', 'u-p', 'product'): u_p_edge,
        ('product', 'p_u', 'user'): p_u_edge
    }

    G_new = dgl.heterograph(data_dict)
    G_new.nodes['user'].data['feature'] = feat_user
    G_new.nodes['user'].data['train_mask'] = torch.cat((train_mask, torch.tensor([1] * (train_final_num-len(train_id)))),0)
    G_new.nodes['user'].data['val_mask'] = torch.cat((G.nodes['user'].data['val_mask'], torch.tensor([0] * (train_final_num-len(train_id)))),0)
    G_new.nodes['user'].data['test_mask'] = torch.cat((G.nodes['user'].data['test_mask'], torch.tensor([0] * (train_final_num-len(train_id)))),0)
    G_new.nodes['user'].data['label'] = torch.cat((label, torch.tensor([1] * (train_final_num-len(train_id)))),0).long()
    G_new.nodes['product'].data['feature'] = feat_product

    return G_new


def graphsmote(G, seed, ratio, uu_threshold, verbose=0, dataset_name='amazon'):
    if verbose:
        logger.info("Starting graph transformation (GraphSMOTE)...")
    random.seed(seed)
    np.random.seed(seed)


    train_mask = G.nodes['user'].data.pop('train_mask')
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
    val_mask = G.nodes['user'].data.pop('val_mask')
    test_mask = G.nodes['user'].data.pop('test_mask')
    labels = G.nodes['user'].data.pop('label')

    user_emb = torch.load(f'embed/{dataset_name}_hgt_user_emb.pt', map_location=torch.device('cpu'))
    product_emb = torch.load(f'embed/{dataset_name}_hgt_product_emb.pt', map_location=torch.device('cpu'))

    user_size = user_emb.shape[0]
    product_size = product_emb.shape[0]
    emb_dim = user_emb.shape[1]


    # define and load model
    node_generator = Generator()
    uu_generator = Decoder(emb_dim=emb_dim, matrix_dim=256)
    up_generator = MLP(emb_dim*2, 1, 512, 4)

    node_generator.load_state_dict(torch.load(f"generator/{dataset_name}_G.pt", map_location="cpu"))
    uu_generator.load_state_dict(torch.load("generator/" + 'uu_generator.pt', map_location="cpu"))
    up_generator.load_state_dict(torch.load("generator/" + 'up_generator.pt', map_location="cpu"))

    minor_idx = train_idx[(labels==1)[train_idx]]
    k = int((ratio*len(train_idx) - len(minor_idx))/(1-ratio))
    if verbose:
        logger.info(f"Generating {k} synthetic user nodes for {int(ratio*100)}% spam ratio")
    minor_embed = user_emb[minor_idx,:]
    minor_label = labels[minor_idx]

    distance = squareform(pdist(minor_embed.detach().numpy()))
    np.fill_diagonal(distance,distance.max()+100)
    neighbor_idx = distance.argmin(axis=-1)

    new_emb = torch.zeros((k, emb_dim), dtype=torch.float)
    new_uu = torch.zeros((k, user_size), dtype=torch.float)
    new_up = torch.zeros((k, product_size), dtype=torch.float)
    finish_cnt = 0
    cnt = 0
    with torch.no_grad():
        for i in range(k):
            while finish_cnt == i:
                cnt = cnt + 1
                select_idx = random.choices(range(minor_idx.size(0)), k = 1)
                interp_place = random.random()
                embed = (minor_embed[select_idx, :]) + (minor_embed[neighbor_idx[select_idx],:] - minor_embed[select_idx, :])*interp_place

                up_emb = torch.empty((product_size, emb_dim*2))
                up_emb[:, :emb_dim] = embed
                up_emb[:, emb_dim:] = product_emb

                uu_emb = torch.empty((user_size, emb_dim*2))
                uu_emb[:, :emb_dim] = embed
                uu_emb[:, emb_dim:] = user_emb

                uu_edge = uu_generator(embed, user_emb)
                up_edge = up_generator(up_emb).view(-1)

                if (uu_edge >= uu_threshold).sum().item() >= 1:
                    new_emb[i, :] = embed[0]
                    new_uu[i, :] = uu_edge[0]
                    new_up[i, :] = up_edge
                    finish_cnt = finish_cnt + 1

                    if finish_cnt % (k//10) == 0:
                        if verbose:
                            print("finished {:.2f}%".format(finish_cnt/k*100))

    new_uu[new_uu >= uu_threshold] = 1
    new_uu[new_uu < uu_threshold] = 0
    new_up = torch.zeros((k, product_size), dtype=torch.float)

    new_uu = new_uu[:finish_cnt, :]
    new_up = new_up[:finish_cnt, :]
    new_emb = new_emb[:finish_cnt, :]

    u1 = G.edges(etype='u-u')[0].tolist()
    u2 = G.edges(etype='u-u')[1].tolist()
    u_u_edge = list(zip(u1, u2))


    (u1, u2) = np.where(new_uu == 1)
    u1 = u1 + user_size
    u_u_edge.extend(list(zip(u1, u2)))
    u_u_edge.extend(list(zip(u2, u1)))
    u_u_edge = list(set(u_u_edge))
    del u1, u2


    U = G.edges(etype='u-p')[0].tolist()
    P = G.edges(etype='u-p')[1].tolist()
    u_p_edge = list(zip(U, P))
    p_u_edge = list(zip(P, U))

    (U, P) = np.where(new_up == 1)
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

    feat_user = torch.cat((user_emb, new_emb), dim = 0)
    feat_product = product_emb
    train_mask = torch.cat((train_mask, torch.ones(finish_cnt)), dim = 0)
    val_mask = torch.cat((val_mask, torch.zeros(finish_cnt)), dim = 0)
    test_mask = torch.cat((test_mask, torch.zeros(finish_cnt)), dim = 0)
    labels = torch.cat((labels, torch.ones(finish_cnt)), dim = 0)


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

    return g


def gan(G, seed, ratio, uu_threshold, up_threshold, verbose=0, dataset_name='amazon'):
    if verbose:
        logger.info("Starting graph transformation (FincGAN)...")
    random.seed(seed)
    np.random.seed(seed)


    train_mask = G.nodes['user'].data.pop('train_mask')
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
    val_mask = G.nodes['user'].data.pop('val_mask')
    test_mask = G.nodes['user'].data.pop('test_mask')
    labels = G.nodes['user'].data.pop('label')

    user_emb = torch.load(f'embed/{dataset_name}_hgt_user_emb.pt', map_location=torch.device('cpu'))
    product_emb = torch.load(f'embed/{dataset_name}_hgt_product_emb.pt', map_location=torch.device('cpu'))

    user_size = user_emb.shape[0]
    product_size = product_emb.shape[0]
    emb_dim = user_emb.shape[1]


    # define and load model
    node_generator = Generator()
    uu_generator = Decoder(emb_dim=emb_dim, matrix_dim=256)
    up_generator = MLP(emb_dim*2, 1, 512, 4)

    node_generator.load_state_dict(torch.load(f"generator/{dataset_name}_G.pt", map_location="cpu"))
    uu_generator.load_state_dict(torch.load("generator/" + 'uu_generator.pt', map_location="cpu"))
    up_generator.load_state_dict(torch.load("generator/" + 'up_generator.pt', map_location="cpu"))

    minor_idx = train_idx[(labels==1)[train_idx]]
    k = int((ratio*len(train_idx) - len(minor_idx))/(1-ratio))
    if verbose:
        logger.info(f"Generating {k} synthetic user nodes for {int(ratio*100)}% spam ratio")
    minor_embed = user_emb[minor_idx,:]
    minor_label = labels[minor_idx]

#     distance = squareform(pdist(minor_embed.detach().numpy()))
#     np.fill_diagonal(distance,distance.max()+100)
#     neighbor_idx = distance.argmin(axis=-1)

    new_emb = torch.zeros((k, emb_dim), dtype=torch.float)
    new_uu = torch.zeros((k, user_size), dtype=torch.float)
    new_up = torch.zeros((k, product_size), dtype=torch.float)
    finish_cnt = 0
    cnt = 0
    with torch.no_grad():
        for i in tqdm(range(k)):
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

                if ((uu_edge >= uu_threshold).sum().item() >= 1) & ((up_edge >= up_threshold).sum().item() >= 1):
                    new_emb[i, :] = embed[0]
                    new_uu[i, :] = uu_edge[0]
                    new_up[i, :] = up_edge
                    finish_cnt = finish_cnt + 1

                    if finish_cnt % (k//10) == 0:
                        if verbose:
                            print("finished {:.2f}%".format(finish_cnt/k*100))

    new_uu[new_uu >= uu_threshold] = 1
    new_uu[new_uu < uu_threshold] = 0
    new_up[new_up >= up_threshold] = 1
    new_up[new_up < up_threshold] = 0

    new_uu = new_uu[:finish_cnt, :]
    new_up = new_up[:finish_cnt, :]
    new_emb = new_emb[:finish_cnt, :]

    u1 = G.edges(etype='u-u')[0].tolist()
    u2 = G.edges(etype='u-u')[1].tolist()
    u_u_edge = list(zip(u1, u2))


    (u1, u2) = np.where(new_uu == 1)
    u1 = u1 + user_size
    u_u_edge.extend(list(zip(u1, u2)))
    u_u_edge.extend(list(zip(u2, u1)))
    u_u_edge = list(set(u_u_edge))
    del u1, u2


    U = G.edges(etype='u-p')[0].tolist()
    P = G.edges(etype='u-p')[1].tolist()
    u_p_edge = list(zip(U, P))
    p_u_edge = list(zip(P, U))

    (U, P) = np.where(new_up == 1)
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

    feat_user = torch.cat((user_emb, new_emb), dim = 0)
    feat_product = product_emb
    train_mask = torch.cat((train_mask, torch.ones(finish_cnt)), dim = 0)
    val_mask = torch.cat((val_mask, torch.zeros(finish_cnt)), dim = 0)
    test_mask = torch.cat((test_mask, torch.zeros(finish_cnt)), dim = 0)
    labels = torch.cat((labels, torch.ones(finish_cnt)), dim = 0)


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

    return g
