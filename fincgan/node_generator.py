#!/usr/bin/env python
# coding: utf-8


# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# # os.environ["CUDA_VISIBLE_DEVICES"] = "0" -> GPU ID: 1 Quadro RTX A6000
# # os.environ["CUDA_VISIBLE_DEVICES"] = "1" -> GPU ID: 5 Quadro RTX A6000
# # os.environ["CUDA_VISIBLE_DEVICES"] = "2" -> GPU ID: 6 Quadro RTX A6000
# # os.environ["CUDA_VISIBLE_DEVICES"] = "3" -> GPU ID: 2 RTX 3090
# # os.environ["CUDA_VISIBLE_DEVICES"] = "4" -> GPU ID: 3 RTX 3090
# # os.environ["CUDA_VISIBLE_DEVICES"] = "5" -> GPU ID: 0 RTX 2080 Ti
# # os.environ["CUDA_VISIBLE_DEVICES"] = "6" -> GPU ID: 4 RTX 2080 Ti
# # os.environ["CUDA_VISIBLE_DEVICES"] = "7" -> GPU ID: 7 RTX 2080 Ti
# import torch


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f'device: {device}')


import argparse
import os
import numpy as np
import random

from dgl.data.utils import load_graphs
from torch._C import device
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torch.autograd import Variable
from fincgan.logger import get_logger

import torch.nn as nn
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help="to train on which GPU")
parser.add_argument('--data_dir', type=str, default="graph/")
parser.add_argument('--emb_dir', type=str, default="embed/")
parser.add_argument('--gan_dir', type=str, default="generator/")
parser.add_argument('--tsne_dir', type=str, default="tsne/")
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--emb_dim", type=int, default=256, help="size of each image dimension")
parser.add_argument("--n_classes", type=int, default=2, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic_D", type=int, default=3, help="number of training steps for discriminator per iter")
parser.add_argument("--n_critic_G", type=int, default=1, help="number of training steps for discriminator per iter")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--gan_verbose", type=int, default=0, help="set 1 to show training details of GAN, otherwise 0")
parser.add_argument("--tsne_verbose", type=int, default=0, help="set 0, 1 or 2, provides additional details of tsne as visualizing synthetic data")
parser.add_argument("--save_tsne_figure", type=bool, default=True, help="save the reulst of tsne as visualizing synthetic data")
parser.add_argument('--dataset-name', type=str, default="amazon", help="name of the dataset (used for file naming)")

''' Use one of the following two lines when parser throws error '''
# opt = parser.parse_args(args = [])
opt = parser.parse_args()

# Setup logger
logger = get_logger('FincGAN.node_generator')
logger.info(f"opt:\n{opt}")

np.random.seed(0)
# torch.cuda.set_device(opt.gpu_id)  # Commented out for CPU mode
torch.manual_seed(0)
random.seed(0)

cuda = False  # Use CPU mode for DGL CPU version compatibility
logger.info(f"Using CPU for training (DGL CPU version)")


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim + opt.emb_dim + opt.n_classes, 128 * self.init_size ** 2))
        self.out = nn.Sequential(nn.Linear(opt.img_size ** 2 * opt.channels, opt.emb_dim))
        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)


        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        z = torch.cat((self.label_emb(labels), z), -1)
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        img = img.view(img.shape[0], opt.img_size ** 2 * opt.channels)
        img = self.out(img)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.l1 = nn.Sequential(nn.Linear(opt.emb_dim + opt.n_classes, opt.img_size ** 2 * opt.channels))
        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))

    def forward(self, img, labels):
        img = torch.cat((img, self.label_embedding(labels)), -1)
        img = self.l1(img)
        img = img.view(img.shape[0], opt.channels, opt.img_size, opt.img_size)
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        validity = validity
        return validity


# Loss function
similarity_loss = torch.nn.MSELoss(reduction='mean')

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    similarity_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)


# Configure data loader
[G], _ = load_graphs(opt.data_dir)
train_mask = G.nodes['user'].data.pop('train_mask')
train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
val_mask = G.nodes['user'].data.pop('val_mask')
val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze()
labels = G.nodes['user'].data.pop('label')
user_emb = torch.load(opt.emb_dir + f'{opt.dataset_name}_hgt_user_emb.pt', map_location=torch.device('cpu'))
user_emb = user_emb[train_idx, :]
labels = labels[train_idx]


# reweight
class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in labels])
samples_weight = torch.from_numpy(samples_weight)
samples_weigth = samples_weight.double()
sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
torch_dataset = torch.utils.data.TensorDataset(user_emb, labels)
dataloader = DataLoader(
    torch_dataset,
    batch_size=opt.batch_size,
    sampler=sampler,
)


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr*2, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


# ----------
#  Training
# ----------
patient = 0
logger.info(f"Now Training Node Generator of FincGAN...")
for epoch in range(opt.n_epochs):
    for i, (imgs, lab) in enumerate(dataloader):
        batches_done = epoch * len(dataloader) + i

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        lab = Variable(lab.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        G_inp = Variable(torch.cat((real_imgs, z), dim = 1))
        # G_inp = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], imgs.shape[1]+opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(G_inp, lab)


        # Loss measures generator's ability to fool the discriminator
        sim_loss = 1*similarity_loss(gen_imgs, real_imgs).clip(min = 0.5)
        g_loss = -torch.mean(discriminator(gen_imgs, lab).clip(-25,25)) + sim_loss


        if batches_done % opt.n_critic_G == 0:
            g_loss.backward()
            optimizer_G.step()


        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        d_loss = -1*torch.mean(discriminator(real_imgs.detach(), lab).clip(-30,30)) + 1*torch.mean(discriminator(gen_imgs.detach(), lab).clip(-20,20)) # Wasserstein loss


        if batches_done % opt.n_critic_D == 0:
            d_loss.backward()
            optimizer_D.step()


        if (batches_done % 51 == 0):
            acc_real = (discriminator(real_imgs.detach(), lab) > 0.5).sum()/real_imgs.size(0)
            acc_fake = (discriminator(gen_imgs.detach(), lab) > 0.5).sum()/gen_imgs.size(0)
            if opt.gan_verbose:

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f] [Sim loss: %.4f] [Acc real: %.4f] [Acc fake: %.4f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), similarity_loss(gen_imgs, real_imgs).item(), acc_real, acc_fake.item())
                )


if not os.path.exists(opt.gan_dir):
        os.makedirs(opt.gan_dir)
        logger.info(f"completed.")


torch.save(discriminator.state_dict(), opt.gan_dir + f"{opt.dataset_name}_D.pt")
torch.save(generator.state_dict(), opt.gan_dir + f"{opt.dataset_name}_G.pt")
logger.info(f"Node Generator of FincGAN, training completed.")
logger.info(f"As {opt.dataset_name}_D.pt, state dictionary of discriminator was successfully saved to: {opt.gan_dir}")
logger.info(f"As {opt.dataset_name}_G.pt, state dictionary of generator was successfully saved to: {opt.gan_dir}\n")


logger.info(f"Visualizing synthetic result by t-SNE")

from sklearn import manifold
import matplotlib.pyplot as plt

generator.cpu()
discriminator.cpu()

num = 5000
indice_normal = np.where(labels == 0)[0]
indice_spam = np.where(labels == 1)[0]

indice1 = random.choices(indice_normal.tolist(), k = int(num/2))
indice2 = random.choices(indice_spam.tolist(), k = int(num/2))
indice = indice1 + indice2

noise = Tensor(np.random.normal(0, 1, (num, opt.latent_dim))).cpu()
z = Variable(torch.cat((user_emb[indice], noise), dim = 1)).cpu()
fake_imgs = generator(z, labels[indice].cpu())
fake_lab = labels[indice]
fake_lab[fake_lab == 0] = 2
fake_lab[fake_lab == 1] = 3


if opt.save_tsne_figure:
    if not os.path.exists(opt.tsne_dir):
                os.makedirs(opt.tsne_dir)
                logger.info(f"completed.")
else:
    logger.info(f"save_tsne_figure was set to False, no figure will be saved.")

data = torch.cat((user_emb[indice].detach().cpu(), fake_imgs), axis = 0)
lab = torch.cat((labels[indice].detach().cpu(), fake_lab), axis = 0)
X_tsne = manifold.TSNE(n_components=2, learning_rate='auto', init='random', random_state=5, verbose=opt.tsne_verbose).fit_transform(data.detach().numpy())
save_tsne_figure = opt.save_tsne_figure

plt.figure()
plt.scatter(x = X_tsne[lab == 0, 0], y = X_tsne[lab == 0, 1], c = 'royalblue', alpha = 0.3, label = 'real_benign', marker = '.')
plt.scatter(x = X_tsne[lab == 1, 0], y = X_tsne[lab == 1, 1], c = '#F39C12', alpha = 0.3, label = 'real_spam', marker = 'x')
plt.scatter(x = X_tsne[lab == 2, 0], y = X_tsne[lab == 2, 1], c = 'darkcyan', alpha = 0.3, label = 'fake_benign', marker = "^")
plt.scatter(x = X_tsne[lab == 3, 0], y = X_tsne[lab == 3, 1], c = 'green', alpha = 0.3, label = 'fake_spam', marker = "P")
plt.xlim(-120,160)
plt.ylim(-120,160)
plt.legend(loc="upper right", ncol=1)
if save_tsne_figure:
    plt.savefig(opt.tsne_dir + 'tsne.jpg')
    logger.info(f"tsne.jpg successfully saved to {opt.tsne_dir}")

plt.figure()
plt.scatter(x = X_tsne[lab == 0, 0], y = X_tsne[lab == 0, 1], c = 'royalblue', alpha = 0.3, label = 'real_benign', marker = '.')
plt.xlim(-120,160)
plt.ylim(-120,160)
plt.legend(loc="upper right", ncol=1)
if save_tsne_figure:
    plt.savefig(opt.tsne_dir + 'tsne_real_benign.jpg')
    logger.info(f"tsne_real_benign.jpg successfully saved to {opt.tsne_dir}")

plt.figure()
plt.scatter(x = X_tsne[lab == 1, 0], y = X_tsne[lab == 1, 1], c = '#F39C12', alpha = 0.3, label = 'real_spam', marker = 'x')
plt.xlim(-120,160)
plt.ylim(-120,160)
plt.legend(loc="upper right", ncol=1)
if save_tsne_figure:
    plt.savefig(opt.tsne_dir + 'tsne_real_spam.jpg')
    logger.info(f"tsne_real_spam.jpg successfully saved to {opt.tsne_dir}")

plt.figure()
plt.scatter(x = X_tsne[lab == 2, 0], y = X_tsne[lab == 2, 1], c = 'darkcyan', alpha = 0.3, label = 'fake_benign', marker = "^")
plt.xlim(-120,160)
plt.ylim(-120,160)
plt.legend(loc="upper right", ncol=1)
if save_tsne_figure:
    plt.savefig(opt.tsne_dir + 'tsne_fake_benign.jpg')
    logger.info(f"tsne_fake_benign.jpg successfully saved to {opt.tsne_dir}")

plt.figure()
plt.scatter(x = X_tsne[lab == 3, 0], y = X_tsne[lab == 3, 1], c = 'green', alpha = 0.3, label = 'fake_spam', marker = "P")
plt.xlim(-120,160)
plt.ylim(-120,160)
plt.legend(loc="upper right", ncol=1)
if save_tsne_figure:
    plt.savefig(opt.tsne_dir + 'tsne_fake_spam.jpg')
    logger.info(f"tsne_fake_spam.jpg successfully saved to {opt.tsne_dir}")


plt.figure()
plt.scatter(x = X_tsne[:5000, 0], y = X_tsne[:5000, 1], c = 'royalblue', alpha = 0.3, label = 'real', marker = '.')
plt.scatter(x = X_tsne[5000:, 0], y = X_tsne[5000:, 1], c = '#F39C12', alpha = 0.3, label = 'fake', marker = 'x')
plt.xlim(-120,160)
plt.ylim(-120,160)
plt.legend(loc="upper right", ncol=1)
if save_tsne_figure:
    plt.savefig(opt.tsne_dir + 'tsne_real_fake.jpg')
    logger.info(f"tsne_real_fake.jpg successfully saved to {opt.tsne_dir}")

logger.info(f"t-SNE visualization completed, program ends.")