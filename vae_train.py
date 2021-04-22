import numpy as np
import math
import random
import torch
import torch.nn.functional as F
import torchtext.vocab
from vae import VAE
from utils import on_cuda
from datasets import get_ptb
import yaml
import os
import argparse
from argparse import Namespace
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/default.yaml')

def str2bool(v):
    if v.lower() == 'true':
        return True
    else:
        return False

parser.add_argument('--resume_training', type=str2bool, default=False)
parser.add_argument('--to_train', type=str2bool, default=True)

args = parser.parse_args()

with open(args.config) as file:
    conf_dict = yaml.load(file, Loader=yaml.FullLoader)
conf = Namespace(**conf_dict)

# seeding
np.random.seed(conf.seed)
torch.manual_seed(conf.seed)
random.seed(conf.seed)

# model save path
save_path = 'data/saved_models/vae_model.pt'
if not os.path.exists('data/saved_models'):
    os.makedirs('data/saved_models')

# TODO cuda visible devices?
# os.environ['CUDA_VISIBLE_DEVICES'] = str(conf.gpu_device)

# create new vae model
def create_vae(conf):
    emb = torchtext.vocab.GloVe(conf.vector, conf.n_embed)
    vae = VAE(conf, emb)
    vae = on_cuda(vae)
    trainer_vae = torch.optim.Adam(vae.parameters(), lr=conf.lr)
    return vae, trainer_vae

# load weights from checkpoint
def load_ckpt(conf, save_path):
    vae, trainer_vae = create_vae(conf)
    checkpoint = torch.load(save_path)
    vae.load_state_dict(checkpoint['vae_dict'])
    trainer_vae.load_state_dict(checkpoint['vae_trainer'])
    return checkpoint['step'], checkpoint['epoch'], vae, trainer_vae

def create_g_input(x, train, vocab, conf):
    # performs random word dropout during training
    # clipping the last word in the sequence
    G_inp = x[:, 0:x.size(1)-1].clone()
    if not train:
        return on_cuda(G_inp)

    # random word dropout
    r = np.random.rand(G_inp.size(0), G_inp.size(1))
    for i in range(len(G_inp)):
        for j in range(1, G_inp.size(1)):
            if r[i, j] < conf.word_dropout and G_inp[i, j] not in [vocab.stoi[conf.pad_token], vocab.stoi[conf.end_token]]:
                G_inp[i, j] = vocab.stoi[conf.unk_token]
    return on_cuda(G_inp)

def train_batch(vae, trainer_vae, x, G_inp, step, conf, train=True):
    logit, _, kld = vae(x, G_inp, None, None)
    # convert into shape (batch_size*(n_seq-1), n_vocab) for ce
    logit = logit.view(-1, conf.n_vocab)
    # target for generator should exclude first word of sequence
    x = x[:, 1:x.size(1)]
    # convert into shape (batch_size*(n_seq-1), 1) for ce
    x = x.contiguous().view(-1)
    rec_loss = F.cross_entropy(logit, x)
    kld_coef = (math.tanh((step - 15000)/1000) + 1) / 2
    loss = conf.rec_coef*rec_loss + kld_coef*kld
    if train:
        trainer_vae.zero_grad()
        loss.backward()
        trainer_vae.step()
    return rec_loss.item(), kld.item()

# training
def train():
    # create model, load weights if necessary
    if args.resume_training:
        step, start_epoch, vae, trainer_vae = load_ckpt(conf, save_path)
    else:
        start_epoch = 0
        step = 0
        vae, trainer_vae = create_vae(conf)

    # data loading
    train_iter, test_iter, valid_iter, vocab = get_ptb(conf)

    # training epochs
    for epoch in tqdm(range(start_epoch, conf.epochs)):
        vae.train()
        # logging
        train_rec_loss = []
        train_kl_loss = []

        for batch in train_iter:
            # batch is encoder input and target ouput for generator
            batch = on_cuda(batch)
            G_inp = create_g_input(batch, True, vocab, conf)
            rec_loss, kl_loss = train_batch(vae, trainer_vae, batch, G_inp, step, conf, train=True)
            train_rec_loss.append(rec_loss)
            train_kl_loss.append(kl_loss)
            step += 1

        # valid
        vae.eval()
        valid_rec_loss = []
        valid_kl_loss = []
        for valid_batch in valid_iter:
            valid_batch = on_cuda(valid_batch)
            G_inp = create_g_input(valid_batch, True, vocab, conf)
            with torch.autograd.no_grad():
                rec_loss, kl_loss = train_batch(vae, trainer_vae, valid_batch, G_inp, step, conf, train=False)
            valid_rec_loss.append(rec_loss)
            valid_kl_loss.append(kl_loss)

        mean_t_rec_loss = np.mean(train_rec_loss)
        mean_t_kl_loss = np.mean(train_kl_loss)
        mean_v_rec_loss = np.mean(valid_rec_loss)
        mean_v_kl_loss = np.mean(valid_kl_loss)

        print('No.', epoch, 'T_rec:', '%.2f'%mean_t_rec_loss,
              'T_kld:', '%.2f'%mean_t_kl_loss, 'V_rec:',
              '%.2f'%mean_v_rec_loss, 'V_kld:', '%.2f'%mean_v_kl_loss)

        if epoch%5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'vae_dict': vae.state_dict(),
                'vae_trainer': trainer_vae.state_dict(),
                'step': step
                }, save_path)


def generate_sentences():
    pass


if __name__ == '__main__':
    if args.to_train:
        train()
    else:
        generate_sentences(50)