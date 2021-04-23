import numpy as np
import math
import random
import torch
import torch.nn.functional as F
import torchtext.vocab
from vae import VAE
from utils import on_cuda
from datasets import get_wiki2, get_ptb, str_to_tensor
import yaml
import os
import argparse
from argparse import Namespace
import tqdm

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
print(conf)
print(args)
# seeding
np.random.seed(conf.seed)
torch.manual_seed(conf.seed)
random.seed(conf.seed)

# model save path
# NOTE: model path
# save_path = 'data/saved_models/vae_model_wiki2.pt'
# save_path = 'data/saved_models/vae_model_ptb.pt'
save_path = 'data/saved_models/vae_model_ptb_embfix_nodropout.pt'
if not os.path.exists('data/saved_models'):
    os.makedirs('data/saved_models')
if not os.path.exists('data/losses_log'):
    os.makedirs('data/losses_log')

# create new vae model
def create_vae(conf, vocab):
    # emb = torchtext.vocab.GloVe(conf.vector, conf.n_embed)
    # vae = VAE(conf, emb)
    vae = VAE(conf)
    vae.embedding.weight.data.copy_(vocab.vectors)
    vae = on_cuda(vae)
    trainer_vae = torch.optim.Adam(vae.parameters(), lr=conf.lr)
    return vae, trainer_vae

# load weights from checkpoint
def load_ckpt(conf, save_path, vocab):
    vae, trainer_vae = create_vae(conf, vocab)
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
    # data loading
    # train_iter, test_iter, valid_iter, vocab = get_wiki2(conf)
    train_iter, test_iter, valid_iter, vocab = get_ptb(conf)

    # create model, load weights if necessary
    if args.resume_training:
        step, start_epoch, vae, trainer_vae = load_ckpt(conf, save_path, vocab)
    else:
        start_epoch = 0
        step = 0
        vae, trainer_vae = create_vae(conf, vocab)

    all_t_rec_loss = []
    all_t_kl_loss = []
    all_v_rec_loss = []
    all_v_kl_loss = []

    # progress bar
    epoch_loop = tqdm.tqdm(total=conf.epochs - start_epoch, desc='Epochs', position=0)
    # loss_log = tqdm.tqdm(total=0, position=3, bar_format='{desc}')

    # training epochs
    for epoch in range(start_epoch, conf.epochs):
        vae.train()
        # logging
        train_rec_loss = []
        train_kl_loss = []

        # progress bar
        train_loop = tqdm.tqdm(total=len(train_iter), desc='Train', position=1)

        for batch in train_iter:
            # batch is encoder input and target ouput for generator
            batch = on_cuda(batch.T)
            G_inp = create_g_input(batch, True, vocab, conf)
            rec_loss, kl_loss = train_batch(vae, trainer_vae, batch, G_inp, step, conf, train=True)
            train_rec_loss.append(rec_loss)
            train_kl_loss.append(kl_loss)
            step += 1
            train_loop.update(1)

        # valid
        vae.eval()
        valid_rec_loss = []
        valid_kl_loss = []

        # progress bar
        valid_loop = tqdm.tqdm(total=len(valid_iter), desc='Valid', position=2)

        for valid_batch in valid_iter:
            valid_batch = on_cuda(valid_batch.T)
            G_inp = create_g_input(valid_batch, True, vocab, conf)
            with torch.autograd.no_grad():
                rec_loss, kl_loss = train_batch(vae, trainer_vae, valid_batch, G_inp, step, conf, train=False)
            valid_rec_loss.append(rec_loss)
            valid_kl_loss.append(kl_loss)
            valid_loop.update(1)

        all_t_rec_loss.append(train_rec_loss)
        all_t_kl_loss.append(train_kl_loss)
        all_v_rec_loss.append(valid_rec_loss)
        all_v_kl_loss.append(valid_kl_loss)
        mean_t_rec_loss = np.mean(train_rec_loss)
        mean_t_kl_loss = np.mean(train_kl_loss)
        mean_v_rec_loss = np.mean(valid_rec_loss)
        mean_v_kl_loss = np.mean(valid_kl_loss)

        # loss_log.set_description_str(f'T_rec: ' + '%.2f'%mean_t_rec_loss +
        #     ' T_kld: ' + '%.2f'%mean_t_kl_loss + ' V_rec: ' +
        #     '%.2f'%mean_v_rec_loss + ' V_kld: ' + '%.2f'%mean_v_kl_loss)
        tqdm.tqdm.write(f'T_rec: ' + '%.2f'%mean_t_rec_loss +
                ' T_kld: ' + '%.2f'%mean_t_kl_loss + ' V_rec: ' +
                '%.2f'%mean_v_rec_loss + ' V_kld: ' + '%.2f'%mean_v_kl_loss)

        if epoch%5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'vae_dict': vae.state_dict(),
                'vae_trainer': trainer_vae.state_dict(),
                'step': step
                }, save_path)

            # NOTE: npz path, still messed up, overwrites with the latest 5 when resume training
            # np.savez_compressed('data/losses_log/losses_wiki2_fixed.npz',
            #                     t_rec=np.array(all_t_rec_loss),
            #                     t_kl=np.array(all_t_kl_loss),
            #                     v_rec=np.array(all_v_rec_loss),
            #                     v_kl=np.array(all_v_kl_loss))

            np.savez_compressed('data/losses_log/losses_ptb_embfix_nodropout.npz',
                                t_rec=np.array(all_t_rec_loss),
                                t_kl=np.array(all_t_kl_loss),
                                v_rec=np.array(all_v_rec_loss),
                                v_kl=np.array(all_v_kl_loss))

        epoch_loop.update(1)


def generate_sentences(n_examples):
    # NOTE: vocab based on datasets
    train_iter, test_iter, valid_iter, vocab = get_ptb(conf)

    ckpt = torch.load(save_path)
    vae, vae_trainer = create_vae(conf, vocab)
    vae.load_state_dict(ckpt['vae_dict'])
    vae.eval()
    del ckpt

    for i in range(n_examples):
        z = on_cuda(torch.randn([1, conf.n_z]))
        h_0 = on_cuda(torch.zeros(conf.n_layers_G, 1, conf.n_hidden_G))
        c_0 = on_cuda(torch.zeros(conf.n_layers_G, 1, conf.n_hidden_G))
        G_hidden = (h_0, c_0)
        # 2 is the index of start token in vocab stoi
        G_inp = torch.LongTensor(1, 1).fill_(vocab.stoi[conf.start_token])
        G_inp = on_cuda(G_inp)
        string = conf.start_token + ' '
        # until we hit end token (index 3 in vocab stoi)
        while G_inp[0][0].item() != vocab.stoi[conf.end_token]:
            with torch.autograd.no_grad():
                logit, G_hidden, _ = vae(None, G_inp, z, G_hidden)
            probs = F.softmax(logit[0], dim=1)
            G_inp = torch.multinomial(probs, 1)
            string += (vocab.itos[G_inp[0][0].item()] + ' ')
        # print(string.encode('utf-8'))
        print(string)

def interpolate_existing_sentences(s1, s2, num=10):
    # NOTE: vocab based on datasets
    train_iter, test_iter, valid_iter, vocab = get_ptb(conf)

    ckpt = torch.load(save_path)
    vae, vae_trainer = create_vae(conf, vocab)
    vae.load_state_dict(ckpt['vae_dict'])
    vae.eval()
    del ckpt

    # string to tensor
    s1_tensor = str_to_tensor(s1, vocab, conf)
    s2_tensor = str_to_tensor(s2, vocab, conf)
    s1_tensor = on_cuda(s1_tensor.unsqueeze(0))
    s2_tensor = on_cuda(s2_tensor.unsqueeze(0))
    z1 = vae.encode(s1_tensor)
    z2 = vae.encode(s2_tensor)

    # interpolate
    int_z = torch.lerp(z1, z2, on_cuda(torch.linspace(0.0, 1.0, num).unsqueeze(1)))

    # z to strings
    for i in range(int_z.size()[0]):
        z = int_z[i, :].unsqueeze(0)
        h_0 = on_cuda(torch.zeros(conf.n_layers_G, 1, conf.n_hidden_G))
        c_0 = on_cuda(torch.zeros(conf.n_layers_G, 1, conf.n_hidden_G))
        G_hidden = (h_0, c_0)
        G_inp = torch.LongTensor(1, 1).fill_(vocab.stoi[conf.start_token])
        G_inp = on_cuda(G_inp)
        string = conf.start_token + ' '
        while G_inp[0][0].item() != vocab.stoi[conf.end_token]:
            with torch.autograd.no_grad():
                logit, G_hidden, _ = vae(None, G_inp, z, G_hidden)
            probs = F.softmax(logit[0], dim=1)
            G_inp = torch.multinomial(probs, 1)
            string += (vocab.itos[G_inp[0][0].item()] + ' ')
        print('----------------------------')
        print(string.encode('utf-8'))

def sampling_around_existing_sentence(s1, num=10):
    # NOTE: vocab based on datasets
    train_iter, test_iter, valid_iter, vocab = get_ptb(conf)

    ckpt = torch.load(save_path)
    vae, vae_trainer = create_vae(conf, vocab)
    vae.load_state_dict(ckpt['vae_dict'])
    vae.eval()
    del ckpt

    # string to tensor
    s1_tensor = str_to_tensor(s1, vocab, conf)
    s1_tensor = on_cuda(s1_tensor.unsqueeze(0))

    for i in range(num):
        z = vae.encode(s1_tensor)
        h_0 = on_cuda(torch.zeros(conf.n_layers_G, 1, conf.n_hidden_G))
        c_0 = on_cuda(torch.zeros(conf.n_layers_G, 1, conf.n_hidden_G))
        G_hidden = (h_0, c_0)
        G_inp = torch.LongTensor(1, 1).fill_(vocab.stoi[conf.start_token])
        G_inp = on_cuda(G_inp)
        string = conf.start_token + ' '
        while G_inp[0][0].item() != vocab.stoi[conf.end_token]:
            with torch.autograd.no_grad():
                logit, G_hidden, _ = vae(None, G_inp, z, G_hidden)
            probs = F.softmax(logit[0], dim=1)
            G_inp = torch.multinomial(probs, 1)
            string += (vocab.itos[G_inp[0][0].item()] + ' ')
        print('----------------------------')
        print(string.encode('utf-8'))

def interpolate_sentences(num=10):
    # NOTE: vocab based on datasets
    train_iter, test_iter, valid_iter, vocab = get_ptb(conf)

    ckpt = torch.load(save_path)
    vae, vae_trainer = create_vae(conf, vocab)
    vae.load_state_dict(ckpt['vae_dict'])
    vae.eval()
    del ckpt

    z1 = on_cuda(torch.randn([1, conf.n_z]))
    z2 = on_cuda(torch.randn([1, conf.n_z]))

    int_z = torch.lerp(z1, z2, on_cuda(torch.linspace(0.0, 1.0, num).unsqueeze(1)))
    # zs to strings
    for i in range(int_z.size()[0]):
        z = int_z[i, :].unsqueeze(0)
        h_0 = on_cuda(torch.zeros(conf.n_layers_G, 1, conf.n_hidden_G))
        c_0 = on_cuda(torch.zeros(conf.n_layers_G, 1, conf.n_hidden_G))
        G_hidden = (h_0, c_0)
        G_inp = torch.LongTensor(1, 1).fill_(vocab.stoi[conf.start_token])
        G_inp = on_cuda(G_inp)
        string = conf.start_token + ' '
        while G_inp[0][0].item() != vocab.stoi[conf.end_token]:
            with torch.autograd.no_grad():
                logit, G_hidden, _ = vae(None, G_inp, z, G_hidden)
            probs = F.softmax(logit[0], dim=1)
            G_inp = torch.multinomial(probs, 1)
            string += (vocab.itos[G_inp[0][0].item()] + ' ')
        print('----------------------------')
        print(string.encode('utf-8'))


if __name__ == '__main__':
    if args.to_train:
        train()
    else:
        # np.random.seed(999)
        # torch.manual_seed(999)
        # random.seed(999)
        print('---------Generating---------')
        print('----------------------------')
        generate_sentences(10)
        print('----------Sampling----------')
        print('----------------------------')
        print('------Original Sentence-----')
        s = 'i went to the kitchen'
        print(s)
        print('----------------------------')
        sampling_around_existing_sentence(s, num=3)
        print('-------Interpolating--------')
        print('----------------------------')
        interpolate_sentences(num=5)