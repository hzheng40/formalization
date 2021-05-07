import torch
import torch.nn.functional as F
import numpy as np
import yaml
from torch.distributions import MultivariateNormal
from vae import VAE
from style_transfer import LinearShift
from utils import on_cuda
from datasets import str_to_tensor, get_formality_set, get_gyafc, get_informal_test_set
from argparse import Namespace

def create_vae(conf, vocab):
    vae = VAE(conf)
    vae.embedding.weight.data.copy_(vocab.vectors)
    vae = on_cuda(vae)
    trainer_vae = torch.optim.Adam(vae.parameters(), lr=conf.lr)
    return vae, trainer_vae

if __name__ == '__main__':
    with open('configs/default.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    print(conf)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)

    best_linear_shift = on_cuda(LinearShift(conf))
    linear_ckpt = torch.load(conf.linear_model_save_path)
    best_linear_shift.load_state_dict(linear_ckpt)
    best_linear_shift.eval()

    _, _, _, vocab = get_gyafc(conf)
    ckpt = torch.load(conf.vae_model_path)
    vae, _ = create_vae(conf, vocab)
    vae.load_state_dict(ckpt['vae_dict'])
    vae.eval()
    del ckpt, linear_ckpt

    test = get_informal_test_set(conf, vocab)

    all_strings = []
    for batch in test:
        print('New Batch')
        batch = on_cuda(batch.T)
        mu, logvar = vae.encode(batch)
        new_mu, new_logvar = best_linear_shift(mu, logvar)

        for i in range(new_mu.size()[0]):
            mvn = MultivariateNormal(new_mu[i, :], scale_tril=torch.diag(torch.exp(new_logvar[i, :])))
            z = mvn.sample().unsqueeze(0)

            h_0 = on_cuda(torch.zeros(conf.n_layers_G, 1, conf.n_hidden_G))
            c_0 = on_cuda(torch.zeros(conf.n_layers_G, 1, conf.n_hidden_G))
            G_hidden = (h_0, c_0)
            G_inp = torch.LongTensor(1, 1).fill_(vocab.stoi[conf.start_token])
            G_inp = on_cuda(G_inp)
            string = ''
            length = 0
            while G_inp[0][0].item() != vocab.stoi[conf.end_token]:
                with torch.autograd.no_grad():
                    logit, G_hidden, _ = vae(None, G_inp, z, G_hidden)
                probs = F.softmax(logit[0], dim=1)
                G_inp = torch.multinomial(probs, 1)
                if G_inp[0][0].item() != vocab.stoi[conf.end_token]:
                    string += vocab.itos[G_inp[0][0].item()] + ' '
                    length += 1
                if length >= 20:
                    break

            all_strings.append(string)

    with open('data/test_output.ref0', 'w') as file:
        for st in all_strings:
            file.write('%s\n' % st)