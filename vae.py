import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import on_cuda

"""
Implementation of:
    Generating Sentences from a Continuous Space
    Bowman et al. 2016
"""

# highway layer
class Highway(nn.Module):
    def __init__(self, config):
        super(Highway, self).__init__()
        self.n_layers = config.n_highway_layers
        self.non_linear = nn.ModuleList([nn.Linear(config.n_embed, config.n_embed) for _ in range(self.n_layers)])
        self.linear = nn.ModuleList([nn.Linear(config.n_embed, config.n_embed) for _ in range(self.n_layers)])
        self.gate = nn.ModuleList([nn.Linear(config.n_embed, config.n_embed) for _ in range(self.n_layers)])

    def forward(self, x):
        for layer in range(self.n_layers):
            # compute percentage of non linear information to be allowed for each element in x
            gate = torch.sigmoid(self.gate[layer](x))
            # compute nonlinear info
            non_linear = F.relu(self.non_linear[layer](x))
            # compute linear info
            linear = self.linear[layer](x)
            # combine nonlinear and linear info based on gate
            x = gate*non_linear + (1-gate)*linear
        return x

# encoder
class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.highway = Highway(config)
        self.n_hidden_E = config.n_hidden_E
        self.n_layers_E = config.n_layers_E
        self.lstm = nn.LSTM(input_size=config.n_embed,
                            hidden_size=config.n_hidden_E,
                            num_layers=config.n_layers_E,
                            batch_first=True,
                            bidirectional=True)

    def init_hidden(self, batch_size):
        h_0 = torch.zeros(2*self.n_layers_E, batch_size, self.n_hidden_E)
        c_0 = torch.zeros(2*self.n_layers_E, batch_size, self.n_hidden_E)
        self.hidden = (on_cuda(h_0), on_cuda(c_0))

    def forward(self, x):
        batch_size, n_seq, n_embed = x.size()
        # highway pass
        x = self.highway(x)
        self.init_hidden(batch_size)
        # exclude c_T and extract only h_T
        _, (self.hidden, _) = self.lstm(x, self.hidden)
        self.hidden = self.hidden.view(self.n_layers_E, 2, batch_size, self.n_hidden_E)
        # select final layer
        self.hidden = self.hidden[-1]
        # merge hidden states of both directions
        # TODO: size check
        e_hidden = torch.cat(list(self.hidden), dim=1)
        return e_hidden

# decoder
class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.n_hidden_G = config.n_hidden_G
        self.n_layers_G = config.n_layers_G
        self.n_z = config.n_z
        self.lstm = nn.LSTM(input_size=config.n_embed+config.n_z,
                            hidden_size=config.n_hidden_G,
                            num_layers=config.n_layers_G,
                            batch_first=True)
        self.fc = nn.Linear(config.n_hidden_G, config.n_vocab)

    def init_hidden(self, batch_size):
        h_0 = torch.zeros(self.n_layers_G, batch_size, self.n_hidden_G)
        c_0 = torch.zeros(self.n_layers_G, batch_size, self.n_hidden_G)
        self.hidden = (on_cuda(h_0), on_cuda(c_0))

    def forward(self, x, z, g_hidden=None):
        batch_size, n_seq, n_embed = x.size()
        # replicate z in order to append same z at each time step
        z = torch.cat([z]*n_seq, 1).view(batch_size, n_seq, self.n_z)
        # append z to generator word input at each time step
        x = torch.cat([x, z], dim=2)

        # validating
        if g_hidden is None:
            self.init_hidden(batch_size)
        # training
        else:
            self.hidden = g_hidden

        # get top layer of h_T at each time step, produce logit vec of vocab words
        output, self.hidden = self.lstm(x, self.hidden)
        output = self.fc(output)

        # return complete (h_T, c_T) incase if we are testing
        return output, self.hidden

# full vae
class VAE(nn.Module):
    # def __init__(self, config, pretrained_embedding):
    def __init__(self, config):
        super(VAE, self).__init__()
        # self.embedding = nn.Embedding.from_pretrained(pretrained_embedding.vectors)
        self.embedding = nn.Embedding(config.n_vocab, config.n_embed)
        self.encoder = Encoder(config)
        self.hidden_to_mu = nn.Linear(2*config.n_hidden_E, config.n_z)
        self.hidden_to_logvar = nn.Linear(2*config.n_hidden_E, config.n_z)
        self.generator = Generator(config)
        self.n_z = config.n_z

    def forward(self, x, G_inp, z=None, G_hidden=None):
        # if testing with z sampled from random noise
        if z is None:
            batch_size, n_seq = x.size()
            # produce embedding from encoder input
            x = self.embedding(x)
            # h_T of encoder
            E_hidden = self.encoder(x)
            # mean of latent z
            mu = self.hidden_to_mu(E_hidden)
            # log variance of latent z
            logvar = self.hidden_to_logvar(E_hidden)
            # noise sampled from Normal(0, 1)
            z = on_cuda(torch.randn([batch_size, self.n_z]))
            # reparam trick: sample z = mu + eps*sigma for back prop
            z = mu + z*torch.exp(0.5*logvar)
            # KL-divergence loss
            # kld = -0.5*torch.sum(logvar-mu.pow(2)-logvar.exp()+1, 1).mean()
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            # training with given text
            kld = None

        # embeddings for generator input
        G_inp = self.embedding(G_inp)
        logit, G_hidden = self.generator(G_inp, z, G_hidden)
        return logit, G_hidden, kld

    def encode(self, x):
        # returns the latent distribution given text
        batch_size, n_seq = x.size()
        x = self.embedding(x)
        E_hidden = self.encoder(x)
        mu = self.hidden_to_mu(E_hidden)
        logvar = self.hidden_to_logvar(E_hidden)
        # z = on_cuda(torch.randn([x.size()[0], self.n_z]))
        # z = mu + z * torch.exp(0.5*logvar)
        z = mu
        # return mu, logvar
        return z