import nevergrad as ng
import numpy as np
import ray
import torch
from torch.distributions import MultivariateNormal
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

from vae import VAE
from style_transfer import LinearShift
from utils import on_cuda
from datasets import get_ptb, get_gyafc, str_to_tensor, get_formality_set
from evals import FeatureExtractor

import yaml
import argparse
import pickle
from argparse import Namespace
from tqdm import tqdm

@ray.remote(num_gpus=1)
class Worker:
    """
    Ray remote worker that:
        1. gets the parameter sampled by optimizer
        2. creates the network based on sampled param
        3. passes input latent through network
        4. decodes output then get score
    """
    def __init__(self, conf):
        # create vae, load weights
        _, _, _, self.vocab = get_gyafc(conf)
        self.vae, _ = create_vae(conf, self.vocab)
        ckpt = torch.load(conf.vae_model_path)
        self.vae.load_state_dict(ckpt['vae_dict'])
        self.vae.eval()
        del(ckpt)

        # create linear shift
        self.linear_shift = on_cuda(LinearShift(conf))

        # save conf
        self.conf = conf
        # init
        self.score = 0
        self.eval_done = False

        # load dataset
        self.test = get_formality_set(conf, self.vocab)

        # scoring
        self.extractor = FeatureExtractor(conf.w2v_path, conf.corpus_dict_path)
        self.pt16_ridge = pickle.load(open(conf.pt16_path, 'rb'))

    def eval(self, work):
        # evaluates quality of given parameters
        # copy weights to linear shift
        mu_weight = work['mu_weight']
        mu_bias = work['mu_bias']
        var_weight = work['var_weight']
        var_bias = work['var_bias']

        with torch.no_grad():
            self.linear_shift.linear_mu[0].weight.copy_(torch.from_numpy(mu_weight).float())
            self.linear_shift.linear_mu[0].bias.copy_(torch.from_numpy(mu_bias).float())
            self.linear_shift.linear_logvar[0].weight.copy_(torch.from_numpy(var_weight).float())
            self.linear_shift.linear_logvar[0].bias.copy_(torch.from_numpy(var_bias).float())

        batch_scores = []

        for batch in self.test:
            print('New Batch')
            current_batch_scores = []
            current_batch_strings = []
            batch = on_cuda(batch.T)
            # encode batch to mu and logvars
            mu, logvar = self.vae.encode(batch)

            # put mu and logvars pass linear shift
            new_mu, new_logvar = self.linear_shift(mu, logvar)

            # loop through each batch
            for i in range(new_mu.size()[0]):
                # create distribution
                mvn = MultivariateNormal(new_mu[i, :], scale_tril=torch.diag(torch.exp(new_logvar[i, :])))

                # sample and decode
                z = mvn.sample().unsqueeze(0)

                h_0 = on_cuda(torch.zeros(self.conf.n_layers_G, 1, self.conf.n_hidden_G))
                c_0 = on_cuda(torch.zeros(self.conf.n_layers_G, 1, self.conf.n_hidden_G))
                G_hidden = (h_0, c_0)
                G_inp = torch.LongTensor(1, 1).fill_(self.vocab.stoi[self.conf.start_token])
                G_inp = on_cuda(G_inp)
                string = ''
                length = 0
                while G_inp[0][0].item() != self.vocab.stoi[self.conf.end_token]:
                    with torch.autograd.no_grad():
                        logit, G_hidden, _ = self.vae(None, G_inp, z, G_hidden)
                    probs = F.softmax(logit[0], dim=1)
                    G_inp = torch.multinomial(probs, 1)
                    if G_inp[0][0].item() != self.vocab.stoi[self.conf.end_token]:
                        string += self.vocab.itos[G_inp[0][0].item()] + ' '
                        length += 1
                    if length >= 20:
                        break
                current_batch_strings.append(string)

            print('Decode on current batch done, scoring now')
            # score on strings
            for i, sent in enumerate(current_batch_strings):
                # PT16 formality
                pt16 = self.get_pt16_score(sent)
                # bleu with original
                # TODO: how to get orignal sentence?
                # bleu = self.get_bleu_with_orig(?, sent)
                # current_batch_scores.append(self.conf.pt16_weight*pt16 + self.conf.bleu_weight*bleu)
                current_batch_scores.append(pt16)

            print('Current batch average score:', np.mean(current_batch_scores))
            batch_scores.append(np.mean(current_batch_scores))

        # TODO: process all scores to a single score?
        # score = 0 # TODO
        score = -np.mean(batch_scores)
        self.score = score
        self.eval_done = True

    def get_pt16_score(self, s):
        # Returns the pt16 formality score on a sentence
        feature = self.extractor.extract_annotations(s)
        parse_tree = self.extractor.extract_parse(s)
        feature_vec = self.extractor.extract_features_pt16(s, feature, parse_tree)
        return self.pt16_ridge.predict(feature_vec)

    def get_bleu_with_orig(self, orig, new):
        orig = word_tokenize(orig)
        new = word_tokenize(new)
        return sentence_bleu(orig, new)

    def collect(self):
        # collect function for ray
        while not self.eval_done:
            continue
        return self.score

# create new vae model
def create_vae(conf, vocab):
    vae = VAE(conf)
    vae.embedding.weight.data.copy_(vocab.vectors)
    vae = on_cuda(vae)
    trainer_vae = torch.optim.Adam(vae.parameters(), lr=conf.lr)
    return vae, trainer_vae

def search(conf):
    """
    Function that creates cma-es, and starts the search
    """
    # seeding
    np.random.seed(conf.seed)

    # number of concurrent workers
    num_workers = conf.num_workers

    # parameterization
    # uniform_scale = np.sqrt(1/conf.n_z)
    # param = ng.p.Dict(
    #     mu_weight=ng.p.Array(init=np.random.uniform(low=-uniform_scale, high=uniform_scale, size=(conf.n_z, conf.n_z))),
    #     mu_bias=ng.p.Array(init=np.random.uniform(low=-uniform_scale, high=uniform_scale, size=conf.n_z)),
    #     var_weight=ng.p.Array(init=np.random.uniform(low=-uniform_scale, high=uniform_scale, size=(conf.n_z, conf.n_z))),
    #     var_bias=ng.p.Array(init=np.random.uniform(low=-uniform_scale, high=uniform_scale, size=conf.n_z)))

    # need to start from no change
    param = ng.p.Dict(
        mu_weight=ng.p.Array(init=np.ones((conf.n_z, conf.n_z))),
        mu_bias=ng.p.Array(init=np.zeros((conf.n_z))),
        var_weight=ng.p.Array(init=np.ones((conf.n_z, conf.n_z))),
        var_bias=ng.p.Array(init=np.zeros((conf.n_z))))

    # optimizer
    optim = ng.optimizers.registry['CMA'](parametrization=param, budget=conf.budget, num_workers=num_workers)
    # seeding
    optim.parametrization.random_state = np.random.RandomState(conf.seed)

    # setting up workers
    workers = [Worker.remote(conf) for _ in range(num_workers)]

    # book keeping
    all_scores = []
    all_individuals = []

    # work distribution loop
    for _ in tqdm(range(conf.budget//num_workers)):
        individuals = [optim.ask() for _ in range(num_workers)]
        results = []

        # distribute
        for ind, worker in zip(individuals, workers):
            worker.eval.remote(ind.args[0])

        # collect
        future_results = [worker.collect.remote() for worker in workers]
        results = ray.get(future_results)

        # update optimization
        for ind, score in zip(individuals, results):
            optim.tell(ind, score)

        # collect all
        all_scores.extend(results)
        all_individuals.extend(individuals)

        # book keeping
        # TODO: change what to save
        optim.dump(conf.optim_filename)
        np.savez_compressed(conf.npz_filename,
                            scores=np.array(all_scores),
                            individuals=all_individuals)

    return optim, all_scores, all_individuals

if __name__ == '__main__':
    with open('configs/default.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    print(conf)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    ray.init()
    optim, all_scores, all_individuals = search(conf)
    best_params = optim.recommend()
    best_linear_shift = on_cuda(LinearShift(conf))
    mu_weight = best_params['mu_weight']
    mu_bias = best_params['mu_bias']
    var_weight = best_params['var_weight']
    var_bias = best_params['var_bias']

    with torch.no_grad():
        best_linear_shift.linear_mu[0].weight.copy_(torch.from_numpy(mu_weight.value).float())
        best_linear_shift.linear_mu[0].bias.copy_(torch.from_numpy(mu_bias.value).float())
        best_linear_shift.linear_logvar[0].weight.copy_(torch.from_numpy(var_weight.value).float())
        best_linear_shift.linear_logvar[0].bias.copy_(torch.from_numpy(var_bias.value).float())

    torch.save(best_linear_shift.state_dict(), conf.linear_model_save_path)

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_context('poster')
plt.plot([-num for num in all_scores])
plt.xlabel('Iterations')
plt.ylabel('Score (Higher is more formal)')
plt.show()
