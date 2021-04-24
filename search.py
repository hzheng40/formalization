import nevergrad as ng
import numpy as np
import ray
import torch

from vae import VAE
from vae_train import create_vae
from style_transfer import LinearShift
from utils import on_cuda
from datasets import get_ptb

import argparse
from argparse import Namespace
from tqdm import tqdm

@ray.remote
class Worker:
    """
    Ray remote worker that:
        1. gets the parameter sampled by optimizer
        2. creates the network based on sampled param
        3. passes input latent through network
        4. decodes output then get score
    """
    def __init__(self, conf):
        # create vae
        _, _, _, self.vocab = get_ptb(conf)
        self.vae = create_vae(conf, self.vocab)
        # create linear shift
        self.linear_shift = LinearShift(conf)
        # save conf
        self.conf = conf
        # init
        self.score = 0
        self.eval_done = False

    def eval(self, work):
        # evaluates quality of given parameters
        pass

    def collect(self):
        # collect function for ray
        while not self.eval_done:
            continue
        return self.score


def search(conf):
    """
    Function that creates cma-es, and starts the search
    """
    # seeding
    np.random.seed(conf.seed)

    # number of concurrent workers
    num_workers = conf.num_workers

    # parameterization
    # TODO: init should follow torch.nn.Linear
    # should both be from Uniform(-sqrt(k), sqrt(k)), where k = 1/features
    param = ng.p.Dict(
        weight=ng.p.Array(init=np.zeros(conf.n_z)),
        bias=ng.p.Array(init=np.zeros(conf.n_z)))
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
                        weights=None,
                        bias=None)