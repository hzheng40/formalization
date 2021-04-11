import torch
from torchtext.vocab import GloVe

def on_cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor