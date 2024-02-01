from functools import cache

import torch


@cache
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def to_device(model):
    return model.to(get_device())
