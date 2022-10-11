import torch
from torch.optim import Adam

import os.path as osp


def to_numpy(x):
    return x.detach().float().cpu().numpy()


def dict_to_numpy(tensor_dict):
    return {
        k: to_numpy(v) for k, v in tensor_dict.items()
    }


