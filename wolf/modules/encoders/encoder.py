__author__ = 'max'

from typing import Dict
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder base class
    """
    _registry = dict()

    def __init__(self):
        super(Encoder, self).__init__()

    def init(self, x, init_scale=1.0):
        raise NotImplementedError

    @classmethod
    def register(cls, name: str):
        Encoder._registry[name] = cls

    @classmethod
    def by_name(cls, name: str):
        return Encoder._registry[name]

    @classmethod
    def from_params(cls, params: Dict):
        raise NotImplementedError
