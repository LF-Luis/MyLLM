import math

import torch.nn as nn

from src.params import HParams


'''
Utils to help with weight initialization.

Inspiration from the following, and the idea of making the variance of the layers feeding into the
residual path smaller than those that are not.
https://github.com/jzhang38/TinyLlama
https://github.com/Lightning-AI/litgpt
https://arxiv.org/pdf/2204.06745.pdf - GPT-NeoX
'''


def init_embedding(module: nn.Module, hParams: HParams):
    '''
    Initialization for embedding layer.
    '''
    std = math.sqrt(2.0 / 5 / hParams.n_embd)
    nn.init.normal_(module.weight, mean=0.0, std=std)


def init_linear(module, hParams: HParams):
    '''
    Initialization for linear layer (all of which have a bias, except for
    weight sharing one, but that one is not initialize).
    '''
    init_embedding(module, hParams)
    nn.init.zeros_(module.bias)


def init_linear_res_proj(module, hParams: HParams):
    '''
    Initialization for linear layer that are output projections before a residual connection.
    These benefit from lower variance.
    '''
    std = 1 / math.sqrt(hParams.n_embd) / hParams.n_layer
    nn.init.normal_(module.weight, mean=0.0, std=std)
    nn.init.zeros_(module.bias)
