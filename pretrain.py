import os
import logging
import math

import tiktoken
import torch

from src.model import LLM
from src.params import HParams, TParams
from src.utils.logger import setup_logging
from src.utils.handle_ddp import DDPHandler
from src.model_utils.adamw_opt import AdamWOptimizer
from src.model_utils.debugging import get_model_size


'''
Main script to train language model.
'''

tokenizer = tiktoken.get_encoding("r50k_base")

# Looking at common trends from assets/some_open_source_models.png to help select n_ctx, n_head and n_layer
hParams = HParams(
    n_vocab = 50_257,
    n_ctx = 2_048,
    n_embd = 1_024,
    n_head = 16,
    n_layer = 16,
    ffn_act_pdrop = 0.15,  # Slightly larger dropout due to larger hidden space
    attn_res_pdrop = 0.1,
)

# See `notebooks/parameters_tuning.ipynb` to see how I came up with my guessed 1.19e-02 ratio and max_lr = 0.0021
tot_train_tokens = 10e9  # Training on 10BT
batch_token_count = 524_288
linear_warm_up_tokens = int(1.19e-02 * tot_train_tokens)
linear_warm_up_steps = int(linear_warm_up_tokens / batch_token_count)
total_training_steps = int(tot_train_tokens / batch_token_count)

tParams = TParams(
    tot_steps = total_training_steps,
    warm_up_steps = linear_warm_up_steps,
    batch_token_count = batch_token_count,
    max_lr = 0.0021,
    min_lr_ratio = 0.1,
    adam_beta_1 = 0.9, 
    adam_beta_2 = 0.95,
    adam_eps = 1e-8,
    clip_grad_max_norm = 1.0,
    weight_decay_rate = 0.1,
)


if __name__ == "__main__":
    ddp = DDPHandler()
    setup_logging()
    log = logging.getLogger(__name__)

    model = LLM(hParams)

    if ddp.is_main:
        print(f'Model size: {math.ceil(get_model_size(model) / 1_000_000)}M')
        print(f'hParams: {hParams}')
        print(f'tParams: {tParams}')

    opt = AdamWOptimizer(tParams, ddp, model)
    opt.zero_grad()
    opt.step(step=0)

    ddp.end()
    