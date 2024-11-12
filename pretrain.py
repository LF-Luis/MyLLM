import os
import logging

import tiktoken
import torch
from torch.distributed import init_process_group, destroy_process_group

from src.params import HParams, TParams
from src.utils.logger import setup_logging
from src.utils.handle_ddp import HandleDDP

'''
Main script to train language model.
Let's pretrain a small (~500M params) model and then finetune it for general knowledge Q&A.
'''

tokenizer = tiktoken.get_encoding("r50k_base")

hParams = HParams(
    n_vocab = 50257,  # Vocab size of r50k_base
    n_ctx = 2048,  # context size
    n_embd = 768,  # embedding dimension
    n_head = 12,  # number of attention heads
    n_layer = 12,  # number of attention blocks
    ffn_act_pdrop = 0.15,  # Dropout after FFN activation
    attn_res_pdrop = 0.1,  # After attention, in residual connection
)
 
tParams = TParams(
    tot_steps = 19073  # Total number of training steps
)


if __name__ == "__main__":

    setup_logging()
    log = logging.getLogger(__name__)

    ddp = HandleDDP()

    ddp.end()
    