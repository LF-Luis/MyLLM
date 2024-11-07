from dataclasses import dataclass


@dataclass
class HParams:
    '''
    Hyperparameters
    '''
    n_vocab: int  # Vocab size
    n_ctx: int  # token context (sequence) len
    n_embd: int  # embedding dimension
    n_head: int  # number of attention heads
    n_layer: int  # number of attention blocks
    ffn_dropout: int = None  # Dropout rate
    # TODO: add more dropout to help small model stabilize/generalize


@dataclass
class TParams:
    '''
    Training parameters
    '''
    max_steps: int # Max number of training steps
