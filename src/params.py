from dataclasses import dataclass


@dataclass
class HParams:
    '''
    Hyperparameters
    '''
    # Lang. processing
    n_vocab: int  # Vocab size
    n_ctx: int  # token context (sequence) len
    # Model arch. 
    n_embd: int  # embedding dimension
    n_head: int  # number of attention heads
    n_layer: int  # number of attention blocks
    # Dropout rates
    # embd_pdrop: float = 0  # Embedding dropout
    # attn_pdrop: float = 0  # Attention dropout
    ffn_act_pdrop: float = 0  # Dropout after FFN activation
    attn_res_pdrop: float = 0  # After attention, in residual connection
    # ffn_res_pdrop: float = 0  # After FFN, in residual connection
    

@dataclass
class TParams:
    '''
    Training parameters
    '''
    tot_steps: int  # Total number of training steps
