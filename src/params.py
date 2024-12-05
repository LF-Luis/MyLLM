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
    grad_acc_steps: int  # Gradient accumulation steps (tune to system being used)
    warm_up_steps: int  # Steps in which to perform initial linear LR warm up
    batch_token_count: int  # Number of tokens that make up one global batch
    max_lr: float  # Max learning rate to reach
    min_lr_ratio: float  # % of max_lr reached at the end of cosine decay
    adam_beta_1: float  # Used in AdamW optimizer
    adam_beta_2: float  # Used in AdamW optimizer
    adam_eps: float  # Used in AdamW optimizer
    clip_grad_max_norm: float  # Max value to clip the gradient norm all parameters
    weight_decay_rate: float  # L2 regularization factor to prevent overfitting

    logging_interval: int  # At what intervals to log
    checkpointing_steps: set  # At intervals to create a checkpoint
    validation_interval: int  # Intervals at which to run validation
    validation_steps: int  # Number of validation batches to evaluate for averaging
    sampling_interval: int  # Intervals during training at which to sample from model
    sampling_batch: int  # Number of times to run a single prompt for sampling
    sampling_tokens: int  # Number of tokens to generate when sampling
    sampling_top_k: int  # top-k value for sampling
    eval_interval: int  # Training intervals at which to run HellaSwag evaluation
    eval_hs_subset_key: str  # Validation subset to use from HellaSwag
