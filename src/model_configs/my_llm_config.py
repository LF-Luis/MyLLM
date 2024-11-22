import os

from src.params import HParams, TParams


def get_llm_config():
    if os.getenv('DEBUG_MODE'):
        return get_debug_config()
    else:
        return get_production_config()
            
    
def get_production_config():
    '''
    Actual model and training values!!
    '''

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
        grad_acc_steps = 2,
        warm_up_steps = linear_warm_up_steps,
        batch_token_count = batch_token_count,
        max_lr = 0.0021,
        min_lr_ratio = 0.1,
        adam_beta_1 = 0.9, 
        adam_beta_2 = 0.95,
        adam_eps = 1e-8,
        clip_grad_max_norm = 1.0,
        weight_decay_rate = 0.1,
        logging_interval = 50,
        checkpointing_steps = set(
            list(
                range(0, total_training_steps, int(total_training_steps * 0.2))
            )[:-1]  # Excluding last since it's very near the actual last step
        ),
    )

    return hParams, tParams


def get_debug_config():
    '''
    Debug model and training values!!
    '''
    
    hParams = HParams(
        n_vocab = 50257,
        n_ctx = 8,
        n_embd = 4,
        n_head = 2,
        n_layer = 1,
        ffn_act_pdrop = 0.15,
        attn_res_pdrop = 0.1,
    )
    tot_train_tokens = 60_000
    batch_token_count = 64
    linear_warm_up_tokens = int(1.19e-02 * tot_train_tokens)
    linear_warm_up_steps = int(linear_warm_up_tokens / batch_token_count)
    total_training_steps = int(tot_train_tokens / batch_token_count)
    tParams = TParams(
        tot_steps = total_training_steps,
        grad_acc_steps = 2,
        warm_up_steps = linear_warm_up_steps,
        batch_token_count = batch_token_count,
        max_lr = 0.0021,
        min_lr_ratio = 0.1,
        adam_beta_1 = 0.9, 
        adam_beta_2 = 0.95,
        adam_eps = 1e-8,
        clip_grad_max_norm = 1.0,
        weight_decay_rate = 0.1,
        logging_interval = 1,
        checkpointing_steps = {int(total_training_steps / 2)},
    )
    return hParams, tParams
