import os

from src.params import HParams, TParams


def get_pre_train_sampling_prompts():
    if os.getenv('DEBUG_MODE'):
        return [
            "Test",
            "Test two",
            "Test number three,",
        ]
    else:
        return [
            "HTML stands for",
            "If animals could talk, my pet would probably say",
            "The clever fox built the strange machine with just a feather, a pebble, and a tiny twig",
        ]


def get_llm_config():
    if os.getenv('DEBUG_MODE'):
        return _get_debug_config()
    else:
        return _get_production_config()
            
    
def _get_production_config():
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
        validation_interval = 100,
        validation_steps = 30,
        sampling_interval = 500,
        sampling_batch = 4,
        sampling_tokens = 20,
        sampling_top_k = 50,  # Moderate, fairly default value to start with
    )

    return hParams, tParams


def _get_debug_config():
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
    tot_train_tokens = 600
    batch_token_count = 64
    linear_warm_up_tokens = 2 * 64  # int(1.19e-02 * tot_train_tokens)  # 2 warm-up steps
    linear_warm_up_steps = int(linear_warm_up_tokens / batch_token_count)
    total_training_steps = int(tot_train_tokens / batch_token_count)    # 9 total steps
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
        validation_interval = 5,
        validation_steps = 5,
        sampling_interval = 5,
        sampling_batch = 2,
        sampling_tokens = 3,
        sampling_top_k = 50,
    )
    return hParams, tParams
