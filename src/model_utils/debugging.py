import time

import torch
import torch.distributed as dist


def get_stats(weights):
    # min_val = weights.min().item()
    max_val = weights.max().item()
    mean_val = weights.mean().item()
    std_val = weights.std().item()
    var_val = weights.var().item()
    print("Weight Statistics:")
    # print(f"Min: {min_val}")
    print(f"Max: {max_val}")
    print(f"Mean: {mean_val}")
    print(f"Standard Deviation: {std_val}")
    print(f"Variance: {var_val}")


def get_model_size(model):
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    double_counted = sum(p.numel() for p in model.out_proj.parameters())  # Due to weight-tying
    return trainable_param_count - double_counted


def log_training_metrics(
        log, ddp, tParams, train_start, train_end, step, loss, grad_norm, lr
    ):
    '''
    Offload all metric logging to one function.
    If running in a DDP env, this is expected to be called by rank 0 only.
    '''

    if ddp.is_avail:
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)

    perplexity = torch.exp(loss)
    perplexity = f'{perplexity.item():.4f}'
    loss = f'{loss.item():.4f}'
    grad_norm = f'{grad_norm:.4f}'

    time_elapsed = train_end - train_start  # secs
    throughput = f'{(tParams.batch_token_count / time_elapsed):,.2f}'
    time_elapsed = f'{time_elapsed * 1_000:.2f}'  # m.secs

    if ddp.is_main:
        log.info(f"Step {step}: Time: {time_elapsed} ms. LR: {lr:.4e}. Avg. loss: {loss}. Perplexity: {perplexity}. Grad Norm: {grad_norm}. Throughput: {throughput} tokens/sec")
