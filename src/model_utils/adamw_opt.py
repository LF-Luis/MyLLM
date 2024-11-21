import math
import logging

import torch.nn as nn
import torch.optim as optim

from src.params import TParams
from src.utils.handle_ddp import DDPHandler


log = logging.getLogger(__name__)


class AdamWOptimizer:
    '''
    Opinionated class that will handle all AdamW optimizer actions.
    '''

    def __init__(self, tParams: TParams, ddp: DDPHandler, model: nn.Module):
        self.ddp = ddp
        self.tot_steps = tParams.tot_steps
        self.warm_up_steps = tParams.warm_up_steps
        self.max_lr = tParams.max_lr
        self.min_lr = self.max_lr * tParams.min_lr_ratio

        param_groups = self._get_param_groups(model, tParams.weight_decay_rate)

        self.optimizer = optim.AdamW(
            params = param_groups,
            lr=tParams.max_lr,
            betas=(tParams.adam_beta_1, tParams.adam_beta_2),
            eps=tParams.adam_eps,
            fused=ddp.is_avail,
        )

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, step):
        # Updating with step count instead of token count to make things easier for now.
        # Might implement curriculum learning later on which would change this.

        # Update learning rate
        lr = self._get_scheduled_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # Update weights
        self.optimizer.step()

        return lr  # Only used for debugging

    def _get_scheduled_lr(self, step):
        '''
        Calculate learning-rate given training step. First perform linear warm-up and 
        then cosine decay.
        '''
        if step < self.warm_up_steps:
            # Linear warm-up
            lr = (self.max_lr / self.warm_up_steps) * (step + 1)
        else:
            # Cosine decay
            progress = (step - self.warm_up_steps) / (self.tot_steps - self.warm_up_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        return lr

    def _get_param_groups(self, model, weight_decay):
        '''
        Set up weight decay for learnable parameters. Avoiding bias and norm layers since
        they don't contribute (much) to overfitting and may actually be harmed by regularization.
        '''
        decay = []
        no_decay = []
        decay_name = []
        no_decay_name = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if "bias" in name.lower() or "norm" in name.lower():
                    no_decay.append(param)
                    no_decay_name.append(name)
                else:
                    decay.append(param)
                    decay_name.append(name)

        if self.ddp.is_main:
            log.info(f'{weight_decay} decay params: {decay_name}')
            log.info(f'0.0 no-decay params: {no_decay_name}')

        return [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}
        ]


if __name__ == "__main__":
    import torch
    from src.model import LLM
    from src.params import HParams
    from src.utils.handle_ddp import DDPHandler
    
    ddp = DDPHandler()
    logging.basicConfig(level=logging.DEBUG)

    hParams = HParams(
        n_vocab = 32,
        n_ctx = 16,
        n_embd = 8,
        n_head = 4,
        n_layer = 4,
        ffn_act_pdrop = 0.15,
        attn_res_pdrop = 0.1,
    )

    tParams = TParams(
        tot_steps = 100,
        warm_up_steps = 20,
        batch_token_count = 32,
        max_lr = 0.0021,
        min_lr_ratio = 0.1,
        adam_beta_1 = 0.9, 
        adam_beta_2 = 0.95,
        adam_eps = 1e-8,
        clip_grad_max_norm = 1.0,
        weight_decay_rate = 0.1,
    )

    model = LLM(hParams)
    opt = AdamWOptimizer(tParams, ddp, model)

    x = torch.tensor([
        [2, 3, 3, 1, 3],
        [2, 1, 3, 2, 0],
    ])
    y = torch.tensor([
        [3, 3, 1, 3, 0],
        [1, 3, 2, 0, 1],
    ])

    opt.zero_grad()  # Zero the parameter gradients
    output, loss = model(x, y)  # Forward pass
    loss.backward()  # Backward pass
    opt.step(step=0)  # Update weights

    ddp.end()
