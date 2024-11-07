from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F

from src.params import HParams
from src.transformer_block import TransformerBlock


class LLM(nn.Module):

    def __init__(self, hParams: HParams):
        '''
        Standard LLM structure, borrowing from GPT-2/3 and newer Llama models
        (also seen in models like MolmoE 1B).
        '''
        super().__init__()
        self.hParams = hParams
        self.embd = nn.Embedding(hParams.n_vocab, hParams.n_embd)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(hParams) for _ in range(hParams.n_layer)]
        )
        self.norm = nn.RMSNorm(hParams.n_embd, eps=1e-5)
        self.out_proj = nn.Linear(hParams.n_embd, hParams.n_vocab, bias=False)
        self.embd.weight = self.out_proj.weight
        
    def forward(
            self, x: torch.Tensor, y: torch.Tensor = None
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        batch_size, n_ctx = x.size()

        assert n_ctx <= self.hParams.n_ctx, f"Input context length {n_ctx} exceeds maximum {self.hParams.n_ctx}"

        '''
        Create high-dimensional embedding of input, pass through the transformer blocks,
        apply efficient post-normalization, and lastly project into logits using weight
        sharing of the last layer.
        ''' 
        x = self.embd(x)        
        x = self.transformer_blocks(x)
        x = self.norm(x)
        logits = self.out_proj(x)

        loss = None
        if y is not None:
            # Get loss if expected target value y is provided
            # Reordering logits and y to work with cross_entropy
            tot_tokens = batch_size * n_ctx
            loss = F.cross_entropy(
                logits.view(tot_tokens, -1),
                y.view(tot_tokens)
            )

        return logits, loss


if __name__ == '__main__':
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)

    x = torch.tensor([
        [2, 3, 3, 1, 3],  # mock input sequence
        [2, 1, 3, 2, 0],
    ])
    y = torch.tensor([
        [3, 3, 1, 3, 0],  # mock next tokens
        [1, 3, 2, 0, 1],
    ])
    batch_size, n_ctx = x.shape
    
    hParams = HParams(
        n_vocab = torch.max(x) + 1,
        n_ctx = n_ctx,
        n_embd = 4,
        n_head = 2,
        n_layer = 2,
        ffn_dropout = 0.1,
    )
    
    expected_output = torch.tensor([[[-0.3571,  0.2121,  0.9920, -0.3854],
         [-0.5856,  0.5658,  0.8923,  0.0022],
         [-0.6292,  0.6958,  0.9037, -0.0424],
         [-0.6026,  0.6961,  0.9856, -0.2327],
         [-0.6149,  0.6599,  0.8561,  0.0293]],

        [[-0.3460,  0.2279,  0.9943, -0.4344],
         [-0.4709,  0.4108,  0.9912, -0.2740],
         [-0.5231,  0.4391,  0.8254,  0.0707],
         [-0.4790,  0.3604,  1.0687, -0.3408],
         [-0.2865,  0.1606,  0.8860, -0.3446]]])
    
    model = LLM(hParams)
    output, _ = model(x)
    output = torch.round(output * 10000) / 10000
    # print(f'output: {output}')
    
    if torch.equal(output, expected_output):
        print('Got expected output!')
    else:
        not_equal = output != expected_output
        different_indices = not_equal.nonzero(as_tuple=True)
        for idx in zip(*different_indices):
            print(f"Diff at index {idx}: output = {output[idx]}, expected_output = {expected_output[idx]}")

    output, loss = model(x, y)
    expected_loss = 1.740107
    output = torch.round(output * 10000) / 10000
    loss = loss.item()
    # print(f'loss: {(loss)}')

    if round(loss, 6) == expected_loss:
        print('Got expected loss!')
    else:
        f'Error, {round(loss, 5)} != {expected_loss}'
    