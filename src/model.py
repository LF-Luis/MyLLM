from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F

from src.params import HParams
from src.transformer_block import TransformerBlock
from src.model_utils.weight_init import init_embedding


class LLM(nn.Module):

    def __init__(self, hParams: HParams):
        '''
        Standard LLM structure, borrowing from GPT-2/3 and newer Llama models
        (also seen in models like MolmoE 1B).
        '''
        super().__init__()
        self.hParams = hParams
        self.embd = nn.Embedding(hParams.n_vocab, hParams.n_embd)
        # Avoiding dropout here for now since it may lead to information loss (e.g. it 
        # would mess with RoPE), and affect training stability since it's so early in the network.
        # self.embd_dropout = nn.Dropout(hParams.embd_pdrop)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(hParams) for _ in range(hParams.n_layer)]
        )
        self.norm = nn.RMSNorm(hParams.n_embd, eps=1e-5)
        self.out_proj = nn.Linear(hParams.n_embd, hParams.n_vocab, bias=False)
        self.embd.weight = self.out_proj.weight
        self.reset_parameters(hParams)

    def reset_parameters(self, hParams: HParams):
        init_embedding(self.embd, hParams)
        
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
        # x = self.embd_dropout(x)
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
        ffn_act_pdrop = 0.1,
    )
    
    expected_output = torch.tensor([
        [[ 0.0643,  0.1504,  1.5457, -0.3968],
         [ 0.2599,  0.5008,  0.1130,  1.2352],
         [ 0.1778,  0.3794,  0.0348,  1.2232],
         [-0.1258,  1.1608, -0.3776,  0.7196],
         [ 0.0292,  0.2619, -0.3227,  1.1662]],
        [[ 0.0333,  0.1185,  1.5608, -0.3510],
         [-0.0135,  1.3279, -0.2422,  0.6294],
         [ 0.1876,  0.3131, -0.0958,  1.2011],
         [-0.0848,  0.2016,  1.4463, -0.3105],
         [ 0.1845,  0.6925, -0.4936, -0.5954]]
    ])
    
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
    expected_loss = 1.576537
    output = torch.round(output * 10000) / 10000
    loss = loss.item()
    # print(f'loss: {(loss)}')

    if round(loss, 6) == expected_loss:
        print('Got expected loss!')
    else:
        print(f'Error, {round(loss, 5)} != {expected_loss}')
    